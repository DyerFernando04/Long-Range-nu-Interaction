#!/usr/bin/env python3
"""analysis/reweight_mc.py
===========================================
Interpolacion de pesos MC evento-a-evento para las LUT de templates HDF5 v3.1.

Entrada:
    results/templates/{standard,Le-Lmu,Le-Ltau,Lmu-Ltau}.h5
    /event_weights/nominal [mass, coupling, event]
    /status/completed_nodes [mass, coupling]
    /axes/log10_mass_eV, /axes/log10_abs_coupling

LRI:
    La interpolacion es BILINEAL en (log10(m/eV), log10(|acoplamiento|))
    sobre los PESOS por evento. El signo del acoplamiento es fijo y se
    comprueba expresamente (importante para Lmu-Ltau y coupling_kind='bigg').

    w_i(m,c) = sum_{a,b=0,1} lambda_ab(m,c) * w_i(m_a,c_b)

    Los histogramas y sumw2 se calculan DESPUES de interpolar cada peso,
    usando analysis/systematics.py: sigma_b^2 = sum_{i en b} w_i^2.
    Nunca interpolar directamente /sumw2 ni los histogramas.

STANDARD:
    El modelo standard tiene ejes ficticios [0] x [0], que NO son masas
    ni acoplamientos fisicos. Se lee solo el nodo [0,0].

La interpolacion en parametros fisicos es una aproximacion numerica de la LUT,
NO equivale a una nueva propagacion nuSQuIDS. No se extrapola. No se ajustan
priors ni se evalua likelihood (corresponde a FrequentistAnalysis).

Se conserva la convencion de unidades/normalizacion ya incorporada a los
pesos nominales por build_templates.py; aqui NO se agrega otro factor 1e4.

Ejemplos, desde la raiz del repositorio:
    python3 analysis/reweight_mc.py standard
    python3 analysis/reweight_mc.py Le-Lmu
    python3 analysis/reweight_mc.py Le-Lmu --mass 1e-14 --coupling 1e-25
    python3 analysis/reweight_mc.py Le-Lmu --node 5 10
    python3 analysis/reweight_mc.py Lmu-Ltau --log10-mass -14 --log10-abs-coupling -25
    python3 analysis/reweight_mc.py Le-Lmu --check
    python3 analysis/reweight_mc.py Le-Lmu --param neutrino_norm=1.05

Para importar:
    from analysis.reweight_mc import MCReweighter

    with MCReweighter('results/templates/Le-Lmu.h5') as rw:
        weights = rw.weights(mass_eV=1e-14, coupling=1e-25)
        prediction = rw.predict(mass_eV=1e-14, coupling=1e-25,
                                params={'neutrino_norm': 1.05})
        mu, sigma2 = prediction.total_mu, prediction.total_sigma2

Dependencias: numpy, h5py, analysis/systematics.py.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

try:
    from .systematics import (
        AXIAL_CONVENTIONS, CATEGORIES, NuisanceParameters, Prediction,
        TemplateSystematics,
    )
except ImportError:  # python3 analysis/reweight_mc.py (ejecucion directa)
    from systematics import (  # type: ignore[no-redef]
        AXIAL_CONVENTIONS, CATEGORIES, NuisanceParameters, Prediction,
        TemplateSystematics,
    )

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / 'results' / 'templates'
MODELS = ('standard', 'Le-Lmu', 'Le-Ltau', 'Lmu-Ltau')


@dataclass(frozen=True)
class InterpolationCell:
    """Punto solicitado y nodos de soporte con sus coeficientes."""

    log10_mass_eV: float | None
    log10_abs_coupling: float | None
    mass_eV: float | None
    coupling: float | None
    mass_lower: int
    coupling_lower: int
    mass_fraction: float
    coupling_fraction: float
    corners: tuple[tuple[int, int, float], ...]
    is_exact_node: bool

    @property
    def weight_sum(self) -> float:
        return float(sum(weight for _, _, weight in self.corners))


class MCReweighter:
    """Interpolador perezoso: solo lee los 1-4 nodos de pesos necesarios.

    Evita cargar una LUT completa de 41 x 61 x ~397k eventos (~3.7 GiB
    crudos). Almacena hasta `cache_nodes` vectores de un solo nodo en un LRU.
    Para likelihoods repetidos, mantiene `TemplateSystematics` en memoria
    desde la primera llamada a `predict` hasta `close`.
    """

    def __init__(self, filename: str | Path, *, cache_nodes: int = 8,
                 axial_mass_convention: str = 'unconfirmed') -> None:
        self.path = Path(filename).expanduser().resolve()
        if cache_nodes < 0:
            raise ValueError('cache_nodes debe ser >= 0')
        if axial_mass_convention not in AXIAL_CONVENTIONS:
            raise ValueError(f'Convencion axial desconocida: {axial_mass_convention}')
        self.cache_nodes = int(cache_nodes)
        self.axial_mass_convention = axial_mass_convention
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._systematics: TemplateSystematics | None = None
        self._closed = False
        self.h5 = h5py.File(self.path, 'r')
        try:
            self._load_metadata()
        except BaseException:
            self.h5.close()
            raise

    def __enter__(self) -> 'MCReweighter':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            if self._systematics is not None:
                self._systematics.close()
                self._systematics = None
            self.h5.close()
            self._cache.clear()
            self._closed = True

    def _load_metadata(self) -> None:
        h5 = self.h5
        if str(h5.attrs.get('format_version', '')) != '3.1':
            raise ValueError(f'Formato de templates HDF5 3.1 requerido: {self.path}')
        if not bool(h5.attrs.get('build_completed', False)):
            raise RuntimeError('El constructor no ha terminado: build_completed=False')
        required = (
            'axes/mass_eV', 'axes/log10_mass_eV',
            'axes/coupling_values', 'axes/log10_abs_coupling',
            'event_weights/nominal', 'status/completed_nodes',
        )
        for name in required:
            if name not in h5:
                raise KeyError(f'Falta /{name} en {self.path}')
        self.model = str(h5.attrs.get('model', ''))
        self.coupling_kind = str(h5.attrs.get('coupling_kind', ''))
        if self.model not in MODELS:
            raise ValueError(f'Modelo desconocido: {self.model}')
        self.is_standard = (self.model == 'standard')
        self.mass_eV_axis = np.asarray(h5['axes/mass_eV'][:], dtype=np.float64)
        self.log10_mass_axis = np.asarray(h5['axes/log10_mass_eV'][:], dtype=np.float64)
        self.coupling_axis = np.asarray(h5['axes/coupling_values'][:], dtype=np.float64)
        self.log10_abs_coupling_axis = np.asarray(
            h5['axes/log10_abs_coupling'][:], dtype=np.float64
        )
        self.ds = h5['event_weights/nominal']
        if self.ds.ndim != 3:
            raise ValueError('/event_weights/nominal debe tener shape [mass,coupling,event]')
        self.n_mass, self.n_coupling, self.n_event = self.ds.shape
        self.done = np.asarray(h5['status/completed_nodes'][:], dtype=bool)
        if self.done.shape != (self.n_mass, self.n_coupling):
            raise ValueError('/status/completed_nodes no coincide con la malla de pesos')
        if any(axis.size != dim for axis, dim in (
            (self.mass_eV_axis, self.n_mass),
            (self.log10_mass_axis, self.n_mass),
            (self.coupling_axis, self.n_coupling),
            (self.log10_abs_coupling_axis, self.n_coupling),
        )):
            raise ValueError('Las dimensiones de los ejes no coinciden con /event_weights/nominal')
        if self.n_event < 1:
            raise ValueError('No hay eventos MC')
        if not np.all(np.isfinite(self.mass_eV_axis)) or not np.all(np.isfinite(self.coupling_axis)):
            raise ValueError('Ejes con elementos no finitos')
        if self.is_standard:
            if self.coupling_kind != 'none' or (self.n_mass, self.n_coupling) != (1, 1):
                raise ValueError('Standard debe tener ejes ficticios [1,1] y coupling_kind=none')
            if np.any(self.mass_eV_axis != 0) or np.any(self.coupling_axis != 0):
                raise ValueError('Standard debe guardar ejes ficticios [0,0]')
            self.coupling_sign = 0
        else:
            if self.coupling_kind not in ('gprime', 'bigg'):
                raise ValueError('Los modelos LRI deben indicar coupling_kind=gprime o bigg')
            if np.any(self.mass_eV_axis <= 0) or np.any(self.coupling_axis == 0):
                raise ValueError('Las masas deben ser positivas y los acoplamientos no nulos')
            self.coupling_sign = 1 if self.coupling_axis[0] > 0 else -1
            if np.any(np.sign(self.coupling_axis) != self.coupling_sign):
                raise ValueError('No se permite interpolar entre signos de acoplamiento diferentes')
            if self.coupling_kind == 'gprime' and self.coupling_sign != 1:
                raise ValueError('gprime debe ser positivo')
            for label, axis in (
                ('log10_mass_eV', self.log10_mass_axis),
                ('log10_abs_coupling', self.log10_abs_coupling_axis),
            ):
                if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                    raise ValueError(f'Eje {label} no finito o no creciente')
            if not np.allclose(self.log10_mass_axis, np.log10(self.mass_eV_axis),
                               rtol=0, atol=1e-10):
                raise ValueError('log10_mass_eV no coincide con mass_eV')
            if not np.allclose(self.log10_abs_coupling_axis, np.log10(np.abs(self.coupling_axis)),
                               rtol=0, atol=1e-10):
                raise ValueError('log10_abs_coupling no coincide con coupling_values')

    @staticmethod
    def _coordinate(axis: np.ndarray, value: float, name: str) -> tuple[int, float]:
        """Devuelve intervalo inferior y coordenada normalizada, sin extrapolar."""
        if not np.isfinite(value):
            raise ValueError(f'{name} debe ser finito')
        if axis.size == 1:
            if not np.isclose(value, axis[0], rtol=0.0, atol=1e-11):
                raise ValueError(f'{name}={value:g}: esta LUT solo permite {axis[0]:g}')
            return 0, 0.0
        tolerance = 1e-11
        if value < axis[0] - tolerance or value > axis[-1] + tolerance:
            raise ValueError(f'{name}={value:.12g} fuera de la malla '
                             f'[{axis[0]:.12g}, {axis[-1]:.12g}]; sin extrapolacion')
        value = float(np.clip(value, axis[0], axis[-1]))
        pos = int(np.searchsorted(axis, value, side='right'))
        lo = min(max(0, pos - 1), axis.size - 2)
        frac = (value - float(axis[lo])) / (float(axis[lo + 1]) - float(axis[lo]))
        if np.isclose(frac, 0.0, atol=5e-12, rtol=0.0):
            frac = 0.0
        elif np.isclose(frac, 1.0, atol=5e-12, rtol=0.0):
            frac = 1.0
        return lo, float(frac)

    def cell(self, *, mass_eV: float | None = None,
             coupling: float | None = None,
             log10_mass_eV: float | None = None,
             log10_abs_coupling: float | None = None) -> InterpolationCell:
        """Define coordenadas por pares físicos o logaritmos decimales.

        Se permiten `mass_eV` y `log10_mass_eV`, y análogamente `coupling`
        y `log10_abs_coupling`, pero nunca ambos para un mismo eje.
        Para bigg negativo, la forma log usa el signo de toda la LUT.
        """
        if self.is_standard:
            if any(v is not None for v in
                   (mass_eV, coupling, log10_mass_eV, log10_abs_coupling)):
                raise ValueError('standard no posee masa ni acoplamiento LRI')
            if not self.done[0, 0]:
                raise RuntimeError('Standard aún no está completo')
            return InterpolationCell(
                log10_mass_eV=None, log10_abs_coupling=None,
                mass_eV=None, coupling=None,
                mass_lower=0, coupling_lower=0,
                mass_fraction=0.0, coupling_fraction=0.0,
                corners=((0, 0, 1.0),), is_exact_node=True,
            )
        if (mass_eV is not None) and (log10_mass_eV is not None):
            raise ValueError('Usa mass_eV O log10_mass_eV, no ambos')
        if (coupling is not None) and (log10_abs_coupling is not None):
            raise ValueError('Usa coupling O log10_abs_coupling, no ambos')
        if (mass_eV is None) and (log10_mass_eV is None):
            raise ValueError('Se requiere mass_eV o log10_mass_eV para LRI')
        if (coupling is None) and (log10_abs_coupling is None):
            raise ValueError('Se requiere coupling o log10_abs_coupling para LRI')
        if mass_eV is not None:
            if not np.isfinite(mass_eV) or mass_eV <= 0:
                raise ValueError('mass_eV debe ser positivo y finito')
            lm = float(np.log10(mass_eV))
        else:
            lm = float(log10_mass_eV)
        if coupling is not None:
            if not np.isfinite(coupling) or coupling == 0:
                raise ValueError('coupling debe ser no nulo y finito')
            if np.sign(coupling) != self.coupling_sign:
                raise ValueError(f'El acoplamiento debe tener signo {self.coupling_sign:+d} '
                                 '(no se mezclan ramas de signo)')
            lc = float(np.log10(abs(coupling)))
        else:
            lc = float(log10_abs_coupling)
        mi, mt = self._coordinate(self.log10_mass_axis, lm, 'log10(m/eV)')
        ci, ct = self._coordinate(self.log10_abs_coupling_axis, lc, 'log10|coupling|')
        ms = ((mi, 1.0-mt), (mi+1, mt)) if self.n_mass > 1 else ((0, 1.0),)
        cs = ((ci, 1.0-ct), (ci+1, ct)) if self.n_coupling > 1 else ((0, 1.0),)
        corners = tuple((i, j, float(a*b)) for i, a in ms for j, b in cs if a*b > 0)
        if not corners or abs(sum(x[2] for x in corners) - 1.0) > 1e-12:
            raise RuntimeError('Coeficientes de interpolacion inconsistentes')
        for i, j, _ in corners:
            if not self.done[i, j]:
                raise RuntimeError(f'Nodo requerido ({i},{j}) no completado. '
                                   'No se construyen predicciones parciales')
        return InterpolationCell(
            log10_mass_eV=lm, log10_abs_coupling=lc,
            mass_eV=float(10**lm), coupling=float(self.coupling_sign * 10**lc),
            mass_lower=mi, coupling_lower=ci,
            mass_fraction=mt, coupling_fraction=ct,
            corners=corners, is_exact_node=len(corners) == 1,
        )

    def node_weights(self, mass_index: int, coupling_index: int) -> np.ndarray:
        """Devuelve pesos nominales de un nodo preciso (sin aplicar nuisances)."""
        if self._closed:
            raise RuntimeError('MCReweighter ya se ha cerrado')
        if (not 0 <= mass_index < self.n_mass or
                not 0 <= coupling_index < self.n_coupling):
            raise IndexError('Indices mass,coupling fuera de la malla')
        if not self.done[mass_index, coupling_index]:
            raise RuntimeError(f'Nodo ({mass_index},{coupling_index}) no completado')
        key = (mass_index, coupling_index)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        weights = np.asarray(self.ds[mass_index, coupling_index, :], dtype=np.float64)
        if weights.shape != (self.n_event,) or not np.all(np.isfinite(weights)):
            raise FloatingPointError(f'Pesos invalidos en nodo ({mass_index},{coupling_index})')
        if np.any(weights < 0):
            raise ValueError(f'Pesos negativos en nodo ({mass_index},{coupling_index})')
        if self.cache_nodes:
            self._cache[key] = weights
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_nodes:
                self._cache.popitem(last=False)
        return weights.copy()

    def weights(self, *, mass_eV: float | None = None,
                coupling: float | None = None,
                log10_mass_eV: float | None = None,
                log10_abs_coupling: float | None = None) -> np.ndarray:
        """Interpola vectores completos, no histogramas ni sus varianzas."""
        cell = self.cell(mass_eV=mass_eV, coupling=coupling,
                         log10_mass_eV=log10_mass_eV,
                         log10_abs_coupling=log10_abs_coupling)
        if cell.is_exact_node:
            i, j, _ = cell.corners[0]
            return self.node_weights(i, j)
        result = np.zeros(self.n_event, dtype=np.float64)
        for i, j, alpha in cell.corners:
            result += alpha * self.node_weights(i, j)
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError('Pesos interpolados no finitos o negativos')
        return result

    @property
    def systematics(self) -> TemplateSystematics:
        """Carga una vez las hypersurfaces, cache de eventos y fondos."""
        if self._closed:
            raise RuntimeError('MCReweighter ya se ha cerrado')
        if self._systematics is None:
            self._systematics = TemplateSystematics(
                self.path, axial_mass_convention=self.axial_mass_convention
            )
            if self._systematics.n_event != self.n_event or (
                self._systematics.n_mass, self._systematics.n_coupling
            ) != (self.n_mass, self.n_coupling):
                self._systematics.close()
                self._systematics = None
                raise ValueError('systematics.py y reweight_mc.py no ven la misma LUT')
        return self._systematics

    def predict(self, *, mass_eV: float | None = None,
                coupling: float | None = None,
                log10_mass_eV: float | None = None,
                log10_abs_coupling: float | None = None,
                params: NuisanceParameters | Mapping[str, float] | None = None) -> Prediction:
        """Interpolar -> aplicar nuisances -> histogramar -> recalcular sumw2."""
        weights = self.weights(mass_eV=mass_eV, coupling=coupling,
                               log10_mass_eV=log10_mass_eV,
                               log10_abs_coupling=log10_abs_coupling)
        return self.systematics.predict_from_weights(weights, params=params)

    def predict_node(self, mass_index: int = 0, coupling_index: int = 0, *,
                     params: NuisanceParameters | Mapping[str, float] | None = None) -> Prediction:
        return self.systematics.predict_from_weights(
            self.node_weights(mass_index, coupling_index), params=params
        )

    def validate_nodes(self, *, max_nodes: int = 9, rtol: float = 1e-11,
                       atol: float = 1e-9) -> list[tuple[int, int]]:
        """Prueba nodos representativos y reproduccion nominal de templates.

        Solo lee nodos en corners/centro; no recorre la malla completa.
        Usa el modulo systematics.py para reconstruir /templates y /sumw2.
        """
        if max_nodes < 1:
            raise ValueError('max_nodes debe ser positivo')
        indices_m = sorted({0, self.n_mass // 2, self.n_mass - 1})
        indices_c = sorted({0, self.n_coupling // 2, self.n_coupling - 1})
        nodes = [(i, j) for i in indices_m for j in indices_c][:max_nodes]
        for i, j in nodes:
            if not self.done[i, j]:
                raise RuntimeError(f'Nodo de validacion ({i},{j}) incompleto')
            if self.is_standard:
                cell = self.cell()
                a = self.weights()
            else:
                cell = self.cell(log10_mass_eV=float(self.log10_mass_axis[i]),
                                 log10_abs_coupling=float(self.log10_abs_coupling_axis[j]))
                a = self.weights(log10_mass_eV=float(self.log10_mass_axis[i]),
                                 log10_abs_coupling=float(self.log10_abs_coupling_axis[j]))
            if not cell.is_exact_node:
                raise AssertionError(f'El nodo ({i},{j}) no se reconocio como exacto')
            b = self.node_weights(i, j)
            if not np.array_equal(a, b):
                raise AssertionError(f'Interpolacion no reproduce exactamente nodo ({i},{j})')
            self.systematics.check_nominal_node(i, j, rtol=rtol, atol=atol)
        return nodes

    def validate_interpolation(self) -> None:
        """Prueba interna de bilinealidad en una celda inicial completa."""
        if self.is_standard or self.n_mass < 2 or self.n_coupling < 2:
            return
        valid = np.argwhere(self.done[:-1, :-1] & self.done[1:, :-1] &
                            self.done[:-1, 1:] & self.done[1:, 1:])
        if not len(valid):
            raise RuntimeError('No hay ninguna celda de 4 nodos completados para probar')
        i, j = map(int, valid[0])
        mid_m = 0.5 * (self.log10_mass_axis[i] + self.log10_mass_axis[i+1])
        mid_c = 0.5 * (self.log10_abs_coupling_axis[j] + self.log10_abs_coupling_axis[j+1])
        cell = self.cell(log10_mass_eV=float(mid_m),
                         log10_abs_coupling=float(mid_c))
        expected = (self.node_weights(i, j) + self.node_weights(i+1, j) +
                    self.node_weights(i, j+1) + self.node_weights(i+1, j+1)) / 4.0
        actual = self.weights(log10_mass_eV=float(mid_m),
                              log10_abs_coupling=float(mid_c))
        if len(cell.corners) != 4 or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError('La interpolacion bilineal del punto medio no coincide')


def _parse_params(entries: list[str], parser: argparse.ArgumentParser) -> dict[str, float]:
    params = {}
    for item in entries:
        if '=' not in item:
            parser.error(f'--param requiere NOMBRE=VALOR, recibido: {item}')
        name, value = item.split('=', 1)
        try:
            params[name] = float(value)
        except ValueError:
            parser.error(f'Valor no numerico para {name}: {value}')
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Interpola pesos MC de LUTs HDF5 v3.1 y evalua predicciones.'
    )
    parser.add_argument('model', nargs='?', default='standard', choices=MODELS)
    parser.add_argument('--templates', type=Path, default=None,
                        help='HDF5 de entrada en vez de results/templates/MODEL.h5')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--node', nargs=2, type=int, metavar=('MASS_INDEX', 'COUPLING_INDEX'),
                       help='Evalua un nodo exacto; no interpola')
    group.add_argument('--mass', type=float, default=None,
                       help='Masa fisica m en eV (requiere --coupling o --log10-abs-coupling)')
    group.add_argument('--log10-mass', type=float, default=None,
                       help='log10(m/eV) (requiere --coupling o --log10-abs-coupling)')
    parser.add_argument('--coupling', type=float, default=None,
                        help='Acoplamiento fisico gprime o G_muta con signo')
    parser.add_argument('--log10-abs-coupling', type=float, default=None,
                        help='log10|gprime o G_muta|; se usa el signo del archivo')
    parser.add_argument('--param', action='append', default=[], metavar='NOMBRE=VALOR',
                        help='Nuisance de systematics.py (puede repetirse)')
    parser.add_argument('--check', action='store_true',
                        help='Prueba nodos nominales, sumw2 y bilinealidad (sin fit)')
    parser.add_argument('--axial-convention', choices=AXIAL_CONVENTIONS,
                        default='unconfirmed')
    parser.add_argument('--cache-nodes', type=int, default=8)
    args = parser.parse_args()
    if args.node is not None and (args.coupling is not None or
                                  args.log10_abs_coupling is not None):
        parser.error('--node no se combina con --coupling/--log10-abs-coupling')
    if args.coupling is not None and args.log10_abs_coupling is not None:
        parser.error('Selecciona --coupling O --log10-abs-coupling')
    path = args.templates or (TEMPLATE_DIR / f'{args.model}.h5')
    changes = _parse_params(args.param, parser)
    with MCReweighter(path, cache_nodes=args.cache_nodes,
                      axial_mass_convention=args.axial_convention) as reader:
        if reader.model != args.model:
            parser.error(f'El HDF5 pertenece al modelo {reader.model}, no {args.model}')
        print('=' * 74)
        print('ANALYSIS / REWEIGHT MC -- EVENT LEVEL HDF5 3.1')
        print('=' * 74)
        print(f'Archivo                 : {reader.path}')
        print(f'Modelo / coupling_kind  : {reader.model} / {reader.coupling_kind}')
        print(f'Malla [mass,coupling]   : {reader.n_mass} x {reader.n_coupling}')
        print(f'Eventos MC              : {reader.n_event:,}')
        print(f'Checkpoints terminados  : {int(reader.done.sum())}/{reader.done.size}')
        if args.check:
            nodes = reader.validate_nodes()
            reader.validate_interpolation()
            print(f'[OK] nodos exactos, templates y sumw2: {nodes}')
            if not reader.is_standard:
                print('[OK] interpolacion bilineal de cuatro esquinas')
        if args.node is not None:
            i, j = args.node
            info = f'nodo exacto ({i},{j})'
            prediction = reader.predict_node(i, j, params=changes)
        elif reader.is_standard:
            if any(x is not None for x in
                   (args.mass, args.log10_mass, args.coupling, args.log10_abs_coupling)):
                parser.error('standard no acepta coordenadas de masa/acoplamiento')
            info = 'standard (nodo unico 0,0)'
            prediction = reader.predict(params=changes)
        else:
            if args.mass is not None:
                m_kw = {'mass_eV': args.mass}
            elif args.log10_mass is not None:
                m_kw = {'log10_mass_eV': args.log10_mass}
            else:
                m_kw = {'log10_mass_eV': float(np.mean([
                    reader.log10_mass_axis[0], reader.log10_mass_axis[-1]
                ]))}
            if args.coupling is not None:
                c_kw = {'coupling': args.coupling}
            elif args.log10_abs_coupling is not None:
                c_kw = {'log10_abs_coupling': args.log10_abs_coupling}
            else:
                c_kw = {'log10_abs_coupling': float(np.mean([
                    reader.log10_abs_coupling_axis[0],
                    reader.log10_abs_coupling_axis[-1]
                ]))}
            cell = reader.cell(**m_kw, **c_kw)
            info = (f'm={cell.mass_eV:.6e} eV; '
                    f'{reader.coupling_kind}={cell.coupling:.6e}; '
                    f'esquinas={len(cell.corners)}; '
                    f'fracciones=({cell.mass_fraction:.6f},{cell.coupling_fraction:.6f})')
            prediction = reader.predict(**m_kw, **c_kw, params=changes)
        print(f'Punto                   : {info}')
        if changes:
            print(f'Nuisances modificados   : {changes}')
        else:
            print('Nuisances               : nominales')
        print('-' * 74)
        for cat in CATEGORIES:
            print(f'{cat:22s} : {float(prediction.category_mu[cat].sum()):14.6f}')
        print(f'Neutrinos               : {float(prediction.neutrino_mu.sum()):14.6f}')
        print(f'Muones                  : {float(prediction.muon_mu.sum()):14.6f}')
        print(f'Total predicho          : {float(prediction.total_mu.sum()):14.6f}')
        print(f'Datos observados        : {float(prediction.observed.sum()):14.6f}')
        print(f'Predicho / observado    : {float(prediction.total_mu.sum()/prediction.observed.sum()):14.6f}')
        print(f'Suma varianza MC/KDE    : {float(prediction.total_sigma2.sum()):14.6f}')
        print('NOTA: Interpolacion en parametros LRI, sin nueva propagacion ni fit.')


if __name__ == '__main__':
    main()
