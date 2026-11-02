#!/usr/bin/env python3
"""
analysis/systematics.py
=======================

Reponderación evento-a-evento y predicción nominal/sistemática para los HDF5
formato 3.1 creados por analysis/build_templates.py.

Este módulo NO ajusta parámetros, NO calcula likelihood y NO modifica HDF5.
La función predict_from_weights permite a FrequentistAnalysis pasarle pesos
interpolados en la malla (masa, acoplamiento) y construir sumw2 exactamente
DESPUÉS de interpolar los pesos individuales.

Parámetros reconocidos (11):
  dom_eff, hole_ice_p0, hole_ice_p1, bulk_ice_abs, bulk_ice_scatter,
  delta_gamma_nu, ma_ccqe, ma_ccres, nc_cc_ratio,
  neutrino_norm, muon_norm.

ADVERTENCIA SOBRE MASAS AXIALES:
  El release contiene MaCCQE_linear/quad y MaCCRES_linear/quad, pero aún no
  verificamos la variable de expansión con la documentación de IceCube/GENIE.
  Con axial_mass_convention='unconfirmed' (por defecto) sólo pueden usarse
  los valores nominales ma_ccqe=0.99 GeV y ma_ccres=1.12 GeV.
  Las tres convenciones alternativas están disponibles SÓLO por selección
  explícita, para estudiar la hipótesis; no se consideran validadas.

Los seis nuisances Barr, DIS CSMS, theta23 y dm31 no se implementan aquí.
No se aplican gaussian priors: eso corresponde a FrequentistAnalysis, para
no contabilizarlos dos veces.

Ejecución desde la raíz:
  python3 analysis/systematics.py standard
  python3 analysis/systematics.py standard --param neutrino_norm=1.05
  python3 analysis/systematics.py Le-Lmu --mass-index 0 --coupling-index 0

Dependencias: numpy, h5py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "results" / "templates"
CATEGORIES = ("nu_nc", "nue_cc", "numu_cc", "nutau_cc")
DETECTOR_NAMES = (
    "dom_eff", "hole_ice_p0", "hole_ice_p1", "bulk_ice_abs", "bulk_ice_scatter"
)
PARAMETER_NAMES = (
    *DETECTOR_NAMES,
    "delta_gamma_nu", "ma_ccqe", "ma_ccres", "nc_cc_ratio",
    "neutrino_norm", "muon_norm",
)
MA_CCQE_NOMINAL_GEV = 0.99
MA_CCRES_NOMINAL_GEV = 1.12

AXIAL_CONVENTIONS = (
    "unconfirmed", "pull_polynomial", "delta_gev_polynomial",
    "fractional_polynomial",
)


@dataclass(frozen=True)
class NuisanceParameters:
    dom_eff: float = 1.0
    hole_ice_p0: float = 0.10
    hole_ice_p1: float = -0.05
    bulk_ice_abs: float = 1.0
    bulk_ice_scatter: float = 1.0
    delta_gamma_nu: float = 0.0
    ma_ccqe: float = MA_CCQE_NOMINAL_GEV
    ma_ccres: float = MA_CCRES_NOMINAL_GEV
    nc_cc_ratio: float = 1.0
    neutrino_norm: float = 1.0
    muon_norm: float = 1.0

    def updated(self, values: Mapping[str, float]) -> "NuisanceParameters":
        unknown = set(values) - set(PARAMETER_NAMES)
        if unknown:
            raise KeyError(f"Nuisances desconocidos: {sorted(unknown)}")
        return replace(self, **{name: float(value) for name, value in values.items()})


@dataclass(frozen=True)
class Prediction:
    total_mu: np.ndarray
    total_sigma2: np.ndarray
    neutrino_mu: np.ndarray
    neutrino_sigma2: np.ndarray
    muon_mu: np.ndarray
    muon_sigma2: np.ndarray
    category_mu: dict[str, np.ndarray]
    category_sigma2: dict[str, np.ndarray]
    detector_corrections: dict[str, np.ndarray]
    observed: np.ndarray
    hist_shape: tuple[int, int, int]

    def reshaped(self, name: str = "total_mu") -> np.ndarray:
        if not hasattr(self, name):
            raise KeyError(name)
        array = getattr(self, name)
        if not isinstance(array, np.ndarray):
            raise ValueError(f"'{name}' no es un array")
        return array.reshape(self.hist_shape)


def _validate_array(name: str, arr: np.ndarray, *, nonnegative: bool = False) -> None:
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"'{name}' contiene NaN/inf")
    if nonnegative and np.any(arr < 0.0):
        idx = int(np.flatnonzero(arr < 0.0)[0])
        raise ValueError(f"'{name}' es negativo en índice {idx}: {arr.flat[idx]}")


def _nominal_axial(value: float, nominal: float) -> bool:
    return bool(np.isclose(value, nominal, atol=1e-12, rtol=0.0))


def _axial_coordinate(value: float, nominal: float, parameter: str,
                      convention: str) -> float:
    if convention == "unconfirmed":
        if _nominal_axial(value, nominal):
            return 0.0
        raise RuntimeError(
            f"No podemos variar {parameter}={value:g} GeV: la convención "
            "del reweight MaCC*_linear/quad no está confirmada. "
            "El valor nominal sí está permitido. No actives una convención "
            "alternativa sin verificarla frente al release."
        )
    if convention == "delta_gev_polynomial":
        return value - nominal
    if convention == "fractional_polynomial":
        return value / nominal - 1.0
    if convention == "pull_polynomial":
        if parameter == "ma_ccqe":
            frac = 0.25 if value >= nominal else 0.15
        else:
            frac = 0.20
        return (value - nominal) / (frac * nominal)
    raise ValueError(f"Convención axial desconocida: {convention}")


class TemplateSystematics:
    """Lector perezoso de pesos de un nodo + cache evento-a-evento estática.

    Las LUT de todos los nodos nunca se cargan enteras a memoria.
    A diferencia de interpolar /sumw2, predict_from_weights construye
    sumw2 a partir de los pesos interpolados de cada evento.
    """

    def __init__(self, filename: str | Path, *,
                 axial_mass_convention: str = "unconfirmed") -> None:
        if axial_mass_convention not in AXIAL_CONVENTIONS:
            raise ValueError(f"Convención axial inválida: {axial_mass_convention}")
        self.axial_mass_convention = axial_mass_convention
        self.path = Path(filename).expanduser().resolve()
        self.h5 = h5py.File(self.path, "r")
        try:
            self._load()
        except BaseException:
            self.h5.close()
            raise

    def close(self) -> None:
        self.h5.close()

    def __enter__(self) -> "TemplateSystematics":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _load(self) -> None:
        h5 = self.h5
        if str(h5.attrs.get("format_version", "")) != "3.1":
            raise ValueError(f"Formato 3.1 requerido: {self.path}")
        if not bool(h5.attrs.get("build_completed", False)):
            raise RuntimeError("build_templates.py no completó el archivo")

        required = [
            "event_weights/nominal", "event_cache/reco_bin_id",
            "event_cache/category_id", "event_cache/spectral_log_energy_ratio",
            "backgrounds/atmospheric_muon", "backgrounds/atmospheric_muon_sumw2",
            "observed/data", "status/completed_nodes",
        ]
        required.extend(f"hypersurfaces/{cat}/intercept" for cat in CATEGORIES)
        for name in required:
            if name not in h5:
                raise KeyError(f"Falta /{name} en {self.path}")

        self.weights_dataset = h5["event_weights/nominal"]
        if self.weights_dataset.ndim != 3:
            raise ValueError("event_weights/nominal debe ser [mass,coupling,event]")
        self.n_mass, self.n_coupling, self.n_event = self.weights_dataset.shape

        self.done = np.asarray(h5["status/completed_nodes"][:], dtype=bool)
        if self.done.shape != (self.n_mass, self.n_coupling):
            raise ValueError("status/completed_nodes tiene dimensiones incorrectas")

        self.category_id = np.asarray(h5["event_cache/category_id"][:], dtype=np.int16)
        self.reco_bin_id = np.asarray(h5["event_cache/reco_bin_id"][:], dtype=np.int64)
        self.spectral_log_energy = np.asarray(
            h5["event_cache/spectral_log_energy_ratio"][:], dtype=np.float64
        )
        if any(x.shape != (self.n_event,) for x in
               (self.category_id, self.reco_bin_id, self.spectral_log_energy)):
            raise ValueError("La cache de eventos no coincide con los pesos")
        if np.any((self.category_id < 0) | (self.category_id >= len(CATEGORIES))):
            raise ValueError("category_id fuera de [0,3]")

        self.observed = np.asarray(h5["observed/data"][:], dtype=np.float64)
        self.muon_base = np.asarray(h5["backgrounds/atmospheric_muon"][:], dtype=np.float64)
        self.muon_variance_base = np.asarray(
            h5["backgrounds/atmospheric_muon_sumw2"][:], dtype=np.float64
        )
        self.n_bins = len(self.observed)
        if (self.muon_base.shape != (self.n_bins,)
                or self.muon_variance_base.shape != (self.n_bins,)):
            raise ValueError("Dimensiones de fondos incorrectas")
        if np.any((self.reco_bin_id < 0) | (self.reco_bin_id >= self.n_bins)):
            raise ValueError("reco_bin_id fuera de los bins")
        for name, values in (
            ("observed", self.observed),
            ("atmospheric_muon", self.muon_base),
            ("atmospheric_muon_sumw2", self.muon_variance_base),
            ("spectral_log_energy_ratio", self.spectral_log_energy),
        ):
            _validate_array(name, values, nonnegative=name != "spectral_log_energy_ratio")

        hist_shape = tuple(int(x) for x in h5["observed/data"].attrs["shape_unflattened"])
        if len(hist_shape) != 3 or np.prod(hist_shape) != self.n_bins:
            raise ValueError("shape_unflattened del histograma inválido")
        self.hist_shape = hist_shape
        self.nominal_detector: dict[str, float] = {}
        self.bounds_detector: dict[str, tuple[float, float]] = {}
        hs_attrs = h5["hypersurfaces"].attrs
        for name in DETECTOR_NAMES:
            self.nominal_detector[name] = float(hs_attrs[f"nominal_{name}"])
            self.bounds_detector[name] = (
                float(hs_attrs[f"lower_{name}"]),
                float(hs_attrs[f"upper_{name}"]),
            )
        self.hs: dict[str, dict[str, np.ndarray]] = {}
        for category in CATEGORIES:
            cg = h5[f"hypersurfaces/{category}"]
            self.hs[category] = {}
            for name in ("intercept", *DETECTOR_NAMES):
                arr = np.asarray(cg[name][:], dtype=np.float64)
                if arr.shape != (self.n_bins,):
                    raise ValueError(f"hypersurfaces/{category}/{name}: shape incorrecto")
                _validate_array(f"hypersurfaces/{category}/{name}", arr)
                self.hs[category][name] = arr

        # Estos cuatro arrays se leen sólo si se solicita variar una masa axial.
        self._axial_coefficients: dict[str, np.ndarray] = {}

    def nominal_parameters(self) -> NuisanceParameters:
        return NuisanceParameters().updated(self.nominal_detector)

    def _coerce(self, params: NuisanceParameters | Mapping[str, float] | None) -> NuisanceParameters:
        if params is None:
            p = self.nominal_parameters()
        elif isinstance(params, NuisanceParameters):
            p = params
        else:
            p = self.nominal_parameters().updated(params)
        for name in PARAMETER_NAMES:
            v = float(getattr(p, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} no es finito")
        for name in ("nc_cc_ratio", "neutrino_norm", "muon_norm", "ma_ccqe", "ma_ccres"):
            if getattr(p, name) <= 0.0:
                raise ValueError(f"{name} debe ser positivo")
        for name, (low, high) in self.bounds_detector.items():
            value = float(getattr(p, name))
            if value < low or value > high:
                raise ValueError(f"{name}={value:g} fuera de [{low:g},{high:g}]")
        return p

    def _load_axial(self, name: str) -> np.ndarray:
        if name not in self._axial_coefficients:
            if f"event_cache/{name}" not in self.h5:
                raise KeyError(f"Falta event_cache/{name}")
            coeff = np.asarray(self.h5[f"event_cache/{name}"][:], dtype=np.float64)
            if coeff.shape != (self.n_event,):
                raise ValueError(f"event_cache/{name}: shape erróneo")
            _validate_array(name, coeff)
            self._axial_coefficients[name] = coeff
        return self._axial_coefficients[name]

    def _event_factor(self, p: NuisanceParameters) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            factor = np.exp(p.delta_gamma_nu * self.spectral_log_energy)
        _validate_array("spectral reweight", factor, nonnegative=True)
        for nuisance, nominal, prefix in (
            ("ma_ccqe", MA_CCQE_NOMINAL_GEV, "MaCCQE"),
            ("ma_ccres", MA_CCRES_NOMINAL_GEV, "MaCCRES"),
        ):
            z = _axial_coordinate(float(getattr(p, nuisance)), nominal,
                                  nuisance, self.axial_mass_convention)
            if z == 0.0:
                continue
            linear = self._load_axial(f"{prefix}_linear")
            quad = self._load_axial(f"{prefix}_quad")
            axial_factor = 1.0 + z * linear + (z * z) * quad
            _validate_array(f"{nuisance} reweight", axial_factor, nonnegative=True)
            factor *= axial_factor
        _validate_array("event reweight", factor, nonnegative=True)
        return factor

    def _correction(self, category: str, p: NuisanceParameters) -> np.ndarray:
        coefficients = self.hs[category]
        factor = coefficients["intercept"].copy()
        for name in DETECTOR_NAMES:
            delta = getattr(p, name) - self.nominal_detector[name]
            if delta:
                factor += delta * coefficients[name]
        _validate_array(f"hypersurface/{category}", factor, nonnegative=True)
        return factor

    def read_node_weights(self, mass_index: int, coupling_index: int) -> np.ndarray:
        if not (0 <= mass_index < self.n_mass and 0 <= coupling_index < self.n_coupling):
            raise IndexError("Nodo (mass_index,coupling_index) fuera de rango")
        if not self.done[mass_index, coupling_index]:
            raise RuntimeError(f"El nodo ({mass_index},{coupling_index}) no está completado")
        return np.asarray(self.weights_dataset[mass_index, coupling_index, :], dtype=np.float64)

    def predict_node(self, mass_index: int = 0, coupling_index: int = 0, *,
                     params: NuisanceParameters | Mapping[str, float] | None = None
                     ) -> Prediction:
        return self.predict_from_weights(self.read_node_weights(mass_index, coupling_index),
                                         params=params)

    def predict_from_weights(self, event_weights: np.ndarray, *,
                             params: NuisanceParameters | Mapping[str, float] | None = None
                             ) -> Prediction:
        """Predice desde pesos nominales por evento de cualquier punto interpolado.

        Se generan sumw2 como Σ_i [w_i(reponderado)]², NO interpolando
        los sumw2 precomputados de la malla de parámetros físicos.
        """
        weights = np.asarray(event_weights, dtype=np.float64)
        if weights.shape != (self.n_event,):
            raise ValueError(f"Se esperaban {self.n_event} pesos; encontrado {weights.shape}")
        _validate_array("event_weights", weights, nonnegative=True)
        p = self._coerce(params)
        weights = weights * self._event_factor(p)
        _validate_array("reweighted event_weights", weights, nonnegative=True)

        cat_mu: dict[str, np.ndarray] = {}
        cat_var: dict[str, np.ndarray] = {}
        correction: dict[str, np.ndarray] = {}
        nu_mu = np.zeros(self.n_bins, dtype=np.float64)
        nu_var = np.zeros(self.n_bins, dtype=np.float64)
        for cid, category in enumerate(CATEGORIES):
            mask = self.category_id == cid
            indices = self.reco_bin_id[mask]
            w = weights[mask]
            mean = np.bincount(indices, weights=w, minlength=self.n_bins)
            var = np.bincount(indices, weights=w*w, minlength=self.n_bins)
            hs_factor = self._correction(category, p)
            xsec = p.nc_cc_ratio if category == "nu_nc" else 1.0
            factor = p.neutrino_norm * xsec * hs_factor
            mean *= factor
            var *= factor * factor
            _validate_array(f"{category} prediction", mean, nonnegative=True)
            _validate_array(f"{category} sumw2", var, nonnegative=True)
            cat_mu[category] = mean
            cat_var[category] = var
            correction[category] = hs_factor
            nu_mu += mean
            nu_var += var

        mu_mu = p.muon_norm * self.muon_base
        mu_var = p.muon_norm**2 * self.muon_variance_base
        return Prediction(
            total_mu=nu_mu + mu_mu,
            total_sigma2=nu_var + mu_var,
            neutrino_mu=nu_mu,
            neutrino_sigma2=nu_var,
            muon_mu=mu_mu,
            muon_sigma2=mu_var,
            category_mu=cat_mu,
            category_sigma2=cat_var,
            detector_corrections=correction,
            observed=self.observed.copy(),
            hist_shape=self.hist_shape,
        )

    def check_nominal_node(self, mass_index: int = 0, coupling_index: int = 0,
                           *, rtol: float = 1e-10, atol: float = 1e-10
                           ) -> None:
        """Comprueba que el módulo reproduzca los templates sin hypersurface.

        La salida nominal de predict_node incluye el intercept de cada
        hypersurface, mientras que /templates/<category> todavía NO lo tiene.
        """
        w = self.read_node_weights(mass_index, coupling_index)
        for cid, category in enumerate(CATEGORIES):
            mask = self.category_id == cid
            idx = self.reco_bin_id[mask]
            val = w[mask]
            mean = np.bincount(idx, weights=val, minlength=self.n_bins)
            var = np.bincount(idx, weights=val*val, minlength=self.n_bins)
            original_mu = self.h5[f"templates/{category}"][mass_index, coupling_index, :]
            original_var = self.h5[f"sumw2/{category}"][mass_index, coupling_index, :]
            if not np.allclose(original_mu, mean, rtol=rtol, atol=atol):
                raise AssertionError(f"Templates nominales inconsistentes: {category}")
            if not np.allclose(original_var, var, rtol=rtol, atol=atol):
                raise AssertionError(f"sumw2 nominal inconsistente: {category}")
        prediction = self.predict_node(mass_index, coupling_index)
        _validate_array("predicción total", prediction.total_mu, nonnegative=True)
        _validate_array("varianza total", prediction.total_sigma2, nonnegative=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evalúa nuisances sobre templates HDF5 3.1")
    parser.add_argument("model", nargs="?", default="standard",
                        help="standard, Le-Lmu, Le-Ltau o Lmu-Ltau")
    parser.add_argument("--templates", type=Path, default=None,
                        help="Ruta al HDF5 si difiere de results/templates/MODEL.h5")
    parser.add_argument("--mass-index", type=int, default=0)
    parser.add_argument("--coupling-index", type=int, default=0)
    parser.add_argument("--param", action="append", default=[], metavar="NOMBRE=VALOR",
                        help="Modifica un nuisance (se puede repetir)")
    parser.add_argument("--axial-convention", choices=AXIAL_CONVENTIONS,
                        default="unconfirmed", help="Convenciones alternativas NO validadas")
    args = parser.parse_args()
    path = args.templates or (TEMPLATE_DIR / f"{args.model}.h5")
    changes: dict[str, float] = {}
    for item in args.param:
        if "=" not in item:
            parser.error(f"Se esperaba NOMBRE=VALOR, recibido: {item}")
        name, value = item.split("=", 1)
        try:
            changes[name] = float(value)
        except ValueError:
            parser.error(f"Valor numérico inválido para {name}: {value}")

    with TemplateSystematics(path, axial_mass_convention=args.axial_convention) as source:
        source.check_nominal_node(args.mass_index, args.coupling_index)
        p = source.nominal_parameters().updated(changes)
        result = source.predict_node(args.mass_index, args.coupling_index, params=p)
        print("=" * 65)
        print("ANALYSIS / SYSTEMATICS (HDF5 v3.1)")
        print("=" * 65)
        print(f"Archivo             : {source.path}")
        print(f"Nodo (masa,acop.)   : {args.mass_index}, {args.coupling_index}")
        print(f"Binning             : {source.hist_shape} = {source.n_bins} bins")
        print(f"Eventos MC          : {source.n_event:,}")
        print(f"Chequeo nominal     : OK (templates, sumw2)")
        print(f"Axiales             : {source.axial_mass_convention}")
        print("-" * 65)
        for category in CATEGORIES:
            print(f"{category:16s}: {result.category_mu[category].sum():14.6f}")
        print(f"Neutrinos           : {result.neutrino_mu.sum():14.6f}")
        print(f"Muones              : {result.muon_mu.sum():14.6f}")
        print(f"Total previsto      : {result.total_mu.sum():14.6f}")
        print(f"Total observado     : {result.observed.sum():14.6f}")
        print(f"Predicho/observado  : {result.total_mu.sum()/result.observed.sum():14.6f}")
        print(f"Suma varianza MC/KDE: {result.total_sigma2.sum():14.6f}")
        print("NOTA: Esto NO es un fit ni incluye penalizaciones de priors.")


if __name__ == "__main__":
    _cli()
