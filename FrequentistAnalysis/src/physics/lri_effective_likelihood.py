#!/usr/bin/env python3
"""IceCube DeepCore likelihood efectivo para templates event-level HDF5 v3.1.

Adaptacion de la arquitectura del likelihood LRI anterior (HDF5 v3.0).
Esta implementacion sigue siendo AUTONOMA dentro de FrequentistAnalysis:
no requiere importar analysis/reweight_mc.py ni analysis/systematics.py.

Conserva el procedimiento estadistico original:
    1) Interpolar los PESOS POR EVENTO bilinealmente en log10(m/eV) y
       log10(|acoplamiento|), sin interpolar histogramas o sus sumw2.
    2) Aplicar reweighting espectral y, solo con convencion verificada,
       reweighting de masas axiales.
    3) Histograma y sum(w**2) por bin y categoria.
    4) Aplicar hypersurfaces detectorales, NC/CC y normalizaciones.
    5) Agregar fondo de muones y la varianza KDE del release.
    6) Evaluar el likelihood efectivo de Argüelles, Schneider & Yuan (2019).

Los PRIORS del fit los aplica src/core/likelihood.py con fallbacks del JSON.
Por ello el kernel debe llamar con include_gaussian_constraints=False.

Los templates 3.1 se generan por analysis/build_templates.py:
    results/templates/Le-Lmu.h5
    results/templates/Le-Ltau.h5
    results/templates/Lmu-Ltau.h5
    results/templates/standard.h5

Esquema de ejes HDF5 v3.1:
    /axes/mass_eV
    /axes/log10_mass_eV
    /axes/coupling_values
    /axes/log10_abs_coupling
    /event_weights/nominal [mass,coupling,event]
    /status/completed_nodes [mass,coupling]

Para Lmu-Ltau el eje coupling_values puede representar G_{mu,tau} negativo;
la interpolacion conserva la rama de signo de la LUT y utiliza log10|G|.
El modelo standard posee solo un nodo ficticio [0,0] sin POIs LRI.

La compatibilidad con el viejo kernel.py conserva:
    DEFAULT_PARAMETER_ORDER
    LRIEffectiveLikelihood
    LRIEffectiveLikelihood.minus2loglike_vector(...)
    effective_log_likelihood_per_bin(...)

NOVEDADES (cross sections y oscilaciones):
  - ma_ccqe/ma_ccres: las columnas de coeficientes YA estaban en el MC;
    esta version permite seleccion explicita de la coordenada polinomial,
    pero NO afirma que ninguna convencion sea correcta antes de validarla.
  - dis_csms: 0=GENIE, 1=CSMS aproximado. Necesita .npz externo con
    deltas de seccion total + forma diferencial NORMALIZADA por evento.
    El programa no inventa razones GENIE/CSMS ni aplica un reweight falso.
  - theta23/dm31: reconocidos para una futura respuesta fisica, pero NO
    flotables aun con los templates 3.1 nominales. Hay un provider opcional
    que debe suministrar pesos de propagaciones reales de nuSQuIDS.
  - DEFAULT_PARAMETER_ORDER conserva los 11 anteriores para no romper
    kernel.py ni los JSON existentes. CROSS_SECTION_PARAMETER_ORDER y
    FULL_PARAMETER_ORDER preparan el paso a 14 o 16 parametros, respectivamente.

Ejemplos:
    export LRI_REPOSITORY_ROOT=/simulation/Long-Range-nu-Interaction
    export LRI_MODEL=Le-Lmu
    python3 -m src.physics.lri_effective_likelihood --model standard --validate
    python3 -m src.physics.lri_effective_likelihood --model Le-Lmu \\
        --log10-mass -14 --log10-g -26
"""
from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import h5py
import numpy as np
from scipy.special import gammaln


FORMAT_VERSION = "3.1"
MODELS = ("standard", "Le-Lmu", "Le-Ltau", "Lmu-Ltau")
CATEGORIES = ("nu_nc", "nue_cc", "numu_cc", "nutau_cc")
CATEGORY_TO_ID = {name: i for i, name in enumerate(CATEGORIES)}
DETECTOR_NUISANCES = (
    "dom_eff", "hole_ice_p0", "hole_ice_p1", "bulk_ice_abs", "bulk_ice_scatter"
)
DEFAULT_PARAMETER_ORDER = (
    "log10_g", "log10_m", *DETECTOR_NUISANCES,
    "delta_gamma_nu", "nc_cc_ratio", "neutrino_norm", "muon_norm"
)
DEFAULT_STANDARD_PARAMETER_ORDER = DEFAULT_PARAMETER_ORDER[2:]

# No sustituir DEFAULT_PARAMETER_ORDER hasta actualizar kernel.py y los JSON.
# Con esta version, el pipeline nominal anterior sigue siendo ejecutable.
CROSS_SECTION_PARAMETER_ORDER = (
    "log10_g", "log10_m", *DETECTOR_NUISANCES,
    "delta_gamma_nu", "ma_ccqe", "ma_ccres", "dis_csms",
    "nc_cc_ratio", "neutrino_norm", "muon_norm",
)
FULL_PARAMETER_ORDER = (
    "log10_g", "log10_m", *DETECTOR_NUISANCES,
    "delta_gamma_nu", "ma_ccqe", "ma_ccres", "dis_csms",
    "nc_cc_ratio", "neutrino_norm", "muon_norm", "theta23", "dm31",
)

MA_CCQE_NOMINAL_GEV = 0.99
MA_CCRES_NOMINAL_GEV = 1.12
MA_CCQE_MINUS_FRAC = 0.15
MA_CCQE_PLUS_FRAC = 0.25
MA_CCRES_SIGMA_FRAC = 0.20
DEFAULT_AXIAL_MASS_REWEIGHT_CONVENTION = "unconfirmed"
ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS = (
    "unconfirmed", "pull_polynomial", "delta_gev_polynomial", "fractional_polynomial"
)
DEFAULT_DETECTOR_NOMINAL = {
    "dom_eff": 1.0, "hole_ice_p0": 0.10, "hole_ice_p1": -0.05,
    "bulk_ice_abs": 1.0, "bulk_ice_scatter": 1.0,
}
# Sin anchos por defecto: el framework aplica los priors desde config/*.json.
DEFAULT_GAUSSIAN_PRIORS: dict[str, tuple[float, float | None]] = {
    **{name: (value, None) for name, value in DEFAULT_DETECTOR_NOMINAL.items()},
    "delta_gamma_nu": (0.0, None),
    "ma_ccqe": (MA_CCQE_NOMINAL_GEV, None),
    "ma_ccres": (MA_CCRES_NOMINAL_GEV, None),
    "dis_csms": (0.0, None),
    "nc_cc_ratio": (1.0, None),
    "neutrino_norm": (1.0, None),
    "muon_norm": (1.0, None),
}


@dataclass(frozen=True)
class DISCSMSResponse:
    """Variaciones fraccionales por evento (de una fuente externa validada).

    total_delta = sigma_tot(CSMS)/sigma_tot(GENIE) - 1
    differential_delta = shape_CSMS(E,y)/shape_GENIE(E,y) - 1

    OJO: differential_delta es una RAZON DE FORMAS normalizadas, no la
    razon bruta de d(sigma)/dy: multiplicar esta ultima por la razon total
    contaria dos veces el efecto en sigma_tot.
    """
    total_delta: np.ndarray
    differential_delta: np.ndarray
    dis_mask: np.ndarray
    source: str


@dataclass(frozen=True)
class CellCoordinate:
    lower_index: int
    fraction: float


@dataclass
class Prediction:
    total_mu: np.ndarray
    total_sigma2: np.ndarray
    neutrino_mu: np.ndarray
    neutrino_sigma2: np.ndarray
    muon_mu: np.ndarray
    muon_sigma2: np.ndarray
    category_mu: dict[str, np.ndarray]
    category_sigma2: dict[str, np.ndarray]
    correction_factors: dict[str, np.ndarray]
    mass_cell: int
    g_cell: int
    mass_fraction: float
    g_fraction: float
    delta_gamma_nu: float
    ma_ccqe_coordinate: float
    ma_ccres_coordinate: float
    minimum_event_reweight: float
    maximum_event_reweight: float
    dis_csms: float = 0.0
    theta23_rad: float | None = None
    dm31_eV2: float | None = None



def effective_log_likelihood_per_bin(
    observed: np.ndarray,
    mu: np.ndarray,
    sigma2: np.ndarray,
    *,
    poisson_relative_variance_threshold: float = 1e-12,
) -> np.ndarray:
    """Log-likelihood efectivo de A. Argüelles, A. Schneider, T. Yuan.

    alpha = mu**2/sigma2 + 1, beta = mu/sigma2.
    Se usa su forma binomial-negativa y el limite de Poisson si sigma2=0.
    NO se impone sigma2 <= mu**2 al total: la incertidumbre KDE muonica
    es agregada por bin y no proviene de un listado de pesos de eventos.
    """
    k = np.asarray(observed, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    v = np.asarray(sigma2, dtype=np.float64)
    if k.shape != m.shape or k.shape != v.shape or k.ndim != 1:
        raise ValueError("observed, mu, sigma2 deben tener el mismo shape unidimensional")
    for name, a in (("observed", k), ("mu", m), ("sigma2", v)):
        if not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"{name} tiene elementos no finitos o negativos")
    if np.any(np.abs(k - np.rint(k)) > 1e-9):
        raise ValueError("observed debe contener conteos enteros")
    if not np.isfinite(poisson_relative_variance_threshold) or poisson_relative_variance_threshold < 0:
        raise ValueError("Umbral Poisson invalido")
    k = np.rint(k)
    out = np.full(k.shape, -np.inf, dtype=np.float64)
    out[(m == 0) & (k == 0)] = 0.0
    positive = m > 0
    if not np.any(positive):
        return out
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        rv = np.zeros_like(m)
        rv[positive] = v[positive] / (m[positive] ** 2)
    poisson = positive & ((v == 0) | (rv <= poisson_relative_variance_threshold))
    if np.any(poisson):
        kp, mp = k[poisson], m[poisson]
        out[poisson] = kp * np.log(mp) - mp - gammaln(kp + 1)
    eff = positive & ~poisson
    if np.any(eff):
        ke, me, ve = k[eff], m[eff], v[eff]
        alpha = me * me / ve + 1.0
        # log(beta/(1+beta)) y log(1/(1+beta)) calculados establemente
        den = np.logaddexp(np.log(me), np.log(ve))
        out[eff] = (
            gammaln(ke + alpha) - gammaln(alpha) - gammaln(ke + 1)
            + alpha * (np.log(me) - den)
            + ke * (np.log(ve) - den)
        )
    return out



def _check_array(name: str, a: np.ndarray, *, nonnegative: bool = False) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: valores no finitos")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name}: valores negativos")



def _find_project_root(root: str | Path | None) -> Path:
    candidates: list[Path] = []
    if root is not None:
        candidates.append(Path(root))
    if os.getenv("LRI_REPOSITORY_ROOT"):
        candidates.append(Path(os.environ["LRI_REPOSITORY_ROOT"]))
    this = Path(__file__).resolve()
    for parent in this.parents:
        candidates.append(parent)
        candidates.append(parent / "Long-Range-nu-Interaction")
    candidates.append(Path.cwd())
    for candidate in candidates:
        c = candidate.expanduser().resolve()
        if (c / "results" / "templates").is_dir() or (
            (c / "analysis" / "build_templates.py").is_file()
        ):
            return c
    raise FileNotFoundError(
        "No se localizo la raiz del proyecto LRI. Usa project_root=... "
        "o export LRI_REPOSITORY_ROOT=/simulation/Long-Range-nu-Interaction"
    )


class LRIEffectiveLikelihood:
    """Likelihood autonomo a partir de HDF5 3.1 event-level.

    La lectura HDF5 de pesos se hace bajo demanda y con cache LRU. Nunca
    carga la malla completa en memoria. Las instancias son seguras frente a
    multiproceso: cualquier HDF5 heredado de otro PID se descarta y reabre.
    """

    def __init__(
        self,
        template_file: str | Path | None = None,
        *,
        model: str | None = None,
        project_root: str | Path | None = None,
        gaussian_priors: Mapping[str, tuple[float, float | None]] | None = None,
        include_muon_mc_uncertainty: bool = True,
        require_exact_sumw2_interpolation: bool | None = None,
        poisson_relative_variance_threshold: float = 1e-12,
        axial_mass_reweight_convention: str = DEFAULT_AXIAL_MASS_REWEIGHT_CONVENTION,
        # Archivo NPZ externo con deltas GENIE -> CSMS de CADA evento.
        # Si no existe, dis_csms=0 esta permitido; una variacion da error.
        dis_csms_response_file: str | Path | None = None,
        # Nominal de theta23: NO esta escrito por el constructor HDF5 3.1.
        # Debe pasarse el valor que realmente se uso en nuSQuIDS, en radianes.
        theta23_nominal_rad: float | None = None,
        # Interfaz opcional para pesos de nuSQuIDS fisicamente reprocesados.
        # Firma: provider(log10_m, log10_g, theta23_rad, dm31_eV2)
        # -> array de n_events, con misma normalizacion que nominal.
        oscillation_weight_provider: Callable[
            [float | None, float | None, float, float], np.ndarray
        ] | None = None,
        node_cache_size: int = 16,
        validate_nominal_node: bool = True,
    ):
        if model is None:
            model = os.getenv("LRI_MODEL")
        if model is not None and model not in MODELS:
            raise ValueError(f"Modelo no reconocido: {model}")
        if template_file is None:
            if model is None:
                model = "Le-Lmu"  # comportamiento historico por defecto
            project_root = _find_project_root(project_root)
            template_file = project_root / "results" / "templates" / f"{model}.h5"
        else:
            template_file = Path(template_file).expanduser().resolve()
        self.template_file = Path(template_file).expanduser().resolve()
        if not self.template_file.is_file():
            raise FileNotFoundError(f"No existe el archivo de templates: {self.template_file}")
        if node_cache_size < 0:
            raise ValueError("node_cache_size debe ser >= 0")
        if not np.isfinite(poisson_relative_variance_threshold) or poisson_relative_variance_threshold < 0:
            raise ValueError("Umbral de varianza invalido")
        if axial_mass_reweight_convention not in ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS:
            raise ValueError("Convencion de masa axial desconocida")
        self.poisson_relative_variance_threshold = float(poisson_relative_variance_threshold)
        self.include_muon_mc_uncertainty = bool(include_muon_mc_uncertainty)
        self.require_exact_sumw2_interpolation = require_exact_sumw2_interpolation
        self.axial_mass_reweight_convention = axial_mass_reweight_convention
        self.node_cache_size = int(node_cache_size)
        self._node_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._h5_handle: h5py.File | None = None
        self._h5_pid: int | None = None
        self._cache_pid: int = os.getpid()
        self._axial_coefficients: dict[str, np.ndarray] = {}
        self.dis_csms_response: DISCSMSResponse | None = None
        self.oscillation_weight_provider = oscillation_weight_provider
        self.theta23_nominal_rad = (
            None if theta23_nominal_rad is None else float(theta23_nominal_rad)
        )
        if self.theta23_nominal_rad is not None and not (
            np.isfinite(self.theta23_nominal_rad)
            and 0 < self.theta23_nominal_rad < np.pi / 2
        ):
            raise ValueError("theta23_nominal_rad debe estar en (0, pi/2), en radianes")

        # Metadatos y arrays O(n_event) se leen una vez; se cierra HDF5 antes
        # de hacer fork. El dataset de pesos se abre solo cuando se necesita.
        with h5py.File(self.template_file, "r") as h5:
            self._load_metadata(h5, model)

        if dis_csms_response_file is None:
            dis_csms_response_file = os.getenv("LRI_DIS_CSMS_RESPONSE_FILE") or None
        if dis_csms_response_file is not None:
            self._load_dis_csms_response(dis_csms_response_file)

        self.gaussian_priors = dict(DEFAULT_GAUSSIAN_PRIORS)
        if gaussian_priors is not None:
            self.gaussian_priors = {
                str(k): (float(v[0]), None if v[1] is None else float(v[1]))
                for k, v in gaussian_priors.items()
            }
        allowed = set(DEFAULT_GAUSSIAN_PRIORS)
        for k, (mean, sigma) in self.gaussian_priors.items():
            if k not in allowed or not np.isfinite(mean) or (
                sigma is not None and (not np.isfinite(sigma) or sigma <= 0)
            ):
                raise ValueError(f"Prior invalido para {k}: {(mean, sigma)}")
        if validate_nominal_node:
            self._validate_nominal_node()

    def _load_metadata(self, h5: h5py.File, requested_model: str | None) -> None:
        if str(h5.attrs.get("format_version", "")) != FORMAT_VERSION:
            raise ValueError("Se requiere HDF5 event-level version 3.1")
        if not bool(h5.attrs.get("build_completed", False)):
            raise RuntimeError("El constructor de templates no completo el HDF5")
        self.model = str(h5.attrs.get("model", ""))
        self.coupling_kind = str(h5.attrs.get("coupling_kind", ""))
        if self.model not in MODELS:
            raise ValueError(f"Modelo invalido en HDF5: {self.model}")
        if requested_model is not None and self.model != requested_model:
            raise ValueError(f"Modelo requerido={requested_model}, encontrado={self.model}")
        self.is_standard = self.model == "standard"
        # La hypersurface y el constructor se hicieron al mismo dm31.
        self.dm31_nominal_eV2 = float(
            h5["hypersurfaces"].attrs["dm31_fixed_eV2"]
        )
        if not np.isfinite(self.dm31_nominal_eV2) or self.dm31_nominal_eV2 <= 0:
            raise ValueError("dm31 nominal invalido en HDF5")
        # A diferencia de dm31, el builder no almacena theta23 nominal.
        # Si se ha agregado esa metadata en una version futura, se respeta.
        if self.theta23_nominal_rad is None and "theta23_fixed_rad" in h5.attrs:
            self.theta23_nominal_rad = float(h5.attrs["theta23_fixed_rad"])
        for name in (
            "axes/mass_eV", "axes/log10_mass_eV", "axes/coupling_values",
            "axes/log10_abs_coupling", "status/completed_nodes",
            "event_weights/nominal", "event_cache/reco_bin_id",
            "event_cache/category_id", "event_cache/spectral_log_energy_ratio",
            "observed/data", "backgrounds/atmospheric_muon",
            "backgrounds/atmospheric_muon_sumw2",
        ):
            if name not in h5:
                raise KeyError(f"Falta /{name} en {self.template_file}")

        self.mass_eV = np.asarray(h5["axes/mass_eV"][:], dtype=np.float64)
        self.log10_mass = np.asarray(h5["axes/log10_mass_eV"][:], dtype=np.float64)
        self.coupling_values = np.asarray(h5["axes/coupling_values"][:], dtype=np.float64)
        self.log10_g = np.asarray(h5["axes/log10_abs_coupling"][:], dtype=np.float64)
        self.n_mass, self.n_coupling, self.n_events = h5["event_weights/nominal"].shape
        self.n_bins = int(h5["observed/data"].size)
        self.hist_shape = tuple(map(int, h5["observed/data"].attrs["shape_unflattened"]))
        if (self.n_bins < 1 or np.prod(self.hist_shape) != self.n_bins or
            self.n_events < 1):
            raise ValueError("Dimensiones de histograma/eventos invalidas")
        for arr, size in (
            (self.mass_eV, self.n_mass), (self.log10_mass, self.n_mass),
            (self.coupling_values, self.n_coupling), (self.log10_g, self.n_coupling),
        ):
            if arr.shape != (size,):
                raise ValueError("Ejes incompatibles con las dimensiones de pesos MC")
        for name, a in (("mass_eV", self.mass_eV), ("log10_mass_eV", self.log10_mass),
                        ("coupling_values", self.coupling_values), ("log10_abs_coupling", self.log10_g)):
            _check_array(name, a)
        if self.is_standard:
            if self.coupling_kind != "none" or (self.n_mass, self.n_coupling) != (1, 1):
                raise ValueError("standard debe tener un nodo [1,1] y coupling_kind=none")
            if not (self.mass_eV[0] == 0.0 and self.coupling_values[0] == 0.0):
                raise ValueError("standard debe guardar ejes ficticios iguales a cero")
            self.coupling_sign = 0
        else:
            if np.any(self.mass_eV <= 0) or np.any(self.coupling_values == 0):
                raise ValueError("Masa positiva y acoplamientos no nulos requeridos")
            if self.coupling_kind not in ("gprime", "bigg"):
                raise ValueError("coupling_kind debe ser gprime o bigg")
            self.coupling_sign = 1 if self.coupling_values[0] > 0 else -1
            if np.any(np.sign(self.coupling_values) != self.coupling_sign):
                raise ValueError("No se permite mezclar signos de acoplamiento")
            if self.coupling_kind == "gprime" and self.coupling_sign != 1:
                raise ValueError("gprime debe ser positivo")
            if not np.all(np.diff(self.log10_mass) > 0) or not np.all(np.diff(self.log10_g) > 0):
                raise ValueError("Ejes logaritmicos deben ser estrictamente crecientes")
            if not np.allclose(self.log10_mass, np.log10(self.mass_eV), rtol=0, atol=1e-10):
                raise ValueError("Los valores de masa no coinciden con log10_mass_eV")
            if not np.allclose(self.log10_g, np.log10(np.abs(self.coupling_values)), rtol=0, atol=1e-10):
                raise ValueError("Acoplamientos incompatibles con log10_abs_coupling")
        self.done = np.asarray(h5["status/completed_nodes"][:], dtype=bool)
        if self.done.shape != (self.n_mass, self.n_coupling):
            raise ValueError("Dimensiones de /status/completed_nodes incorrectas")
        if not np.all(self.done):
            raise RuntimeError("La LUT de templates esta incompleta")

        self.observed = np.asarray(h5["observed/data"][:], dtype=np.float64)
        self.muon_template = np.asarray(h5["backgrounds/atmospheric_muon"][:], dtype=np.float64)
        self.muon_sumw2 = np.asarray(h5["backgrounds/atmospheric_muon_sumw2"][:], dtype=np.float64)
        self.category_id = np.asarray(h5["event_cache/category_id"][:], dtype=np.int8)
        self.reco_bin_id = np.asarray(h5["event_cache/reco_bin_id"][:], dtype=np.int32)
        self.spectral_log_energy_ratio = np.asarray(
            h5["event_cache/spectral_log_energy_ratio"][:], dtype=np.float64
        )
        for name, a in (("observed", self.observed), ("muon", self.muon_template),
                        ("muon_sumw2", self.muon_sumw2)):
            if a.shape != (self.n_bins,):
                raise ValueError(f"{name}: numero de bins incorrecto")
            _check_array(name, a, nonnegative=True)
        for name, a in (("category_id", self.category_id), ("reco_bin_id", self.reco_bin_id),
                        ("spectral_log_energy_ratio", self.spectral_log_energy_ratio)):
            if a.shape != (self.n_events,):
                raise ValueError(f"{name}: numero de eventos incorrecto")
            _check_array(name, a)
        if (np.any(self.category_id < 0) or np.any(self.category_id >= 4) or
            np.any(self.reco_bin_id < 0) or np.any(self.reco_bin_id >= self.n_bins)):
            raise ValueError("Indices de categoria/reconstruccion invalidos")

        self.hypersurface_nominal: dict[str, float] = {}
        self.hypersurface_bounds: dict[str, tuple[float, float]] = {}
        for name in DETECTOR_NUISANCES:
            base = h5["hypersurfaces"].attrs
            self.hypersurface_nominal[name] = float(base[f"nominal_{name}"])
            self.hypersurface_bounds[name] = (
                float(base[f"lower_{name}"]), float(base[f"upper_{name}"])
            )
        self.hypersurfaces: dict[str, dict[str, np.ndarray]] = {}
        for cat in CATEGORIES:
            group = h5[f"hypersurfaces/{cat}"]
            self.hypersurfaces[cat] = {}
            for coefficient in ("intercept", *DETECTOR_NUISANCES):
                a = np.asarray(group[coefficient][:], dtype=np.float64)
                if a.shape != (self.n_bins,):
                    raise ValueError(f"Hypersurface {cat}/{coefficient} tiene shape incorrecto")
                _check_array(f"Hypersurface {cat}/{coefficient}", a)
                self.hypersurfaces[cat][coefficient] = a
        self.category_event_indices = {
            cat: np.flatnonzero(self.category_id == i)
            for i, cat in enumerate(CATEGORIES)
        }
        # Las cuatro columnas axiales se leen solo si se intentan variar.
        self._axial_coefficients = {}

    def _event_fingerprint_sha256(self) -> str:
        """Huella de orden de eventos que previene usar respuestas de otro MC.

        Usar los tipos canonicos definidos aqui al generar la tabla externa.
        Se incluyen valores y el orden de las filas del event cache.
        """
        h = hashlib.sha256()
        with h5py.File(self.template_file, "r") as f:
            for name, dtype in (
                ("true_energy_GeV", "<f8"),
                ("inelasticity_y", "<f4"),
                ("pdg", "<i2"),
                ("interaction", "<i2"),
                ("reco_bin_id", "<i2"),
            ):
                a = np.ascontiguousarray(f[f"event_cache/{name}"][...], dtype=dtype)
                h.update(name.encode("ascii") + b"\x00")
                h.update(a.tobytes())
        return h.hexdigest()

    @property
    def dis_csms_fingerprint(self) -> str:
        return self._event_fingerprint_sha256()

    def _load_dis_csms_response(self, filename: str | Path) -> None:
        """Carga ratios fisicos calculados externamente para estos eventos.

        Archivo .npz requerido:
            total_delta       float64[n_events]
            differential_delta float64[n_events]
            event_fingerprint_sha256  str
            source             str (referencia, version, procedimiento)
            differential_definition str,
                EXACTAMENTE "normalized_differential_shape_ratio_minus_one"

        Las entradas para eventos NO DIS deben ser cero. Para eventos DIS:
            R_i(t) = (1 + t*total_delta_i) *
                     (1 + t*differential_delta_i)

        t=0 GENIE; t=1 aproxima CSMS. No se inventan tablas CSMS.
        """
        filename = Path(filename).expanduser().resolve()
        if not filename.is_file():
            raise FileNotFoundError(f"No existe la respuesta DIS/CSMS: {filename}")
        with np.load(filename, allow_pickle=False) as data:
            mandatory = {
                "total_delta", "differential_delta", "event_fingerprint_sha256",
                "source", "differential_definition",
            }
            missing = mandatory - set(data.files)
            if missing:
                raise ValueError(f"DIS/CSMS: faltan campos {sorted(missing)} en {filename}")
            total_delta = np.asarray(data["total_delta"], dtype=np.float64)
            shape_delta = np.asarray(data["differential_delta"], dtype=np.float64)
            fingerprint = str(data["event_fingerprint_sha256"].item())
            source = str(data["source"].item()).strip()
            definition = str(data["differential_definition"].item())
        if fingerprint != self.dis_csms_fingerprint:
            raise ValueError(
                "DIS/CSMS: la huella del orden de eventos no coincide con "
                "este HDF5. No se pueden asociar ratios a eventos distintos."
            )
        if not source:
            raise ValueError("DIS/CSMS: debes documentar la fuente de los ratios")
        if definition != "normalized_differential_shape_ratio_minus_one":
            raise ValueError(
                "DIS/CSMS: differential_delta debe ser una razon diferencial "
                "normalizada de FORMA, no la razon bruta dσ/dy."
            )
        if total_delta.shape != (self.n_events,) or shape_delta.shape != (self.n_events,):
            raise ValueError("DIS/CSMS: tamaño de deltas incompatible con el MC")
        _check_array("DIS/CSMS total_delta", total_delta)
        _check_array("DIS/CSMS differential_delta", shape_delta)
        if np.any(1 + total_delta <= 0) or np.any(1 + shape_delta <= 0):
            raise ValueError("DIS/CSMS: ratios finales no positivos")
        with h5py.File(self.template_file, "r") as h5:
            interaction = np.asarray(h5["event_cache/interaction"], dtype=np.int16)
        # De la codificacion de interacciones del MC IceCube:
        # 0=QE, 1=RES, 2=DIS, 3=otros. Verificar con release al tabular ratios.
        mask = interaction == 2
        if np.any(total_delta[~mask] != 0) or np.any(shape_delta[~mask] != 0):
            raise ValueError("DIS/CSMS: los eventos no DIS deben tener deltas cero")
        self.dis_csms_response = DISCSMSResponse(
            total_delta=total_delta, differential_delta=shape_delta,
            dis_mask=mask, source=source,
        )

    def _dis_csms_factor(self, z: float) -> np.ndarray | float:
        if z == 0.0:
            return 1.0
        if self.dis_csms_response is None:
            raise RuntimeError(
                "dis_csms se ha variado, pero faltan los ratios fisicos GENIE->CSMS. "
                "Proporciona dis_csms_response_file=... con una respuesta "
                "validada y alineada con event_cache. No se puede inferir "
                "de E_true e y solamente."
            )
        d = self.dis_csms_response
        total = 1.0 + z * d.total_delta
        differential = 1.0 + z * d.differential_delta
        _check_array("DIS/CSMS factor total", total, nonnegative=True)
        _check_array("DIS/CSMS factor diferencial", differential, nonnegative=True)
        if np.any(total <= 0.0) or np.any(differential <= 0.0):
            raise ValueError("DIS/CSMS factor fisico no positivo para este pull")
        return total * differential

    def _physics_event_weights(self, p: Mapping[str, float]) -> tuple[np.ndarray, CellCoordinate, CellCoordinate]:
        """Oscilaciones fijas por defecto, o respuesta real proporcionada externamente.

        main.cpp --theta23 y --dm31 permite crear nuevas propagaciones, pero
        no genera por si mismo la dependencia de los pesos en el fit.
        """
        log_m = None if self.is_standard else p["log10_m"]
        log_g = None if self.is_standard else p["log10_g"]
        base, cm, cg = self._interpolate_event_weights(log_m, log_g)
        dm31 = p.get("dm31", self.dm31_nominal_eV2)
        theta23 = p.get("theta23", self.theta23_nominal_rad)
        if theta23 is None:
            if dm31 != self.dm31_nominal_eV2:
                raise RuntimeError(
                    "Para variar dm31 debes indicar theta23 nominal en radianes "
                    "y una respuesta de oscilaciones nueva."
                )
            return base, cm, cg
        if self.theta23_nominal_rad is None:
            raise RuntimeError(
                "No consta theta23 nominal en HDF5 3.1. "
                "Proporciona theta23_nominal_rad usado en nuSQuIDS."
            )
        if (np.isclose(theta23, self.theta23_nominal_rad, rtol=0.0, atol=1e-14)
                and np.isclose(dm31, self.dm31_nominal_eV2, rtol=0.0, atol=1e-16)):
            return base, cm, cg
        if self.oscillation_weight_provider is None:
            raise RuntimeError(
                "theta23/dm31 no pueden flotarse con una LUT HDF5 3.1 "
                "generada a oscilaciones fijas. Falta una respuesta fisica "
                "validada de nuSQuIDS que incluya esos parametros. "
                "Activarlos en main.cpp o en el JSON no es suficiente."
            )
        weights = np.asarray(
            self.oscillation_weight_provider(log_m, log_g, float(theta23), float(dm31)),
            dtype=np.float64,
        )
        if weights.shape != (self.n_events,):
            raise ValueError("Respuesta de oscilaciones: numero de eventos incorrecto")
        _check_array("Respuesta de oscilaciones", weights, nonnegative=True)
        return weights, cm, cg

    def _handle(self) -> h5py.File:
        pid = os.getpid()
        if self._h5_pid is not None and self._h5_pid != pid:
            # Handle heredado de fork: no usarlo en el proceso hijo.
            self._h5_handle = None
            self._node_cache.clear()
            self._h5_pid = None
        if self._cache_pid != pid:
            self._node_cache.clear()
            self._cache_pid = pid
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.template_file, "r")
            self._h5_pid = pid
        return self._h5_handle

    def close(self) -> None:
        if self._h5_handle is not None:
            if self._h5_pid == os.getpid():
                self._h5_handle.close()
            self._h5_handle = None
        self._h5_pid = None
        self._node_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getstate__(self):
        obj = self.__dict__.copy()
        obj["_h5_handle"] = None
        obj["_h5_pid"] = None
        obj["_node_cache"] = OrderedDict()
        return obj

    def _node_weights(self, i: int, j: int) -> np.ndarray:
        if not (0 <= i < self.n_mass and 0 <= j < self.n_coupling):
            raise IndexError("Indice de nodo fuera de rango")
        if not self.done[i, j]:
            raise RuntimeError("Nodo no completado")
        key = (i, j)
        self._handle()  # Verificar PID ANTES de consultar la cache.
        if key in self._node_cache:
            self._node_cache.move_to_end(key)
            return self._node_cache[key]
        w = np.asarray(self._h5_handle["event_weights/nominal"][i, j, :], dtype=np.float64)
        if w.shape != (self.n_events,):
            raise ValueError(f"Nodo {key}: numero de pesos incorrecto")
        _check_array(f"weights{key}", w, nonnegative=True)
        if self.node_cache_size:
            self._node_cache[key] = w
            while len(self._node_cache) > self.node_cache_size:
                self._node_cache.popitem(last=False)
        return w

    @staticmethod
    def _coordinate(axis: np.ndarray, value: float, name: str) -> CellCoordinate:
        if not np.isfinite(value):
            raise ValueError(f"{name}: valor no finito")
        if axis.size == 1:
            tol = 2e-12
            if abs(value - float(axis[0])) > tol:
                raise ValueError(f"{name} fuera del unico nodo")
            return CellCoordinate(0, 0.0)
        if value < axis[0] - 1e-11 or value > axis[-1] + 1e-11:
            raise ValueError(f"{name}={value:g} fuera del rango [{axis[0]:g},{axis[-1]:g}]")
        v = float(np.clip(value, axis[0], axis[-1]))
        j = int(np.searchsorted(axis, v, side="right") - 1)
        j = min(max(j, 0), axis.size - 2)
        f = float((v - axis[j]) / (axis[j + 1] - axis[j]))
        return CellCoordinate(j, min(1.0, max(0.0, f)))

    def _interpolate_event_weights(
        self, log10_m: float | None, log10_g: float | None,
    ) -> tuple[np.ndarray, CellCoordinate, CellCoordinate]:
        if self.is_standard:
            if log10_m is not None or log10_g is not None:
                raise ValueError("standard carece de parametros LRI")
            return self._node_weights(0, 0).copy(), CellCoordinate(0, 0.0), CellCoordinate(0, 0.0)
        if log10_m is None or log10_g is None:
            raise ValueError("Los modelos LRI necesitan log10_m y log10_g")
        cm = self._coordinate(self.log10_mass, float(log10_m), "log10_m")
        cg = self._coordinate(self.log10_g, float(log10_g), "log10_g")
        ms = ((cm.lower_index, 1-cm.fraction), (cm.lower_index+1, cm.fraction)) if self.n_mass > 1 else ((0, 1.0),)
        gs = ((cg.lower_index, 1-cg.fraction), (cg.lower_index+1, cg.fraction)) if self.n_coupling > 1 else ((0, 1.0),)
        result = np.zeros(self.n_events, dtype=np.float64)
        for i, a in ms:
            for j, b in gs:
                if a*b:
                    result += (a*b) * self._node_weights(i, j)
        _check_array("Pesos interpolados", result, nonnegative=True)
        return result, cm, cg

    def _axial_coordinate(self, name: str, mass_gev: float) -> float:
        nominal = MA_CCQE_NOMINAL_GEV if name == "ma_ccqe" else MA_CCRES_NOMINAL_GEV
        delta = mass_gev - nominal
        convention = self.axial_mass_reweight_convention
        if convention == "unconfirmed":
            if abs(delta) <= 1e-12 * nominal:
                return 0.0
            raise RuntimeError(
                f"Se intento variar {name}, pero la convencion de las columnas "
                "MaCC*_linear / MaCC*_quad aun NO esta confirmada. "
                "Mantener ma_ccqe=0.99 y ma_ccres=1.12 hasta validarla."
            )
        if convention == "delta_gev_polynomial":
            return delta
        if convention == "fractional_polynomial":
            return delta / nominal
        if name == "ma_ccqe":
            sigma = (MA_CCQE_PLUS_FRAC if delta >= 0 else MA_CCQE_MINUS_FRAC) * nominal
        else:
            sigma = MA_CCRES_SIGMA_FRAC * nominal
        return delta / sigma

    def _axial_coeff(self, name: str) -> np.ndarray:
        if name not in self._axial_coefficients:
            a = np.asarray(self._handle()[f"event_cache/{name}"][:], dtype=np.float64)
            if a.shape != (self.n_events,):
                raise ValueError(f"Coeficiente {name}: shape incorrecto")
            _check_array(name, a)
            self._axial_coefficients[name] = a
        return self._axial_coefficients[name]

    def default_parameters(self) -> dict[str, float]:
        p = {
            **self.hypersurface_nominal,
            "delta_gamma_nu": 0.0, "ma_ccqe": MA_CCQE_NOMINAL_GEV,
            "ma_ccres": MA_CCRES_NOMINAL_GEV, "dis_csms": 0.0,
            "nc_cc_ratio": 1.0, "neutrino_norm": 1.0, "muon_norm": 1.0,
            "dm31": self.dm31_nominal_eV2,
        }
        if self.theta23_nominal_rad is not None:
            p["theta23"] = self.theta23_nominal_rad
        if not self.is_standard:
            p["log10_g"] = float((self.log10_g[0] + self.log10_g[-1]) / 2)
            p["log10_m"] = float((self.log10_mass[0] + self.log10_mass[-1]) / 2)
        return p

    def _complete_parameters(self, parameters: Mapping[str, float]) -> dict[str, float]:
        p = self.default_parameters()
        unknown = set(parameters) - (set(p) | {"theta23"})
        if unknown:
            raise KeyError(f"Parametros desconocidos: {sorted(unknown)}")
        for key, value in parameters.items():
            v = float(value)
            if not np.isfinite(v):
                raise ValueError(f"{key} debe ser finito")
            p[key] = v
        for name in ("neutrino_norm", "nc_cc_ratio", "ma_ccqe", "ma_ccres"):
            if p[name] <= 0:
                raise ValueError(f"{name} debe ser positivo")
        if p["muon_norm"] < 0:
            raise ValueError("muon_norm debe ser no negativo")
        if not (0 < p["dm31"] < 1e-1):
            raise ValueError("dm31 debe ser positivo (en eV^2)")
        if "theta23" in p and not (0 < p["theta23"] < np.pi / 2):
            raise ValueError("theta23 debe estar en radianes y pertenecer a (0,pi/2)")
        for name, (lo, hi) in self.hypersurface_bounds.items():
            if not (lo <= p[name] <= hi):
                raise ValueError(f"{name} fuera del dominio de hypersurface [{lo}, {hi}]")
        return p

    def _event_reweight(self, p: Mapping[str, float]) -> tuple[np.ndarray, float, float]:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            f = np.exp(p["delta_gamma_nu"] * self.spectral_log_energy_ratio)
        _check_array("reweight espectral", f, nonnegative=True)
        coords = []
        for name, prefix in (("ma_ccqe", "MaCCQE"), ("ma_ccres", "MaCCRES")):
            z = self._axial_coordinate(name, p[name])
            coords.append(z)
            if z:
                f *= 1.0 + z*self._axial_coeff(f"{prefix}_linear") + z*z*self._axial_coeff(f"{prefix}_quad")
        f *= self._dis_csms_factor(float(p["dis_csms"]))
        _check_array("reweight final", f, nonnegative=True)
        return f, coords[0], coords[1]

    def _correction(self, category: str, p: Mapping[str, float]) -> np.ndarray:
        hs = self.hypersurfaces[category]
        corr = hs["intercept"].copy()
        for name in DETECTOR_NUISANCES:
            delta = p[name] - self.hypersurface_nominal[name]
            if delta != 0:
                corr += delta * hs[name]
        _check_array(f"hypersurface {category}", corr, nonnegative=True)
        return corr

    def predict(self, parameters: Mapping[str, float] | None = None) -> Prediction:
        p = self._complete_parameters(parameters or {})
        weights, cm, cg = self._physics_event_weights(p)
        factors, ccqe_z, ccres_z = self._event_reweight(p)
        w = weights * factors
        _check_array("Pesos finales MC", w, nonnegative=True)

        nu_mu = np.zeros(self.n_bins, dtype=np.float64)
        nu_var = np.zeros(self.n_bins, dtype=np.float64)
        per_cat_mu: dict[str, np.ndarray] = {}
        per_cat_var: dict[str, np.ndarray] = {}
        corrections: dict[str, np.ndarray] = {}
        for cat, indices in self.category_event_indices.items():
            bins = self.reco_bin_id[indices]
            wi = w[indices]
            raw = np.bincount(bins, weights=wi, minlength=self.n_bins)
            raw_var = np.bincount(bins, weights=wi**2, minlength=self.n_bins)
            corr = self._correction(cat, p)
            scale = p["neutrino_norm"] * (p["nc_cc_ratio"] if cat == "nu_nc" else 1.0)
            applied = corr * scale
            mu_i = raw * applied
            var_i = raw_var * applied**2
            per_cat_mu[cat] = mu_i
            per_cat_var[cat] = var_i
            corrections[cat] = corr
            nu_mu += mu_i
            nu_var += var_i
        muon_mu = p["muon_norm"] * self.muon_template
        muon_var = (p["muon_norm"]**2) * self.muon_sumw2 if self.include_muon_mc_uncertainty else np.zeros(self.n_bins)
        total_mu = nu_mu + muon_mu
        total_var = nu_var + muon_var
        _check_array("Prediccion total", total_mu, nonnegative=True)
        _check_array("Varianza total", total_var, nonnegative=True)
        return Prediction(
            total_mu=total_mu, total_sigma2=total_var,
            neutrino_mu=nu_mu, neutrino_sigma2=nu_var,
            muon_mu=muon_mu, muon_sigma2=muon_var,
            category_mu=per_cat_mu, category_sigma2=per_cat_var,
            correction_factors=corrections,
            mass_cell=cm.lower_index, g_cell=cg.lower_index,
            mass_fraction=cm.fraction, g_fraction=cg.fraction,
            delta_gamma_nu=p["delta_gamma_nu"],
            ma_ccqe_coordinate=ccqe_z, ma_ccres_coordinate=ccres_z,
            minimum_event_reweight=float(np.min(factors)),
            maximum_event_reweight=float(np.max(factors)),
            dis_csms=p["dis_csms"],
            theta23_rad=p.get("theta23"),
            dm31_eV2=p.get("dm31"),
        )

    def gaussian_penalty(self, parameters: Mapping[str, float]) -> float:
        p = self._complete_parameters(parameters)
        penalty = 0.0
        for name, (mean, sigma) in self.gaussian_priors.items():
            if sigma is not None:
                penalty += ((p[name] - mean) / sigma) ** 2
        return float(penalty)

    def minus2loglike(self, parameters: Mapping[str, float] | None = None,
                      *, include_gaussian_constraints: bool = False) -> float:
        pred = self.predict(parameters)
        value = -2.0 * np.sum(effective_log_likelihood_per_bin(
            self.observed, pred.total_mu, pred.total_sigma2,
            poisson_relative_variance_threshold=self.poisson_relative_variance_threshold,
        ))
        if include_gaussian_constraints:
            value += self.gaussian_penalty(parameters or {})
        return float(value)

    def minus2loglike_vector(self, vector: Sequence[float],
                             *, parameter_order: Sequence[str] = DEFAULT_PARAMETER_ORDER,
                             include_gaussian_constraints: bool = False) -> float:
        return self.minus2loglike(
            self.vector_to_parameters(vector, parameter_order),
            include_gaussian_constraints=include_gaussian_constraints,
        )

    def vector_to_parameters(self, vector: Sequence[float],
                             parameter_order: Sequence[str] = DEFAULT_PARAMETER_ORDER) -> dict[str, float]:
        order = tuple(parameter_order)
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim != 1 or arr.size != len(order):
            raise ValueError(f"Vector de parametros de longitud incorrecta: {arr.shape} != {len(order)}")
        if len(set(order)) != len(order):
            raise ValueError("Hay nombres de parametros repetidos")
        return self._complete_parameters(dict(zip(order, arr)))

    def parameters_to_vector(self, parameters: Mapping[str, float],
                             parameter_order: Sequence[str] = DEFAULT_PARAMETER_ORDER) -> np.ndarray:
        p = self._complete_parameters(parameters)
        return np.asarray([p[name] for name in parameter_order], dtype=np.float64)

    def default_bounds(self, parameter_order: Sequence[str] = DEFAULT_PARAMETER_ORDER):
        p = self.default_parameters()
        bounds: list[tuple[float | None, float | None]] = []
        for name in parameter_order:
            if name == "log10_m" and not self.is_standard:
                bounds.append((float(self.log10_mass[0]), float(self.log10_mass[-1])))
            elif name == "log10_g" and not self.is_standard:
                bounds.append((float(self.log10_g[0]), float(self.log10_g[-1])))
            elif name in self.hypersurface_bounds:
                bounds.append(self.hypersurface_bounds[name])
            elif name == "ma_ccqe":
                bounds.append((max(0.001, 0.99 * 0.55), 0.99 * 1.75))
            elif name == "ma_ccres":
                bounds.append((max(0.001, 1.12 * 0.4), 1.12 * 1.6))
            elif name == "dis_csms":
                bounds.append((-1.0, 2.0))
            elif name == "theta23":
                bounds.append((0.0, np.pi / 2))
            elif name == "dm31":
                bounds.append((0.0005, 0.005))
            elif name in ("neutrino_norm", "nc_cc_ratio", "muon_norm"):
                bounds.append((0.0, None))
            else:
                bounds.append((None, None))
        return bounds

    def _validate_nominal_node(self) -> None:
        # Comprobar solamente el primer nodo y sus cuatro categorias.
        i, j = 0, 0
        with h5py.File(self.template_file, "r") as h5:
            w = np.asarray(h5["event_weights/nominal"][i, j, :], dtype=np.float64)
            for cat, indices in self.category_event_indices.items():
                b = self.reco_bin_id[indices]
                actual_mu = np.bincount(b, weights=w[indices], minlength=self.n_bins)
                actual_var = np.bincount(b, weights=w[indices]**2, minlength=self.n_bins)
                stored_mu = np.asarray(h5[f"templates/{cat}"][i, j, :], dtype=np.float64)
                stored_var = np.asarray(h5[f"sumw2/{cat}"][i, j, :], dtype=np.float64)
                if not np.allclose(actual_mu, stored_mu, rtol=2e-6, atol=1e-9):
                    raise RuntimeError(f"Template nominal {cat} no reproduce pesos de eventos")
                if not np.allclose(actual_var, stored_var, rtol=2e-6, atol=1e-10):
                    raise RuntimeError(f"Sumw2 nominal {cat} no reproduce pesos de eventos")


def _parse_parameter_assignment(s: str) -> tuple[str, float]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Usar NOMBRE=VALOR")
    key, value = s.split("=", 1)
    try:
        return key, float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Valor invalido: {s}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Likelihood IceCube event-level HDF5 3.1")
    ap.add_argument("--model", choices=MODELS, default=None)
    ap.add_argument("--templates", type=Path, default=None)
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument("--log10-mass", type=float, default=None)
    ap.add_argument("--log10-g", type=float, default=None)
    ap.add_argument("--param", action="append", type=_parse_parameter_assignment, default=[])
    ap.add_argument("--axial-convention", choices=ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS,
                    default=DEFAULT_AXIAL_MASS_REWEIGHT_CONVENTION)
    ap.add_argument("--dis-response", type=Path, default=None)
    ap.add_argument("--theta23-nominal", type=float, default=None,
                    help="theta23 realmente usado en nuSQuIDS, en RADIANES")
    ap.add_argument("--validate", action="store_true", help="Comprueba templates nominales")
    args = ap.parse_args()
    with LRIEffectiveLikelihood(
        template_file=args.templates, model=args.model,
        project_root=args.project_root, validate_nominal_node=args.validate,
        axial_mass_reweight_convention=args.axial_convention,
        dis_csms_response_file=args.dis_response,
        theta23_nominal_rad=args.theta23_nominal,
    ) as likelihood:
        p = likelihood.default_parameters()
        if args.log10_mass is not None:
            p["log10_m"] = args.log10_mass
        if args.log10_g is not None:
            p["log10_g"] = args.log10_g
        p.update(dict(args.param))
        pred = likelihood.predict(p)
        chi2 = likelihood.minus2loglike(p, include_gaussian_constraints=False)
        print("=" * 72)
        print("FREQUENTIST ANALYSIS / LIKELIHOOD EFFECTIVE HDF5 v3.1")
        print("=" * 72)
        print(f"Archivo                 : {likelihood.template_file}")
        print(f"Modelo / coupling_kind  : {likelihood.model} / {likelihood.coupling_kind}")
        print(f"Eje acoplamiento        : {'ninguno' if likelihood.is_standard else 'log10|'+likelihood.coupling_kind+'|'}")
        print(f"Eventos MC              : {likelihood.n_events:,}")
        print(f"Bins reconstruidos      : {likelihood.n_bins}")
        print(f"Neutrinos                : {pred.neutrino_mu.sum():.6f}")
        print(f"Muones                   : {pred.muon_mu.sum():.6f}")
        print(f"Total previsto           : {pred.total_mu.sum():.6f}")
        print(f"Total observado          : {likelihood.observed.sum():.6f}")
        print(f"Varianza MC/KDE          : {pred.total_sigma2.sum():.6f}")
        print(f"-2 log L_eff (sin priors): {chi2:.9f}")
        print(f"MA reweight convention  : {likelihood.axial_mass_reweight_convention}")
        print(f"DIS response cargada    : {likelihood.dis_csms_response is not None}")
        print(f"theta23 nominal         : {likelihood.theta23_nominal_rad}")
        print(f"dm31 nominal (eV^2)     : {likelihood.dm31_nominal_eV2:.9e}")
        print("Priors                  : externos, definidos en lri_config.json")


if __name__ == "__main__":
    main()
