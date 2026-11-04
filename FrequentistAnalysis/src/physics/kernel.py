#!/usr/bin/env python3
"""Adaptador FrequentistAnalysis <-> LRI event-level HDF5 3.1.

Compatible con:
    src/core/likelihood.py:    from src.physics.kernel import (
                                    PARAMETER_ORDER, compute_user_model_single)
    src/engines/optimizers.py: parámetros flotantes definidos en start_values
    src/main.py:            python3 -m src.main --config_file config/lri_config.json

Funciones públicas preservadas:
    PARAMETER_ORDER, LATEX_LABELS, get_label, format_title_stats,
    compute_user_model_single, get_active_likelihood.

IMPORTANTE:
  - Devuelve únicamente -2 ln L_eff. Los priors son EXTERNOS en
    GenericLikelihoodWrapper (config/*.json), evitando el doble conteo.
  - Los 11 parámetros anteriores funcionan sin cambios.
  - Al añadir ma_ccqe, ma_ccres, dis_csms, theta23 o dm31 al JSON, el orden
    se amplía automáticamente; no hace falta editar este archivo.
  - La existencia de un parámetro en el JSON NO crea su respuesta física.
    Las masas axiales requieren convención GENIE verificable, DIS/CSMS
    requiere ratios validados, y theta23/dm31 requiere pesos de nuSQuIDS
    reprocesados mediante un provider compatible.
  - Los tres modelos se seleccionan desde lri_model o el nombre del JSON.
  - Una instancia por proceso. No se heredan handles HDF5 entre procesos.

Campos adicionales opcionales del JSON (al mismo nivel que start_values):
    "lri_model": "Le-Lmu" | "Le-Ltau" | "Lmu-Ltau"
    "lri_template_file": "/ruta/a/templates/Le-Lmu.h5"
    "axial_mass_reweight_convention": "unconfirmed" | \
        "pull_polynomial" | "delta_gev_polynomial" | "fractional_polynomial"
    "dis_csms_response_file": "data/dis_csms_response.npz"
    "theta23_nominal_rad": <valor usado realmente en nuSQuIDS, radianes>
    "oscillation_weight_provider": "paquete.modulo:nombre_funcion"

Los valores por defecto son conservadores: ninguna convención no
verificada es activada silenciosamente.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence

import numpy as np

from .lri_effective_likelihood import (
    ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS,
    CROSS_SECTION_PARAMETER_ORDER,
    DEFAULT_PARAMETER_ORDER,
    DEFAULT_STANDARD_PARAMETER_ORDER,
    FULL_PARAMETER_ORDER,
    MA_CCQE_NOMINAL_GEV,
    MA_CCRES_NOMINAL_GEV,
    LRIEffectiveLikelihood,
)


ALL_MODELS = ("standard", "Le-Lmu", "Le-Ltau", "Lmu-Ltau")
SUPPORTED_LRI_MODELS = ALL_MODELS[1:]
EXTRA_PARAMETERS = ("ma_ccqe", "ma_ccres", "dis_csms", "theta23", "dm31")


def _normalize_model(raw: object) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(raw).lower())
    mapping = {
        "standard": "standard", "sm": "standard",
        "lelmu": "Le-Lmu", "leminuslmu": "Le-Lmu",
        "leltau": "Le-Ltau", "leminusltau": "Le-Ltau",
        "lmultau": "Lmu-Ltau", "lmuminusltau": "Lmu-Ltau",
    }
    if key not in mapping:
        raise ValueError(f"Modelo desconocido {raw!r}. Modelos: {ALL_MODELS}")
    return mapping[key]


def _model_hint(value: object) -> str | None:
    if value is None:
        return None
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    guesses = {
        model for label, model in (
            ("lmultau", "Lmu-Ltau"),
            ("leltau", "Le-Ltau"),
            ("lelmu", "Le-Lmu"),
        ) if label in key
    }
    if len(guesses) > 1:
        raise ValueError(f"Nombre de modelo ambiguo: {value!r}")
    return next(iter(guesses), None)


def _config_path_from_argv(argv: Sequence[str]) -> Path | None:
    for index, arg in enumerate(argv):
        if arg in ("--config_file", "--config-file"):
            if index + 1 >= len(argv):
                raise ValueError(f"Falta archivo después de {arg}")
            return Path(argv[index + 1]).expanduser().resolve()
        for prefix in ("--config_file=", "--config-file="):
            if arg.startswith(prefix):
                return Path(arg[len(prefix):]).expanduser().resolve()
    return None


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"No existe el JSON: {path}")
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"El JSON debe ser un objeto: {path}")
    return result


ACTIVE_CONFIG_FILE = _config_path_from_argv(sys.argv[1:])
ACTIVE_CONFIG = _load_config(ACTIVE_CONFIG_FILE)


def _select_model() -> str:
    field = ACTIVE_CONFIG.get("lri_model", ACTIVE_CONFIG.get("physics_model"))
    explicit = _normalize_model(field) if field is not None else None
    if explicit is None:
        explicit = (
            _model_hint(ACTIVE_CONFIG_FILE.stem if ACTIVE_CONFIG_FILE else None)
            or _model_hint(ACTIVE_CONFIG.get("output_dir"))
            or _model_hint(ACTIVE_CONFIG.get("experiment"))
        )
    environment = os.getenv("LRI_MODEL", "").strip()
    env_model = _normalize_model(environment) if environment else None
    if env_model and explicit and env_model != explicit:
        raise ValueError(
            f"LRI_MODEL={env_model} contradice --config_file (modelo={explicit}). "
            "Elimina la variable de entorno o corrige el JSON."
        )
    return env_model or explicit or "Le-Lmu"


ACTIVE_MODEL = _select_model()
# Necesario si multiprocessing usa spawn en lugar de fork.
os.environ.setdefault("LRI_MODEL", ACTIVE_MODEL)


def _extract_mapping(name: str) -> dict:
    value = ACTIVE_CONFIG.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"La clave {name!r} del JSON debe contener un objeto")
    return value


def _configured_parameter_order() -> tuple[str, ...]:
    """Usa el mismo JSON que estimate_param_names en optimizers.py.

    El orden del vector incluye TAMBIÉN parámetros fijos, porque el wrapper
    construye model_args leyendo start_values y fixed_params. Esto impide
    inyectar ceros por error para parámetros omitidos.
    """
    full = (
        tuple(n for n in FULL_PARAMETER_ORDER if n not in ("log10_g", "log10_m"))
        if ACTIVE_MODEL == "standard" else tuple(FULL_PARAMETER_ORDER)
    )
    if not ACTIVE_CONFIG:
        return (
            tuple(DEFAULT_STANDARD_PARAMETER_ORDER)
            if ACTIVE_MODEL == "standard" else tuple(DEFAULT_PARAMETER_ORDER)
        )
    starts = _extract_mapping("start_values")
    fixed = _extract_mapping("fixed_params")
    if not starts:
        raise ValueError("El JSON no define parámetros flotantes en start_values")
    intersection = set(starts) & set(fixed)
    if intersection:
        raise ValueError(
            "Parámetros presentes simultáneamente en start_values y "
            f"fixed_params: {sorted(intersection)}"
        )
    names = set(starts) | set(fixed)
    missing = set(DEFAULT_PARAMETER_ORDER) - names
    if ACTIVE_MODEL == "standard":
        missing -= {"log10_g", "log10_m"}
    if missing:
        raise ValueError(
            "Para evitar usar ceros silenciosos en los parámetros nominales, "
            f"declara cada parámetro en start_values o fixed_params: {sorted(missing)}"
        )
    unknown = names - set(full)
    if unknown:
        raise ValueError(
            f"Parámetros no reconocidos por lri_effective_likelihood.py: {sorted(unknown)}"
        )
    explicit_order = ACTIVE_CONFIG.get("parameter_order")
    if explicit_order is not None:
        if not isinstance(explicit_order, list) or not all(
            isinstance(x, str) for x in explicit_order
        ):
            raise ValueError("parameter_order debe ser una lista de nombres")
        if len(explicit_order) != len(set(explicit_order)) or set(explicit_order) != names:
            raise ValueError(
                "parameter_order debe contener exactamente los nombres de "
                "start_values y fixed_params, sin repetidos"
            )
        return tuple(explicit_order)
    return tuple(n for n in full if n in names)


PARAMETER_ORDER = _configured_parameter_order()
FLOATING_PARAMETERS = tuple(_extract_mapping("start_values"))
FIXED_PARAMETERS = tuple(_extract_mapping("fixed_params"))


LATEX_LABELS: dict[str, str] = {
    "log10_g": (
        r"\log_{10}|G_{\mu\tau}|" if ACTIVE_MODEL == "Lmu-Ltau"
        else r"\log_{10}g'"
    ),
    "log10_m": r"\log_{10}(m'/\mathrm{eV})",
    "dom_eff": r"\epsilon_{\mathrm{DOM}}",
    "hole_ice_p0": r"p_{0}^{\mathrm{hole}}",
    "hole_ice_p1": r"p_{1}^{\mathrm{hole}}",
    "bulk_ice_abs": r"A_{\mathrm{abs}}",
    "bulk_ice_scatter": r"A_{\mathrm{scat}}",
    "delta_gamma_nu": r"\Delta\gamma_{\nu}",
    "ma_ccqe": r"M_A^{\mathrm{CCQE}}\,(\mathrm{GeV})",
    "ma_ccres": r"M_A^{\mathrm{CCRES}}\,(\mathrm{GeV})",
    "dis_csms": r"\eta_{\mathrm{DIS/CSMS}}",
    "nc_cc_ratio": r"R_{\mathrm{NC/CC}}",
    "neutrino_norm": r"N_{\nu}",
    "muon_norm": r"N_{\mu}",
    "theta23": r"\theta_{23}\,(\mathrm{rad})",
    "dm31": r"\Delta m_{31}^{2}\,(\mathrm{eV}^{2})",
}


def get_label(param: str) -> str:
    label = LATEX_LABELS.get(param, param.replace("_", r"\_"))
    return rf"${label}$"


def format_title_stats(param: str, results_dict: Mapping) -> str:
    base = LATEX_LABELS.get(param, param.replace("_", r"\_"))
    value = results_dict.get(param, {})
    if not isinstance(value, Mapping) or value.get("best_fit") is None:
        return rf"${base}$"
    best = float(value["best_fit"])
    if not math.isfinite(best):
        return rf"${base}$"
    intervals = value.get("1sigma")
    if not intervals:
        return rf"${base} = {best:.3g}$"
    try:
        low, high = next(
            (float(a), float(b)) for a, b in intervals
            if float(a) <= best <= float(b)
        )
    except (StopIteration, TypeError, ValueError):
        low, high = map(float, intervals[0])
    return (
        rf"${base} = {best:.3g}"
        rf"_{{-{best - low:.3g}}}"
        rf"^{{+{high - best:.3g}}}$"
    )


def _resolve_file(value: str | os.PathLike | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Interpretación principal: rutas relativas al cwd de src.main.
    return (Path.cwd() / path).resolve()


def _axial_convention() -> str:
    value = str(ACTIVE_CONFIG.get(
        "axial_mass_reweight_convention",
        os.getenv("LRI_AXIAL_MASS_REWEIGHT_CONVENTION", "unconfirmed"),
    )).strip().lower()
    if value not in ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS:
        raise ValueError(
            f"Convención axial desconocida: {value!r}. "
            f"Permitidas: {ALLOWED_AXIAL_MASS_REWEIGHT_CONVENTIONS}"
        )
    return value


def _dis_response_file() -> Path | None:
    return _resolve_file(ACTIVE_CONFIG.get(
        "dis_csms_response_file", os.getenv("LRI_DIS_CSMS_RESPONSE_FILE")
    ))


def _theta23_nominal() -> float | None:
    raw = ACTIVE_CONFIG.get("theta23_nominal_rad", os.getenv("LRI_THETA23_NOMINAL_RAD"))
    if raw is None:
        return None
    value = float(raw)
    if not np.isfinite(value) or not (0.0 < value < np.pi / 2):
        raise ValueError("theta23_nominal_rad debe estar en (0,pi/2), radianes")
    return value


def _provider_import_path() -> str | None:
    raw = ACTIVE_CONFIG.get("oscillation_weight_provider", "")
    if not raw:
        return None
    if not isinstance(raw, str) or ":" not in raw:
        raise ValueError(
            "oscillation_weight_provider debe ser 'paquete.modulo:funcion'"
        )
    return raw.strip()


def _load_provider() -> Callable | None:
    spec = _provider_import_path()
    if spec is None:
        return None
    module_name, member = spec.split(":", 1)
    if not module_name or not member or "." in member:
        raise ValueError("Formato de provider inválido; usar paquete.modulo:funcion")
    module = importlib.import_module(module_name)
    provider = getattr(module, member)
    if not callable(provider):
        raise TypeError(f"El provider {spec} no es callable")
    return provider


def _preflight_response_requirements() -> None:
    """Fallar ANTES de que GenericLikelihoodWrapper esconda una excepción.

    Esta comprobación evita que el framework cambie una excepción física
    en una penalización 1e9 y produzca un fit aparentemente 'válido'.
    """
    if not ACTIVE_CONFIG:
        return
    floating = _extract_mapping("start_values")
    fixed = _extract_mapping("fixed_params")
    axial = _axial_convention()

    if any(k in floating for k in ("ma_ccqe", "ma_ccres")) and axial == "unconfirmed":
        raise RuntimeError(
            "ma_ccqe/ma_ccres están en start_values, pero la convención de "
            "MaCC*_linear/quad sigue 'unconfirmed'. No habilites ese fit "
            "sin verificar qué variable usó GENIE."
        )
    for name, nominal in (
        ("ma_ccqe", MA_CCQE_NOMINAL_GEV),
        ("ma_ccres", MA_CCRES_NOMINAL_GEV),
    ):
        if name in fixed and not math.isclose(
            float(fixed[name]), nominal, rel_tol=0.0, abs_tol=1e-12
        ) and axial == "unconfirmed":
            raise RuntimeError(
                f"{name} fijado fuera de su nominal sin convención axial validada"
            )

    dis_file = _dis_response_file()
    need_dis = "dis_csms" in floating or (
        "dis_csms" in fixed and float(fixed["dis_csms"]) != 0.0
    )
    if need_dis and (dis_file is None or not dis_file.is_file()):
        raise RuntimeError(
            "dis_csms aparece en el fit pero falta un archivo físico de "
            "respuesta GENIE->CSMS. Configura dis_csms_response_file con "
            "ratios validados, alineados por evento."
        )

    oscillation = ("theta23", "dm31")
    if any(name in floating for name in oscillation):
        if _theta23_nominal() is None:
            raise RuntimeError(
                "Para flotar theta23/dm31 configura theta23_nominal_rad "
                "con el valor REAL de la propagación (radianes)."
            )
        if _provider_import_path() is None:
            raise RuntimeError(
                "theta23/dm31 requieren un oscillation_weight_provider "
                "basado en nuevas propagaciones nuSQuIDS. El HDF5 3.1 "
                "nominal NO permite flotarlos por sí solo."
            )
    if "theta23" in fixed and _theta23_nominal() is None:
        raise RuntimeError(
            "theta23 fijo exige theta23_nominal_rad en el JSON o retirar "
            "theta23 de fixed_params (se mantiene el nominal implícito)."
        )


_preflight_response_requirements()


LRI_LIKELIHOOD: LRIEffectiveLikelihood | None = None
_LIKELIHOOD_PID: int | None = None


def get_active_likelihood() -> LRIEffectiveLikelihood:
    """Singleton por proceso (compatible con multiprocessing fork/spawn)."""
    global LRI_LIKELIHOOD, _LIKELIHOOD_PID
    pid = os.getpid()
    if LRI_LIKELIHOOD is not None and _LIKELIHOOD_PID == pid:
        return LRI_LIKELIHOOD
    if LRI_LIKELIHOOD is not None:
        LRI_LIKELIHOOD.close()
        LRI_LIKELIHOOD = None

    filename = _resolve_file(
        ACTIVE_CONFIG.get("lri_template_file", os.getenv("LRI_TEMPLATE_FILE"))
    )
    LRI_LIKELIHOOD = LRIEffectiveLikelihood(
        template_file=filename,
        model=ACTIVE_MODEL,
        include_muon_mc_uncertainty=True,
        axial_mass_reweight_convention=_axial_convention(),
        dis_csms_response_file=_dis_response_file(),
        theta23_nominal_rad=_theta23_nominal(),
        oscillation_weight_provider=_load_provider(),
        node_cache_size=int(ACTIVE_CONFIG.get("node_cache_size", 16)),
        validate_nominal_node=bool(ACTIVE_CONFIG.get("validate_nominal_node", True)),
    )
    _LIKELIHOOD_PID = pid
    return LRI_LIKELIHOOD


def compute_user_model_single(params: Sequence[float], static_data: object) -> float:
    """Retorna -2 ln L_eff en los parámetros de PARAMETER_ORDER; SIN priors."""
    del static_data
    values = np.asarray(params, dtype=np.float64)
    if values.ndim != 1 or values.size != len(PARAMETER_ORDER):
        raise ValueError(
            f"Se recibieron {values.size} elementos, se esperan "
            f"{len(PARAMETER_ORDER)} en {PARAMETER_ORDER}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Vector contiene NaN/inf")
    return float(get_active_likelihood().minus2loglike_vector(
        values,
        parameter_order=PARAMETER_ORDER,
        include_gaussian_constraints=False,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Kernel LRI / HDF5 3.1")
    parser.add_argument("--model", choices=ALL_MODELS, default=None)
    parser.add_argument("--templates", type=Path, default=None)
    parser.add_argument("--config_file", "--config-file", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        parser.print_help()
        return
    requested_model = _normalize_model(args.model) if args.model else ACTIVE_MODEL
    if args.config_file is not None and requested_model != ACTIVE_MODEL:
        raise ValueError("--model no coincide con lri_model del JSON")

    if args.config_file is not None:
        like = get_active_likelihood()
        if args.templates is not None:
            raise ValueError("--templates no debe combinarse con --config_file")
        close_likelihood = False
    else:
        like = LRIEffectiveLikelihood(
            template_file=args.templates,
            model=requested_model,
            include_muon_mc_uncertainty=True,
            axial_mass_reweight_convention=_axial_convention(),
            dis_csms_response_file=_dis_response_file(),
            theta23_nominal_rad=_theta23_nominal(),
            oscillation_weight_provider=_load_provider(),
            validate_nominal_node=True,
        )
        close_likelihood = True
    try:
        order = PARAMETER_ORDER if args.config_file is not None else (
            tuple(DEFAULT_STANDARD_PARAMETER_ORDER)
            if like.is_standard else tuple(DEFAULT_PARAMETER_ORDER)
        )
        p = like.default_parameters()
        vec = like.parameters_to_vector(p, parameter_order=order)
        prediction = like.predict(p)
        objective = like.minus2loglike_vector(
            vec, parameter_order=order, include_gaussian_constraints=False,
        )
        print("=" * 74)
        print("FREQUENTISTANALYSIS / KERNEL -- HDF5 3.1")
        print("=" * 74)
        print(f"Modelo                    : {like.model}")
        print(f"Template                  : {like.template_file}")
        print(f"Acoplamiento              : {like.coupling_kind}")
        print(f"Malla                     : {like.n_mass} x {like.n_coupling}")
        print(f"Eventos MC                : {like.n_events:,}")
        print(f"Bins reconstruidos        : {like.n_bins}")
        print(f"Parámetros del kernel     : {len(order)} ({', '.join(order)})")
        print(f"Neutrinos                 : {prediction.neutrino_mu.sum():.6f}")
        print(f"Muones                    : {prediction.muon_mu.sum():.6f}")
        print(f"Total previsto            : {prediction.total_mu.sum():.6f}")
        print(f"Observados                : {like.observed.sum():.6f}")
        print(f"-2 ln L_eff (sin priors)  : {objective:.9f}")
        print(f"Convención axial          : {like.axial_mass_reweight_convention}")
        print(f"DIS/CSMS respuesta        : {like.dis_csms_response is not None}")
        print(f"theta23 nominal (rad)     : {like.theta23_nominal_rad}")
        print(f"dm31 nominal (eV²)       : {like.dm31_nominal_eV2:.9e}")
        print("[OK] Kernel y likelihood HDF5 3.1 integrados")
    finally:
        if close_likelihood:
            like.close()


if __name__ == "__main__":
    main()
