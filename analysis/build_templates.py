#!/usr/bin/env python3
"""Construye LUT de pesos evento-a-evento para IceCube/DeepCore desde el
propagador actual LRI/propagation/generate_flux_lut.

Uso desde cualquier directorio:
    python3 analysis/build_templates.py standard
    python3 analysis/build_templates.py Le-Lmu
    python3 analysis/build_templates.py Le-Ltau
    python3 analysis/build_templates.py Lmu-Ltau

Entradas:
    results/oscillated_flux/<modelo>.h5
    data/icecube/{observed,mc,hypersurfaces,derived}/...

Salidas:
    results/templates/<modelo>.h5

Se conservan los pesos nominales por evento, histogramas, errores MC,
variables cinemáticas e hypersurfaces del constructor anterior.
No se recalculan los flujos ni las oscilaciones; no se inventan nuisances.
La construcción exige todos los ángulos de todos los nodos completos.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd


# =============================================================================
# Configuración
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
ICECUBE_DIR = REPO_ROOT / "data" / "icecube"
FLUX_DIR = REPO_ROOT / "results" / "oscillated_flux"
TEMPLATE_DIR = REPO_ROOT / "results" / "templates"

MODELS = ("standard", "Le-Lmu", "Le-Ltau", "Lmu-Ltau")

# Debe coincidir con el valor usado al generar la LUT nuSQuIDS.
DM31_FIXED_EV2 = 2.47e-3

# No fijamos 41 x 61: smoke/reduced grids y futuras resoluciones pueden
# variar. Se inspeccionan las dimensiones reales del HDF5.
EXPECTED_MASS_COUNT = None
EXPECTED_G_COUNT = None

FLUX_TO_ICECUBE_WEIGHT_FACTOR = 1.0e4

# Pivot usado para guardar ln(E_true/E_pivot). La variación delta_gamma_nu
# se aplicará después, en el likelihood, como exp(delta_gamma_nu * log_ratio).
SPECTRAL_INDEX_PIVOT_GEV = 24.0

# Los pesos por evento son, con diferencia, el dataset más grande.
# float32 deja ~3.7 GiB crudos para 41 x 61 x ~397k eventos, antes de compresión.
EVENT_WEIGHT_DTYPE = np.float32
EVENT_CHUNK_SIZE = 65536

# Compresión para datasets pequeños/medianos.
AUX_GZIP_LEVEL = 4

# Progreso: imprime al menos el último g de cada masa y cada N valores de g.
PROGRESS_EVERY_G = 10

FORMAT_NAME = "LRI IceCube event-level template LUT"
FORMAT_VERSION = "3.1"


# =============================================================================
# Categorías y especies
# =============================================================================

SPECIES = (
    "nue",
    "numu",
    "nutau",
    "antinue",
    "antinumu",
    "antinutau",
)

SPECIES_TO_ID = {
    species: index
    for index, species in enumerate(SPECIES)
}

PDG_TO_SPECIES = {
    12: "nue",
    -12: "antinue",
    14: "numu",
    -14: "antinumu",
    16: "nutau",
    -16: "antinutau",
}

CATEGORY_FILES = {
    "nu_nc": "mc_nu_nc.csv",
    "nue_cc": "mc_nue_cc.csv",
    "numu_cc": "mc_numu_cc.csv",
    "nutau_cc": "mc_nutau_cc.csv",
}

CATEGORIES = tuple(CATEGORY_FILES)

CATEGORY_TO_ID = {
    category: index
    for index, category in enumerate(CATEGORIES)
}

HYPERSURFACE_FILES = {
    "nu_nc": "hs_nu_nc_nue_cc.csv",
    "nue_cc": "hs_nu_nc_nue_cc.csv",
    "numu_cc": "hs_numu_cc.csv",
    "nutau_cc": "hs_nutau_cc.csv",
}

def icecube_file(icecube_directory: Path, filename: str) -> Path:
    """Resuelve las rutas del release reorganizadas en data/icecube/."""
    derived = {"fixed_data.npz", "detector_reference.npz", "daemon_reference.npz"}
    hypersurfaces = set(HYPERSURFACE_FILES.values())
    if filename in derived:
        return icecube_directory / "derived" / filename
    if filename in hypersurfaces:
        return icecube_directory / "hypersurfaces" / filename
    if filename == "data.csv":
        return icecube_directory / "observed" / filename
    if filename.startswith("mc_") and filename.endswith(".csv"):
        return icecube_directory / "mc" / filename
    raise ValueError(f"Archivo IceCube desconocido: {filename}")


def input_files_fingerprint(icecube_directory: Path) -> str:
    """SHA256 reproducible de TODOS los archivos físicos que usa el constructor."""
    digest = hashlib.sha256()
    for name in sorted(set(ICECUBE_REQUIRED_FILES)):
        digest.update(name.encode("utf-8") + b"\0")
        with icecube_file(icecube_directory, name).open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


DETECTOR_NUISANCES = (
    "dom_eff",
    "hole_ice_p0",
    "hole_ice_p1",
    "bulk_ice_abs",
    "bulk_ice_scatter",
)

HYPERSURFACE_VALUE_COLUMNS = (
    "intercept",
    "intercept_sigma",
    "dom_eff",
    "dom_eff_sigma",
    "hole_ice_p0",
    "hole_ice_p0_sigma",
    "hole_ice_p1",
    "hole_ice_p1_sigma",
    "bulk_ice_abs",
    "bulk_ice_abs_sigma",
    "bulk_ice_scatter",
    "bulk_ice_scatter_sigma",
)

NOMINAL_SYSTEMATICS_FALLBACK = {
    "dom_eff": 1.00,
    "hole_ice_p0": 0.10,
    "hole_ice_p1": -0.05,
    "bulk_ice_abs": 1.00,
    "bulk_ice_scatter": 1.00,
}

ICECUBE_REQUIRED_FILES = (
    "fixed_data.npz",
    "detector_reference.npz",
    "data.csv",
    "mc_mu.csv",
    "mc_nu_nc.csv",
    "mc_nue_cc.csv",
    "mc_numu_cc.csv",
    "mc_nutau_cc.csv",
    "hs_nu_nc_nue_cc.csv",
    "hs_numu_cc.csv",
    "hs_nutau_cc.csv",
)

MC_REQUIRED_COLUMNS = (
    "reco_energy",
    "reco_coszen",
    "pid",
    "pdg",
    "true_energy",
    "true_coszen",
    "weight",
    "type",
    "interaction",
    "MaCCQE_linear",
    "MaCCQE_quad",
    "MaCCRES_linear",
    "MaCCRES_quad",
    "Q2",
    "W",
    "x",
    "y",
)

SUPPORTED_POST_PROPAGATION_NUISANCES = (
    "dom_eff",
    "hole_ice_p0",
    "hole_ice_p1",
    "bulk_ice_abs",
    "bulk_ice_scatter",
    "delta_gamma_nu",
    "ma_ccqe",
    "ma_ccres",
    "nc_cc_ratio",
    "neutrino_norm",
    "muon_norm",
)

OMITTED_BARR_NUISANCES = (
    "barr_pi_A_F",
    "barr_pi_G",
    "barr_pi_H",
    "barr_Kplus_W",
    "barr_Kminus_W",
    "barr_Kplus_Y",
)

FIXED_OSCILLATION_PARAMETERS = (
    "theta23",
    "dm31",
)


# =============================================================================
# Contenedores
# =============================================================================

@dataclass(frozen=True)
class Binning:
    reco_energy_edges_GeV: np.ndarray
    reco_coszen_edges: np.ndarray
    pid_edges: np.ndarray
    hist_shape: tuple[int, int, int]
    n_bins: int
    nucount: float


@dataclass(frozen=True)
class FluxLUTMetadata:
    model: str
    coupling_kind: str
    is_standard: bool
    source_signature: str
    mass_eV: np.ndarray
    log10_mass_eV: np.ndarray
    coupling_values: np.ndarray
    log10_abs_coupling: np.ndarray
    energy_GeV: np.ndarray
    coszen: np.ndarray
    shape: tuple[int, ...]



@dataclass
class MCEventCache:
    true_energy_GeV: np.ndarray
    true_coszen: np.ndarray
    base_mc_weight: np.ndarray

    species_id: np.ndarray
    category_id: np.ndarray
    pdg: np.ndarray
    reco_bin_id: np.ndarray

    event_type: np.ndarray
    interaction: np.ndarray

    ma_ccqe_linear: np.ndarray
    ma_ccqe_quad: np.ndarray
    ma_ccres_linear: np.ndarray
    ma_ccres_quad: np.ndarray

    q2: np.ndarray
    w_hadronic: np.ndarray
    bjorken_x: np.ndarray
    inelasticity_y: np.ndarray

    spectral_log_energy_ratio: np.ndarray

    energy_lower: np.ndarray
    energy_fraction: np.ndarray
    coszen_lower: np.ndarray
    coszen_fraction: np.ndarray

    species_event_indices: tuple[np.ndarray, ...]
    category_event_indices: tuple[np.ndarray, ...]

    source_counts_total: dict[str, int]
    source_counts_inside_bins: dict[str, int]

    @property
    def size(self) -> int:
        return int(self.base_mc_weight.size)


# =============================================================================
# Utilidades
# =============================================================================

def require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} debe ser finito y > 0. Encontrado: {value}")


def ensure_required_files(
    directory: Path,
    filenames: Iterable[str],
    description: str,
) -> None:
    missing = [
        filename
        for filename in filenames
        if not (icecube_file(directory, filename)).is_file()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {icecube_file(directory, filename)}"
            for filename in missing
        )
        raise FileNotFoundError(
            f"Faltan archivos de {description}:\n{formatted}"
        )


def validate_strictly_increasing(values: np.ndarray, name: str) -> None:
    values = np.asarray(values)

    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"{name} debe ser un vector 1D con al menos dos puntos."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contiene valores no finitos.")

    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} no es estrictamente creciente.")


def create_string_dataset(
    group: h5py.Group,
    name: str,
    values: Sequence[str],
) -> h5py.Dataset:
    string_dtype = h5py.string_dtype(encoding="utf-8")

    return group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=string_dtype,
    )


def aux_dataset_options() -> dict[str, object]:
    if AUX_GZIP_LEVEL <= 0:
        return {}

    return {
        "compression": "gzip",
        "compression_opts": int(AUX_GZIP_LEVEL),
        "shuffle": True,
        "fletcher32": True,
    }


def event_weight_dataset_options() -> dict[str, object]:
    # LZF suele ser bastante más rápido para el patrón de lectura del
    # likelihood (cuatro nodos por evaluación). Si la instalación de HDF5
    # no lo expone, usamos gzip nivel 1.
    try:
        lzf_available = bool(
            h5py.h5z.filter_avail(h5py.h5z.FILTER_LZF)
        )
    except Exception:
        lzf_available = False

    if lzf_available:
        return {
            "compression": "lzf",
            "shuffle": True,
            "fletcher32": True,
        }

    return {
        "compression": "gzip",
        "compression_opts": 1,
        "shuffle": True,
        "fletcher32": True,
    }


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    if seconds < 60.0:
        return f"{seconds:.1f} s"

    minutes = seconds / 60.0

    if minutes < 60.0:
        return f"{minutes:.1f} min"

    return f"{minutes / 60.0:.2f} h"


# =============================================================================
# Binning oficial del release
# =============================================================================

def load_binning(icecube_directory: Path) -> Binning:
    fixed_file = icecube_file(icecube_directory, "fixed_data.npz")

    with np.load(fixed_file, allow_pickle=False) as fixed:
        required = (
            "BINS_reco_energy",
            "BINS_reco_coszen",
            "BINS_pid",
            "HIST_SHAPE",
            "NUCOUNT",
        )

        missing = [
            key
            for key in required
            if key not in fixed
        ]

        if missing:
            raise KeyError(
                f"Faltan keys en {fixed_file}: {missing}"
            )

        energy_edges = np.asarray(
            fixed["BINS_reco_energy"],
            dtype=np.float64,
        )

        coszen_edges = np.asarray(
            fixed["BINS_reco_coszen"],
            dtype=np.float64,
        )

        pid_edges = np.asarray(
            fixed["BINS_pid"],
            dtype=np.float64,
        )

        hist_shape_array = np.asarray(
            fixed["HIST_SHAPE"],
            dtype=np.int64,
        )

        nucount = float(
            np.asarray(fixed["NUCOUNT"]).item()
        )

    validate_strictly_increasing(
        energy_edges,
        "BINS_reco_energy",
    )

    validate_strictly_increasing(
        coszen_edges,
        "BINS_reco_coszen",
    )

    validate_strictly_increasing(
        pid_edges,
        "BINS_pid",
    )

    if hist_shape_array.shape != (3,):
        raise ValueError(
            f"HIST_SHAPE debe tener shape (3,), no {hist_shape_array.shape}."
        )

    expected_shape = (
        energy_edges.size - 1,
        coszen_edges.size - 1,
        pid_edges.size - 1,
    )

    hist_shape = tuple(
        int(value)
        for value in hist_shape_array
    )

    if hist_shape != expected_shape:
        raise ValueError(
            f"HIST_SHAPE={hist_shape} es inconsistente con los edges; "
            f"se esperaba {expected_shape}."
        )

    require_finite_positive(
        nucount,
        "NUCOUNT",
    )

    data_file = icecube_file(icecube_directory, "data.csv")
    data = pd.read_csv(data_file)

    if "count" not in data.columns:
        raise ValueError(
            f"{data_file} no contiene la columna count."
        )

    data_total = float(
        data["count"].sum()
    )

    if not np.isclose(
        data_total,
        nucount,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            f"NUCOUNT={nucount:.12e} pero sum(data.csv count)="
            f"{data_total:.12e}."
        )

    return Binning(
        reco_energy_edges_GeV=energy_edges,
        reco_coszen_edges=coszen_edges,
        pid_edges=pid_edges,
        hist_shape=hist_shape,
        n_bins=int(np.prod(hist_shape)),
        nucount=nucount,
    )


def load_detector_reference(
    icecube_directory: Path,
) -> tuple[
    dict[str, float],
    dict[str, tuple[float, float]],
]:
    reference_file = (
        icecube_file(icecube_directory, "detector_reference.npz")
    )

    with np.load(
        reference_file,
        allow_pickle=False,
    ) as reference:
        keys = [
            str(value)
            for value in reference[
                "SYSTEMATIC_KEYS"
            ].tolist()
        ]

        nominal_values = np.asarray(
            reference["NOMINAL_SYSTEMATICS"],
            dtype=np.float64,
        )

        lower_values = np.asarray(
            reference["SYSTEMATIC_LOWER"],
            dtype=np.float64,
        )

        upper_values = np.asarray(
            reference["SYSTEMATIC_UPPER"],
            dtype=np.float64,
        )

    if tuple(keys) != DETECTOR_NUISANCES:
        raise ValueError(
            "SYSTEMATIC_KEYS no coincide con los cinco nuisances "
            f"esperados. Encontrado: {keys}"
        )

    if not (
        nominal_values.shape
        == lower_values.shape
        == upper_values.shape
        == (len(DETECTOR_NUISANCES),)
    ):
        raise ValueError(
            "Shapes inesperados en detector_reference.npz."
        )

    nominal = {
        name: float(value)
        for name, value
        in zip(
            DETECTOR_NUISANCES,
            nominal_values,
        )
    }

    bounds = {
        name: (
            float(lower),
            float(upper),
        )
        for name, lower, upper
        in zip(
            DETECTOR_NUISANCES,
            lower_values,
            upper_values,
        )
    }

    for name, fallback in (
        NOMINAL_SYSTEMATICS_FALLBACK.items()
    ):
        if not np.isclose(
            nominal[name],
            fallback,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Nominal detectoral inesperado para {name}: "
                f"{nominal[name]} (esperado {fallback})."
            )

    return nominal, bounds


# =============================================================================
# Nueva LUT HDF5 de flujos
# =============================================================================

def require_hdf5_path(
    file: h5py.File,
    path: str,
) -> None:
    if path not in file:
        raise KeyError(
            f"Falta '{path}' en {file.filename}."
        )


def load_flux_lut_metadata(filename: Path, requested_model: str) -> FluxLUTMetadata:
    """Lee y valida el formato actual del propagador C++.

    LRI:      /flux [mass, coupling, coszen, energy, 6]
              /done [mass, coupling, coszen]
    Standard: /flux [coszen, energy, 6]
              /done [coszen]
    En modo standard se usan ejes de índice ficticios 1x1; NO son
    parámetros físicos. Se documenta explícitamente en el HDF5 de salida.
    """
    if not filename.is_file():
        raise FileNotFoundError(f"No existe la LUT de flujos: {filename}")
    with h5py.File(filename, "r") as file:
        for name in ("energy_GeV", "coszen", "flux", "done"):
            require_hdf5_path(file, name)
        def attr_str(name, default=""):
            value = file.attrs.get(name, default)
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        mode = attr_str("mode")
        model = attr_str("model")
        source_signature = attr_str("config_signature")
        if not source_signature:
            raise ValueError(
                "El HDF5 de propagación no tiene config_signature; "
                "se necesita el formato nuevo con checkpoints identificables."
            )
        # En nuestro propagador actual la signatura registra ;dm=dm21,dm31.
        # Evitamos mezclar las hypersurfaces de dm31 fijo con otras oscilaciones.
        if source_signature:
            import re
            match = re.search(r"(?:^|;)dm=([^,;]+),([^;]+);", source_signature)
            if match:
                source_dm31 = float(match.group(2))
                if not np.isclose(source_dm31, DM31_FIXED_EV2, rtol=1e-12, atol=0.0):
                    raise ValueError(
                        f"dm31 en nuSQuIDS={source_dm31:.6e}, pero las "
                        f"hypersurfaces se preparan con {DM31_FIXED_EV2:.6e} eV^2")
        species_order = attr_str("species_order")
        if species_order:
            actual_order = tuple(part.strip() for part in species_order.split(","))
            if actual_order != SPECIES:
                raise ValueError(f"Orden de especies inesperado: {actual_order}")
        if requested_model == "standard":
            if mode != "standard" or model != "standard":
                raise ValueError(f"Se esperaba standard, HDF5: mode={mode}, model={model}")
            mass_eV = np.asarray([0.0], dtype=np.float64)
            couplings = np.asarray([0.0], dtype=np.float64)
            log_mass = np.asarray([0.0], dtype=np.float64)  # placeholder de indice
            log_c = np.asarray([0.0], dtype=np.float64)     # placeholder de indice
            kind = "none"
        else:
            if mode != "lri" or model != requested_model:
                raise ValueError(f"Se esperaba {requested_model}, HDF5: mode={mode}, model={model}")
            for name in ("mass_eV", "coupling_values"):
                require_hdf5_path(file, name)
            mass_eV = np.asarray(file["mass_eV"], dtype=np.float64)
            couplings = np.asarray(file["coupling_values"], dtype=np.float64)
            validate_strictly_increasing(mass_eV, "mass_eV")
            if np.any(mass_eV <= 0):
                raise ValueError("mass_eV debe ser positiva")
            if not np.all(np.isfinite(couplings)) or np.any(couplings == 0):
                raise ValueError("Los acoplamientos LRI deben ser finitos y no nulos")
            kind = attr_str("coupling_kind")
            if kind not in ("gprime", "bigg"):
                raise ValueError(f"coupling_kind desconocido: {kind}")
            if kind == "gprime" and np.any(couplings <= 0):
                raise ValueError("gprime debe ser positivo")
            if np.any(np.sign(couplings) != np.sign(couplings[0])):
                raise ValueError("Acoplamientos de signos mixtos; no hay un único eje log|c|")
            log_mass = np.log10(mass_eV)
            log_c = np.log10(np.abs(couplings))
            validate_strictly_increasing(log_mass, "log10_mass_eV")
            validate_strictly_increasing(log_c, "log10_abs_coupling")

        energy = np.asarray(file["energy_GeV"], dtype=np.float64)
        coszen = np.asarray(file["coszen"], dtype=np.float64)
        validate_strictly_increasing(energy, "energy_GeV")
        validate_strictly_increasing(coszen, "coszen")
        if np.any(energy <= 0):
            raise ValueError("energy_GeV debe ser positiva")
        if coszen[0] < -1.0 - 1e-12 or coszen[-1] > 1.0 + 1e-12:
            raise ValueError("coszen fuera de [-1,1]")

        if mode == "standard":
            expected_flux_shape = (coszen.size, energy.size, len(SPECIES))
            expected_done_shape = (coszen.size,)
        else:
            expected_flux_shape = (mass_eV.size, couplings.size,
                                   coszen.size, energy.size, len(SPECIES))
            expected_done_shape = (mass_eV.size, couplings.size, coszen.size)
        if file["flux"].shape != expected_flux_shape:
            raise ValueError(f"/flux tiene shape {file['flux'].shape}; se esperaba {expected_flux_shape}")
        if file["done"].shape != expected_done_shape:
            raise ValueError(f"/done tiene shape {file['done'].shape}; se esperaba {expected_done_shape}")
        done = np.asarray(file["done"], dtype=np.bool_)
        if not np.all(done):
            incomplete = np.argwhere(~done)
            raise RuntimeError(
                f"La propagación {requested_model} aún está incompleta: "
                f"{len(incomplete)} ángulos pendientes; primeros índices "
                f"{incomplete[:10].tolist()}. El constructor necesita toda "
                "la malla angular; un archivo smoke no basta para MC completo."
            )
        if EXPECTED_MASS_COUNT is not None and len(mass_eV) != EXPECTED_MASS_COUNT:
            raise ValueError(f"Número de masas inesperado: {len(mass_eV)}")
        if EXPECTED_G_COUNT is not None and len(couplings) != EXPECTED_G_COUNT:
            raise ValueError(f"Número de acoplamientos inesperado: {len(couplings)}")
        return FluxLUTMetadata(
            model=model, coupling_kind=kind, is_standard=(mode == "standard"),
            source_signature=source_signature,
            mass_eV=mass_eV, log10_mass_eV=log_mass,
            coupling_values=couplings, log10_abs_coupling=log_c,
            energy_GeV=energy, coszen=coszen,
            shape=expected_flux_shape,
        )


# =============================================================================
# Binning de eventos reconstruidos
# =============================================================================

def histogram_bin_index(
    values: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce la convención de numpy.histogram:
      [edge_i, edge_{i+1}) salvo el último bin,
      que incluye el borde superior.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    indices = np.searchsorted(
        edges,
        values,
        side="right",
    ) - 1

    tolerance = (
        1.0e-12
        * max(
            1.0,
            abs(float(edges[-1])),
        )
    )

    on_upper_edge = np.isclose(
        values,
        edges[-1],
        rtol=0.0,
        atol=tolerance,
    )

    indices[on_upper_edge] = (
        edges.size - 2
    )

    valid = (
        np.isfinite(values)
        & (indices >= 0)
        & (
            indices
            < edges.size - 1
        )
    )

    return (
        indices.astype(np.int64),
        valid,
    )


def compute_flat_reco_bins(
    reco_energy: np.ndarray,
    reco_coszen: np.ndarray,
    pid: np.ndarray,
    binning: Binning,
) -> tuple[np.ndarray, np.ndarray]:

    energy_bin, valid_energy = (
        histogram_bin_index(
            reco_energy,
            binning.reco_energy_edges_GeV,
        )
    )

    coszen_bin, valid_coszen = (
        histogram_bin_index(
            reco_coszen,
            binning.reco_coszen_edges,
        )
    )

    pid_bin, valid_pid = (
        histogram_bin_index(
            pid,
            binning.pid_edges,
        )
    )

    valid = (
        valid_energy
        & valid_coszen
        & valid_pid
    )

    flat_bin = (
        (
            energy_bin
            * binning.hist_shape[1]
            + coszen_bin
        )
        * binning.hist_shape[2]
        + pid_bin
    )

    flat_bin[~valid] = -1

    return (
        flat_bin.astype(np.int64),
        valid,
    )


# =============================================================================
# Coordenadas de interpolación en la LUT de flujo
# =============================================================================

def interpolation_coordinates(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    grid_name: str,
    value_name: str,
    use_log10: bool,
) -> tuple[np.ndarray, np.ndarray]:

    grid = np.asarray(
        grid,
        dtype=np.float64,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{value_name} contiene valores no finitos."
        )

    if use_log10:
        if (
            np.any(grid <= 0.0)
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                f"{grid_name} y {value_name} deben "
                "ser > 0 para interpolar en log10."
            )

        working_grid = np.log10(
            grid
        )

        working_values = np.log10(
            values
        )
    else:
        working_grid = grid
        working_values = values

    tolerance = (
        1.0e-11
        * max(
            1.0,
            abs(float(working_grid[0])),
            abs(float(working_grid[-1])),
        )
    )

    below = (
        working_values
        < working_grid[0]
        - tolerance
    )

    above = (
        working_values
        > working_grid[-1]
        + tolerance
    )

    if np.any(below) or np.any(above):
        n_below = int(
            np.count_nonzero(below)
        )

        n_above = int(
            np.count_nonzero(above)
        )

        raise ValueError(
            f"{value_name} queda fuera de "
            f"{grid_name}. "
            f"Debajo={n_below}, encima={n_above}; "
            f"rango values=["
            f"{np.min(working_values):.18e}, "
            f"{np.max(working_values):.18e}], "
            f"rango grid=["
            f"{working_grid[0]:.18e}, "
            f"{working_grid[-1]:.18e}]. "
            "No se hace extrapolación."
        )

    clipped = np.clip(
        working_values,
        working_grid[0],
        working_grid[-1],
    )

    lower = (
        np.searchsorted(
            working_grid,
            clipped,
            side="right",
        )
        - 1
    )

    lower = np.clip(
        lower,
        0,
        working_grid.size - 2,
    ).astype(np.int64)

    denominator = (
        working_grid[lower + 1]
        - working_grid[lower]
    )

    fraction = (
        clipped
        - working_grid[lower]
    ) / denominator

    fraction = np.clip(
        fraction,
        0.0,
        1.0,
    )

    return (
        lower,
        fraction.astype(np.float64),
    )


# =============================================================================
# MC de neutrinos
# =============================================================================

def load_mc_cache(
    icecube_directory: Path,
    metadata: FluxLUTMetadata,
    binning: Binning,
) -> MCEventCache:

    frames: list[pd.DataFrame] = []
    source_counts_total: dict[str, int] = {}

    for category in CATEGORIES:
        filename = (
            icecube_file(icecube_directory, CATEGORY_FILES[category])
        )

        dataframe = pd.read_csv(
            filename,
            low_memory=False,
        )

        missing_columns = sorted(
            set(MC_REQUIRED_COLUMNS)
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Faltan columnas en {filename}: "
                f"{missing_columns}"
            )

        selected = dataframe[
            list(MC_REQUIRED_COLUMNS)
        ].copy()

        selected["category_id"] = (
            CATEGORY_TO_ID[category]
        )

        selected[
            "source_row_index"
        ] = np.arange(
            selected.shape[0],
            dtype=np.int64,
        )

        frames.append(selected)

        source_counts_total[
            category
        ] = int(
            selected.shape[0]
        )

    mc = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    pdg = mc[
        "pdg"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    unknown_pdg = sorted(
        set(
            np.unique(pdg).tolist()
        )
        - set(PDG_TO_SPECIES)
    )

    if unknown_pdg:
        raise ValueError(
            "PDG no soportados en el MC: "
            f"{unknown_pdg}"
        )

    species_id_all = np.asarray(
        [
            SPECIES_TO_ID[
                PDG_TO_SPECIES[
                    int(code)
                ]
            ]
            for code in pdg
        ],
        dtype=np.int8,
    )

    category_id_all = mc[
        "category_id"
    ].to_numpy(
        dtype=np.int8,
        copy=True,
    )

    reco_bin_id_all, valid_reco = (
        compute_flat_reco_bins(
            mc[
                "reco_energy"
            ].to_numpy(
                dtype=np.float64,
            ),
            mc[
                "reco_coszen"
            ].to_numpy(
                dtype=np.float64,
            ),
            mc[
                "pid"
            ].to_numpy(
                dtype=np.float64,
            ),
            binning,
        )
    )

    valid_indices = np.flatnonzero(
        valid_reco
    )

    if valid_indices.size == 0:
        raise RuntimeError(
            "Ningún evento MC cae dentro "
            "de los bins reconstruidos."
        )

    source_counts_inside_bins = {
        category: int(
            np.count_nonzero(
                valid_reco
                & (
                    category_id_all
                    == category_id
                )
            )
        )
        for category, category_id
        in CATEGORY_TO_ID.items()
    }

    def take_float64(
        column: str,
        *,
        require_finite: bool = True,
    ) -> np.ndarray:
        values = mc[
            column
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )[valid_indices]

        if (
            require_finite
            and not np.all(
                np.isfinite(values)
            )
        ):
            raise ValueError(
                f"{column} contiene valores "
                "no finitos entre los eventos usados."
            )

        return values

    def take_int16(
        column: str,
    ) -> np.ndarray:
        return mc[
            column
        ].to_numpy(
            dtype=np.int16,
            copy=True,
        )[valid_indices]

    true_energy = take_float64(
        "true_energy"
    )

    true_coszen = take_float64(
        "true_coszen"
    )

    base_mc_weight = take_float64(
        "weight"
    )

    if np.any(
        base_mc_weight < 0.0
    ):
        raise ValueError(
            "Hay pesos MC negativos."
        )

    if np.any(
        true_energy <= 0.0
    ):
        raise ValueError(
            "Hay true_energy <= 0."
        )

    if np.any(
        (true_coszen < -1.0 - 1.0e-10)
        | (
            true_coszen
            > 1.0 + 1.0e-10
        )
    ):
        raise ValueError(
            "Hay true_coszen fuera de [-1,1]."
        )

    energy_lower, energy_fraction = (
        interpolation_coordinates(
            metadata.energy_GeV,
            true_energy,
            grid_name="flux energy_GeV",
            value_name=(
                "MC true_energy "
                "(eventos dentro de reco bins)"
            ),
            use_log10=True,
        )
    )

    coszen_lower, coszen_fraction = (
        interpolation_coordinates(
            metadata.coszen,
            true_coszen,
            grid_name="flux coszen",
            value_name=(
                "MC true_coszen "
                "(eventos dentro de reco bins)"
            ),
            use_log10=False,
        )
    )

    species_id = species_id_all[
        valid_indices
    ]

    category_id = category_id_all[
        valid_indices
    ]

    reco_bin_id = reco_bin_id_all[
        valid_indices
    ]

    pdg_valid = pdg[
        valid_indices
    ].astype(
        np.int16,
        copy=False,
    )

    species_event_indices = tuple(
        np.flatnonzero(
            species_id == species_index
        )
        for species_index
        in range(len(SPECIES))
    )

    category_event_indices = tuple(
        np.flatnonzero(
            category_id == category_index
        )
        for category_index
        in range(len(CATEGORIES))
    )

    spectral_log_energy_ratio = (
        np.log(
            true_energy
            / SPECTRAL_INDEX_PIVOT_GEV
        )
    )

    return MCEventCache(
        true_energy_GeV=true_energy,
        true_coszen=true_coszen,
        base_mc_weight=base_mc_weight,
        species_id=species_id,
        category_id=category_id,
        pdg=pdg_valid,
        reco_bin_id=reco_bin_id,
        event_type=take_int16("type"),
        interaction=take_int16(
            "interaction"
        ),
        ma_ccqe_linear=take_float64(
            "MaCCQE_linear"
        ),
        ma_ccqe_quad=take_float64(
            "MaCCQE_quad"
        ),
        ma_ccres_linear=take_float64(
            "MaCCRES_linear"
        ),
        ma_ccres_quad=take_float64(
            "MaCCRES_quad"
        ),
        # Estas variables cinemáticas pueden contener NaN en eventos para
        # los que el generador no define la cantidad. Se conservan tal cual;
        # un futuro reweight DIS deberá seleccionar primero los eventos DIS.
        q2=take_float64(
            "Q2",
            require_finite=False,
        ),
        w_hadronic=take_float64(
            "W",
            require_finite=False,
        ),
        bjorken_x=take_float64(
            "x",
            require_finite=False,
        ),
        inelasticity_y=take_float64(
            "y",
            require_finite=False,
        ),
        spectral_log_energy_ratio=(
            spectral_log_energy_ratio
        ),
        energy_lower=energy_lower,
        energy_fraction=energy_fraction,
        coszen_lower=coszen_lower,
        coszen_fraction=coszen_fraction,
        species_event_indices=(
            species_event_indices
        ),
        category_event_indices=(
            category_event_indices
        ),
        source_counts_total=(
            source_counts_total
        ),
        source_counts_inside_bins=(
            source_counts_inside_bins
        ),
    )


# =============================================================================
# Interpolación de Phi(E_true, coszen_true) dentro de un nodo (m,g)
# =============================================================================

def evaluate_log_bilinear_flux(
    flux_coszen_energy: np.ndarray,
    cache: MCEventCache,
    event_indices: np.ndarray,
) -> np.ndarray:
    """
    Interpola, para eventos de una especie dada:

        x = log10(E_true)
        y = coszen_true
        z = log(Phi)

    La slice de flujo tiene orden [coszen, energy].
    """

    if event_indices.size == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    flux = np.asarray(
        flux_coszen_energy,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(flux)
    ):
        raise FloatingPointError(
            "La slice de flujo contiene "
            "valores no finitos."
        )

    if np.any(
        flux < 0.0
    ):
        raise FloatingPointError(
            "La slice de flujo contiene "
            "valores negativos."
        )

    i_energy = cache.energy_lower[
        event_indices
    ]

    i_coszen = cache.coszen_lower[
        event_indices
    ]

    t_energy = cache.energy_fraction[
        event_indices
    ]

    t_coszen = cache.coszen_fraction[
        event_indices
    ]

    phi_00 = flux[
        i_coszen,
        i_energy,
    ]

    phi_10 = flux[
        i_coszen,
        i_energy + 1,
    ]

    phi_01 = flux[
        i_coszen + 1,
        i_energy,
    ]

    phi_11 = flux[
        i_coszen + 1,
        i_energy + 1,
    ]

    positive = flux[
        flux > 0.0
    ]

    if positive.size == 0:
        return np.zeros(
            event_indices.size,
            dtype=np.float64,
        )

    flux_floor = max(
        np.finfo(np.float64).tiny,
        float(
            np.min(positive)
        )
        * 1.0e-12,
    )

    log_00 = np.log(
        np.maximum(
            phi_00,
            flux_floor,
        )
    )

    log_10 = np.log(
        np.maximum(
            phi_10,
            flux_floor,
        )
    )

    log_01 = np.log(
        np.maximum(
            phi_01,
            flux_floor,
        )
    )

    log_11 = np.log(
        np.maximum(
            phi_11,
            flux_floor,
        )
    )

    one_minus_energy = (
        1.0 - t_energy
    )

    one_minus_coszen = (
        1.0 - t_coszen
    )

    interpolated_log_flux = (
        one_minus_energy
        * one_minus_coszen
        * log_00
        + t_energy
        * one_minus_coszen
        * log_10
        + one_minus_energy
        * t_coszen
        * log_01
        + t_energy
        * t_coszen
        * log_11
    )

    result = np.exp(
        interpolated_log_flux
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise FloatingPointError(
            "La interpolación de flujo produjo "
            "valores no finitos."
        )

    return result


def build_event_weights_for_node(
    flux_file: h5py.File,
    metadata: FluxLUTMetadata,
    mass_index: int,
    g_index: int,
    cache: MCEventCache,
    nucount: float,
) -> np.ndarray:
    """Un nodo completo se lee UNA VEZ (las seis especies juntas)."""
    if metadata.is_standard:
        flux_node = np.asarray(flux_file["flux"][...], dtype=np.float64)
    else:
        flux_node = np.asarray(
            flux_file["flux"][mass_index, g_index, :, :, :], dtype=np.float64)
    expected_shape = (metadata.coszen.size, metadata.energy_GeV.size, len(SPECIES))
    if flux_node.shape != expected_shape:
        raise ValueError(f"Shape de flujo incorrecta: {flux_node.shape}; esperado {expected_shape}")

    event_flux = np.zeros(cache.size, dtype=np.float64)
    for species_index in range(len(SPECIES)):
        indices = cache.species_event_indices[species_index]
        if len(indices):
            event_flux[indices] = evaluate_log_bilinear_flux(
                flux_node[:, :, species_index], cache, indices)
    event_weight = (nucount * cache.base_mc_weight *
                    FLUX_TO_ICECUBE_WEIGHT_FACTOR * event_flux)
    if not np.all(np.isfinite(event_weight)) or np.any(event_weight < 0.0):
        raise FloatingPointError("Pesos nominales no finitos o negativos")
    return event_weight


# =============================================================================
# Histogramas nominales
# =============================================================================

def histogram_event_weights(
    event_weight: np.ndarray,
    cache: MCEventCache,
    binning: Binning,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    templates: dict[
        str,
        np.ndarray,
    ] = {}

    sumw2: dict[
        str,
        np.ndarray,
    ] = {}

    for category_index, category in enumerate(
        CATEGORIES
    ):
        indices = (
            cache.category_event_indices[
                category_index
            ]
        )

        bins = cache.reco_bin_id[
            indices
        ]

        weights = event_weight[
            indices
        ]

        templates[
            category
        ] = np.bincount(
            bins,
            weights=weights,
            minlength=binning.n_bins,
        ).astype(np.float64)

        sumw2[
            category
        ] = np.bincount(
            bins,
            weights=(
                weights
                * weights
            ),
            minlength=binning.n_bins,
        ).astype(np.float64)

    return (
        templates,
        sumw2,
    )


# =============================================================================
# Datos observados y muones
# =============================================================================

def require_exact_one_row_per_bin(
    coordinates: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    binning: Binning,
    description: str,
) -> None:
    occupancy, _ = np.histogramdd(
        coordinates,
        bins=(
            binning.reco_energy_edges_GeV,
            binning.reco_coszen_edges,
            binning.pid_edges,
        ),
    )

    if not np.all(
        occupancy == 1.0
    ):
        unique, counts = np.unique(
            occupancy,
            return_counts=True,
        )

        summary = {
            float(value): int(count)
            for value, count
            in zip(unique, counts)
        }

        raise ValueError(
            f"{description} no tiene exactamente "
            "una fila por bin. "
            f"Ocupaciones: {summary}"
        )


def build_observed_histogram(
    icecube_directory: Path,
    binning: Binning,
) -> np.ndarray:
    filename = (
        icecube_file(icecube_directory, "data.csv")
    )

    dataframe = pd.read_csv(
        filename
    )

    required = {
        "reco_energy",
        "reco_coszen",
        "pid",
        "count",
    }

    missing = sorted(
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas en {filename}: "
            f"{missing}"
        )

    coordinates = (
        dataframe[
            "reco_energy"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "reco_coszen"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "pid"
        ].to_numpy(
            dtype=np.float64,
        ),
    )

    require_exact_one_row_per_bin(
        coordinates,
        binning,
        "data.csv",
    )

    histogram, _ = np.histogramdd(
        coordinates,
        bins=(
            binning.reco_energy_edges_GeV,
            binning.reco_coszen_edges,
            binning.pid_edges,
        ),
        weights=dataframe[
            "count"
        ].to_numpy(
            dtype=np.float64,
        ),
    )

    return histogram.reshape(-1)


def build_muon_histograms(
    icecube_directory: Path,
    binning: Binning,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    filename = (
        icecube_file(icecube_directory, "mc_mu.csv")
    )

    dataframe = pd.read_csv(
        filename
    )

    required = {
        "reco_energy",
        "reco_coszen",
        "pid",
        "count",
        "abs_uncertainty",
    }

    missing = sorted(
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas en {filename}: "
            f"{missing}"
        )

    coordinates = (
        dataframe[
            "reco_energy"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "reco_coszen"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "pid"
        ].to_numpy(
            dtype=np.float64,
        ),
    )

    require_exact_one_row_per_bin(
        coordinates,
        binning,
        "mc_mu.csv",
    )

    counts = dataframe[
        "count"
    ].to_numpy(
        dtype=np.float64,
    )

    absolute_uncertainty = dataframe[
        "abs_uncertainty"
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(counts)
    ):
        raise ValueError(
            "mc_mu.csv contiene count "
            "no finito."
        )

    if not np.all(
        np.isfinite(
            absolute_uncertainty
        )
    ):
        raise ValueError(
            "mc_mu.csv contiene "
            "abs_uncertainty no finito."
        )

    if np.any(
        counts < 0.0
    ):
        raise ValueError(
            "mc_mu.csv contiene count < 0."
        )

    if np.any(
        absolute_uncertainty < 0.0
    ):
        raise ValueError(
            "mc_mu.csv contiene "
            "abs_uncertainty < 0."
        )

    muon_histogram, _ = (
        np.histogramdd(
            coordinates,
            bins=(
                binning.reco_energy_edges_GeV,
                binning.reco_coszen_edges,
                binning.pid_edges,
            ),
            weights=counts,
        )
    )

    # Este dataset se llama sumw2 para ser consumido de forma natural por
    # el likelihood, pero físicamente contiene la varianza KDE publicada:
    # sigma_mu^2 = abs_uncertainty^2.
    muon_variance, _ = (
        np.histogramdd(
            coordinates,
            bins=(
                binning.reco_energy_edges_GeV,
                binning.reco_coszen_edges,
                binning.pid_edges,
            ),
            weights=(
                absolute_uncertainty
                * absolute_uncertainty
            ),
        )
    )

    return (
        muon_histogram.reshape(-1),
        muon_variance.reshape(-1),
    )


# =============================================================================
# Hypersurfaces detectorales
# =============================================================================

def interpolate_hypersurface_in_dm31(
    dataframe: pd.DataFrame,
    dm31_target: float,
    filename: Path,
) -> pd.DataFrame:

    coordinate_columns = (
        "reco_energy",
        "reco_coszen",
        "pid",
    )

    required_columns = {
        "deltam31",
        *coordinate_columns,
        *HYPERSURFACE_VALUE_COLUMNS,
    }

    missing = sorted(
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Faltan columnas en {filename}: "
            f"{missing}"
        )

    dm31_grid = np.asarray(
        sorted(
            dataframe[
                "deltam31"
            ].unique()
        ),
        dtype=np.float64,
    )

    if dm31_grid.size == 0:
        raise ValueError(
            f"No hay valores deltam31 "
            f"en {filename}."
        )

    if (
        dm31_target
        < dm31_grid[0]
        or dm31_target
        > dm31_grid[-1]
    ):
        raise ValueError(
            f"dm31={dm31_target:.18e} queda "
            f"fuera de [{dm31_grid[0]:.18e}, "
            f"{dm31_grid[-1]:.18e}] "
            f"en {filename}."
        )

    exact = np.where(
        np.isclose(
            dm31_grid,
            dm31_target,
            rtol=1.0e-12,
            atol=0.0,
        )
    )[0]

    sorting_columns = list(
        coordinate_columns
    )

    if exact.size:
        selected_dm31 = float(
            dm31_grid[
                int(exact[0])
            ]
        )

        result = (
            dataframe.loc[
                np.isclose(
                    dataframe[
                        "deltam31"
                    ].to_numpy(
                        dtype=np.float64,
                    ),
                    selected_dm31,
                    rtol=1.0e-12,
                    atol=0.0,
                )
            ]
            .sort_values(
                sorting_columns
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        result[
            "deltam31"
        ] = dm31_target

        return result

    upper_index = int(
        np.searchsorted(
            dm31_grid,
            dm31_target,
            side="right",
        )
    )

    lower_dm31 = float(
        dm31_grid[
            upper_index - 1
        ]
    )

    upper_dm31 = float(
        dm31_grid[
            upper_index
        ]
    )

    lower = (
        dataframe.loc[
            np.isclose(
                dataframe[
                    "deltam31"
                ].to_numpy(
                    dtype=np.float64,
                ),
                lower_dm31,
                rtol=1.0e-12,
                atol=0.0,
            )
        ]
        .sort_values(
            sorting_columns
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    upper = (
        dataframe.loc[
            np.isclose(
                dataframe[
                    "deltam31"
                ].to_numpy(
                    dtype=np.float64,
                ),
                upper_dm31,
                rtol=1.0e-12,
                atol=0.0,
            )
        ]
        .sort_values(
            sorting_columns
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    if (
        lower.shape[0]
        != upper.shape[0]
    ):
        raise ValueError(
            "Los planos dm31 vecinos tienen "
            f"distinto número de filas en "
            f"{filename}."
        )

    for coordinate in (
        coordinate_columns
    ):
        if not np.allclose(
            lower[
                coordinate
            ].to_numpy(
                dtype=np.float64,
            ),
            upper[
                coordinate
            ].to_numpy(
                dtype=np.float64,
            ),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Las coordenadas "
                f"{coordinate} no coinciden "
                "entre planos dm31 en "
                f"{filename}."
            )

    fraction = (
        dm31_target
        - lower_dm31
    ) / (
        upper_dm31
        - lower_dm31
    )

    result = lower.copy()

    result[
        "deltam31"
    ] = dm31_target

    for column in (
        HYPERSURFACE_VALUE_COLUMNS
    ):
        result[
            column
        ] = (
            (1.0 - fraction)
            * lower[
                column
            ].to_numpy(
                dtype=np.float64,
            )
            + fraction
            * upper[
                column
            ].to_numpy(
                dtype=np.float64,
            )
        )

    return result


def hypersurface_columns_to_histograms(
    dataframe: pd.DataFrame,
    filename: Path,
    binning: Binning,
) -> dict[str, np.ndarray]:

    coordinates = (
        dataframe[
            "reco_energy"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "reco_coszen"
        ].to_numpy(
            dtype=np.float64,
        ),
        dataframe[
            "pid"
        ].to_numpy(
            dtype=np.float64,
        ),
    )

    require_exact_one_row_per_bin(
        coordinates,
        binning,
        str(filename),
    )

    result: dict[
        str,
        np.ndarray,
    ] = {}

    for column in (
        HYPERSURFACE_VALUE_COLUMNS
    ):
        values = dataframe[
            column
        ].to_numpy(
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{column} contiene valores "
                f"no finitos en {filename}."
            )

        histogram, _ = (
            np.histogramdd(
                coordinates,
                bins=(
                    binning.reco_energy_edges_GeV,
                    binning.reco_coszen_edges,
                    binning.pid_edges,
                ),
                weights=values,
            )
        )

        result[
            column
        ] = histogram.reshape(-1)

    return result


def load_all_hypersurfaces(
    icecube_directory: Path,
    dm31_target: float,
    binning: Binning,
) -> dict[
    str,
    dict[str, np.ndarray],
]:

    cache_by_filename: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    result: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for category in CATEGORIES:
        csv_name = (
            HYPERSURFACE_FILES[
                category
            ]
        )

        if (
            csv_name
            not in cache_by_filename
        ):
            filename = (
                icecube_file(icecube_directory, csv_name)
            )

            dataframe = pd.read_csv(
                filename
            )

            interpolated = (
                interpolate_hypersurface_in_dm31(
                    dataframe,
                    dm31_target,
                    filename,
                )
            )

            cache_by_filename[
                csv_name
            ] = (
                hypersurface_columns_to_histograms(
                    interpolated,
                    filename,
                    binning,
                )
            )

        result[
            category
        ] = {
            name: values.copy()
            for name, values
            in cache_by_filename[
                csv_name
            ].items()
        }

    return result


# =============================================================================
# Creación de la estructura HDF5
# =============================================================================

def write_event_cache(
    h5: h5py.File,
    cache: MCEventCache,
) -> None:
    group = h5.create_group(
        "event_cache"
    )

    group.attrs[
        "number_events"
    ] = cache.size

    group.attrs[
        "selection"
    ] = (
        "Only neutrino MC events whose reconstructed "
        "variables lie inside the 10x10x2 analysis bins."
    )

    group.attrs[
        "spectral_index_pivot_GeV"
    ] = SPECTRAL_INDEX_PIVOT_GEV

    group.attrs[
        "spectral_log_energy_ratio_definition"
    ] = (
        "ln(true_energy_GeV / spectral_index_pivot_GeV)"
    )

    group.attrs[
        "ma_reweight_note"
    ] = (
        "Raw MaCCQE_linear, MaCCQE_quad, MaCCRES_linear and "
        "MaCCRES_quad coefficients from the public IceCube MC "
        "are stored unchanged. build_templates.py does not assume the "
        "polynomial convention; it is applied in the likelihood "
        "only after that convention is verified."
    )

    group.attrs[
        "dis_csms_note"
    ] = (
        "true_energy_GeV and inelasticity_y, plus Q2, W and x, "
        "are retained so the official DIS CSMS reweight can be "
        "added later without rebuilding this file."
    )

    options = aux_dataset_options()

    datasets = {
        "true_energy_GeV": (
            cache.true_energy_GeV,
            np.float64,
        ),
        "true_coszen": (
            cache.true_coszen,
            np.float64,
        ),
        "base_mc_weight": (
            cache.base_mc_weight,
            np.float64,
        ),
        "species_id": (
            cache.species_id,
            np.int8,
        ),
        "category_id": (
            cache.category_id,
            np.int8,
        ),
        "pdg": (
            cache.pdg,
            np.int16,
        ),
        "reco_bin_id": (
            cache.reco_bin_id,
            np.int16,
        ),
        "event_type": (
            cache.event_type,
            np.int16,
        ),
        "interaction": (
            cache.interaction,
            np.int16,
        ),
        "MaCCQE_linear": (
            cache.ma_ccqe_linear,
            np.float32,
        ),
        "MaCCQE_quad": (
            cache.ma_ccqe_quad,
            np.float32,
        ),
        "MaCCRES_linear": (
            cache.ma_ccres_linear,
            np.float32,
        ),
        "MaCCRES_quad": (
            cache.ma_ccres_quad,
            np.float32,
        ),
        "Q2": (
            cache.q2,
            np.float32,
        ),
        "W": (
            cache.w_hadronic,
            np.float32,
        ),
        "x": (
            cache.bjorken_x,
            np.float32,
        ),
        "y": (
            cache.inelasticity_y,
            np.float32,
        ),
        "spectral_log_energy_ratio": (
            cache.spectral_log_energy_ratio,
            np.float64,
        ),
    }

    for name, (
        values,
        dtype,
    ) in datasets.items():
        group.create_dataset(
            name,
            data=np.asarray(
                values,
                dtype=dtype,
            ),
            **options,
        )

    create_string_dataset(
        group,
        "species_labels",
        SPECIES,
    )

    create_string_dataset(
        group,
        "category_labels",
        CATEGORIES,
    )

    counts_total = group.create_group(
        "source_counts_total"
    )

    counts_inside = group.create_group(
        "source_counts_inside_bins"
    )

    for category in CATEGORIES:
        counts_total.attrs[
            category
        ] = (
            cache.source_counts_total[
                category
            ]
        )

        counts_inside.attrs[
            category
        ] = (
            cache.source_counts_inside_bins[
                category
            ]
        )


def write_nuisance_support(
    h5: h5py.File,
) -> None:
    group = h5.create_group(
        "nuisance_support"
    )

    create_string_dataset(
        group,
        "post_propagation_supported",
        SUPPORTED_POST_PROPAGATION_NUISANCES,
    )

    create_string_dataset(
        group,
        "omitted_requires_mceq_flux_derivatives",
        OMITTED_BARR_NUISANCES,
    )

    create_string_dataset(
        group,
        "fixed_requires_new_oscillation_propagation",
        FIXED_OSCILLATION_PARAMETERS,
    )

    create_string_dataset(
        group,
        "pending_exact_reweight_definition",
        ("dis_csms",),
    )

    group.attrs[
        "delta_gamma_nu_note"
    ] = (
        "The event-level cache stores ln(E_true/24 GeV), allowing "
        "a post-propagation spectral-index reweight. With transport "
        "interactions enabled this is an approximation to changing "
        "the atmospheric spectrum before propagation; no new "
        "nuSQuIDS runs are performed."
    )

    group.attrs[
        "barr_note"
    ] = (
        "The six Barr/hadron-yield parameters are intentionally "
        "not floated because the required MCEq flux derivatives "
        "are not present in the current propagated-flux LUT."
    )

    group.attrs[
        "oscillation_note"
    ] = (
        "theta23 and dm31 are fixed to the values used to generate "
        "oscillated_flux_lut.h5. Floating them would require new "
        "oscillation propagations in the current pipeline."
    )

    group.attrs[
        "dis_csms_note"
    ] = (
        "Event-level E_true, y, Q2, W and x are retained, but no "
        "numerical GENIE-to-CSMS correction is invented here."
    )


def initialize_output_file(
    output_filename: Path,
    flux_lut_filename: Path,
    icecube_directory: Path,
    metadata: FluxLUTMetadata,
    binning: Binning,
    detector_nominal: dict[str, float],
    detector_bounds: dict[
        str,
        tuple[float, float],
    ],
    observed: np.ndarray,
    muon_template: np.ndarray,
    muon_variance: np.ndarray,
    hypersurfaces: dict[
        str,
        dict[str, np.ndarray],
    ],
    cache: MCEventCache,
) -> None:

    output_filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        output_filename,
        "w",
    ) as h5:

        # -----------------------------------------------------------------
        # Root metadata
        # -----------------------------------------------------------------

        h5.attrs[
            "format_name"
        ] = FORMAT_NAME

        h5.attrs[
            "format_version"
        ] = FORMAT_VERSION

        h5.attrs[
            "build_completed"
        ] = False

        h5.attrs[
            "created_unix_time"
        ] = time.time()

        h5.attrs[
            "source_flux_lut"
        ] = str(
            flux_lut_filename.resolve()
        )

        h5.attrs[
            "source_icecube_directory"
        ] = str(
            icecube_directory.resolve()
        )

        h5.attrs[
            "dm31_fixed_eV2"
        ] = DM31_FIXED_EV2

        h5.attrs[
            "flux_to_icecube_weight_factor"
        ] = (
            FLUX_TO_ICECUBE_WEIGHT_FACTOR
        )

        h5.attrs[
            "nucount"
        ] = binning.nucount

        h5.attrs[
            "hist_shape"
        ] = binning.hist_shape

        h5.attrs[
            "n_hist_bins"
        ] = binning.n_bins

        h5.attrs[
            "event_weight_storage_dtype"
        ] = "float32"

        h5.attrs[
            "event_weight_definition"
        ] = (
            "NUCOUNT * MC_weight * "
            "FLUX_TO_ICECUBE_WEIGHT_FACTOR * "
            "Phi_final_nuSQuIDS(E_true, coszen_true)"
        )

        h5.attrs[
            "flux_interpolation"
        ] = (
            "bilinear in log10(E_true), coszen_true, log(Phi)"
        )

        h5.attrs["model"] = metadata.model
        h5.attrs["coupling_kind"] = metadata.coupling_kind
        h5.attrs["source_flux_signature"] = metadata.source_signature
        h5.attrs["standard_dummy_axes"] = metadata.is_standard
        h5.attrs["icecube_inputs_sha256"] = input_files_fingerprint(icecube_directory)
        h5.attrs["lri_interpolation_recommendation"] = (
            "Interpolate event-weight vectors on (log10_mass_eV, "
            "log10_abs_coupling) with fixed coupling sign; then apply "
            "event-level nuisances, then histogram. Standard is an independent 1x1 baseline."
        )

        h5.attrs[
            "cross_sumw_present"
        ] = False

        h5.attrs[
            "old_histogram_only_likelihood_compatible"
        ] = False

        h5.attrs[
            "likelihood_requirement"
        ] = (
            "Use the event-level format-3 likelihood. The old "
            "cross_sumw-only likelihood cannot apply delta_gamma_nu "
            "or axial-mass event reweights."
        )

        # -----------------------------------------------------------------
        # Axes
        # -----------------------------------------------------------------

        axes = h5.create_group(
            "axes"
        )

        axes.create_dataset(
            "mass_eV",
            data=metadata.mass_eV,
        )

        axes.create_dataset(
            "log10_mass_eV",
            data=metadata.log10_mass_eV,
        )

        axes.create_dataset("coupling_values", data=metadata.coupling_values)
        axes.create_dataset("log10_abs_coupling", data=metadata.log10_abs_coupling)
        axes.attrs["coupling_kind"] = metadata.coupling_kind
        axes.attrs["standard_dummy_axes"] = metadata.is_standard
        if metadata.is_standard:
            axes.attrs["standard_axes_warning"] = (
                "Both axes are dummy indices [0], not physical mass/coupling. "
                "Load event_weights/nominal[0,0,:] as independent standard reference."
            )
        if metadata.coupling_kind == "gprime":
            # Aliases solo cuando el parámetro ES físicamente gprime.
            axes.create_dataset("gprime", data=metadata.coupling_values)
            axes.create_dataset("log10_gprime", data=metadata.log10_abs_coupling)

        axes.create_dataset(
            "flux_energy_GeV",
            data=metadata.energy_GeV,
        )

        axes.create_dataset(
            "flux_coszen",
            data=metadata.coszen,
        )

        axes.create_dataset(
            "reco_energy_edges_GeV",
            data=(
                binning.reco_energy_edges_GeV
            ),
        )

        axes.create_dataset(
            "reco_coszen_edges",
            data=(
                binning.reco_coszen_edges
            ),
        )

        axes.create_dataset(
            "pid_edges",
            data=binning.pid_edges,
        )

        create_string_dataset(
            axes,
            "category_labels",
            CATEGORIES,
        )

        create_string_dataset(
            axes,
            "species_labels",
            SPECIES,
        )

        # -----------------------------------------------------------------
        # Observed data
        # -----------------------------------------------------------------

        observed_group = h5.create_group(
            "observed"
        )

        observed_ds = (
            observed_group.create_dataset(
                "data",
                data=observed,
            )
        )

        observed_ds.attrs[
            "shape_unflattened"
        ] = binning.hist_shape

        observed_ds.attrs[
            "total_count"
        ] = float(
            np.sum(observed)
        )

        # -----------------------------------------------------------------
        # Atmospheric muons
        # -----------------------------------------------------------------

        backgrounds = h5.create_group(
            "backgrounds"
        )

        muon_ds = (
            backgrounds.create_dataset(
                "atmospheric_muon",
                data=muon_template,
            )
        )

        muon_ds.attrs[
            "shape_unflattened"
        ] = binning.hist_shape

        muon_ds.attrs[
            "definition"
        ] = "count from mc_mu.csv"

        muon_var_ds = (
            backgrounds.create_dataset(
                "atmospheric_muon_sumw2",
                data=muon_variance,
            )
        )

        muon_var_ds.attrs[
            "shape_unflattened"
        ] = binning.hist_shape

        muon_var_ds.attrs[
            "definition"
        ] = (
            "Published binwise KDE variance: "
            "(abs_uncertainty from mc_mu.csv)^2"
        )

        muon_var_ds.attrs[
            "scaling_with_muon_norm"
        ] = (
            "variance -> muon_norm^2 * variance"
        )

        # -----------------------------------------------------------------
        # Detector hypersurfaces
        # -----------------------------------------------------------------

        hs_group = h5.create_group(
            "hypersurfaces"
        )

        hs_group.attrs[
            "dm31_fixed_eV2"
        ] = DM31_FIXED_EV2

        hs_group.attrs[
            "correction_definition"
        ] = (
            "intercept + sum_p slope_p * "
            "(parameter_p - nominal_p)"
        )

        hs_group.attrs[
            "sigma_columns_note"
        ] = (
            "The *_sigma and intercept_sigma columns from the "
            "release are retained for provenance/diagnostics. "
            "They are not automatically interpreted as nuisance priors."
        )

        for nuisance in (
            DETECTOR_NUISANCES
        ):
            hs_group.attrs[
                f"nominal_{nuisance}"
            ] = detector_nominal[
                nuisance
            ]

            lower, upper = (
                detector_bounds[
                    nuisance
                ]
            )

            hs_group.attrs[
                f"lower_{nuisance}"
            ] = lower

            hs_group.attrs[
                f"upper_{nuisance}"
            ] = upper

        for category in CATEGORIES:
            category_group = (
                hs_group.create_group(
                    category
                )
            )

            category_group.attrs[
                "source_csv"
            ] = (
                HYPERSURFACE_FILES[
                    category
                ]
            )

            for column in (
                HYPERSURFACE_VALUE_COLUMNS
            ):
                category_group.create_dataset(
                    column,
                    data=(
                        hypersurfaces[
                            category
                        ][
                            column
                        ]
                    ),
                )

        # -----------------------------------------------------------------
        # Event-level cache
        # -----------------------------------------------------------------

        write_event_cache(
            h5,
            cache,
        )

        write_nuisance_support(
            h5
        )

        # -----------------------------------------------------------------
        # Node-dependent event weights
        # -----------------------------------------------------------------

        event_weight_group = (
            h5.create_group(
                "event_weights"
            )
        )

        event_weight_group.attrs[
            "dimension_order"
        ] = "mass, coupling, event"

        event_weight_group.attrs[
            "n_events"
        ] = cache.size

        chunk_size = min(
            EVENT_CHUNK_SIZE,
            cache.size,
        )

        event_weights_ds = (
            event_weight_group.create_dataset(
                "nominal",
                shape=(
                    metadata.mass_eV.size,
                    metadata.coupling_values.size,
                    cache.size,
                ),
                dtype=EVENT_WEIGHT_DTYPE,
                chunks=(
                    1,
                    1,
                    chunk_size,
                ),
                **event_weight_dataset_options(),
            )
        )

        event_weights_ds.attrs[
            "dimension_order"
        ] = "mass, coupling, event"

        event_weights_ds.attrs[
            "definition"
        ] = (
            "Nominal event weights before detector, flux-index "
            "and cross-section nuisance reweights."
        )

        # -----------------------------------------------------------------
        # Nominal node histograms
        # -----------------------------------------------------------------

        templates_group = (
            h5.create_group(
                "templates"
            )
        )

        sumw2_group = h5.create_group(
            "sumw2"
        )

        dataset_shape = (
            metadata.mass_eV.size,
            metadata.coupling_values.size,
            binning.n_bins,
        )

        chunk_shape = (
            1,
            1,
            binning.n_bins,
        )

        options = aux_dataset_options()

        for category in CATEGORIES:
            template_ds = (
                templates_group.create_dataset(
                    category,
                    shape=dataset_shape,
                    dtype=np.float64,
                    chunks=chunk_shape,
                    **options,
                )
            )

            template_ds.attrs[
                "dimension_order"
            ] = (
                "mass, coupling, "
                "flattened_reco_bin"
            )

            template_ds.attrs[
                "shape_unflattened"
            ] = (
                metadata.mass_eV.size,
                metadata.coupling_values.size,
                *binning.hist_shape,
            )

            template_ds.attrs[
                "definition"
            ] = (
                "Nominal validation template reconstructed from "
                "/event_weights/nominal."
            )

            sumw2_ds = (
                sumw2_group.create_dataset(
                    category,
                    shape=dataset_shape,
                    dtype=np.float64,
                    chunks=chunk_shape,
                    **options,
                )
            )

            sumw2_ds.attrs[
                "definition"
            ] = (
                "Nominal sum of squared stored event weights "
                "in each reconstructed bin."
            )

        # -----------------------------------------------------------------
        # Build status
        # -----------------------------------------------------------------

        status = h5.create_group(
            "status"
        )

        status.create_dataset(
            "completed_nodes",
            shape=(
                metadata.mass_eV.size,
                metadata.coupling_values.size,
            ),
            dtype=np.bool_,
            data=np.zeros(
                (
                    metadata.mass_eV.size,
                    metadata.coupling_values.size,
                ),
                dtype=np.bool_,
            ),
        )

        status.attrs[
            "completed_node_definition"
        ] = (
            "True only after event weights, templates and sumw2 "
            "for a complete (mass,coupling) node have been flushed."
        )

        h5.flush()


# =============================================================================
# Reanudación / compatibilidad
# =============================================================================

def is_compatible_v3_file(
    filename: Path,
    metadata: FluxLUTMetadata,
    cache: MCEventCache,
    binning: Binning,
    icecube_directory: Path,
    flux_lut_filename: Path,
) -> bool:

    try:
        with h5py.File(
            filename,
            "r",
        ) as h5:
            if (
                str(
                    h5.attrs.get(
                        "format_name",
                        "",
                    )
                )
                != FORMAT_NAME
            ):
                return False

            if (
                str(
                    h5.attrs.get(
                        "format_version",
                        "",
                    )
                )
                != FORMAT_VERSION
            ):
                return False

            if str(h5.attrs.get("model", "")) != metadata.model:
                return False
            if str(h5.attrs.get("coupling_kind", "")) != metadata.coupling_kind:
                return False
            if str(h5.attrs.get("source_flux_signature", "")) != metadata.source_signature:
                return False
            if str(h5.attrs.get("source_flux_lut", "")) != str(flux_lut_filename.resolve()):
                return False
            if str(h5.attrs.get("icecube_inputs_sha256", "")) != input_files_fingerprint(icecube_directory):
                return False

            required = (
                "axes/mass_eV",
                "axes/coupling_values",
                "event_cache/reco_bin_id",
                "event_weights/nominal",
                "templates",
                "sumw2",
                "status/completed_nodes",
                "backgrounds/atmospheric_muon_sumw2",
            )

            for path in required:
                if path not in h5:
                    return False

            if not np.array_equal(
                h5[
                    "axes/mass_eV"
                ][...],
                metadata.mass_eV,
            ):
                return False

            if not np.array_equal(
                h5[
                    "axes/coupling_values"
                ][...],
                metadata.coupling_values,
            ):
                return False

            if (
                h5[
                    "event_cache/reco_bin_id"
                ].shape
                != (cache.size,)
            ):
                return False

            if (
                h5[
                    "event_weights/nominal"
                ].shape
                != (
                    metadata.mass_eV.size,
                    metadata.coupling_values.size,
                    cache.size,
                )
            ):
                return False

            if (
                tuple(
                    int(value)
                    for value
                    in h5.attrs[
                        "hist_shape"
                    ]
                )
                != binning.hist_shape
            ):
                return False

    except Exception:
        return False

    return True




# =============================================================================
# Validación del archivo estático al reanudar
# =============================================================================

def validate_resume_event_cache(
    h5: h5py.File,
    cache: MCEventCache,
) -> None:
    checks = {
        "event_cache/reco_bin_id": (
            cache.reco_bin_id
        ),
        "event_cache/species_id": (
            cache.species_id
        ),
        "event_cache/category_id": (
            cache.category_id
        ),
        "event_cache/pdg": (
            cache.pdg
        ),
    }

    for path, reference in checks.items():
        stored = h5[
            path
        ][...]

        if not np.array_equal(
            stored,
            reference,
        ):
            raise RuntimeError(
                "El archivo v3.1 existente no coincide "
                f"con el MC actual en {path}. "
                "No es seguro reanudar."
            )

    for path, reference in (
        (
            "event_cache/true_energy_GeV",
            cache.true_energy_GeV,
        ),
        (
            "event_cache/true_coszen",
            cache.true_coszen,
        ),
        (
            "event_cache/base_mc_weight",
            cache.base_mc_weight,
        ),
    ):
        stored = h5[
            path
        ][...]

        if not np.array_equal(
            stored,
            reference,
        ):
            raise RuntimeError(
                "El archivo v3.1 existente no coincide "
                f"con el MC actual en {path}. "
                "No es seguro reanudar."
            )


# =============================================================================
# Construcción / reanudación de nodos
# =============================================================================

def build_nodes(
    output_filename: Path,
    flux_lut_filename: Path,
    metadata: FluxLUTMetadata,
    cache: MCEventCache,
    binning: Binning,
) -> None:

    total_nodes = int(
        metadata.mass_eV.size
        * metadata.coupling_values.size
    )

    build_start = time.time()

    with h5py.File(
        flux_lut_filename,
        "r",
    ) as flux_file, h5py.File(
        output_filename,
        "r+",
    ) as output_file:

        validate_resume_event_cache(
            output_file,
            cache,
        )

        completed = output_file[
            "status/completed_nodes"
        ]

        event_weights_ds = output_file[
            "event_weights/nominal"
        ]

        template_datasets = {
            category: output_file[
                f"templates/{category}"
            ]
            for category in CATEGORIES
        }

        sumw2_datasets = {
            category: output_file[
                f"sumw2/{category}"
            ]
            for category in CATEGORIES
        }

        completed_array = np.asarray(
            completed[...],
            dtype=np.bool_,
        )

        already_completed = int(
            np.count_nonzero(
                completed_array
            )
        )

        if bool(
            output_file.attrs.get(
                "build_completed",
                False,
            )
        ):
            if already_completed != total_nodes:
                raise RuntimeError(
                    "build_completed=True pero "
                    "completed_nodes no está completo."
                )

            print(
                "El archivo formato 3.1 ya está "
                "completamente construido."
            )
            return

        print()
        print("=" * 84)
        print("Construcción de pesos evento-a-evento")
        print("=" * 84)
        print(
            f"Nodos totales          : "
            f"{total_nodes}"
        )
        print(
            f"Ya completados         : "
            f"{already_completed}"
        )
        print(
            f"Pendientes              : "
            f"{total_nodes - already_completed}"
        )
        print(
            f"Eventos almacenados     : "
            f"{cache.size}"
        )
        print(
            "Shape event weights     : "
            f"[{metadata.mass_eV.size}, "
            f"{metadata.coupling_values.size}, "
            f"{cache.size}]"
        )
        print(
            f"Storage dtype           : "
            f"{np.dtype(EVENT_WEIGHT_DTYPE)}"
        )
        print(
            "Compresión              : "
            f"{event_weights_ds.compression}"
        )
        print("=" * 84)

        completed_now = 0

        for mass_index in range(
            metadata.mass_eV.size
        ):
            mass_start = time.time()

            print()
            print(
                f"[mass {mass_index + 1:02d}/"
                f"{metadata.mass_eV.size:02d}] "
                f"m={metadata.mass_eV[mass_index]:.6e} eV "
                f"(log10={metadata.log10_mass_eV[mass_index]:.3f})"
            )

            for g_index in range(
                metadata.coupling_values.size
            ):
                if bool(
                    completed[
                        mass_index,
                        g_index,
                    ]
                ):
                    continue

                node_start = time.time()

                weights_float64 = (
                    build_event_weights_for_node(
                        flux_file,
                        metadata,
                        mass_index,
                        g_index,
                        cache,
                        binning.nucount,
                    )
                )

                # La representación almacenada define el dato que usará
                # el likelihood. Construimos también los histogramas nominales
                # desde esta misma cuantización float32.
                weights_stored = np.asarray(
                    weights_float64,
                    dtype=EVENT_WEIGHT_DTYPE,
                )

                if not np.all(
                    np.isfinite(
                        weights_stored
                    )
                ):
                    raise FloatingPointError(
                        "Conversión a float32 produjo "
                        "valores no finitos."
                    )

                if np.any(
                    weights_stored < 0.0
                ):
                    raise FloatingPointError(
                        "Conversión a float32 produjo "
                        "pesos negativos."
                    )

                weights_for_hist = (
                    weights_stored.astype(
                        np.float64
                    )
                )

                templates, sumw2 = (
                    histogram_event_weights(
                        weights_for_hist,
                        cache,
                        binning,
                    )
                )

                # Primero escribimos toda la información del nodo.
                event_weights_ds[
                    mass_index,
                    g_index,
                    :,
                ] = weights_stored

                for category in CATEGORIES:
                    template_datasets[
                        category
                    ][
                        mass_index,
                        g_index,
                        :,
                    ] = templates[
                        category
                    ]

                    sumw2_datasets[
                        category
                    ][
                        mass_index,
                        g_index,
                        :,
                    ] = sumw2[
                        category
                    ]

                # Flush de los datos ANTES de marcar el nodo como completo.
                output_file.flush()

                completed[
                    mass_index,
                    g_index,
                ] = True

                output_file.flush()

                completed_now += 1

                node_seconds = (
                    time.time()
                    - node_start
                )

                done_total = (
                    already_completed
                    + completed_now
                )

                elapsed = (
                    time.time()
                    - build_start
                )

                if completed_now > 0:
                    seconds_per_new_node = (
                        elapsed
                        / completed_now
                    )
                else:
                    seconds_per_new_node = 0.0

                remaining_nodes = (
                    total_nodes
                    - done_total
                )

                eta = (
                    seconds_per_new_node
                    * remaining_nodes
                )

                should_print = (
                    (g_index + 1)
                    % PROGRESS_EVERY_G
                    == 0
                    or g_index
                    == metadata.coupling_values.size - 1
                    or completed_now == 1
                )

                if should_print:
                    print(
                        f"    g {g_index + 1:02d}/"
                        f"{metadata.coupling_values.size:02d} "
                        f"{metadata.coupling_kind}={metadata.coupling_values[g_index]:.3e}  "
                        f"node={done_total}/{total_nodes}  "
                        f"time={node_seconds:.2f}s  "
                        f"ETA={format_seconds(eta)}"
                    )

            mass_seconds = (
                time.time()
                - mass_start
            )

            print(
                f"    masa terminada/revisada en "
                f"{format_seconds(mass_seconds)}"
            )

        final_completed = np.asarray(
            completed[...],
            dtype=np.bool_,
        )

        if not np.all(
            final_completed
        ):
            missing = np.argwhere(
                ~final_completed
            )

            raise RuntimeError(
                "La construcción terminó el loop pero "
                "quedan nodos incompletos: "
                f"{missing[:20].tolist()}"
            )

        output_file.attrs[
            "build_completed"
        ] = True

        output_file.attrs[
            "completed_unix_time"
        ] = time.time()

        output_file.attrs[
            "build_elapsed_seconds_last_run"
        ] = (
            time.time()
            - build_start
        )

        output_file.flush()


# =============================================================================
# Validación final
# =============================================================================

def validate_final_output(
    filename: Path,
    metadata: FluxLUTMetadata,
    cache: MCEventCache,
    binning: Binning,
) -> None:

    with h5py.File(
        filename,
        "r",
    ) as h5:

        if not bool(
            h5.attrs.get(
                "build_completed",
                False,
            )
        ):
            raise RuntimeError(
                "build_completed no es True."
            )

        completed = np.asarray(
            h5[
                "status/completed_nodes"
            ][...],
            dtype=np.bool_,
        )

        if not np.all(
            completed
        ):
            raise RuntimeError(
                "completed_nodes no está completo."
            )

        event_weights = h5[
            "event_weights/nominal"
        ]

        expected_event_shape = (
            metadata.mass_eV.size,
            metadata.coupling_values.size,
            cache.size,
        )

        if (
            event_weights.shape
            != expected_event_shape
        ):
            raise RuntimeError(
                "Shape final de event_weights incorrecta: "
                f"{event_weights.shape}, esperado "
                f"{expected_event_shape}."
            )

        observed = h5[
            "observed/data"
        ][...]

        muon = h5[
            "backgrounds/atmospheric_muon"
        ][...]

        muon_var = h5[
            "backgrounds/atmospheric_muon_sumw2"
        ][...]

        for name, array in (
            ("observed/data", observed),
            (
                "backgrounds/atmospheric_muon",
                muon,
            ),
            (
                "backgrounds/atmospheric_muon_sumw2",
                muon_var,
            ),
        ):
            if (
                array.shape
                != (binning.n_bins,)
            ):
                raise RuntimeError(
                    f"{name} tiene shape "
                    f"{array.shape}."
                )

            if not np.all(
                np.isfinite(array)
            ):
                raise RuntimeError(
                    f"{name} contiene "
                    "valores no finitos."
                )

            if np.any(
                array < 0.0
            ):
                raise RuntimeError(
                    f"{name} contiene "
                    "valores negativos."
                )

        # Validación directa de algunos nodos: histogramamos nuevamente
        # desde los pesos almacenados y exigimos que coincidan con /templates
        # y /sumw2.
        validation_nodes = {
            (0, 0),
            (
                metadata.mass_eV.size - 1,
                metadata.coupling_values.size - 1,
            ),
            (
                metadata.mass_eV.size // 2,
                metadata.coupling_values.size // 2,
            ),
        }

        for (
            mass_index,
            g_index,
        ) in sorted(validation_nodes):

            weights = np.asarray(
                event_weights[
                    mass_index,
                    g_index,
                    :,
                ],
                dtype=np.float64,
            )

            if not np.all(
                np.isfinite(weights)
            ):
                raise RuntimeError(
                    "event_weights contiene "
                    "no finitos en un nodo de validación."
                )

            templates, sumw2 = (
                histogram_event_weights(
                    weights,
                    cache,
                    binning,
                )
            )

            for category in CATEGORIES:
                stored_template = h5[
                    f"templates/{category}"
                ][
                    mass_index,
                    g_index,
                    :,
                ]

                stored_sumw2 = h5[
                    f"sumw2/{category}"
                ][
                    mass_index,
                    g_index,
                    :,
                ]

                if not np.allclose(
                    templates[
                        category
                    ],
                    stored_template,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                ):
                    raise RuntimeError(
                        "Template nominal no reproduce "
                        "event_weights para "
                        f"{category}, nodo "
                        f"({mass_index},{g_index})."
                    )

                if not np.allclose(
                    sumw2[
                        category
                    ],
                    stored_sumw2,
                    rtol=1.0e-12,
                    atol=1.0e-18,
                ):
                    raise RuntimeError(
                        "sumw2 nominal no reproduce "
                        "event_weights para "
                        f"{category}, nodo "
                        f"({mass_index},{g_index})."
                    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convertir flujos nuSQuIDS a pesos y templates IceCube.")
    parser.add_argument("model", choices=MODELS, nargs="?", default="Le-Lmu",
                        help="standard, Le-Lmu, Le-Ltau o Lmu-Ltau")
    parser.add_argument("--flux", type=Path, default=None,
                        help="HDF5 propagado alternativo; por defecto results/oscillated_flux/<model>.h5")
    parser.add_argument("--output", type=Path, default=None,
                        help="HDF5 de salida; por defecto results/templates/<model>.h5")
    parser.add_argument("--icecube-dir", type=Path, default=ICECUBE_DIR,
                        help="Directorio data/icecube con las cuatro subcarpetas")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reemplazar de forma EXPLÍCITA el HDF5 de templates existente")
    args = parser.parse_args()
    flux_lut_filename = (args.flux or FLUX_DIR / f"{args.model}.h5").resolve()
    icecube_directory = args.icecube_dir.resolve()
    output_filename = (args.output or TEMPLATE_DIR / f"{args.model}.h5").resolve()
    if flux_lut_filename == output_filename:
        parser.error("La salida no puede coincidir con el flujo de entrada")

    require_finite_positive(
        DM31_FIXED_EV2,
        "DM31_FIXED_EV2",
    )

    require_finite_positive(
        FLUX_TO_ICECUBE_WEIGHT_FACTOR,
        "FLUX_TO_ICECUBE_WEIGHT_FACTOR",
    )

    require_finite_positive(
        SPECTRAL_INDEX_PIVOT_GEV,
        "SPECTRAL_INDEX_PIVOT_GEV",
    )

    if not (
        1
        <= EVENT_CHUNK_SIZE
    ):
        raise ValueError(
            "EVENT_CHUNK_SIZE debe ser >= 1."
        )

    if not (
        0
        <= AUX_GZIP_LEVEL
        <= 9
    ):
        raise ValueError(
            "AUX_GZIP_LEVEL debe estar entre 0 y 9."
        )

    if not icecube_directory.is_dir():
        raise NotADirectoryError(
            "No existe el directorio IceCube: "
            f"{icecube_directory}"
        )

    ensure_required_files(
        icecube_directory,
        ICECUBE_REQUIRED_FILES,
        "IceCube",
    )

    print("=" * 84)
    print("BUILD LRI / ICECUBE EVENT-LEVEL LUT  v3.1")
    print("=" * 84)

    print(
        "Leyendo binning oficial..."
    )

    binning = load_binning(
        icecube_directory
    )

    print(
        "Leyendo referencia detectoral..."
    )

    detector_nominal, detector_bounds = (
        load_detector_reference(
            icecube_directory
        )
    )

    print(
        "Validando LUT de flujos..."
    )

    metadata = load_flux_lut_metadata(
        flux_lut_filename, args.model
    )

    print(
        "Construyendo cache evento-a-evento "
        "del MC..."
    )

    cache = load_mc_cache(
        icecube_directory,
        metadata,
        binning,
    )

    print(
        "Construyendo observación y fondo "
        "muónico..."
    )

    observed = (
        build_observed_histogram(
            icecube_directory,
            binning,
        )
    )

    muon_template, muon_variance = (
        build_muon_histograms(
            icecube_directory,
            binning,
        )
    )

    print(
        "Interpolando hypersurfaces en "
        f"dm31={DM31_FIXED_EV2:.6e} eV^2..."
    )

    hypersurfaces = (
        load_all_hypersurfaces(
            icecube_directory,
            DM31_FIXED_EV2,
            binning,
        )
    )

    print()
    print("-" * 84)
    print("Resumen de entrada")
    print("-" * 84)
    print(
        f"Flux LUT               : "
        f"{flux_lut_filename}"
    )
    print(
        f"IceCube dir            : "
        f"{icecube_directory}"
    )
    print(
        f"Output                 : "
        f"{output_filename}"
    )
    print(
        f"Mass points            : "
        f"{metadata.mass_eV.size}"
    )
    print(
        f"Coupling points        : "
        f"{metadata.coupling_values.size}"
    )
    print(f"Model / coupling kind  : {metadata.model} / {metadata.coupling_kind}")
    print(
        f"Flux coszen points     : "
        f"{metadata.coszen.size}"
    )
    print(
        f"Flux energy points     : "
        f"{metadata.energy_GeV.size}"
    )
    print(
        f"Reco histogram         : "
        f"{binning.hist_shape} "
        f"= {binning.n_bins} bins"
    )
    print(
        f"NUCOUNT                : "
        f"{binning.nucount:.0f}"
    )
    print(
        f"MC events total        : "
        f"{sum(cache.source_counts_total.values())}"
    )
    print(
        f"MC events in reco bins : "
        f"{cache.size}"
    )
    print(
        f"True-E range used      : "
        f"{np.min(cache.true_energy_GeV):.6e} -- "
        f"{np.max(cache.true_energy_GeV):.6e} GeV"
    )
    print(
        f"True-coszen range used : "
        f"{np.min(cache.true_coszen):.6f} -- "
        f"{np.max(cache.true_coszen):.6f}"
    )
    print(
        f"Muon total             : "
        f"{np.sum(muon_template):.6f}"
    )
    print(
        f"Muon variance sum      : "
        f"{np.sum(muon_variance):.6f}"
    )

    for category in CATEGORIES:
        print(
            f"  {category:<9} "
            f"total={cache.source_counts_total[category]:>7d}  "
            f"used={cache.source_counts_inside_bins[category]:>7d}"
        )

    print("-" * 84)

    # ---------------------------------------------------------------------
    # Crear / reanudar
    # ---------------------------------------------------------------------

    if output_filename.exists() and args.overwrite:
        output_filename.unlink()
        print(f"--overwrite: se reemplazará {output_filename}")

    if output_filename.exists():
        if is_compatible_v3_file(
            output_filename,
            metadata,
            cache,
            binning,
            icecube_directory,
            flux_lut_filename,
        ):
            print(
                "Se encontró un output v3.1 compatible. "
                "Se reanudará la construcción."
            )
        else:
            raise RuntimeError(
                "Ya existe un HDF5 de templates incompatible. No se modificará. "
                "Usa --output NUEVA_RUTA o --overwrite explícitamente."
            )
    else:
        initialize_output_file(
            output_filename,
            flux_lut_filename,
            icecube_directory,
            metadata,
            binning,
            detector_nominal,
            detector_bounds,
            observed,
            muon_template,
            muon_variance,
            hypersurfaces,
            cache,
        )

    build_nodes(
        output_filename,
        flux_lut_filename,
        metadata,
        cache,
        binning,
    )

    print()
    print(
        "Validando output final..."
    )

    validate_final_output(
        output_filename,
        metadata,
        cache,
        binning,
    )

    size_gib = (
        output_filename.stat().st_size
        / (1024.0 ** 3)
    )

    print()
    print("=" * 84)
    print("BUILD COMPLETADO CORRECTAMENTE")
    print("=" * 84)
    print(
        f"Archivo  : {output_filename}"
    )
    print(
        f"Tamaño   : {size_gib:.3f} GiB"
    )
    print(
        "Formato  : 3.1 event-level"
    )
    print(
        "Mu sigma : incluido desde "
        "mc_mu.csv/abs_uncertainty"
    )
    print(
        "Nuisances preparados sin nuevas "
        "propagaciones:"
    )

    for name in (
        SUPPORTED_POST_PROPAGATION_NUISANCES
    ):
        print(
            f"  - {name}"
        )

    print()
    print(
        "DIS CSMS: kinemática almacenada, "
        "reweight numérico pendiente."
    )
    print(
        "Barr x6: omitidos (requieren "
        "derivadas MCEq)."
    )
    print(
        "theta23, dm31: fijos (requieren "
        "nuevas propagaciones en este pipeline)."
    )
    print("=" * 84)


if __name__ == "__main__":
    main()

