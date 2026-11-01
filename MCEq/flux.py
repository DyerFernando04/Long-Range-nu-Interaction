#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from MCEq.core import MCEqRun
import crflux.models as crf


# ============================================================
# Configuración general
# ============================================================

OUTPUT_DIR = Path("mceq_initial_flux")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERACTION_MODEL = "SIBYLL23C"
PRIMARY_MODEL_NAME = "HillasGaisser2012-H3a"
PRIMARY_MODEL = (crf.HillasGaisser2012, "H3a")

DENSITY_MODEL_NAME = "MSIS00_IC-SouthPole-January"
DENSITY_MODEL = (
    "MSIS00_IC",
    ("SouthPole", "January"),
)

# MCEq devuelve E^MAG * Phi.
# Para nuSQuIDS necesitamos directamente Phi.
MAG = 0.0

# Evitamos entregar exactamente 0 o 90 grados a MCEq.
THETA_EPS_DEG = 1.0e-4

SPECIES = (
    "nue",
    "numu",
    "antinue",
    "antinumu",
)


# ============================================================
# Malla energética final
# ============================================================
#
# El MC tiene aproximadamente:
#
#   E_true,min = 1.02 GeV
#   E_true,max = 9.29 TeV
#
# Usamos:
#
#   1 GeV -- 500 GeV:
#       600 puntos, porque allí está la mayor parte del MC y
#       las oscilaciones pueden variar rápidamente.
#
#   500 GeV -- 20 TeV:
#       200 puntos, para cubrir toda la cola del MC con margen.
#
# Total: 800 puntos.
# ============================================================

energy_low = np.logspace(
    np.log10(1.0),
    np.log10(500.0),
    1200,
    endpoint=False,
)

energy_high = np.logspace(
    np.log10(500.0),
    np.log10(2.0e4),
    400,
    endpoint=True,
)

energy_grid = np.concatenate([
    energy_low,
    energy_high,
])

N_ENERGY = energy_grid.size

if not np.all(np.diff(energy_grid) > 0.0):
    raise RuntimeError("The energy grid is not strictly increasing.")


# ============================================================
# Malla angular
# ============================================================
#
# MCEq solo necesita |cos(theta)| entre 0 y 1.
#
# Usamos más puntos cerca del horizonte:
#
#   0 <= |cos(theta)| < 0.2:
#       101 puntos
#
#   0.2 <= |cos(theta)| <= 1:
#       151 puntos
#
# Total de soluciones MCEq: 252.
#
# Después expandimos la tabla a valores firmados:
#
#   -1 <= cos(theta_true) <= 1.
#
# Para MCEq:
#
#   theta_MCEq = arccos(|cos(theta_true)|).
#
# Para nuSQuIDS conservaremos el signo de cos(theta_true).
# ============================================================

abs_coszen_horizon = np.linspace(
    0.0,
    0.2,
    101,
    endpoint=False,
)

abs_coszen_rest = np.linspace(
    0.2,
    1.0,
    151,
    endpoint=True,
)

abs_coszen_grid = np.concatenate([
    abs_coszen_horizon,
    abs_coszen_rest,
])

N_ABS_COSZEN = abs_coszen_grid.size

# Construir la malla firmada:
#
# [-1, ..., -u, ..., 0, ..., u, ..., 1]
negative_coszen = -abs_coszen_grid[1:][::-1]

coszen_grid = np.concatenate([
    negative_coszen,
    abs_coszen_grid,
])

N_SIGNED_COSZEN = coszen_grid.size

if not np.all(np.diff(abs_coszen_grid) > 0.0):
    raise RuntimeError(
        "The absolute-coszen grid is not strictly increasing."
    )

if not np.all(np.diff(coszen_grid) > 0.0):
    raise RuntimeError(
        "The signed-coszen grid is not strictly increasing."
    )


# ============================================================
# Funciones auxiliares
# ============================================================

def abs_coszen_to_mceq_theta(abs_coszen: float) -> float:
    """
    Convierte |cos(theta_true)| al ángulo atmosférico de MCEq.

    La salida siempre está dentro de:

        THETA_EPS_DEG
        <= theta_MCEq
        <= 90 - THETA_EPS_DEG.
    """
    abs_coszen = float(
        np.clip(abs_coszen, 0.0, 1.0)
    )

    theta_deg = float(
        np.degrees(np.arccos(abs_coszen))
    )

    theta_deg = float(
        np.clip(
            theta_deg,
            THETA_EPS_DEG,
            90.0 - THETA_EPS_DEG,
        )
    )

    return theta_deg


def interpolate_flux_loglog(
    energy_old: np.ndarray,
    flux_old: np.ndarray,
    energy_new: np.ndarray,
    *,
    species: str,
    theta_deg: float,
) -> np.ndarray:
    """
    Interpola un flujo positivo en log(E)-log(Phi).

    No se permite extrapolación. El rango pedido debe estar
    completamente cubierto por la solución válida de MCEq.
    """
    energy_old = np.asarray(
        energy_old,
        dtype=np.float64,
    )

    flux_old = np.asarray(
        flux_old,
        dtype=np.float64,
    )

    energy_new = np.asarray(
        energy_new,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(energy_old)
        & np.isfinite(flux_old)
        & (energy_old > 0.0)
        & (flux_old > 0.0)
    )

    if np.count_nonzero(valid) < 2:
        raise RuntimeError(
            f"Insufficient valid MCEq points for {species} "
            f"at theta={theta_deg:.8f} deg."
        )

    energy_valid = energy_old[valid]
    flux_valid = flux_old[valid]

    order = np.argsort(energy_valid)

    energy_valid = energy_valid[order]
    flux_valid = flux_valid[order]

    energy_min_valid = energy_valid[0]
    energy_max_valid = energy_valid[-1]

    if energy_new[0] < energy_min_valid:
        raise RuntimeError(
            f"Requested minimum energy {energy_new[0]:.6e} GeV "
            f"is below the valid MCEq range "
            f"{energy_min_valid:.6e} GeV for {species}."
        )

    if energy_new[-1] > energy_max_valid:
        raise RuntimeError(
            f"Requested maximum energy {energy_new[-1]:.6e} GeV "
            f"is above the valid MCEq range "
            f"{energy_max_valid:.6e} GeV for {species}."
        )

    log_energy_old = np.log(energy_valid)
    log_flux_old = np.log(flux_valid)
    log_energy_new = np.log(energy_new)

    log_flux_new = np.interp(
        log_energy_new,
        log_energy_old,
        log_flux_old,
    )

    flux_new = np.exp(log_flux_new)

    if not np.all(np.isfinite(flux_new)):
        raise RuntimeError(
            f"Non-finite interpolated flux for {species} "
            f"at theta={theta_deg:.8f} deg."
        )

    if np.any(flux_new <= 0.0):
        raise RuntimeError(
            f"Non-positive interpolated flux for {species} "
            f"at theta={theta_deg:.8f} deg."
        )

    return flux_new


# ============================================================
# Información de las mallas
# ============================================================

print("=" * 72)
print("MCEq initial atmospheric flux generation")
print("=" * 72)

print(f"Energy points              : {N_ENERGY}")
print(
    f"Energy range               : "
    f"{energy_grid[0]:.6g} -- "
    f"{energy_grid[-1]:.6g} GeV"
)

print(f"Unique MCEq angles         : {N_ABS_COSZEN}")
print(f"Signed coszen points       : {N_SIGNED_COSZEN}")

print(
    f"Signed coszen range        : "
    f"{coszen_grid[0]:.6g} -- "
    f"{coszen_grid[-1]:.6g}"
)

print(f"MCEq magnification MAG     : {MAG}")
print()


# ============================================================
# Contenedores para las soluciones en |coszen|
# ============================================================
#
# Shape:
#
#   (N_ENERGY, N_ABS_COSZEN)
#
# Las filas son energías y las columnas son ángulos.
# ============================================================

flux_abs_coszen = {
    species: np.zeros(
        (N_ENERGY, N_ABS_COSZEN),
        dtype=np.float64,
    )
    for species in SPECIES
}


# ============================================================
# Inicializar MCEq una sola vez
# ============================================================

initial_theta_deg = abs_coszen_to_mceq_theta(
    abs_coszen_grid[0]
)

print("Initializing MCEq...")
print(
    f"Initial theta_MCEq = "
    f"{initial_theta_deg:.8f} deg"
)

mceq = MCEqRun(
    interaction_model=INTERACTION_MODEL,
    primary_model=PRIMARY_MODEL,
    theta_deg=initial_theta_deg,
    density_model=DENSITY_MODEL,
)


# ============================================================
# Resolver MCEq para cada |cos(theta)|
# ============================================================

for angular_index, abs_coszen in enumerate(
    abs_coszen_grid
):
    theta_mceq_deg = abs_coszen_to_mceq_theta(
        abs_coszen
    )

    print(
        f"[{angular_index + 1:03d}/"
        f"{N_ABS_COSZEN:03d}]  "
        f"|coszen|={abs_coszen:.10f}  "
        f"theta_MCEq={theta_mceq_deg:.8f} deg"
    )

    mceq.set_theta_deg(theta_mceq_deg)
    mceq.solve()

    mceq_energy_grid = np.asarray(
        mceq.e_grid,
        dtype=np.float64,
    )

    for species in SPECIES:
        mceq_flux = np.asarray(
            mceq.get_solution(
                species,
                mag=MAG,
            ),
            dtype=np.float64,
        )

        interpolated_flux = interpolate_flux_loglog(
            energy_old=mceq_energy_grid,
            flux_old=mceq_flux,
            energy_new=energy_grid,
            species=species,
            theta_deg=theta_mceq_deg,
        )

        flux_abs_coszen[species][
            :,
            angular_index,
        ] = interpolated_flux


# ============================================================
# Expandir a cos(theta) firmado
# ============================================================
#
# El flujo atmosférico inicial se refleja:
#
#   Phi(E, -|c|) = Phi(E, +|c|).
#
# La primera parte corresponde a:
#
#   coszen = -1, ..., valor negativo cercano a 0.
#
# La segunda parte contiene:
#
#   coszen = 0, ..., +1.
#
# nuSQuIDS usará luego el signo para la propagación terrestre.
# ============================================================

flux_signed_coszen = {}

for species in SPECIES:
    negative_side = flux_abs_coszen[species][
        :,
        1:,
    ][:, ::-1]

    positive_side = flux_abs_coszen[species]

    signed_flux = np.concatenate(
        [
            negative_side,
            positive_side,
        ],
        axis=1,
    )

    expected_shape = (
        N_ENERGY,
        N_SIGNED_COSZEN,
    )

    if signed_flux.shape != expected_shape:
        raise RuntimeError(
            f"Wrong signed flux shape for {species}: "
            f"{signed_flux.shape}; "
            f"expected {expected_shape}."
        )

    flux_signed_coszen[species] = signed_flux


# ============================================================
# Flujos tau iniciales
# ============================================================
#
# En esta implementación, el flujo atmosférico inicial de tau
# se establece en cero. nuSQuIDS generará las componentes tau
# durante la propagación mediante las oscilaciones.
# ============================================================

zero_tau_flux = np.zeros(
    (N_ENERGY, N_SIGNED_COSZEN),
    dtype=np.float64,
)

flux_signed_coszen["nutau"] = zero_tau_flux.copy()
flux_signed_coszen["antinutau"] = zero_tau_flux.copy()


# ============================================================
# Validaciones finales
# ============================================================

ALL_SPECIES = (
    "nue",
    "numu",
    "nutau",
    "antinue",
    "antinumu",
    "antinutau",
)

for species in ALL_SPECIES:
    array = flux_signed_coszen[species]

    expected_shape = (
        N_ENERGY,
        N_SIGNED_COSZEN,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Invalid final shape for {species}: "
            f"{array.shape}; expected {expected_shape}."
        )

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"Non-finite values found in {species}."
        )

    if np.any(array < 0.0):
        raise RuntimeError(
            f"Negative flux values found in {species}."
        )

# Comprobar explícitamente la simetría inicial.
for species in SPECIES:
    reflected = flux_signed_coszen[species][:, ::-1]

    if not np.allclose(
        flux_signed_coszen[species],
        reflected,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise RuntimeError(
            f"Reflection symmetry failed for {species}."
        )

print()
print("All flux arrays passed the validation checks.")


# ============================================================
# Guardar las mallas
# ============================================================

np.savetxt(
    OUTPUT_DIR / "energy_grid_GeV.txt",
    energy_grid,
    fmt="%.18e",
    header=(
        "Energy grid in GeV. "
        "Rows of every flux matrix correspond to this grid."
    ),
)

np.savetxt(
    OUTPUT_DIR / "coszen_grid.txt",
    coszen_grid,
    fmt="%.18e",
    header=(
        "Signed true cosine of zenith. "
        "-1 is vertically upgoing, "
        "0 is horizontal, "
        "+1 is vertically downgoing. "
        "Columns of every flux matrix correspond to this grid."
    ),
)

theta_true_grid_deg = np.degrees(
    np.arccos(
        np.clip(coszen_grid, -1.0, 1.0)
    )
)

np.savetxt(
    OUTPUT_DIR / "theta_true_grid_deg.txt",
    theta_true_grid_deg,
    fmt="%.18e",
    header=(
        "Signed-analysis true zenith angle in degrees, "
        "computed as arccos(coszen_grid)."
    ),
)

theta_mceq_grid_deg = np.array([
    abs_coszen_to_mceq_theta(value)
    for value in abs_coszen_grid
])

np.savetxt(
    OUTPUT_DIR / "abs_coszen_mceq_grid.txt",
    abs_coszen_grid,
    fmt="%.18e",
    header=(
        "Absolute cosine-of-zenith values solved with MCEq."
    ),
)

np.savetxt(
    OUTPUT_DIR / "theta_mceq_grid_deg.txt",
    theta_mceq_grid_deg,
    fmt="%.18e",
    header=(
        "Atmospheric zenith angles passed to MCEq, "
        "theta_MCEq = arccos(abs(coszen))."
    ),
)


# ============================================================
# Guardar matrices individuales
# ============================================================
#
# Cada archivo contiene únicamente la matriz de flujo:
#
#   filas    -> energy_grid_GeV.txt
#   columnas -> coszen_grid.txt
# ============================================================

for species in ALL_SPECIES:
    output_file = (
        OUTPUT_DIR
        / f"{species}_flux_E_coszen.txt"
    )

    np.savetxt(
        output_file,
        flux_signed_coszen[species],
        fmt="%.18e",
        header=(
            "Differential atmospheric neutrino flux from MCEq "
            "with MAG=0. "
            "Rows correspond to energy_grid_GeV.txt; "
            "columns correspond to coszen_grid.txt."
        ),
    )

    print(f"Saved: {output_file}")


# ============================================================
# Guardar todos los datos juntos en NPZ
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "mceq_initial_fluxes.npz",
    energy_GeV=energy_grid,
    coszen=coszen_grid,
    theta_true_deg=theta_true_grid_deg,
    abs_coszen_mceq=abs_coszen_grid,
    theta_mceq_deg=theta_mceq_grid_deg,
    nue=flux_signed_coszen["nue"],
    numu=flux_signed_coszen["numu"],
    nutau=flux_signed_coszen["nutau"],
    antinue=flux_signed_coszen["antinue"],
    antinumu=flux_signed_coszen["antinumu"],
    antinutau=flux_signed_coszen["antinutau"],
)


# ============================================================
# Guardar metadatos
# ============================================================

metadata = {
    "interaction_model": INTERACTION_MODEL,
    "primary_model": PRIMARY_MODEL_NAME,
    "density_model": DENSITY_MODEL_NAME,
    "magnification_mag": MAG,
    "flux_quantity": "Phi(E), not E^3 Phi(E)",
    "energy_unit": "GeV",
    "energy_min_GeV": float(energy_grid[0]),
    "energy_max_GeV": float(energy_grid[-1]),
    "number_energy_points": int(N_ENERGY),
    "number_abs_coszen_points": int(
        N_ABS_COSZEN
    ),
    "number_signed_coszen_points": int(
        N_SIGNED_COSZEN
    ),
    "coszen_min": float(coszen_grid[0]),
    "coszen_max": float(coszen_grid[-1]),
    "mceq_angle_definition": (
        "theta_MCEq = arccos(abs(coszen_true))"
    ),
    "reflection": (
        "Phi_initial(E,-abs(coszen)) "
        "= Phi_initial(E,+abs(coszen))"
    ),
    "tau_initial_flux": (
        "nutau and antinutau set to zero"
    ),
    "matrix_rows": "energy_grid_GeV.txt",
    "matrix_columns": "coszen_grid.txt",
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
    encoding="utf-8",
) as metadata_file:
    json.dump(
        metadata,
        metadata_file,
        indent=4,
    )


# ============================================================
# Resumen final
# ============================================================

print()
print("=" * 72)
print("Generation completed successfully")
print("=" * 72)

print(f"Output directory     : {OUTPUT_DIR.resolve()}")
print(f"Energy points        : {N_ENERGY}")
print(f"Signed angle points  : {N_SIGNED_COSZEN}")
print(f"MCEq solutions       : {N_ABS_COSZEN}")
print(f"MAG                   : {MAG}")
print()
