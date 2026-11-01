
#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import h5py

from MCEq.core import MCEqRun
from MCEq.ddm import DataDrivenModel
import crflux.models as crf


# ============================================================
# 1. CONFIGURACION
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "LRI"
    / "INITIAL_FLUX"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "mceq_gsf2019_ddm.h5"

INTERACTION_MODEL = "SIBYLL23D"
PRIMARY_MODEL = (crf.GlobalSplineFit, "2019")
DENSITY_MODEL = ("CORSIKA", ("BK_USStd", None))

SPECIES = (
    "nue",
    "numu",
    "antinue",
    "antinumu",
)


# ============================================================
# 2. MALLA ENERGETICA
# ============================================================

energy_low = np.logspace(
    np.log10(1.0),
    np.log10(500.0),
    1200,
    endpoint=False,
)

energy_high = np.logspace(
    np.log10(500.0),
    np.log10(20000.0),
    400,
)

energy = np.concatenate([
    energy_low,
    energy_high,
])


# ============================================================
# 3. MALLA ANGULAR
# ============================================================

abs_coszen = np.concatenate([
    np.linspace(0.0, 0.2, 101, endpoint=False),
    np.linspace(0.2, 1.0, 151),
])

# MCEq resuelve angulos entre 0 y 90 grados.
theta_mceq = np.degrees(np.arccos(abs_coszen))

theta_mceq = np.clip(
    theta_mceq,
    1e-4,
    90.0 - 1e-4,
)

# Malla firmada utilizada por nuSQuIDS.
coszen = np.concatenate([
    -abs_coszen[1:][::-1],
    abs_coszen,
])

theta_true = np.degrees(np.arccos(coszen))

print("=" * 60)
print("MCEq atmospheric neutrino flux generation")
print("=" * 60)

print(f"Energy points : {len(energy)}")
print(f"Energy range  : {energy[0]:g} - {energy[-1]:g} GeV")
print(f"MCEq angles   : {len(abs_coszen)}")
print(f"Final angles  : {len(coszen)}")
print()


# ============================================================
# 4. INICIALIZAR MCEq
# ============================================================

print("Initializing MCEq...")

mceq = MCEqRun(
    interaction_model=INTERACTION_MODEL,
    primary_model=PRIMARY_MODEL,
    theta_deg=float(theta_mceq[0]),
    density_model=DENSITY_MODEL,
)

# Incorporar el modelo hadronico Data-Driven Model.
ddm = DataDrivenModel()

mceq.inject_ddm(ddm)


# ============================================================
# 5. GENERAR FLUJOS CONVENCIONALES
# ============================================================

# Dimensiones: [angulo, energia]

flux = {
    sp: np.zeros(
        (len(abs_coszen), len(energy)),
        dtype=np.float64,
    )
    for sp in SPECIES
}

print("\nGenerating atmospheric neutrino fluxes...")

start = time.perf_counter()

for i, theta in enumerate(theta_mceq):

    mceq.set_theta_deg(float(theta))
    mceq.solve()

    native_energy = np.asarray(mceq.e_grid)

    for sp in SPECIES:

        # Flujo convencional: piones, kaones, etc.
        # mag=0 devuelve Phi(E), no E^3 Phi(E).

        native_flux = np.asarray(
            mceq.get_solution(
                "conv_" + sp,
                mag=0.0,
            )
        )

        valid = (
            np.isfinite(native_flux)
            & (native_flux > 0.0)
        )

        e_valid = native_energy[valid]
        f_valid = native_flux[valid]

        # Evitar extrapolaciones fuera del rango valido.
        if (
            len(e_valid) < 2
            or energy[0] < e_valid[0]
            or energy[-1] > e_valid[-1]
        ):
            raise ValueError(
                f"Insufficient MCEq energy range for {sp}"
            )

        # Interpolacion log-log.
        flux[sp][i] = np.exp(
            np.interp(
                np.log(energy),
                np.log(e_valid),
                np.log(f_valid),
            )
        )

    if (i + 1) % 10 == 0 or i == len(theta_mceq) - 1:
        print(
            f"Completed {i + 1}/{len(theta_mceq)} angles",
            flush=True,
        )

elapsed = time.perf_counter() - start

print(f"\nMCEq generation time: {elapsed:.2f} s")


# ============================================================
# 6. EXPANDIR A ANGULOS FIRMADOS
# ============================================================

# Aproximacion atmosferica simetrica:
#
# Phi(E, -|coszen|) = Phi(E, +|coszen|)
#
# El signo se conserva para la propagacion terrestre.

flux_signed = {}

for sp in SPECIES:

    flux_signed[sp] = np.concatenate([
        flux[sp][1:][::-1],
        flux[sp],
    ], axis=0)


# Flujo tau atmosferico convencional inicial despreciable.
flux_signed["nutau"] = np.zeros_like(
    flux_signed["nue"]
)

flux_signed["antinutau"] = np.zeros_like(
    flux_signed["nue"]
)


# ============================================================
# 7. GUARDAR FLUJOS EN HDF5
# ============================================================

print("\nSaving atmospheric fluxes...")

with h5py.File(OUTPUT_FILE, "x") as h5:

    # Ejes
    axes = h5.create_group("axes")

    axes.create_dataset(
        "energy_GeV",
        data=energy,
    )

    axes.create_dataset(
        "coszen",
        data=coszen,
    )

    axes.create_dataset(
        "theta_true_deg",
        data=theta_true,
    )

    # Matrices de flujo: [coszen, energy]
    flux_group = h5.create_group("flux")

    for sp, array in flux_signed.items():

        flux_group.create_dataset(
            sp,
            data=array,
            compression="gzip",
            compression_opts=4,
        )

    # Metadatos
    h5.attrs["primary_model"] = "GSF 2019"
    h5.attrs["interaction_model"] = INTERACTION_MODEL
    h5.attrs["hadronic_correction"] = "DDM 1.0"
    h5.attrs["density_model"] = str(DENSITY_MODEL)

    h5.attrs["energy_unit"] = "GeV"
    h5.attrs["flux_unit"] = "GeV^-1 cm^-2 s^-1 sr^-1"
    h5.attrs["flux_type"] = "conventional"

    h5.attrs["dimension_order"] = "coszen, energy"
    h5.attrs["coszen_convention"] = (
        "-1 upgoing, +1 downgoing"
    )

    h5.attrs["angular_symmetry"] = (
        "Phi(E,-abs(coszen)) = Phi(E,+abs(coszen))"
    )

    h5.attrs["generation_time_seconds"] = elapsed


print("\nGeneration completed successfully.")
print(f"Output: {OUTPUT_FILE.resolve()}")
print(f"Shape : ({len(coszen)}, {len(energy)})")
