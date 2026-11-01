
#!/usr/bin/env python3

import os

# Un hilo numerico por proceso.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from multiprocessing import get_context
import time

import h5py
import numpy as np

from MCEq.core import MCEqRun
from MCEq.ddm import DataDrivenModel
from MCEq import config
import crflux.models as crf


# ============================================================
# 1. CONFIGURACION
# ============================================================

N_PROCESSES = 32  # Prueba 4 si tienes RAM suficiente.

SPECIES = (
    "nue",
    "numu",
    "antinue",
    "antinumu",
)

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "LRI"
    / "INITIAL_FLUX"
    / "mceq_gsf2019_ddm.h5"
)

if OUTPUT.exists():
    raise FileExistsError(
        f"El archivo ya existe: {OUTPUT}"
    )


# ============================================================
# 2. MALLA ENERGETICA
# ============================================================

energy = np.concatenate([
    np.logspace(
        np.log10(1.0),
        np.log10(500.0),
        1200,
        endpoint=False,
    ),
    np.logspace(
        np.log10(500.0),
        np.log10(20000.0),
        400,
    ),
])


# ============================================================
# 3. MALLA ANGULAR
# ============================================================

abs_coszen = np.concatenate([
    np.linspace(0.0, 0.2, 101, endpoint=False),
    np.linspace(0.2, 1.0, 151),
])

theta_mceq = np.degrees(
    np.arccos(abs_coszen)
)

theta_mceq = np.clip(
    theta_mceq,
    1e-4,
    90.0 - 1e-4,
)

coszen = np.concatenate([
    -abs_coszen[1:][::-1],
    abs_coszen,
])


# ============================================================
# 4. INICIALIZACION DE CADA PROCESO
# ============================================================

def initialize_worker():

    global mceq

    # Evitar varios hilos MKL dentro de cada proceso.
    if config.has_mkl:
        config.set_mkl_threads(1)

    mceq = MCEqRun(
        interaction_model="SIBYLL23D",
        primary_model=(crf.GlobalSplineFit, "2019"),
        theta_deg=float(theta_mceq[0]),
        density_model=(
            "CORSIKA",
            ("BK_USStd", None),
        ),
    )

    mceq.inject_ddm(
        DataDrivenModel()
    )


# ============================================================
# 5. CALCULO DE UN ANGULO
# ============================================================

def calculate_angle(job):

    i, theta = job

    mceq.set_theta_deg(float(theta))
    mceq.solve()

    native_energy = np.asarray(
        mceq.e_grid
    )

    spectra = []

    for sp in SPECIES:

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

        if (
            len(e_valid) < 2
            or energy[0] < e_valid[0]
            or energy[-1] > e_valid[-1]
        ):
            raise ValueError(
                f"Insufficient MCEq energy range for {sp}"
            )

        # Interpolacion log-log.
        interpolated = np.exp(
            np.interp(
                np.log(energy),
                np.log(e_valid),
                np.log(f_valid),
            )
        )

        spectra.append(interpolated)

    return i, np.asarray(spectra)


# ============================================================
# 6. GENERACION EN PARALELO
# ============================================================

print("=" * 60)
print("MCEq atmospheric neutrino flux generation")
print("=" * 60)

print(f"MCEq solver    : {config.kernel_config}")
print(f"Processes      : {N_PROCESSES}")
print(f"Energy points  : {len(energy)}")
print(f"MCEq angles    : {len(theta_mceq)}")
print(f"Final angles   : {len(coszen)}")
print(flush=True)

flux = np.empty((
    len(SPECIES),
    len(abs_coszen),
    len(energy),
))

start = time.perf_counter()

# Linux: fork permite usar multiprocessing sin __main__.
with get_context("fork").Pool(
    processes=N_PROCESSES,
    initializer=initialize_worker,
) as pool:

    jobs = enumerate(theta_mceq)

    results = pool.imap_unordered(
        calculate_angle,
        jobs,
        chunksize=1,
    )

    for completed, (i, spectra) in enumerate(
        results,
        start=1,
    ):

        flux[:, i, :] = spectra

        if (
            completed % 10 == 0
            or completed == len(theta_mceq)
        ):
            elapsed = time.perf_counter() - start

            print(
                f"Completed {completed}/{len(theta_mceq)} "
                f"angles | {elapsed:.1f} s",
                flush=True,
            )


# ============================================================
# 7. EXPANDIR A COSZEN FIRMADO
# ============================================================

flux_signed = {}

for j, sp in enumerate(SPECIES):

    flux_signed[sp] = np.concatenate([
        flux[j, 1:][::-1],
        flux[j],
    ], axis=0)


# Flujo tau convencional inicialmente nulo.
flux_signed["nutau"] = np.zeros_like(
    flux_signed["nue"]
)

flux_signed["antinutau"] = np.zeros_like(
    flux_signed["nue"]
)


# ============================================================
# 8. GUARDAR HDF5 PARA NUSQUIDS
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with h5py.File(OUTPUT, "x") as h5:

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
        data=np.degrees(np.arccos(coszen)),
    )

    group = h5.create_group("flux")

    for sp, values in flux_signed.items():

        group.create_dataset(
            sp,
            data=values,
            compression="gzip",
        )

    h5.attrs["primary_model"] = "GSF 2019"
    h5.attrs["interaction_model"] = "SIBYLL23D"
    h5.attrs["hadronic_correction"] = "DDM 1.0"
    h5.attrs["density_model"] = "CORSIKA BK_USStd"

    h5.attrs["energy_unit"] = "GeV"
    h5.attrs["flux_unit"] = "GeV^-1 cm^-2 s^-1 sr^-1"
    h5.attrs["flux_type"] = "conventional"

    h5.attrs["dimension_order"] = "coszen, energy"
    h5.attrs["angular_symmetry"] = (
        "Phi(E,-|coszen|) = Phi(E,+|coszen|)"
    )


print("\nGeneration completed successfully.")
print(f"Output: {OUTPUT.resolve()}")
print(
    f"Total time: "
    f"{time.perf_counter() - start:.1f} s"
)
