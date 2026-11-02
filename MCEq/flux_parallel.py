
#!/usr/bin/env python3

import os

# Evitar que cada proceso cree multiples hilos numericos.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from multiprocessing import get_context
from time import perf_counter

import h5py
import numpy as np

from MCEq import config
from MCEq.core import MCEqRun
from MCEq.ddm import DataDrivenModel
import crflux.models as crf


# ============================================================
# 1. CONFIGURACION
# ============================================================

N_PROCESSES = 32
N_HEIGHT_LAYERS = 40

SPECIES = (
    "nue",
    "numu",
    "nutau",
    "antinue",
    "antinumu",
    "antinutau",
)

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "atmospheric"
    / "mceq_gsf2019_ddm_height.h5"
)

# Usar el mismo radio terrestre que nuestro EarthDepthBody.
# Configurarlo antes de crear las instancias MCEq.
config.r_E = 6371.0e3  # metros


# ============================================================
# 2. MALLA ENERGETICA
# ============================================================

energy = np.concatenate([
    np.logspace(
        np.log10(1.0),
        np.log10(500.0),
        300, # 1200 for a higher points
        endpoint=False,
    ),
    np.logspace(
        np.log10(500.0),
        np.log10(20000.0),
        100, # 400 for a higher points
    ),
])


# ============================================================
# 3. MALLA ANGULAR
# ============================================================

abs_coszen = np.concatenate([
    np.linspace(0.0, 0.2, 41, endpoint=False), #101 for a higher points
    np.linspace(0.2, 1.0, 61), #151 for a higher points
])

coszen = np.concatenate([
    -abs_coszen[1:][::-1],
    abs_coszen,
])

theta_mceq = np.degrees(np.arccos(abs_coszen))

theta_mceq = np.clip(
    theta_mceq,
    1e-4,
    90.0 - 1e-4,
)


# ============================================================
# 4. CAPAS ATMOSFERICAS
# ============================================================

# Bordes desde el techo de la atmosfera hasta la superficie.
# config.h_atm esta expresado en metros.

height_edges_km = np.linspace(
    config.h_atm / 1000.0,
    0.0,
    N_HEIGHT_LAYERS + 1,
)


# ============================================================
# 5. INICIALIZACION DE CADA PROCESO
# ============================================================

def initialize_worker():

    global mceq

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
# 6. CALCULO DE UN ANGULO
# ============================================================

def calculate_angle(job):

    i, theta = job

    mceq.set_theta_deg(float(theta))

    # Convertir alturas a profundidad atmosférica inclinada.
    # La conversion depende del angulo cenital.

    X_internal = np.asarray(
        mceq.density_model.h2X(
            height_edges_km[1:-1] * 1e5
        ),
        dtype=np.float64,
    )

    X_surface = float(
        mceq.density_model.max_X
    )

    X_edges = np.concatenate([
        [0.0],
        X_internal,
        [X_surface],
    ])

    if not (
        np.all(np.isfinite(X_edges))
        and np.all(np.diff(X_edges) > 0.0)
    ):
        raise ValueError(
            f"Invalid atmospheric depths at angle {i}"
        )

    # Resolver una sola vez, guardando los estados
    # a diferentes profundidades atmosfericas.

    mceq.solve(
        int_grid=X_internal
    )

    native_energy = np.asarray(
        mceq.e_grid,
        dtype=np.float64,
    )

    if (
        energy[0] < native_energy[0]
        or energy[-1] > native_energy[-1]
    ):
        raise ValueError(
            "Requested energies outside MCEq grid"
        )

    log_native = np.log(native_energy)
    log_target = np.log(energy)

    # cumulative[species, height_edge, energy]
    #
    # Primer borde: inicio de la cascada, flujo nu = 0.
    # Ultimo borde: superficie terrestre.

    cumulative = np.zeros((
        len(SPECIES),
        N_HEIGHT_LAYERS + 1,
        len(energy),
    ))

    for k, sp in enumerate(SPECIES):

        # Flujos acumulados en alturas intermedias.
        for j in range(1, N_HEIGHT_LAYERS):

            native_flux = mceq.get_solution(
                "total_" + sp,
                mag=0.0,
                grid_idx=j - 1,
            )

            cumulative[k, j] = np.interp(
                log_target,
                log_native,
                native_flux,
            )

        # Flujo total al nivel de observacion.
        surface_flux = mceq.get_solution(
            "total_" + sp,
            mag=0.0,
        )

        cumulative[k, -1] = np.interp(
            log_target,
            log_native,
            surface_flux,
        )

    if not np.all(np.isfinite(cumulative)):
        raise ValueError(
            f"Non-finite fluxes at angle {i}"
        )

    # Produccion en cada capa: diferencia de flujos
    # acumulados entre dos bordes consecutivos.

    layers = np.diff(
        cumulative,
        axis=1,
    )

    surface = cumulative[:, -1, :]

    # Permitir solamente pequeñas diferencias negativas
    # causadas por precisión numerica.
    tolerance = (
        1e-5
        * np.maximum(surface[:, None, :], 1e-300)
    )

    if np.any(layers < -tolerance):
        raise ValueError(
            f"Significant negative production at angle {i}"
        )

    layers = np.maximum(layers, 0.0)

    # Verificar que las capas reconstruyen el flujo superficial.
    relative_error = (
        np.abs(layers.sum(axis=1) - surface)
        / np.maximum(surface, 1e-300)
    )

    if np.max(relative_error) > 1e-4:
        raise ValueError(
            f"Height layers do not reproduce surface flux at angle {i}"
        )

    return i, surface, layers, X_edges


# ============================================================
# 7. GENERACION EN PARALELO
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}"
    )

print("=" * 60)
print("MCEq atmospheric neutrino production")
print("=" * 60)

print(f"Processes      : {N_PROCESSES}")
print(f"Energy points  : {len(energy)}")
print(f"MCEq angles    : {len(abs_coszen)}")
print(f"Final angles   : {len(coszen)}")
print(f"Height layers  : {N_HEIGHT_LAYERS}")
print(flush=True)

start = perf_counter()

# El HDF5 se abre despues de iniciar los procesos.
# Solo el proceso principal escribe los resultados.

with get_context("fork").Pool(
    N_PROCESSES,
    initializer=initialize_worker,
) as pool:

    with h5py.File(OUTPUT, "x") as h5:

        # ====================================================
        # EJES
        # ====================================================

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

        axes.create_dataset(
            "abs_coszen_mceq",
            data=abs_coszen,
        )

        # ====================================================
        # FLUJO SUPERFICIAL Y PRODUCCION POR CAPAS
        # ====================================================

        flux_group = h5.create_group("flux")
        production = h5.create_group("production")

        production.create_dataset(
            "height_edges_km",
            data=height_edges_km,
        )

        X_dataset = production.create_dataset(
            "slant_depth_edges_gcm2",
            shape=(
                len(abs_coszen),
                N_HEIGHT_LAYERS + 1,
            ),
            dtype="f8",
        )

        flux_datasets = {}
        layer_datasets = {}

        for sp in SPECIES:

            # Flujo superficial: [coszen, energy]
            flux_datasets[sp] = flux_group.create_dataset(
                sp,
                shape=(len(coszen), len(energy)),
                dtype="f8",
                chunks=(1, len(energy)),
                compression="gzip",
                compression_opts=3,
            )

            # Produccion por altura: [abs_coszen, layer, energy]
            #
            # Guardamos solo 252 angulos independientes.
            # Los ascendentes se obtienen por simetria.

            layer_datasets[sp] = production.create_dataset(
                sp,
                shape=(
                    len(abs_coszen),
                    N_HEIGHT_LAYERS,
                    len(energy),
                ),
                dtype="f4",
                chunks=(1, 4, len(energy)),
                compression="gzip",
                compression_opts=3,
            )

        # ====================================================
        # RECOGER RESULTADOS
        # ====================================================

        jobs = enumerate(theta_mceq)

        results = pool.imap_unordered(
            calculate_angle,
            jobs,
            chunksize=1,
        )

        for completed, result in enumerate(
            results,
            start=1,
        ):

            i, surface, layers, X_edges = result

            positive_index = len(abs_coszen) - 1 + i

            X_dataset[i, :] = X_edges

            for k, sp in enumerate(SPECIES):

                flux_datasets[sp][positive_index, :] = surface[k]

                # Simetria angular para flujo inicial.
                if i > 0:
                    negative_index = len(abs_coszen) - 1 - i
                    flux_datasets[sp][negative_index, :] = surface[k]

                layer_datasets[sp][i, :, :] = layers[k]

            if (
                completed % 10 == 0
                or completed == len(abs_coszen)
            ):
                elapsed = perf_counter() - start

                print(
                    f"Completed {completed}/{len(abs_coszen)} "
                    f"angles | {elapsed:.1f} s",
                    flush=True,
                )

        # ====================================================
        # METADATOS
        # ====================================================

        h5.attrs["primary_model"] = "GSF 2019"
        h5.attrs["interaction_model"] = "SIBYLL23D + DDM 1.0"
        h5.attrs["density_model"] = "CORSIKA BK_USStd"

        h5.attrs["flux_type"] = "total (conventional + prompt)"
        h5.attrs["energy_unit"] = "GeV"
        h5.attrs["flux_unit"] = "GeV^-1 cm^-2 s^-1 sr^-1"

        h5.attrs["surface_dimension_order"] = (
            "coszen, energy"
        )

        h5.attrs["production_dimension_order"] = (
            "abs_coszen_mceq, layer, energy"
        )

        h5.attrs["production_quantity"] = (
            "delta Phi produced in each height layer"
        )

        h5.attrs["angular_symmetry"] = (
            "Phi(E,-abs(coszen))=Phi(E,+abs(coszen))"
        )

        h5.attrs["earth_radius_km"] = 6371.0

        h5.attrs["production_height_convention"] = (
            "height above local surface"
        )

        h5.attrs["calculation_seconds"] = (
            perf_counter() - start
        )


print("\nGeneration completed successfully.")
print(f"Output: {OUTPUT.resolve()}")
print(f"Total time: {perf_counter() - start:.1f} s")
