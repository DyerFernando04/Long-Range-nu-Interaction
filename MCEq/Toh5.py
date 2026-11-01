from pathlib import Path
import json

import h5py
import numpy as np


# ============================================================
# Rutas
# ============================================================

INPUT_DIR = Path("mceq_initial_flux2")

NPZ_FILE = INPUT_DIR / "mceq_initial_fluxes.npz"
METADATA_FILE = INPUT_DIR / "metadata.json"

OUTPUT_FILE = INPUT_DIR / "mceq_initial_fluxes.h5"


# ============================================================
# Comprobar archivos de entrada
# ============================================================

if not NPZ_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo NPZ:\n{NPZ_FILE.resolve()}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de metadatos:\n"
        f"{METADATA_FILE.resolve()}"
    )


# ============================================================
# Cargar datos numéricos
# ============================================================

print(f"Leyendo: {NPZ_FILE.resolve()}")

with np.load(NPZ_FILE) as npz_data:
    available_keys = list(npz_data.keys())

    print("Variables encontradas en el NPZ:")
    for key in available_keys:
        print(f"  - {key}")

    required_keys = [
        "energy_GeV",
        "coszen",
        "theta_true_deg",
        "abs_coszen_mceq",
        "theta_mceq_deg",
        "nue",
        "numu",
        "nutau",
        "antinue",
        "antinumu",
        "antinutau",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in npz_data
    ]

    if missing_keys:
        raise KeyError(
            "Faltan las siguientes variables en el NPZ:\n"
            + "\n".join(f"  - {key}" for key in missing_keys)
        )

    energy_GeV = np.asarray(
        npz_data["energy_GeV"],
        dtype=np.float64,
    )

    coszen = np.asarray(
        npz_data["coszen"],
        dtype=np.float64,
    )

    theta_true_deg = np.asarray(
        npz_data["theta_true_deg"],
        dtype=np.float64,
    )

    abs_coszen_mceq = np.asarray(
        npz_data["abs_coszen_mceq"],
        dtype=np.float64,
    )

    theta_mceq_deg = np.asarray(
        npz_data["theta_mceq_deg"],
        dtype=np.float64,
    )

    # En el NPZ original:
    #
    # shape = (N_energy, N_coszen)
    #
    # Para el HDF5 lo cambiaremos a:
    #
    # shape = (N_coszen, N_energy)
    flux_npz = {
        "nue": np.asarray(
            npz_data["nue"],
            dtype=np.float64,
        ),
        "numu": np.asarray(
            npz_data["numu"],
            dtype=np.float64,
        ),
        "nutau": np.asarray(
            npz_data["nutau"],
            dtype=np.float64,
        ),
        "antinue": np.asarray(
            npz_data["antinue"],
            dtype=np.float64,
        ),
        "antinumu": np.asarray(
            npz_data["antinumu"],
            dtype=np.float64,
        ),
        "antinutau": np.asarray(
            npz_data["antinutau"],
            dtype=np.float64,
        ),
    }


# ============================================================
# Cargar metadatos
# ============================================================

print(f"\nLeyendo: {METADATA_FILE.resolve()}")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8",
) as metadata_handle:
    metadata = json.load(metadata_handle)


# ============================================================
# Validar las mallas
# ============================================================

if energy_GeV.ndim != 1:
    raise ValueError(
        f"energy_GeV debe ser un vector 1D, "
        f"pero tiene shape {energy_GeV.shape}."
    )

if coszen.ndim != 1:
    raise ValueError(
        f"coszen debe ser un vector 1D, "
        f"pero tiene shape {coszen.shape}."
    )

if theta_true_deg.ndim != 1:
    raise ValueError(
        f"theta_true_deg debe ser un vector 1D, "
        f"pero tiene shape {theta_true_deg.shape}."
    )

if abs_coszen_mceq.ndim != 1:
    raise ValueError(
        f"abs_coszen_mceq debe ser un vector 1D, "
        f"pero tiene shape {abs_coszen_mceq.shape}."
    )

if theta_mceq_deg.ndim != 1:
    raise ValueError(
        f"theta_mceq_deg debe ser un vector 1D, "
        f"pero tiene shape {theta_mceq_deg.shape}."
    )

if not np.all(np.isfinite(energy_GeV)):
    raise ValueError(
        "energy_GeV contiene valores no finitos."
    )

if not np.all(np.isfinite(coszen)):
    raise ValueError(
        "coszen contiene valores no finitos."
    )

if not np.all(np.diff(energy_GeV) > 0.0):
    raise ValueError(
        "La malla energética no es estrictamente creciente."
    )

if not np.all(np.diff(coszen) > 0.0):
    raise ValueError(
        "La malla de coszen no es estrictamente creciente."
    )

if energy_GeV[0] <= 0.0:
    raise ValueError(
        "La malla energética contiene energías no positivas."
    )

if coszen[0] < -1.0 or coszen[-1] > 1.0:
    raise ValueError(
        "La malla coszen está fuera del intervalo [-1, 1]."
    )

if theta_true_deg.size != coszen.size:
    raise ValueError(
        "theta_true_deg y coszen tienen longitudes diferentes."
    )

if theta_mceq_deg.size != abs_coszen_mceq.size:
    raise ValueError(
        "theta_mceq_deg y abs_coszen_mceq "
        "tienen longitudes diferentes."
    )


N_ENERGY = energy_GeV.size
N_COSZEN = coszen.size
N_MCEQ_ANGLES = abs_coszen_mceq.size

expected_flux_shape = (
    N_ENERGY,
    N_COSZEN,
)

print("\nDimensiones encontradas:")
print(f"  N_energy      = {N_ENERGY}")
print(f"  N_coszen      = {N_COSZEN}")
print(f"  N_MCEq_angles = {N_MCEQ_ANGLES}")
print(
    f"  Energy range  = "
    f"{energy_GeV[0]:.8g} -- "
    f"{energy_GeV[-1]:.8g} GeV"
)
print(
    f"  coszen range  = "
    f"{coszen[0]:.8g} -- "
    f"{coszen[-1]:.8g}"
)


# ============================================================
# Validar flujos y cambiar orden de dimensiones
# ============================================================

flux_h5 = {}

for species, array in flux_npz.items():
    if array.shape != expected_flux_shape:
        raise ValueError(
            f"Shape incorrecto para {species}:\n"
            f"  encontrado = {array.shape}\n"
            f"  esperado   = {expected_flux_shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"El flujo {species} contiene valores no finitos."
        )

    if np.any(array < 0.0):
        minimum = np.min(array)

        raise ValueError(
            f"El flujo {species} contiene valores negativos. "
            f"Mínimo = {minimum:.8e}"
        )

    # NPZ:
    #     array[energy, coszen]
    #
    # HDF5:
    #     array[coszen, energy]
    #
    # np.ascontiguousarray asegura una disposición conveniente
    # en memoria para guardar y leer desde C++.
    flux_h5[species] = np.ascontiguousarray(
        array.T,
        dtype=np.float64,
    )

    print(
        f"  {species:10s}: "
        f"{array.shape} -> {flux_h5[species].shape}"
    )


# ============================================================
# Comprobar los flujos tau iniciales
# ============================================================

for tau_species in ["nutau", "antinutau"]:
    tau_maximum = np.max(
        np.abs(flux_h5[tau_species])
    )

    if tau_maximum != 0.0:
        print(
            f"Advertencia: {tau_species} no es exactamente cero. "
            f"Máximo absoluto = {tau_maximum:.8e}"
        )


# ============================================================
# Crear archivo HDF5
# ============================================================

if OUTPUT_FILE.exists():
    print(
        f"\nEl archivo ya existe y será reemplazado:\n"
        f"{OUTPUT_FILE.resolve()}"
    )

print(f"\nCreando: {OUTPUT_FILE.resolve()}")

# Como nuSQuIDS leerá normalmente una fila angular completa:
#
#     flux[coszen_index, :]
#
# usamos chunks de:
#
#     (1, hasta 512 energías)
#
# Esto facilita la lectura de un espectro por cada trayectoria.
energy_chunk_size = min(N_ENERGY, 512)

with h5py.File(
    OUTPUT_FILE,
    "w",
    libver="latest",
) as h5_file:

    # ========================================================
    # Atributos generales
    # ========================================================

    h5_file.attrs["format_name"] = (
        "MCEq atmospheric neutrino initial flux"
    )

    h5_file.attrs["format_version"] = "1.0"

    h5_file.attrs["description"] = (
        "Atmospheric neutrino fluxes generated with MCEq "
        "for use as initial conditions in nuSQuIDS."
    )

    h5_file.attrs["flux_dimension_order"] = (
        "coszen, energy"
    )

    h5_file.attrs["flux_quantity"] = (
        "differential flux Phi(E), with MCEq mag=0"
    )

    h5_file.attrs["energy_unit"] = "GeV"

    h5_file.attrs["coszen_definition"] = (
        "cosine of the true neutrino zenith angle"
    )

    h5_file.attrs["mceq_reflection"] = (
        "MCEq uses theta_MCEq = arccos(abs(coszen_true))"
    )

    h5_file.attrs["number_energy_points"] = N_ENERGY
    h5_file.attrs["number_coszen_points"] = N_COSZEN
    h5_file.attrs["number_mceq_angles"] = N_MCEQ_ANGLES

    # Añadir metadatos originales como atributos.
    for key, value in metadata.items():
        attribute_name = f"mceq_{key}"

        if value is None:
            h5_file.attrs[attribute_name] = "None"
        elif isinstance(
            value,
            (str, int, float, bool, np.integer, np.floating),
        ):
            h5_file.attrs[attribute_name] = value
        else:
            h5_file.attrs[attribute_name] = json.dumps(value)

    # ========================================================
    # Grupo de ejes
    # ========================================================

    axes_group = h5_file.create_group("axes")

    axes_group.attrs["description"] = (
        "Axes used by all flux datasets."
    )

    energy_dataset = axes_group.create_dataset(
        "energy_GeV",
        data=energy_GeV,
        dtype=np.float64,
    )

    energy_dataset.attrs["unit"] = "GeV"
    energy_dataset.attrs["description"] = (
        "Neutrino energy grid."
    )
    energy_dataset.attrs["ordering"] = (
        "strictly increasing"
    )

    coszen_dataset = axes_group.create_dataset(
        "coszen",
        data=coszen,
        dtype=np.float64,
    )

    coszen_dataset.attrs["unit"] = "dimensionless"
    coszen_dataset.attrs["description"] = (
        "Signed cosine of true zenith angle."
    )
    coszen_dataset.attrs["convention"] = (
        "-1 vertically upgoing, "
        "0 horizontal, "
        "+1 vertically downgoing"
    )
    coszen_dataset.attrs["ordering"] = (
        "strictly increasing"
    )

    theta_true_dataset = axes_group.create_dataset(
        "theta_true_deg",
        data=theta_true_deg,
        dtype=np.float64,
    )

    theta_true_dataset.attrs["unit"] = "degree"
    theta_true_dataset.attrs["description"] = (
        "True zenith angle computed as arccos(coszen)."
    )

    # ========================================================
    # Información específica de MCEq
    # ========================================================

    mceq_group = h5_file.create_group("mceq")

    mceq_group.attrs["description"] = (
        "Angular grid actually solved by MCEq "
        "before reflection to signed coszen."
    )

    abs_coszen_dataset = mceq_group.create_dataset(
        "abs_coszen",
        data=abs_coszen_mceq,
        dtype=np.float64,
    )

    abs_coszen_dataset.attrs["unit"] = "dimensionless"
    abs_coszen_dataset.attrs["description"] = (
        "Absolute cosine-of-zenith values solved by MCEq."
    )

    theta_mceq_dataset = mceq_group.create_dataset(
        "theta_deg",
        data=theta_mceq_deg,
        dtype=np.float64,
    )

    theta_mceq_dataset.attrs["unit"] = "degree"
    theta_mceq_dataset.attrs["description"] = (
        "Zenith angles passed to MCEq."
    )

    theta_mceq_dataset.attrs["definition"] = (
        "theta_MCEq = arccos(abs(coszen_true))"
    )

    # ========================================================
    # Grupo de flujos
    # ========================================================

    flux_group = h5_file.create_group("flux")

    flux_group.attrs["description"] = (
        "Initial atmospheric neutrino flux matrices."
    )

    flux_group.attrs["dimension_order"] = (
        "coszen, energy"
    )

    flux_group.attrs["shape"] = (
        f"{N_COSZEN}, {N_ENERGY}"
    )

    flux_group.attrs["quantity"] = (
        "Phi(E), generated with MCEq mag=0"
    )

    for species, array in flux_h5.items():
        dataset = flux_group.create_dataset(
            species,
            data=array,
            dtype=np.float64,
            chunks=(
                1,
                energy_chunk_size,
            ),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )

        dataset.attrs["species"] = species
        dataset.attrs["dimension_order"] = (
            "coszen, energy"
        )
        dataset.attrs["shape_description"] = (
            "dataset[coszen_index, energy_index]"
        )
        dataset.attrs["energy_axis"] = (
            "/axes/energy_GeV"
        )
        dataset.attrs["coszen_axis"] = (
            "/axes/coszen"
        )
        dataset.attrs["magnification_mag"] = 0
        dataset.attrs["quantity"] = (
            "differential atmospheric neutrino flux Phi(E)"
        )

        if species in ("nutau", "antinutau"):
            dataset.attrs["initial_condition"] = (
                "set to zero before nuSQuIDS propagation"
            )

    # Forzar la escritura antes de cerrar.
    h5_file.flush()


# ============================================================
# Verificar el archivo recién creado
# ============================================================

print("\nVerificando el archivo HDF5...")

with h5py.File(
    OUTPUT_FILE,
    "r",
) as h5_file:

    required_datasets = [
        "/axes/energy_GeV",
        "/axes/coszen",
        "/axes/theta_true_deg",
        "/mceq/abs_coszen",
        "/mceq/theta_deg",
        "/flux/nue",
        "/flux/numu",
        "/flux/nutau",
        "/flux/antinue",
        "/flux/antinumu",
        "/flux/antinutau",
    ]

    for dataset_path in required_datasets:
        if dataset_path not in h5_file:
            raise RuntimeError(
                f"No se encontró {dataset_path} "
                f"en el archivo HDF5."
            )

    energy_check = h5_file[
        "/axes/energy_GeV"
    ][...]

    coszen_check = h5_file[
        "/axes/coszen"
    ][...]

    if not np.array_equal(
        energy_check,
        energy_GeV,
    ):
        raise RuntimeError(
            "La malla energética cambió al guardarse."
        )

    if not np.array_equal(
        coszen_check,
        coszen,
    ):
        raise RuntimeError(
            "La malla angular cambió al guardarse."
        )

    expected_h5_shape = (
        N_COSZEN,
        N_ENERGY,
    )

    for species in flux_h5:
        dataset = h5_file[
            f"/flux/{species}"
        ]

        if dataset.shape != expected_h5_shape:
            raise RuntimeError(
                f"Shape incorrecto para /flux/{species}: "
                f"{dataset.shape}; "
                f"esperado {expected_h5_shape}."
            )

        if dataset.dtype != np.dtype("float64"):
            raise RuntimeError(
                f"Tipo incorrecto para /flux/{species}: "
                f"{dataset.dtype}."
            )


# ============================================================
# Resumen final
# ============================================================

file_size_mb = (
    OUTPUT_FILE.stat().st_size
    / (1024.0 ** 2)
)

print()
print("=" * 72)
print("Archivo HDF5 creado correctamente")
print("=" * 72)
print(f"Archivo       : {OUTPUT_FILE.resolve()}")
print(f"Tamaño        : {file_size_mb:.3f} MB")
print(f"Energías      : {N_ENERGY}")
print(f"Ángulos       : {N_COSZEN}")
print(f"Orden         : flux[coszen, energy]")
print(f"Tipo numérico : float64")
print(f"Compresión    : gzip nivel 4")
print("=" * 72)
