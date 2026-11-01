#!/usr/bin/env python3

from pathlib import Path
import hashlib
import os

import h5py
import numpy as np


# =============================================================================
# FILES
# =============================================================================

FILE_A = Path(
    "/simulation/LRI/RESULTS/Le-Ltau/"
    "event_templates_effective.h5"
)

FILE_B = Path(
    "/simulation/LRI/RESULTS/Lmu-Ltau/"
    "event_templates_effective.h5"
)

LABEL_A = "Le-Ltau"
LABEL_B = "Lmu-Ltau"


# =============================================================================
# CONFIG
# =============================================================================

CATEGORIES = (
    "nu_nc",
    "nue_cc",
    "numu_cc",
    "nutau_cc",
)

# Benchmark node for a detailed comparison.
BENCHMARK_LOG10_M = -13.0
BENCHMARK_LOG10_G = -25.0

# event_weights/nominal is several GiB.
# False = inspect several representative nodes only.
# True  = scan the whole huge dataset chunk by chunk.
FULL_EVENT_WEIGHT_SCAN = False

# Exact equality tolerance.
RTOL = 0.0
ATOL = 0.0


# =============================================================================
# HELPERS
# =============================================================================

def human_size(number):
    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    value = float(number)

    for unit in units:
        if value < 1024.0:
            return f"{value:.3f} {unit}"
        value /= 1024.0

    return f"{value:.3f} PiB"


def exact_index(
    axis,
    requested,
    name,
    tolerance=2.0e-8,
):
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    index = int(
        np.argmin(
            np.abs(
                axis - requested
            )
        )
    )

    difference = abs(
        float(axis[index])
        - float(requested)
    )

    if difference > tolerance:
        raise ValueError(
            f"{name}={requested} no es nodo exacto. "
            f"Más cercano={axis[index]}"
        )

    return index


def array_stats_difference(
    a,
    b,
):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        return {
            "same_shape": False,
            "shape_a": a.shape,
            "shape_b": b.shape,
        }

    diff = (
        a.astype(np.float64)
        -
        b.astype(np.float64)
    )

    absolute = np.abs(
        diff
    )

    n_total = diff.size

    n_different = int(
        np.count_nonzero(
            a != b
        )
    )

    return {
        "same_shape": True,
        "shape": a.shape,
        "exact_equal": bool(
            np.array_equal(
                a,
                b
            )
        ),
        "n_total": n_total,
        "n_different": n_different,
        "fraction_different": (
            n_different / n_total
            if n_total
            else 0.0
        ),
        "max_abs_diff": float(
            np.max(
                absolute
            )
        ) if n_total else 0.0,
        "mean_abs_diff": float(
            np.mean(
                absolute
            )
        ) if n_total else 0.0,
        "rms_diff": float(
            np.sqrt(
                np.mean(
                    diff * diff
                )
            )
        ) if n_total else 0.0,
    }


def print_stats(
    name,
    stats,
):
    print()
    print(
        f"[{name}]"
    )

    if not stats[
        "same_shape"
    ]:
        print(
            "  SHAPE DIFFERENT"
        )
        print(
            f"  A: {stats['shape_a']}"
        )
        print(
            f"  B: {stats['shape_b']}"
        )
        return

    print(
        f"  shape              = {stats['shape']}"
    )

    print(
        f"  exact_equal        = {stats['exact_equal']}"
    )

    print(
        f"  different elements = "
        f"{stats['n_different']:,} / "
        f"{stats['n_total']:,}"
    )

    print(
        f"  fraction different = "
        f"{stats['fraction_different']:.6e}"
    )

    print(
        f"  max |A-B|          = "
        f"{stats['max_abs_diff']:.12e}"
    )

    print(
        f"  mean |A-B|         = "
        f"{stats['mean_abs_diff']:.12e}"
    )

    print(
        f"  RMS(A-B)           = "
        f"{stats['rms_diff']:.12e}"
    )


def dataset_hash(
    dataset,
    block_size=1024 * 1024,
):
    """
    Hash a dataset without intentionally converting
    the whole thing into one Python byte object.
    """

    sha = hashlib.sha256()

    if dataset.ndim == 0:
        sha.update(
            np.asarray(
                dataset[()]
            ).tobytes()
        )
        return sha.hexdigest()

    # For normal small datasets:
    if dataset.size <= 5_000_000:
        sha.update(
            np.asarray(
                dataset[...]
            ).tobytes()
        )
        return sha.hexdigest()

    # Large first-axis chunking.
    for i in range(
        dataset.shape[0]
    ):
        sha.update(
            np.asarray(
                dataset[i]
            ).tobytes()
        )

    return sha.hexdigest()


# =============================================================================
# BASIC FILE CHECK
# =============================================================================

def basic_file_check():

    print(
        "=" * 72
    )
    print(
        "BASIC FILE CHECK"
    )
    print(
        "=" * 72
    )

    for label, filename in (
        (LABEL_A, FILE_A),
        (LABEL_B, FILE_B),
    ):

        if not filename.is_file():
            raise FileNotFoundError(
                f"{label}: no existe\n"
                f"{filename}"
            )

        stat = filename.stat()

        print()
        print(
            label
        )

        print(
            f"  path  = {filename}"
        )

        print(
            f"  size  = "
            f"{human_size(stat.st_size)}"
        )

        print(
            f"  inode = {stat.st_ino}"
        )

    print()
    print(
        "os.path.samefile =",
        os.path.samefile(
            FILE_A,
            FILE_B,
        ),
    )

    if os.path.samefile(
        FILE_A,
        FILE_B,
    ):
        print()
        print(
            "*** ERROR: ambos paths apuntan "
            "literalmente al mismo archivo. ***"
        )


# =============================================================================
# STRUCTURE
# =============================================================================

def compare_structure(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "HDF5 STRUCTURE"
    )
    print(
        "=" * 72
    )

    paths_a = set()
    paths_b = set()

    a.visit(
        paths_a.add
    )

    b.visit(
        paths_b.add
    )

    only_a = sorted(
        paths_a
        -
        paths_b
    )

    only_b = sorted(
        paths_b
        -
        paths_a
    )

    print()
    print(
        f"Objects in {LABEL_A}: "
        f"{len(paths_a)}"
    )

    print(
        f"Objects in {LABEL_B}: "
        f"{len(paths_b)}"
    )

    print(
        f"Only in {LABEL_A}: "
        f"{len(only_a)}"
    )

    for item in only_a:
        print(
            f"  {item}"
        )

    print(
        f"Only in {LABEL_B}: "
        f"{len(only_b)}"
    )

    for item in only_b:
        print(
            f"  {item}"
        )


# =============================================================================
# AXES
# =============================================================================

def compare_axes(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "AXES"
    )
    print(
        "=" * 72
    )

    axes = (
        "axes/mass_eV",
        "axes/log10_mass_eV",
        "axes/gprime",
        "axes/log10_gprime",
        "axes/reco_energy_edges_GeV",
        "axes/reco_coszen_edges",
        "axes/pid_edges",
    )

    for path in axes:

        if (
            path not in a
            or path not in b
        ):
            print(
                f"{path}: missing"
            )
            continue

        stats = array_stats_difference(
            a[path][...],
            b[path][...],
        )

        print_stats(
            path,
            stats,
        )


# =============================================================================
# THINGS EXPECTED TO BE IDENTICAL
# =============================================================================

def compare_common_detector_data(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "DATA EXPECTED TO BE IDENTICAL"
    )
    print(
        "=" * 72
    )

    paths = (
        "observed/data",
        "backgrounds/atmospheric_muon",
        "event_cache/reco_bin_id",
        "event_cache/species_id",
        "event_cache/category_id",
    )

    for path in paths:

        if (
            path not in a
            or path not in b
        ):
            continue

        stats = array_stats_difference(
            a[path][...],
            b[path][...],
        )

        print_stats(
            path,
            stats,
        )


# =============================================================================
# TEMPLATES
# =============================================================================

def compare_templates(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "PHYSICS-SENSITIVE TEMPLATES"
    )
    print(
        "=" * 72
    )

    all_equal = True

    for category in CATEGORIES:

        path = (
            f"templates/{category}"
        )

        array_a = np.asarray(
            a[path][...],
            dtype=np.float64,
        )

        array_b = np.asarray(
            b[path][...],
            dtype=np.float64,
        )

        stats = (
            array_stats_difference(
                array_a,
                array_b,
            )
        )

        print_stats(
            path,
            stats,
        )

        all_equal &= (
            stats.get(
                "exact_equal",
                False,
            )
        )

    print()
    print(
        "ALL templates exactly identical =",
        all_equal,
    )

    if all_equal:
        print()
        print(
            "*** WARNING: esto es MUY sospechoso. ***"
        )
        print(
            "Le-Ltau y Lmu-Ltau deberían producir "
            "templates distintos."
        )


# =============================================================================
# ONE BENCHMARK NODE
# =============================================================================

def compare_benchmark_node(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "BENCHMARK NODE"
    )
    print(
        "=" * 72
    )

    mass_a = np.asarray(
        a[
            "axes/log10_mass_eV"
        ][...]
    )

    g_a = np.asarray(
        a[
            "axes/log10_gprime"
        ][...]
    )

    mass_b = np.asarray(
        b[
            "axes/log10_mass_eV"
        ][...]
    )

    g_b = np.asarray(
        b[
            "axes/log10_gprime"
        ][...]
    )

    im_a = exact_index(
        mass_a,
        BENCHMARK_LOG10_M,
        "mass A",
    )

    ig_a = exact_index(
        g_a,
        BENCHMARK_LOG10_G,
        "g A",
    )

    im_b = exact_index(
        mass_b,
        BENCHMARK_LOG10_M,
        "mass B",
    )

    ig_b = exact_index(
        g_b,
        BENCHMARK_LOG10_G,
        "g B",
    )

    print()
    print(
        "Point:"
    )

    print(
        f"  log10(m/eV) = "
        f"{BENCHMARK_LOG10_M}"
    )

    print(
        f"  log10(g)    = "
        f"{BENCHMARK_LOG10_G}"
    )

    for category in CATEGORIES:

        path = (
            f"templates/{category}"
        )

        node_a = np.asarray(
            a[path][
                im_a,
                ig_a,
                :
            ],
            dtype=np.float64,
        )

        node_b = np.asarray(
            b[path][
                im_b,
                ig_b,
                :
            ],
            dtype=np.float64,
        )

        stats = (
            array_stats_difference(
                node_a,
                node_b,
            )
        )

        print_stats(
            f"{path} @ benchmark",
            stats,
        )

        print(
            f"  sum {LABEL_A} = "
            f"{np.sum(node_a):.12e}"
        )

        print(
            f"  sum {LABEL_B} = "
            f"{np.sum(node_b):.12e}"
        )


# =============================================================================
# EVENT WEIGHTS
# =============================================================================

def compare_event_weight_nodes(
    a,
    b,
):
    print()
    print(
        "=" * 72
    )
    print(
        "EVENT WEIGHTS / NOMINAL"
    )
    print(
        "=" * 72
    )

    path = (
        "event_weights/nominal"
    )

    if (
        path not in a
        or path not in b
    ):
        print(
            "event_weights/nominal missing."
        )
        return

    da = a[path]
    db = b[path]

    print()
    print(
        f"{LABEL_A}: shape={da.shape}, "
        f"dtype={da.dtype}"
    )

    print(
        f"{LABEL_B}: shape={db.shape}, "
        f"dtype={db.dtype}"
    )

    if da.shape != db.shape:
        print(
            "Shapes DIFFER -> files definitely different."
        )
        return

    nm, ng, _ = da.shape

    sample_nodes = (
        (0, 0),
        (0, ng - 1),
        (nm // 2, ng // 2),
        (nm // 2, ng - 1),
        (nm - 1, ng // 2),
        (nm - 1, ng - 1),
    )

    any_difference = False

    for im, ig in sample_nodes:

        wa = np.asarray(
            da[
                im,
                ig,
                :
            ],
            dtype=np.float64,
        )

        wb = np.asarray(
            db[
                im,
                ig,
                :
            ],
            dtype=np.float64,
        )

        stats = (
            array_stats_difference(
                wa,
                wb,
            )
        )

        print_stats(
            f"event_weights node "
            f"(im={im}, ig={ig})",
            stats,
        )

        if not stats.get(
            "exact_equal",
            False,
        ):
            any_difference = True

    print()
    print(
        "At least one sampled event-weight node differs =",
        any_difference,
    )


# =============================================================================
# OPTIONAL FULL EVENT WEIGHT SCAN
# =============================================================================

def full_event_weight_scan(
    a,
    b,
):
    if not FULL_EVENT_WEIGHT_SCAN:
        return

    print()
    print(
        "=" * 72
    )
    print(
        "FULL EVENT-WEIGHT SCAN"
    )
    print(
        "=" * 72
    )

    da = a[
        "event_weights/nominal"
    ]

    db = b[
        "event_weights/nominal"
    ]

    if da.shape != db.shape:
        print(
            "Different shapes."
        )
        return

    nm, ng, _ = da.shape

    global_max = 0.0
    total_different = 0
    total_elements = 0

    for im in range(
        nm
    ):

        print(
            f"mass {im + 1}/{nm}"
        )

        for ig in range(
            ng
        ):

            wa = np.asarray(
                da[
                    im,
                    ig,
                    :
                ]
            )

            wb = np.asarray(
                db[
                    im,
                    ig,
                    :
                ]
            )

            different = (
                wa != wb
            )

            total_different += int(
                np.count_nonzero(
                    different
                )
            )

            total_elements += (
                wa.size
            )

            max_here = float(
                np.max(
                    np.abs(
                        wa.astype(
                            np.float64
                        )
                        -
                        wb.astype(
                            np.float64
                        )
                    )
                )
            )

            global_max = max(
                global_max,
                max_here,
            )

    print()
    print(
        "Full event-weight result:"
    )

    print(
        f"  different = "
        f"{total_different:,} / "
        f"{total_elements:,}"
    )

    print(
        f"  fraction  = "
        f"{total_different / total_elements:.12e}"
    )

    print(
        f"  max diff  = "
        f"{global_max:.12e}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    basic_file_check()

    with h5py.File(
        FILE_A,
        "r",
    ) as a, h5py.File(
        FILE_B,
        "r",
    ) as b:

        compare_structure(
            a,
            b,
        )

        compare_axes(
            a,
            b,
        )

        compare_common_detector_data(
            a,
            b,
        )

        compare_templates(
            a,
            b,
        )

        compare_benchmark_node(
            a,
            b,
        )

        compare_event_weight_nodes(
            a,
            b,
        )

        full_event_weight_scan(
            a,
            b,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "INTERPRETATION"
    )
    print(
        "=" * 72
    )

    print(
        """
Expected:

- axes                         -> identical
- observed/data                -> identical
- atmospheric muon background -> identical
- event_cache                  -> identical

BUT:

- templates/*                  -> should differ
- sumw2/*                      -> should differ
- event_weights/nominal        -> should differ

If templates and event_weights are exactly identical, then the Lmu-Ltau
build almost certainly used the wrong oscillated_flux_lut.h5, or the
Le-Ltau event template was copied into the Lmu-Ltau directory.
"""
    )


if __name__ == "__main__":
    main()
