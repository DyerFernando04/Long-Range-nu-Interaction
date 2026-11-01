#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_effective_probability_diagnostic.py

Diagnostic of the flavor-level origin of the LRI event excess/deficit.

For atmospheric neutrinos, neglecting the initial tau flux,

    Phi_e
        = Phi_e^0 P_ee
        + Phi_mu^0 P_mu_e

    Phi_mu
        = Phi_e^0 P_e_mu
        + Phi_mu^0 P_mu_mu

Define

    r(E, coszen)
        = Phi_mu^0 / Phi_e^0

Then

    C_e
        = P_ee + r P_mu_e

    C_mu
        = P_e_mu + r P_mu_mu

The figure shows

    Delta P_ee       Delta P_mu_e       Delta C_e / C_e^SM

    Delta P_e_mu     Delta P_mu_mu      Delta C_mu / C_mu^SM

where

    Delta P = P_LRI - P_SM

and the third column is expressed as a percentage.

The probabilities are computed with transition_probabilities.cpp.
"""

from pathlib import Path

import shlex
import subprocess
import tempfile

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import scienceplots


# =============================================================================
# STYLE
# =============================================================================

plt.style.use("science")

plt.rcParams.update({

    "font.size": 12.5,

    "axes.labelsize": 14.5,

    "axes.titlesize": 14.5,

    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,

    "axes.linewidth": 0.9,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,

    "xtick.minor.size": 2.7,
    "ytick.minor.size": 2.7,

    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,

    "xtick.minor.width": 0.65,
    "ytick.minor.width": 0.65,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SCRIPT_DIR = Path(__file__).resolve().parent

CPP_SOURCE = (
    SCRIPT_DIR
    / "transition_probabilities.cpp"
)

BUILD_DIR = (
    SCRIPT_DIR
    / ".probability_build"
)

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "probability"
)

INITIAL_FLUX_FILE = (
    ROOT
    / "INITIAL_FLUX"
    / "mceq_initial_fluxes1.h5"
)


# =============================================================================
# PHYSICAL CONFIGURATION
# =============================================================================

MODEL = "Le-Lmu"

# -------------------------------------------------------------------------
# Benchmark requested:
#
#     g' = 1e-24
#     m' = 1e-13 eV
# -------------------------------------------------------------------------

LOG10_G = -24.0
LOG10_M = -13.0


# Upgoing atmospheric neutrinos
COSZEN_MIN = -1.0
COSZEN_MAX = 0.0


# Probability energy grid
N_ENERGY = 160

E_MIN_GEV = 5.0
E_MAX_GEV = 200.0


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

CASES = {

    "Le-Lmu": {

        "label":
            r"$L_e-L_\mu$",

        "include_dir":
            ROOT
            / "LUT_FLUX",

        "lut":
            ROOT
            / "LUT_VLRI"
            / "vlri_unit.h5",
    },

    "Le-Ltau": {

        "label":
            r"$L_e-L_\tau$",

        "include_dir":
            ROOT
            / "LUT_FLUX"
            / "Le-Ltau",

        "lut":
            ROOT
            / "LUT_VLRI"
            / "vlri_unit.h5",
    },

    "Lmu-Ltau": {

        "label":
            r"$L_\mu-L_\tau$",

        "include_dir":
            ROOT
            / "LUT_FLUX"
            / "Lmu-Ltau",

        "lut":
            ROOT
            / "LUT_VLRI"
            / "Lmu-Ltau"
            / "vlri_unit.h5",
    },
}


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

CMAP = "RdBu_r"

ROBUST_PERCENTILE = 99.0

GRID_MAJOR_ALPHA = 0.14
GRID_MAJOR_LINEWIDTH = 0.45

GRID_MINOR_ALPHA = 0.035
GRID_MINOR_LINEWIDTH = 0.28


# =============================================================================
# COMMAND HELPERS
# =============================================================================

def run_checked(command):

    print(
        " ".join(
            shlex.quote(str(x))
            for x in command
        )
    )

    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def pkg_config_tokens(mode):

    result = subprocess.check_output(
        [
            "pkg-config",
            mode,
            "nusquids",
        ],
        text=True,
    )

    return shlex.split(
        result.strip()
    )


# =============================================================================
# GRID HELPERS
# =============================================================================

def centers_to_edges_linear(centers):

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    if centers.size < 2:

        raise ValueError(
            "Need at least two linear centers."
        )

    edges = np.empty(
        centers.size + 1,
        dtype=np.float64,
    )

    edges[1:-1] = (
        0.5
        *
        (
            centers[:-1]
            +
            centers[1:]
        )
    )

    edges[0] = (
        centers[0]
        -
        0.5
        *
        (
            centers[1]
            -
            centers[0]
        )
    )

    edges[-1] = (
        centers[-1]
        +
        0.5
        *
        (
            centers[-1]
            -
            centers[-2]
        )
    )

    return edges


def centers_to_edges_log(centers):

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    if np.any(
        centers <= 0.0
    ):

        raise ValueError(
            "Energy centers must be positive."
        )

    log_centers = np.log10(
        centers
    )

    log_edges = np.empty(
        log_centers.size + 1,
        dtype=np.float64,
    )

    log_edges[1:-1] = (
        0.5
        *
        (
            log_centers[:-1]
            +
            log_centers[1:]
        )
    )

    log_edges[0] = (
        log_centers[0]
        -
        0.5
        *
        (
            log_centers[1]
            -
            log_centers[0]
        )
    )

    log_edges[-1] = (
        log_centers[-1]
        +
        0.5
        *
        (
            log_centers[-1]
            -
            log_centers[-2]
        )
    )

    return (
        10.0
        **
        log_edges
    )


# =============================================================================
# COLOR-SCALE HELPER
# =============================================================================

def robust_abs_max(
    arrays,
    percentile=99.0,
):

    values = []

    for array in arrays:

        array = np.asarray(
            array,
            dtype=np.float64,
        )

        finite = array[
            np.isfinite(
                array
            )
        ]

        if finite.size:

            values.append(
                np.abs(
                    finite
                ).ravel()
            )

    if not values:

        return 1.0

    values = np.concatenate(
        values
    )

    vmax = float(
        np.percentile(
            values,
            percentile,
        )
    )

    return max(
        vmax,
        1.0e-12,
    )


# =============================================================================
# LOG-LOG INTERPOLATION
# =============================================================================

def interpolate_positive_loglog(
    x_old,
    y_old,
    x_new,
):

    x_old = np.asarray(
        x_old,
        dtype=np.float64,
    )

    y_old = np.asarray(
        y_old,
        dtype=np.float64,
    )

    x_new = np.asarray(
        x_new,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            x_old
        )
        &
        np.isfinite(
            y_old
        )
        &
        (
            x_old > 0.0
        )
        &
        (
            y_old > 0.0
        )
    )

    if np.count_nonzero(
        valid
    ) < 2:

        raise RuntimeError(
            "Not enough positive flux values "
            "for interpolation."
        )

    return np.power(
        10.0,
        np.interp(
            np.log10(
                x_new
            ),
            np.log10(
                x_old[
                    valid
                ]
            ),
            np.log10(
                y_old[
                    valid
                ]
            ),
        ),
    )


# =============================================================================
# COMPILE transition_probabilities.cpp
# =============================================================================

def compile_probability_program(case):

    if not CPP_SOURCE.is_file():

        raise FileNotFoundError(
            f"Missing C++ source:\n"
            f"{CPP_SOURCE}"
        )

    BUILD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    executable = (
        BUILD_DIR
        /
        f"transition_probabilities_{MODEL}_effective"
    )

    cflags = pkg_config_tokens(
        "--cflags"
    )

    libs = pkg_config_tokens(
        "--libs"
    )

    command = [

        "g++",

        "-O3",
        "-std=c++17",

        str(
            CPP_SOURCE
        ),

        "-I"
        +
        str(
            case[
                "include_dir"
            ]
        ),

        "-I"
        +
        str(
            ROOT
            / "LUT_FLUX"
        ),

        "-Wno-abi",

        *cflags,

        *libs,

        "-lhdf5_hl",

        "-o",

        str(
            executable
        ),
    ]

    print(
        "Compiling probability calculator:"
    )

    run_checked(
        command
    )

    return executable


# =============================================================================
# INITIAL MCEQ FLUX
# =============================================================================

def load_initial_flux():

    if not INITIAL_FLUX_FILE.is_file():

        raise FileNotFoundError(
            f"Missing initial flux file:\n"
            f"{INITIAL_FLUX_FILE}"
        )

    with h5py.File(
        INITIAL_FLUX_FILE,
        "r",
    ) as h5:

        energy = np.asarray(
            h5[
                "/axes/energy_GeV"
            ][...],
            dtype=np.float64,
        )

        coszen = np.asarray(
            h5[
                "/axes/coszen"
            ][...],
            dtype=np.float64,
        )

        nue = np.asarray(
            h5[
                "/flux/nue"
            ][...],
            dtype=np.float64,
        )

        numu = np.asarray(
            h5[
                "/flux/numu"
            ][...],
            dtype=np.float64,
        )

    return {

        "energy":
            energy,

        "coszen":
            coszen,

        "nue":
            nue,

        "numu":
            numu,
    }


# =============================================================================
# PROBABILITY GRID
# =============================================================================

def build_probability_grid(
    executable,
    case,
    coszen_grid,
):

    probability_names = (

        "sm_Pee",
        "sm_Pemu",
        "sm_Pmue",
        "sm_Pmumu",

        "lri_Pee",
        "lri_Pemu",
        "lri_Pmue",
        "lri_Pmumu",
    )

    result = {

        name:
            np.empty(
                (
                    coszen_grid.size,
                    N_ENERGY,
                ),
                dtype=np.float64,
            )

        for name in probability_names
    }

    energy_grid = None


    with tempfile.TemporaryDirectory(
        prefix="effective_probability_"
    ) as tmpdir:

        tmpdir = Path(
            tmpdir
        )


        for i, coszen in enumerate(
            coszen_grid
        ):

            output_csv = (
                tmpdir
                /
                f"prob_{i:04d}.csv"
            )

            command = [

                str(
                    executable
                ),

                str(
                    case[
                        "lut"
                    ]
                ),

                str(
                    output_csv
                ),

                str(
                    LOG10_G
                ),

                str(
                    LOG10_M
                ),

                f"{coszen:.16g}",

                str(
                    N_ENERGY
                ),

                str(
                    E_MIN_GEV
                ),

                str(
                    E_MAX_GEV
                ),
            ]

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
            )

            df = pd.read_csv(
                output_csv
            )

            if energy_grid is None:

                energy_grid = np.asarray(
                    df[
                        "energy_GeV"
                    ],
                    dtype=np.float64,
                )

            for name in probability_names:

                result[
                    name
                ][
                    i,
                    :
                ] = np.asarray(
                    df[
                        name
                    ],
                    dtype=np.float64,
                )

            print(
                f"\rProbability angle "
                f"{i + 1}/"
                f"{coszen_grid.size}",
                end="",
                flush=True,
            )

    print()

    result[
        "energy"
    ] = energy_grid

    result[
        "coszen"
    ] = np.asarray(
        coszen_grid,
        dtype=np.float64,
    )

    return result


# =============================================================================
# ATMOSPHERIC FLUX RATIO
# =============================================================================

def build_flux_ratio(
    initial_flux,
    energy_grid,
    coszen_grid,
):

    initial_energy = (
        initial_flux[
            "energy"
        ]
    )

    initial_coszen = (
        initial_flux[
            "coszen"
        ]
    )

    ratio = np.empty(
        (
            coszen_grid.size,
            energy_grid.size,
        ),
        dtype=np.float64,
    )

    for i, coszen in enumerate(
        coszen_grid
    ):

        coszen_index = int(
            np.argmin(
                np.abs(
                    initial_coszen
                    -
                    coszen
                )
            )
        )

        difference = abs(
            float(
                initial_coszen[
                    coszen_index
                ]
            )
            -
            float(
                coszen
            )
        )

        if difference > 1.0e-10:

            raise RuntimeError(
                "Could not match coszen="
                f"{coszen}"
            )

        phi_e = (
            interpolate_positive_loglog(

                initial_energy,

                initial_flux[
                    "nue"
                ][
                    coszen_index,
                    :
                ],

                energy_grid,
            )
        )

        phi_mu = (
            interpolate_positive_loglog(

                initial_energy,

                initial_flux[
                    "numu"
                ][
                    coszen_index,
                    :
                ],

                energy_grid,
            )
        )

        ratio[
            i,
            :
        ] = (
            phi_mu
            /
            phi_e
        )

    return ratio


# =============================================================================
# EFFECTIVE COMBINATIONS
# =============================================================================

def build_effective_combinations(
    probabilities,
    r,
):

    # =========================================================================
    # ELECTRON FINAL FLAVOR
    #
    # C_e = P_ee + r P_mu_e
    # =========================================================================

    ce_sm = (
        probabilities[
            "sm_Pee"
        ]
        +
        r
        *
        probabilities[
            "sm_Pmue"
        ]
    )

    ce_lri = (
        probabilities[
            "lri_Pee"
        ]
        +
        r
        *
        probabilities[
            "lri_Pmue"
        ]
    )


    # =========================================================================
    # MUON FINAL FLAVOR
    #
    # C_mu = P_e_mu + r P_mu_mu
    # =========================================================================

    cmu_sm = (
        probabilities[
            "sm_Pemu"
        ]
        +
        r
        *
        probabilities[
            "sm_Pmumu"
        ]
    )

    cmu_lri = (
        probabilities[
            "lri_Pemu"
        ]
        +
        r
        *
        probabilities[
            "lri_Pmumu"
        ]
    )


    # =========================================================================
    # RELATIVE DIFFERENCES
    # =========================================================================

    delta_ce_percent = np.full_like(
        ce_sm,
        np.nan,
    )

    delta_cmu_percent = np.full_like(
        cmu_sm,
        np.nan,
    )


    valid_e = (
        np.isfinite(
            ce_sm
        )
        &
        np.isfinite(
            ce_lri
        )
        &
        (
            ce_sm > 0.0
        )
    )

    valid_mu = (
        np.isfinite(
            cmu_sm
        )
        &
        np.isfinite(
            cmu_lri
        )
        &
        (
            cmu_sm > 0.0
        )
    )


    delta_ce_percent[
        valid_e
    ] = (
        100.0
        *
        (
            ce_lri[
                valid_e
            ]
            /
            ce_sm[
                valid_e
            ]
            -
            1.0
        )
    )

    delta_cmu_percent[
        valid_mu
    ] = (
        100.0
        *
        (
            cmu_lri[
                valid_mu
            ]
            /
            cmu_sm[
                valid_mu
            ]
            -
            1.0
        )
    )


    return {

        "ce_sm":
            ce_sm,

        "ce_lri":
            ce_lri,

        "cmu_sm":
            cmu_sm,

        "cmu_lri":
            cmu_lri,

        "delta_ce_percent":
            delta_ce_percent,

        "delta_cmu_percent":
            delta_cmu_percent,
    }


# =============================================================================
# AXIS STYLE
# =============================================================================

def style_axis(
    ax,
    show_xlabel,
    show_ylabel,
):

    ax.set_xscale(
        "log"
    )

    ax.minorticks_on()

    ax.grid(
        True,
        which="major",
        alpha=GRID_MAJOR_ALPHA,
        linewidth=GRID_MAJOR_LINEWIDTH,
    )

    ax.grid(
        True,
        which="minor",
        alpha=GRID_MINOR_ALPHA,
        linewidth=GRID_MINOR_LINEWIDTH,
    )

    if show_xlabel:

        ax.set_xlabel(
            r"$E_\nu\,[\mathrm{GeV}]$"
        )

    if show_ylabel:

        ax.set_ylabel(
            r"$\cos\theta_z$"
        )


# =============================================================================
# FIGURE
# =============================================================================

def make_plot(
    case,
    probabilities,
    combinations,
):

    energy = (
        probabilities[
            "energy"
        ]
    )

    coszen = (
        probabilities[
            "coszen"
        ]
    )

    energy_edges = (
        centers_to_edges_log(
            energy
        )
    )

    coszen_edges = (
        centers_to_edges_linear(
            coszen
        )
    )


    # =========================================================================
    # PROBABILITY DIFFERENCES
    # =========================================================================

    delta_pee = (
        probabilities[
            "lri_Pee"
        ]
        -
        probabilities[
            "sm_Pee"
        ]
    )

    delta_pmue = (
        probabilities[
            "lri_Pmue"
        ]
        -
        probabilities[
            "sm_Pmue"
        ]
    )

    delta_pemu = (
        probabilities[
            "lri_Pemu"
        ]
        -
        probabilities[
            "sm_Pemu"
        ]
    )

    delta_pmumu = (
        probabilities[
            "lri_Pmumu"
        ]
        -
        probabilities[
            "sm_Pmumu"
        ]
    )


    # One scale for the four probability panels
    probability_vmax = (
        robust_abs_max(

            [
                delta_pee,
                delta_pmue,
                delta_pemu,
                delta_pmumu,
            ],

            percentile=(
                ROBUST_PERCENTILE
            ),
        )
    )

    probability_norm = TwoSlopeNorm(
        vmin=-probability_vmax,
        vcenter=0.0,
        vmax=probability_vmax,
    )


    # One scale for the two effective-combination panels
    combination_vmax = (
        robust_abs_max(

            [
                combinations[
                    "delta_ce_percent"
                ],

                combinations[
                    "delta_cmu_percent"
                ],
            ],

            percentile=(
                ROBUST_PERCENTILE
            ),
        )
    )

    combination_norm = TwoSlopeNorm(
        vmin=-combination_vmax,
        vcenter=0.0,
        vmax=combination_vmax,
    )


    # =========================================================================
    # EXPLICIT GRID
    #
    # col 0 : probability panel
    # col 1 : probability panel
    # col 2 : probability colorbar
    # col 3 : effective-combination panel
    # col 4 : effective-combination colorbar
    #
    # This prevents any overlap.
    # =========================================================================

    fig = plt.figure(
        figsize=(
            13.2,
            6.7,
        )
    )


    gs = fig.add_gridspec(

        nrows=2,
        ncols=5,

        width_ratios=[
            1.0,
            1.0,
            0.055,
            1.0,
            0.055,
        ],

        height_ratios=[
            1.0,
            1.0,
        ],

        left=0.065,
        right=0.965,

        bottom=0.11,
        top=0.895,

        wspace=0.18,
        hspace=0.16,
    )


    # =========================================================================
    # PANELS
    # =========================================================================

    ax_pee = fig.add_subplot(
        gs[0, 0]
    )

    ax_pmue = fig.add_subplot(
        gs[0, 1],
        sharex=ax_pee,
        sharey=ax_pee,
    )

    ax_ce = fig.add_subplot(
        gs[0, 3],
        sharex=ax_pee,
        sharey=ax_pee,
    )


    ax_pemu = fig.add_subplot(
        gs[1, 0],
        sharex=ax_pee,
        sharey=ax_pee,
    )

    ax_pmumu = fig.add_subplot(
        gs[1, 1],
        sharex=ax_pee,
        sharey=ax_pee,
    )

    ax_cmu = fig.add_subplot(
        gs[1, 3],
        sharex=ax_pee,
        sharey=ax_pee,
    )


    # =========================================================================
    # DEDICATED COLORBAR AXES
    # =========================================================================

    cax_probability = fig.add_subplot(
        gs[:, 2]
    )

    cax_combination = fig.add_subplot(
        gs[:, 4]
    )


    # =========================================================================
    # PROBABILITY PANELS
    # =========================================================================

    mesh_probability = ax_pee.pcolormesh(

        energy_edges,
        coszen_edges,
        delta_pee,

        shading="auto",

        cmap=CMAP,
        norm=probability_norm,

        rasterized=True,
    )


    ax_pmue.pcolormesh(

        energy_edges,
        coszen_edges,
        delta_pmue,

        shading="auto",

        cmap=CMAP,
        norm=probability_norm,

        rasterized=True,
    )


    ax_pemu.pcolormesh(

        energy_edges,
        coszen_edges,
        delta_pemu,

        shading="auto",

        cmap=CMAP,
        norm=probability_norm,

        rasterized=True,
    )


    ax_pmumu.pcolormesh(

        energy_edges,
        coszen_edges,
        delta_pmumu,

        shading="auto",

        cmap=CMAP,
        norm=probability_norm,

        rasterized=True,
    )


    # =========================================================================
    # EFFECTIVE COMBINATIONS
    # =========================================================================

    mesh_combination = ax_ce.pcolormesh(

        energy_edges,
        coszen_edges,

        combinations[
            "delta_ce_percent"
        ],

        shading="auto",

        cmap=CMAP,
        norm=combination_norm,

        rasterized=True,
    )


    ax_cmu.pcolormesh(

        energy_edges,
        coszen_edges,

        combinations[
            "delta_cmu_percent"
        ],

        shading="auto",

        cmap=CMAP,
        norm=combination_norm,

        rasterized=True,
    )


    # =========================================================================
    # TITLES
    # =========================================================================

    ax_pee.set_title(
        r"$\Delta P_{ee}$"
    )

    ax_pmue.set_title(
        r"$\Delta P_{\mu e}$"
    )

    ax_ce.set_title(
        r"$100\,\Delta C_e/C_e^{\rm SM}$"
    )

    ax_pemu.set_title(
        r"$\Delta P_{e\mu}$"
    )

    ax_pmumu.set_title(
        r"$\Delta P_{\mu\mu}$"
    )

    ax_cmu.set_title(
        r"$100\,\Delta C_\mu/C_\mu^{\rm SM}$"
    )


    # =========================================================================
    # AXES
    # =========================================================================

    axes = [

        ax_pee,
        ax_pmue,
        ax_ce,

        ax_pemu,
        ax_pmumu,
        ax_cmu,
    ]


    for ax in axes:

        ax.set_xlim(
            E_MIN_GEV,
            E_MAX_GEV,
        )

        ax.set_ylim(
            COSZEN_MIN,
            COSZEN_MAX,
        )


    style_axis(
        ax_pee,
        show_xlabel=False,
        show_ylabel=True,
    )

    style_axis(
        ax_pmue,
        show_xlabel=False,
        show_ylabel=False,
    )

    style_axis(
        ax_ce,
        show_xlabel=False,
        show_ylabel=False,
    )

    style_axis(
        ax_pemu,
        show_xlabel=True,
        show_ylabel=True,
    )

    style_axis(
        ax_pmumu,
        show_xlabel=True,
        show_ylabel=False,
    )

    style_axis(
        ax_cmu,
        show_xlabel=True,
        show_ylabel=False,
    )


    # Hide redundant labels explicitly
    ax_pmue.tick_params(
        labelleft=False
    )

    ax_ce.tick_params(
        labelleft=False
    )

    ax_pmumu.tick_params(
        labelleft=False
    )

    ax_cmu.tick_params(
        labelleft=False
    )


    ax_pee.tick_params(
        labelbottom=False
    )

    ax_pmue.tick_params(
        labelbottom=False
    )

    ax_ce.tick_params(
        labelbottom=False
    )


    # =========================================================================
    # PROBABILITY COLORBAR
    # =========================================================================

    cbar_probability = fig.colorbar(

        mesh_probability,

        cax=cax_probability,

        extend="both",
    )


    cbar_probability.set_label(

        r"$P_{\rm LRI}-P_{\rm SM}$",

        fontsize=14.0,
        labelpad=8,
    )


    cbar_probability.ax.tick_params(
        labelsize=11.0
    )


    # =========================================================================
    # EFFECTIVE-COMBINATION COLORBAR
    # =========================================================================

    cbar_combination = fig.colorbar(

        mesh_combination,

        cax=cax_combination,

        extend="both",
    )


    cbar_combination.set_label(

        r"Relative deviation [\%]",

        fontsize=14.0,
        labelpad=8,
    )


    cbar_combination.ax.tick_params(
        labelsize=11.0
    )


    # =========================================================================
    # MAIN TITLE
    # =========================================================================

    fig.suptitle(

        case[
            "label"
        ]
        +
        r", "
        +
        rf"$\log_{{10}}(m'/\mathrm{{eV}})"
        rf"={LOG10_M:.1f}$, "
        +
        rf"$\log_{{10}}(g')"
        rf"={LOG10_G:.1f}$",

        fontsize=16.0,

        y=0.975,
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_png = (
        OUTPUT_DIR
        /
        (
            f"effective_probability_diagnostic_"
            f"{MODEL}_"
            f"logm_m13_logg_m24.png"
        )
    )


    output_pdf = (
        OUTPUT_DIR
        /
        (
            f"effective_probability_diagnostic_"
            f"{MODEL}_"
            f"logm_m13_logg_m24.pdf"
        )
    )


    fig.savefig(

        output_png,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.035,
    )


    fig.savefig(

        output_pdf,

        bbox_inches="tight",

        pad_inches=0.035,
    )


    plt.close(
        fig
    )


    print()
    print(
        "Saved PNG:"
    )

    print(
        f"  {output_png}"
    )


    print(
        "Saved PDF:"
    )

    print(
        f"  {output_pdf}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    if MODEL not in CASES:

        raise ValueError(
            f"Unknown MODEL: "
            f"{MODEL}"
        )


    case = (
        CASES[
            MODEL
        ]
    )


    print(
        "============================================================"
    )

    print(
        "Effective oscillation-probability diagnostic"
    )

    print(
        "============================================================"
    )


    print(
        f"Model        : "
        f"{MODEL}"
    )

    print(
        f"log10(m/eV) : "
        f"{LOG10_M}"
    )

    print(
        f"log10(g)     : "
        f"{LOG10_G}"
    )

    print(
        f"g'           : "
        f"{10.0 ** LOG10_G:.3e}"
    )

    print(
        f"m'           : "
        f"{10.0 ** LOG10_M:.3e} eV"
    )

    print(
        f"Energy       : "
        f"{E_MIN_GEV} -> "
        f"{E_MAX_GEV} GeV"
    )

    print(
        f"coszen       : "
        f"{COSZEN_MIN} -> "
        f"{COSZEN_MAX}"
    )

    print()


    # =========================================================================
    # INITIAL FLUX
    # =========================================================================

    print(
        "Reading initial atmospheric flux:"
    )

    print(
        f"  {INITIAL_FLUX_FILE}"
    )


    initial_flux = (
        load_initial_flux()
    )


    # =========================================================================
    # UPGOING MCEQ ZENITH NODES
    # =========================================================================

    coszen_mask = (
        (
            initial_flux[
                "coszen"
            ]
            >=
            COSZEN_MIN
        )
        &
        (
            initial_flux[
                "coszen"
            ]
            <=
            COSZEN_MAX
        )
    )


    coszen_grid = (
        initial_flux[
            "coszen"
        ][
            coszen_mask
        ]
    )


    print(
        f"Number of coszen nodes: "
        f"{coszen_grid.size}"
    )


    # =========================================================================
    # COMPILE
    # =========================================================================

    print()

    executable = (
        compile_probability_program(
            case
        )
    )


    # =========================================================================
    # PROBABILITIES
    # =========================================================================

    print()

    print(
        "Computing Standard Model "
        "and LRI probabilities..."
    )


    probabilities = (
        build_probability_grid(

            executable,

            case,

            coszen_grid,
        )
    )


    # =========================================================================
    # REAL ATMOSPHERIC FLUX RATIO
    # =========================================================================

    print()

    print(
        "Building:"
    )

    print(
        "  r(E,coszen) = "
        "Phi_mu^0 / Phi_e^0"
    )


    r = (
        build_flux_ratio(

            initial_flux,

            probabilities[
                "energy"
            ],

            probabilities[
                "coszen"
            ],
        )
    )


    print(
        f"r minimum = "
        f"{np.nanmin(r):.6f}"
    )

    print(
        f"r maximum = "
        f"{np.nanmax(r):.6f}"
    )


    # =========================================================================
    # EFFECTIVE COMBINATIONS
    # =========================================================================

    combinations = (
        build_effective_combinations(

            probabilities,

            r,
        )
    )


    delta_ce = (
        combinations[
            "delta_ce_percent"
        ]
    )

    delta_cmu = (
        combinations[
            "delta_cmu_percent"
        ]
    )


    print()

    print(
        "Relative differences:"
    )


    print(
        "  electron:"
        f" {np.nanmin(delta_ce):+.4f}%"
        " -> "
        f"{np.nanmax(delta_ce):+.4f}%"
    )


    print(
        "  muon    :"
        f" {np.nanmin(delta_cmu):+.4f}%"
        " -> "
        f"{np.nanmax(delta_cmu):+.4f}%"
    )


    # =========================================================================
    # FIGURE
    # =========================================================================

    make_plot(

        case,

        probabilities,

        combinations,
    )


    print()

    print(
        "Done."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
