#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shlex
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scienceplots


# =============================================================================
# SCIENCEPLOTS / LATEX STYLE
# =============================================================================

plt.style.use("science")

plt.rcParams.update({

    # -------------------------------------------------------------------------
    # Typography
    # -------------------------------------------------------------------------

    "font.size": 12.5,

    "axes.labelsize": 15.0,

    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,

    "legend.fontsize": 8.7,

    # -------------------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------------------

    "axes.linewidth": 0.9,

    # -------------------------------------------------------------------------
    # Ticks
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # PDF
    # -------------------------------------------------------------------------

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
    / "probability2"
)

DATA_DIR = (
    OUTPUT_DIR
    / "data"
)


# =============================================================================
# PHYSICAL CONFIGURATION
# =============================================================================

LOG10_M = -13.0
LOG10_G = -25.0

COSZENS = (
    -1.0,
    -0.4,
)

N_ENERGY = 240

E_MIN_GEV = 5.0
E_MAX_GEV = 200.0


# =============================================================================
# INITIAL FLAVORS
# =============================================================================

INITIAL_FLAVORS_TO_PLOT = (
    "mu",
)


# =============================================================================
# MODELS
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
# FLAVOR LABELS
# =============================================================================

FLAVOR_LATEX = {
    "e": r"e",
    "mu": r"\mu",
    "tau": r"\tau",
}


# Because the y-axis is
#
#     P(nu_alpha -> nu_beta)
#
# beta itself is:
#
#     e, mu, tau
#
FINAL_FLAVORS = (
    ("e",   r"$e$"),
    ("mu",  r"$\mu$"),
    ("tau", r"$\tau$"),
)


# =============================================================================
# CURVE STYLE
# =============================================================================

SM_LINESTYLE = "--"
LRI_LINESTYLE = "-"

SM_LINEWIDTH = 1.85
LRI_LINEWIDTH = 2.15

SM_ALPHA = 0.72


# =============================================================================
# GRID
# =============================================================================

# Slightly more visible than the previous version.
GRID_MAJOR_ALPHA = 0.23
GRID_MAJOR_LINEWIDTH = 0.55

GRID_MINOR_ALPHA = 0.08
GRID_MINOR_LINEWIDTH = 0.35


# =============================================================================
# LEGENDS
# =============================================================================

LEGEND_FONTSIZE = 8.7

# Flavor legend in TOP panel.
FLAVOR_LEGEND_X = 0.975
FLAVOR_LEGEND_Y = 0.590

# Model legend in BOTTOM panel.
MODEL_LEGEND_X = 0.975
MODEL_LEGEND_Y = 0.590

LEGEND_BOX_LINEWIDTH = 0.70
LEGEND_BOX_ALPHA = 0.94


# =============================================================================
# PARAMETER / COSZEN BOX
# =============================================================================

# Same fontsize as legend.
ANNOTATION_FONTSIZE = LEGEND_FONTSIZE

ANNOTATION_BOX_LINEWIDTH = 0.70
ANNOTATION_BOX_ALPHA = 0.94

# One common box:
#
# cos(theta_z) = ...
# m' = ...
# g' = ...
#
ANNOTATION_X = 0.955
ANNOTATION_Y = 0.955


# =============================================================================
# EXACT PHYSICAL PANEL SIZE
# =============================================================================

PANEL_SIZE_IN = 3.35


# =============================================================================
# VERTICAL 2x1 GEOMETRY
# =============================================================================

VERTICAL_FIG_WIDTH_IN = 4.80
VERTICAL_FIG_HEIGHT_IN = 7.60

VERTICAL_LEFT_MARGIN_IN = 0.93
VERTICAL_BOTTOM_MARGIN_IN = 0.58

VERTICAL_PANEL_GAP_IN = 0.16


# =============================================================================
# HORIZONTAL 1x2 GEOMETRY
# =============================================================================

HORIZONTAL_FIG_WIDTH_IN = 8.35
HORIZONTAL_FIG_HEIGHT_IN = 4.55

HORIZONTAL_LEFT_MARGIN_IN = 0.82
HORIZONTAL_BOTTOM_MARGIN_IN = 0.68

HORIZONTAL_PANEL_GAP_IN = 0.42


# =============================================================================
# BUILD UTILITIES
# =============================================================================

def pkg_config(flag):

    text = subprocess.check_output(
        [
            "pkg-config",
            flag,
            "nusquids",
        ],
        text=True,
    ).strip()

    return shlex.split(
        text
    )


def compile_helper(
    case_name,
    case,
):

    if not CPP_SOURCE.is_file():

        raise FileNotFoundError(
            f"Missing C++ helper: "
            f"{CPP_SOURCE}"
        )

    include_dir = (
        case[
            "include_dir"
        ]
    )

    lri_header = (
        include_dir
        / "LRI.h"
    )

    if not lri_header.is_file():

        raise FileNotFoundError(
            f"Missing {case_name} "
            f"LRI.h: {lri_header}"
        )

    if not case[
        "lut"
    ].is_file():

        raise FileNotFoundError(
            f"Missing {case_name} "
            f"LUT: {case['lut']}"
        )

    BUILD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    executable = (
        BUILD_DIR
        / f"transition_probabilities_{case_name}"
    )

    cflags = pkg_config(
        "--cflags"
    )

    libs = pkg_config(
        "--libs"
    )

    command = (
        [
            "g++",
            "-O3",
            "-std=c++17",

            str(
                CPP_SOURCE
            ),

            f"-I{include_dir}",

            f"-I{ROOT / 'LUT_FLUX'}",
        ]
        +
        cflags
        +
        libs
        +
        [
            "-lhdf5_hl",

            "-o",

            str(
                executable
            ),
        ]
    )

    print()

    print(
        f"Compiling {case_name}:"
    )

    print(
        " ".join(
            shlex.quote(x)
            for x in command
        )
    )

    subprocess.run(
        command,
        check=True,
    )

    return executable


# =============================================================================
# RUN UTILITIES
# =============================================================================

def coszen_tag(
    coszen,
):

    return (
        f"{coszen:+.2f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )


def run_case(
    case_name,
    case,
    executable,
    coszen,
):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        DATA_DIR
        /
        (
            f"probabilities_"
            f"{case_name}"
            f"_cz_"
            f"{coszen_tag(coszen)}"
            f".csv"
        )
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

        str(
            coszen
        ),

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

    print()

    print(
        f"Running {case_name}, "
        f"coszen={coszen}:"
    )

    print(
        " ".join(
            shlex.quote(x)
            for x in command
        )
    )

    subprocess.run(
        command,
        check=True,
    )

    return output_csv


# =============================================================================
# DATA COLUMN
# =============================================================================

def probability_column(
    prefix,
    alpha,
    beta,
):

    return (
        f"{prefix}_P"
        f"{alpha}"
        f"{beta}"
    )


# =============================================================================
# PARAMETER LABELS
# =============================================================================

def format_log10_exponent(
    value,
):

    nearest = round(
        value
    )

    if abs(
        value
        -
        nearest
    ) < 1.0e-10:

        return str(
            int(
                nearest
            )
        )

    return (
        f"{value:.2f}"
    )


def mass_label():

    exponent = (
        format_log10_exponent(
            LOG10_M
        )
    )

    return (
        rf"$m'=10^{{{exponent}}}"
        rf"\,\mathrm{{eV}}$"
    )


def coupling_label():

    exponent = (
        format_log10_exponent(
            LOG10_G
        )
    )

    return (
        rf"$g'=10^{{{exponent}}}$"
    )


# =============================================================================
# COMBINED ANNOTATION
# =============================================================================

def combined_annotation_text(
    coszen,
):

    return (
        rf"$\cos\theta_z={coszen:g}$"
        "\n"
        +
        mass_label()
        +
        "\n"
        +
        coupling_label()
    )


def annotate_parameters(
    ax,
    coszen,
):

    ax.text(
        ANNOTATION_X,
        ANNOTATION_Y,

        combined_annotation_text(
            coszen
        ),

        transform=(
            ax.transAxes
        ),

        ha="right",
        va="top",

        fontsize=(
            ANNOTATION_FONTSIZE
        ),

        linespacing=1.05,

        zorder=20,

        bbox={
            "boxstyle":
                "square,pad=0.16",

            "facecolor":
                "white",

            "edgecolor":
                "black",

            "linewidth":
                ANNOTATION_BOX_LINEWIDTH,

            "alpha":
                ANNOTATION_BOX_ALPHA,
        },
    )


# =============================================================================
# 2x1 VERTICAL FIGURE
# =============================================================================

def create_vertical_square_panel_figure():

    fig = plt.figure(
        figsize=(
            VERTICAL_FIG_WIDTH_IN,
            VERTICAL_FIG_HEIGHT_IN,
        )
    )

    panel_width_norm = (
        PANEL_SIZE_IN
        /
        VERTICAL_FIG_WIDTH_IN
    )

    panel_height_norm = (
        PANEL_SIZE_IN
        /
        VERTICAL_FIG_HEIGHT_IN
    )

    left_norm = (
        VERTICAL_LEFT_MARGIN_IN
        /
        VERTICAL_FIG_WIDTH_IN
    )

    bottom_norm = (
        VERTICAL_BOTTOM_MARGIN_IN
        /
        VERTICAL_FIG_HEIGHT_IN
    )

    gap_norm = (
        VERTICAL_PANEL_GAP_IN
        /
        VERTICAL_FIG_HEIGHT_IN
    )

    # -------------------------------------------------------------------------
    # Bottom panel
    # -------------------------------------------------------------------------

    ax_bottom = fig.add_axes(
        [
            left_norm,
            bottom_norm,
            panel_width_norm,
            panel_height_norm,
        ]
    )

    # -------------------------------------------------------------------------
    # Top panel
    # -------------------------------------------------------------------------

    top_bottom_norm = (
        bottom_norm
        +
        panel_height_norm
        +
        gap_norm
    )

    ax_top = fig.add_axes(
        [
            left_norm,
            top_bottom_norm,
            panel_width_norm,
            panel_height_norm,
        ]
    )

    return (
        fig,
        [
            ax_top,
            ax_bottom,
        ],
    )


# =============================================================================
# 1x2 HORIZONTAL FIGURE
# =============================================================================

def create_horizontal_square_panel_figure():

    fig = plt.figure(
        figsize=(
            HORIZONTAL_FIG_WIDTH_IN,
            HORIZONTAL_FIG_HEIGHT_IN,
        )
    )

    panel_width_norm = (
        PANEL_SIZE_IN
        /
        HORIZONTAL_FIG_WIDTH_IN
    )

    panel_height_norm = (
        PANEL_SIZE_IN
        /
        HORIZONTAL_FIG_HEIGHT_IN
    )

    left_norm = (
        HORIZONTAL_LEFT_MARGIN_IN
        /
        HORIZONTAL_FIG_WIDTH_IN
    )

    bottom_norm = (
        HORIZONTAL_BOTTOM_MARGIN_IN
        /
        HORIZONTAL_FIG_HEIGHT_IN
    )

    gap_norm = (
        HORIZONTAL_PANEL_GAP_IN
        /
        HORIZONTAL_FIG_WIDTH_IN
    )

    ax_left = fig.add_axes(
        [
            left_norm,
            bottom_norm,
            panel_width_norm,
            panel_height_norm,
        ]
    )

    right_left_norm = (
        left_norm
        +
        panel_width_norm
        +
        gap_norm
    )

    ax_right = fig.add_axes(
        [
            right_left_norm,
            bottom_norm,
            panel_width_norm,
            panel_height_norm,
        ]
    )

    return (
        fig,
        [
            ax_left,
            ax_right,
        ],
    )


# =============================================================================
# AXIS STYLE
# =============================================================================

def style_probability_axis(
    ax,
    show_xlabel=True,
    show_xticklabels=True,
):

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        E_MIN_GEV,
        E_MAX_GEV,
    )

    # Probability stops exactly at 1.
    ax.set_ylim(
        0.0,
        1.0,
    )

    if show_xlabel:

        ax.set_xlabel(
            r"$E_\nu\,[\mathrm{GeV}]$"
        )

    if not show_xticklabels:

        ax.tick_params(
            labelbottom=False
        )

    ax.minorticks_on()

    ax.grid(
        True,
        which="major",

        alpha=(
            GRID_MAJOR_ALPHA
        ),

        linewidth=(
            GRID_MAJOR_LINEWIDTH
        ),
    )

    ax.grid(
        True,
        which="minor",

        alpha=(
            GRID_MINOR_ALPHA
        ),

        linewidth=(
            GRID_MINOR_LINEWIDTH
        ),
    )


# =============================================================================
# VERIFY PHYSICAL SIZE
# =============================================================================

def report_axes_size(
    fig,
    axes,
):

    fig.canvas.draw()

    fig_width = (
        fig.get_figwidth()
    )

    fig_height = (
        fig.get_figheight()
    )

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        "Actual axes physical dimensions"
    )

    print(
        "------------------------------------------------------------"
    )

    for i, ax in enumerate(
        axes
    ):

        bbox = (
            ax.get_position()
        )

        width_inches = (
            bbox.width
            *
            fig_width
        )

        height_inches = (
            bbox.height
            *
            fig_height
        )

        ratio = (
            width_inches
            /
            height_inches
        )

        print(
            f"Panel {i}: "
            f"{width_inches:.4f} x "
            f"{height_inches:.4f} in"
        )

        print(
            f"         aspect ratio = "
            f"{ratio:.8f}"
        )


# =============================================================================
# INDIVIDUAL MODEL — 2x1
# =============================================================================

def plot_one_initial_flavor(
    case_name,
    case,
    csv_by_coszen,
    alpha,
):

    fig, axes = (
        create_vertical_square_panel_figure()
    )

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    color_by_beta = {

        beta:
            default_colors[i]

        for i, (
            beta,
            _
        ) in enumerate(
            FINAL_FLAVORS
        )
    }


    # =========================================================================
    # CURVES
    # =========================================================================

    for panel_index, (
        ax,
        coszen,
    ) in enumerate(
        zip(
            axes,
            COSZENS,
        )
    ):

        df = pd.read_csv(
            csv_by_coszen[
                coszen
            ]
        )

        energy = (
            df[
                "energy_GeV"
            ]
            .to_numpy()
        )

        for (
            beta,
            _
        ) in FINAL_FLAVORS:

            col_sm = (
                probability_column(
                    "sm",
                    alpha,
                    beta,
                )
            )

            col_lri = (
                probability_column(
                    "lri",
                    alpha,
                    beta,
                )
            )

            # -----------------------------------------------------------------
            # Standard Model
            # -----------------------------------------------------------------

            ax.plot(
                energy,

                df[
                    col_sm
                ],

                linestyle=(
                    SM_LINESTYLE
                ),

                linewidth=(
                    SM_LINEWIDTH
                ),

                alpha=(
                    SM_ALPHA
                ),

                color=(
                    color_by_beta[
                        beta
                    ]
                ),
            )

            # -----------------------------------------------------------------
            # Long-range Interaction
            # -----------------------------------------------------------------

            ax.plot(
                energy,

                df[
                    col_lri
                ],

                linestyle=(
                    LRI_LINESTYLE
                ),

                linewidth=(
                    LRI_LINEWIDTH
                ),

                color=(
                    color_by_beta[
                        beta
                    ]
                ),
            )


        # ---------------------------------------------------------------------
        # Axes
        # ---------------------------------------------------------------------

        if panel_index == 0:

            style_probability_axis(
                ax,

                show_xlabel=False,

                show_xticklabels=False,
            )

        else:

            style_probability_axis(
                ax,

                show_xlabel=True,

                show_xticklabels=True,
            )


        # ---------------------------------------------------------------------
        # Combined cos(theta_z), m', g' box
        # ---------------------------------------------------------------------

        annotate_parameters(
            ax,
            coszen,
        )


    # =========================================================================
    # SHARED EXTERNAL Y LABEL
    # =========================================================================

    alpha_latex = (
        FLAVOR_LATEX[
            alpha
        ]
    )

    fig.text(
        0.065,
        0.50,

        rf"$P("
        rf"\nu_{{{alpha_latex}}}"
        rf"\rightarrow"
        rf"\nu_\beta"
        rf")$",

        rotation="vertical",

        va="center",
        ha="center",

        fontsize=15.0,
    )


    # =========================================================================
    # FLAVOR LEGEND
    #
    # Goes ONLY in the TOP panel:
    #
    # e
    # mu
    # tau
    # =========================================================================

    flavor_handles = [

        Line2D(
            [0],
            [0],

            linewidth=(
                LRI_LINEWIDTH
            ),

            color=(
                color_by_beta[
                    beta
                ]
            ),

            label=(
                beta_label
            ),
        )

        for (
            beta,
            beta_label
        ) in FINAL_FLAVORS
    ]


    flavor_legend = axes[
        0
    ].legend(

        handles=(
            flavor_handles
        ),

        loc="upper right",

        bbox_to_anchor=(
            FLAVOR_LEGEND_X,
            FLAVOR_LEGEND_Y,
        ),

        # One flavor per row:
        ncol=1,

        frameon=True,

        facecolor="white",

        framealpha=(
            LEGEND_BOX_ALPHA
        ),

        edgecolor="black",

        fontsize=(
            LEGEND_FONTSIZE
        ),

        borderpad=0.30,

        labelspacing=0.24,

        handlelength=1.55,

        handletextpad=0.42,

        borderaxespad=0.0,
    )

    flavor_legend.get_frame().set_linewidth(
        LEGEND_BOX_LINEWIDTH
    )


    # =========================================================================
    # MODEL LEGEND
    #
    # Goes ONLY in the BOTTOM panel:
    #
    # -- Standard Model
    #
    # -  Long-range
    #    Interaction
    #    (Le-Lmu)
    # =========================================================================

    model_handles = [

        Line2D(
            [0],
            [0],

            linewidth=(
                SM_LINEWIDTH
            ),

            linestyle=(
                SM_LINESTYLE
            ),

            color="black",

            label=(
                "Standard Model"
            ),
        ),

        Line2D(
            [0],
            [0],

            linewidth=(
                LRI_LINEWIDTH
            ),

            linestyle=(
                LRI_LINESTYLE
            ),

            color="black",

            label=(
                "Long-range\n"
                "Interaction\n"
                f"({case['label']})"
            ),
        ),

    ]


    model_legend = axes[
        1
    ].legend(

        handles=(
            model_handles
        ),

        loc="upper right",

        bbox_to_anchor=(
            MODEL_LEGEND_X,
            MODEL_LEGEND_Y,
        ),

        frameon=True,

        facecolor="white",

        framealpha=(
            LEGEND_BOX_ALPHA
        ),

        edgecolor="black",

        fontsize=(
            LEGEND_FONTSIZE
        ),

        borderpad=0.30,

        labelspacing=0.24,

        handlelength=1.85,

        handletextpad=0.48,

        borderaxespad=0.0,
    )

    model_legend.get_frame().set_linewidth(
        LEGEND_BOX_LINEWIDTH
    )


    # =========================================================================
    # VERIFY
    # =========================================================================

    report_axes_size(
        fig,
        axes,
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    output_pdf = (
        OUTPUT_DIR
        /
        (
            f"probability_"
            f"{case_name}"
            f"_initial_"
            f"{alpha}"
            f"_2x1_1x1.pdf"
        )
    )

    output_png = (
        OUTPUT_DIR
        /
        (
            f"probability_"
            f"{case_name}"
            f"_initial_"
            f"{alpha}"
            f"_2x1_1x1.png"
        )
    )

    fig.savefig(
        output_pdf,

        bbox_inches="tight",

        pad_inches=0.035,
    )

    fig.savefig(
        output_png,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.035,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_pdf}"
    )

    print(
        f"Saved: {output_png}"
    )


# =============================================================================
# P_mu_mu — ALL MODELS
# =============================================================================

def plot_pmumu_comparison(
    all_csv,
):

    fig, axes = (
        create_horizontal_square_panel_figure()
    )

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()[
            "color"
        ]
    )

    color_by_case = {

        case_name:
            default_colors[i]

        for i, case_name in enumerate(
            CASES
        )
    }


    # =========================================================================
    # PANELS
    # =========================================================================

    for ax, coszen in zip(
        axes,
        COSZENS,
    ):

        first_case = next(
            iter(
                CASES
            )
        )

        standard_df = pd.read_csv(
            all_csv[
                first_case
            ][
                coszen
            ]
        )


        # ---------------------------------------------------------------------
        # Standard Model
        # ---------------------------------------------------------------------

        ax.plot(
            standard_df[
                "energy_GeV"
            ],

            standard_df[
                "sm_Pmumu"
            ],

            linestyle=(
                SM_LINESTYLE
            ),

            linewidth=(
                SM_LINEWIDTH
            ),

            color="black",

            alpha=0.80,

            label=(
                "Standard Model"
            ),
        )


        # ---------------------------------------------------------------------
        # Long-range Interaction models
        # ---------------------------------------------------------------------

        for (
            case_name,
            case
        ) in CASES.items():

            df = pd.read_csv(
                all_csv[
                    case_name
                ][
                    coszen
                ]
            )

            ax.plot(
                df[
                    "energy_GeV"
                ],

                df[
                    "lri_Pmumu"
                ],

                linewidth=(
                    LRI_LINEWIDTH
                ),

                color=(
                    color_by_case[
                        case_name
                    ]
                ),

                label=(
                    "Long-range Interaction "
                    f"({case['label']})"
                ),
            )


        style_probability_axis(
            ax,

            show_xlabel=True,

            show_xticklabels=True,
        )


        # Combined box also for P_mu_mu comparison.
        annotate_parameters(
            ax,
            coszen,
        )


    # =========================================================================
    # SHARED Y LABEL
    # =========================================================================

    fig.text(
        0.052,
        0.50,

        r"$P(\nu_\mu\rightarrow\nu_\mu)$",

        rotation="vertical",

        va="center",
        ha="center",

        fontsize=15.0,
    )


    axes[
        1
    ].tick_params(
        labelleft=False
    )


    # =========================================================================
    # LEGEND
    # =========================================================================

    legend = axes[
        1
    ].legend(

        loc="center right",

        frameon=True,

        facecolor="white",

        framealpha=(
            LEGEND_BOX_ALPHA
        ),

        edgecolor="black",

        fontsize=(
            LEGEND_FONTSIZE
        ),

        borderpad=0.32,

        labelspacing=0.24,

        handlelength=1.85,

        handletextpad=0.48,
    )


    legend.get_frame().set_linewidth(
        LEGEND_BOX_LINEWIDTH
    )


    # =========================================================================
    # VERIFY
    # =========================================================================

    report_axes_size(
        fig,
        axes,
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    output_pdf = (
        OUTPUT_DIR
        / "probability_Pmumu_all_models_1x1.pdf"
    )

    output_png = (
        OUTPUT_DIR
        / "probability_Pmumu_all_models_1x1.png"
    )

    fig.savefig(
        output_pdf,

        bbox_inches="tight",

        pad_inches=0.035,
    )

    fig.savefig(
        output_png,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.035,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_pdf}"
    )

    print(
        f"Saved: {output_png}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "============================================================"
    )

    print(
        "Flavor-transition probability pipeline"
    )

    print(
        "============================================================"
    )

    print(
        f"log10(m/eV) = {LOG10_M}"
    )

    print(
        f"log10(g)    = {LOG10_G}"
    )

    print(
        f"coszen      = {COSZENS}"
    )

    print(
        f"E [GeV]     = "
        f"{E_MIN_GEV} -> {E_MAX_GEV}"
    )

    print()

    print(
        "Non-coherent interactions are disabled, "
        "so these are genuine oscillation probabilities."
    )


    all_csv = {}


    # =========================================================================
    # ALL MODELS
    # =========================================================================

    for (
        case_name,
        case
    ) in CASES.items():

        executable = (
            compile_helper(
                case_name,
                case,
            )
        )

        csv_by_coszen = {}

        for coszen in COSZENS:

            csv_by_coszen[
                coszen
            ] = run_case(
                case_name,
                case,
                executable,
                coszen,
            )

        all_csv[
            case_name
        ] = csv_by_coszen


        # ---------------------------------------------------------------------
        # Individual-model plot
        # ---------------------------------------------------------------------

        for alpha in INITIAL_FLAVORS_TO_PLOT:

            plot_one_initial_flavor(
                case_name,
                case,
                csv_by_coszen,
                alpha,
            )


    # =========================================================================
    # P_mu_mu ALL-MODEL COMPARISON
    # =========================================================================

    plot_pmumu_comparison(
        all_csv
    )


    print()

    print(
        "Done."
    )

    print(
        f"Plots: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"CSV data: "
        f"{DATA_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
