#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_flux_diagnostics.py

Diagnostic heatmaps for the electron-flavor atmospheric-neutrino flux.

This script is intentionally SEPARATE from the 8-panel absolute-flux figure.

It generates TWO independent figures:

1) log10(Phi_osc / Phi_0)

       Standard nu_e       | Standard anti-nu_e
       LRI      nu_e       | LRI      anti-nu_e

   where Phi_0 is the INITIAL MCEq atmospheric flux used as the
   nuSQuIDS initial condition.

2) log10(Phi_LRI / Phi_SM)

       nu_e | anti-nu_e

Benchmark:
    log10(m'/eV) = -13
    log10(g')    = -24

Important:
    Phi_0 is NOT searched for inside oscillated_flux_std.h5.
    It is read from the separate MCEq initial-flux HDF5 file.
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib.colors import TwoSlopeNorm

import scienceplots


# =============================================================================
# SCIENCEPLOTS / LATEX STYLE
# =============================================================================

plt.style.use("science")

plt.rcParams.update({

    "font.size": 12.5,
    "axes.labelsize": 15.0,

    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,

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

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "flux_diagnostics"
)

STANDARD_FILE = (
    ROOT
    / "RESULTS"
    / "standard"
    / "oscillated_flux_std.h5"
)

LRI_FILE = (
    ROOT
    / "RESULTS"
    / "Le-Lmu"
    / "oscillated_flux_lut.h5"
)


# =============================================================================
# INITIAL MCEq FLUX FILE
# =============================================================================
#
# This is the actual Phi_0 used as the nuSQuIDS initial condition.
#
# Your converter previously used:
#
#     mceq_initial_flux1/mceq_initial_fluxes.h5
#
# The small fallback list below only helps if you later renamed the directory.
# We are NOT searching for different dataset names inside the oscillated files.
# =============================================================================

INITIAL_FILE_CANDIDATES = (

    ROOT
    / "mceq_initial_flux1"
    / "mceq_initial_fluxes.h5",

    ROOT
    / "mceq_initial_flux_direct_angles"
    / "mceq_initial_fluxes.h5",

    ROOT
    / "mceq_initial_flux"
    / "mceq_initial_fluxes.h5",
)


def resolve_initial_file():

    for filename in INITIAL_FILE_CANDIDATES:

        if filename.is_file():

            return filename

    attempted = "\n".join(
        f"  - {path}"
        for path in INITIAL_FILE_CANDIDATES
    )

    raise FileNotFoundError(
        "No encontré el HDF5 de flujo inicial de MCEq.\n"
        "Probé:\n"
        f"{attempted}\n\n"
        "Si está en otra carpeta, cambia INITIAL_FILE_CANDIDATES."
    )


# =============================================================================
# PHYSICAL POINT
# =============================================================================

LOG10_M = -13.0
LOG10_G = -24.0

COSZEN_MIN = -1.0
COSZEN_MAX = 0.0


# =============================================================================
# SPECIES
# =============================================================================

SPECIES = (

    (
        "nue",
        r"$\nu_e$",
    ),

    (
        "antinue",
        r"$\bar{\nu}_e$",
    ),
)


# =============================================================================
# PLOT STYLE
# =============================================================================

RATIO_CMAP = "RdBu_r"

PANEL_LABEL_FONTSIZE = 10.0

GRID_MAJOR_ALPHA = 0.14
GRID_MAJOR_LINEWIDTH = 0.45

GRID_MINOR_ALPHA = 0.035
GRID_MINOR_LINEWIDTH = 0.28


# =============================================================================
# NUMERICAL SETTINGS
# =============================================================================

# Mask ratios where the denominator is essentially zero.
DENOMINATOR_MASK_FRACTION = 1.0e-8

# Robust symmetric color limits.
RATIO_COLOR_PERCENTILE = 99.5


# =============================================================================
# HELPERS
# =============================================================================

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
                axis
                -
                requested
            )
        )
    )

    difference = abs(
        float(
            axis[index]
        )
        -
        float(
            requested
        )
    )

    if difference > tolerance:

        raise ValueError(
            f"{name}={requested} "
            f"no coincide con un nodo exacto. "
            f"El nodo más cercano es {axis[index]}."
        )

    return index


def centers_to_edges_linear(
    centers,
):

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    if (
        centers.ndim != 1
        or
        centers.size < 2
    ):

        raise ValueError(
            "Se necesitan al menos dos centros lineales."
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


def centers_to_edges_log(
    centers,
):

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    if (
        centers.ndim != 1
        or
        centers.size < 2
    ):

        raise ValueError(
            "Se necesitan al menos dos centros logarítmicos."
        )

    if np.any(
        centers <= 0.0
    ):

        raise ValueError(
            "Los centros logarítmicos deben ser positivos."
        )

    logc = np.log10(
        centers
    )

    loge = np.empty(
        logc.size + 1,
        dtype=np.float64,
    )

    loge[1:-1] = (
        0.5
        *
        (
            logc[:-1]
            +
            logc[1:]
        )
    )

    loge[0] = (
        logc[0]
        -
        0.5
        *
        (
            logc[1]
            -
            logc[0]
        )
    )

    loge[-1] = (
        logc[-1]
        +
        0.5
        *
        (
            logc[-1]
            -
            logc[-2]
        )
    )

    return (
        10.0
        **
        loge
    )


def safe_log10_ratio(
    numerator,
    denominator,
):

    numerator = np.asarray(
        numerator,
        dtype=np.float64,
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64,
    )

    result = np.full_like(
        numerator,
        np.nan,
        dtype=np.float64,
    )

    finite_denominator = denominator[
        np.isfinite(
            denominator
        )
        &
        (
            denominator > 0.0
        )
    ]

    if finite_denominator.size == 0:

        return result

    threshold = (
        DENOMINATOR_MASK_FRACTION
        *
        float(
            np.max(
                finite_denominator
            )
        )
    )

    valid = (
        np.isfinite(
            numerator
        )
        &
        np.isfinite(
            denominator
        )
        &
        (
            numerator > 0.0
        )
        &
        (
            denominator > threshold
        )
    )

    result[
        valid
    ] = np.log10(
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return result


def robust_symmetric_limit(
    arrays,
    percentile=99.5,
    minimum=0.02,
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

        raise RuntimeError(
            "No hay valores finitos para definir el colorbar."
        )

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
        minimum,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_standard_case(
    filename,
):

    if not filename.is_file():

        raise FileNotFoundError(
            f"No existe:\n{filename}"
        )

    with h5py.File(
        filename,
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

        cz_mask = (
            (
                coszen
                >=
                COSZEN_MIN
            )
            &
            (
                coszen
                <=
                COSZEN_MAX
            )
        )

        coszen_plot = (
            coszen[
                cz_mask
            ]
        )

        flux_std = {}

        for (
            species,
            _
        ) in SPECIES:

            flux_std[
                species
            ] = np.asarray(
                h5[
                    f"/flux/{species}"
                ][...],
                dtype=np.float64,
            )[
                cz_mask,
                :
            ]

    return {

        "energy_GeV":
            energy,

        "energy_edges_GeV":
            centers_to_edges_log(
                energy
            ),

        "coszen":
            coszen_plot,

        "coszen_edges":
            centers_to_edges_linear(
                coszen_plot
            ),

        "flux_std":
            flux_std,
    }


def load_lri_case(
    filename,
):

    if not filename.is_file():

        raise FileNotFoundError(
            f"No existe:\n{filename}"
        )

    with h5py.File(
        filename,
        "r",
    ) as h5:

        log10_mass = np.asarray(
            h5[
                "/axes/log10_mass_eV"
            ][...],
            dtype=np.float64,
        )

        log10_g = np.asarray(
            h5[
                "/axes/log10_gprime"
            ][...],
            dtype=np.float64,
        )

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

        mass_index = exact_index(
            log10_mass,
            LOG10_M,
            "log10(m/eV)",
        )

        g_index = exact_index(
            log10_g,
            LOG10_G,
            "log10(g)",
        )

        cz_mask = (
            (
                coszen
                >=
                COSZEN_MIN
            )
            &
            (
                coszen
                <=
                COSZEN_MAX
            )
        )

        coszen_plot = (
            coszen[
                cz_mask
            ]
        )

        flux_lri = {}

        for (
            species,
            _
        ) in SPECIES:

            flux_lri[
                species
            ] = np.asarray(
                h5[
                    f"/flux/{species}"
                ][
                    mass_index,
                    g_index,
                    :,
                    :
                ],
                dtype=np.float64,
            )[
                cz_mask,
                :
            ]

    return {

        "energy_GeV":
            energy,

        "energy_edges_GeV":
            centers_to_edges_log(
                energy
            ),

        "coszen":
            coszen_plot,

        "coszen_edges":
            centers_to_edges_linear(
                coszen_plot
            ),

        "flux_lri":
            flux_lri,

        "actual_log10_m":
            float(
                log10_mass[
                    mass_index
                ]
            ),

        "actual_log10_g":
            float(
                log10_g[
                    g_index
                ]
            ),
    }


def load_initial_mceq_flux(
    filename,
):

    if not filename.is_file():

        raise FileNotFoundError(
            f"No existe:\n{filename}"
        )

    with h5py.File(
        filename,
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

        cz_mask = (
            (
                coszen
                >=
                COSZEN_MIN
            )
            &
            (
                coszen
                <=
                COSZEN_MAX
            )
        )

        coszen_plot = (
            coszen[
                cz_mask
            ]
        )

        flux_initial = {}

        for (
            species,
            _
        ) in SPECIES:

            flux_initial[
                species
            ] = np.asarray(
                h5[
                    f"/flux/{species}"
                ][...],
                dtype=np.float64,
            )[
                cz_mask,
                :
            ]

    return {

        "energy_GeV":
            energy,

        "energy_edges_GeV":
            centers_to_edges_log(
                energy
            ),

        "coszen":
            coszen_plot,

        "coszen_edges":
            centers_to_edges_linear(
                coszen_plot
            ),

        "flux_initial":
            flux_initial,
    }


def check_common_axes(
    first,
    second,
    first_name,
    second_name,
):

    if not np.allclose(
        first[
            "energy_GeV"
        ],
        second[
            "energy_GeV"
        ],
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "Energy grids do not match: "
            f"{first_name} vs {second_name}."
        )

    if not np.allclose(
        first[
            "coszen"
        ],
        second[
            "coszen"
        ],
        rtol=0.0,
        atol=1.0e-12,
    ):

        raise RuntimeError(
            "Coszen grids do not match: "
            f"{first_name} vs {second_name}."
        )


# =============================================================================
# STYLE
# =============================================================================

def style_axis(
    ax,
    energy_edges,
    show_xlabel,
    show_ylabel,
    hide_xlabels=False,
    hide_ylabels=False,
):

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        float(
            energy_edges[
                0
            ]
        ),
        float(
            energy_edges[
                -1
            ]
        ),
    )

    ax.set_ylim(
        COSZEN_MIN,
        COSZEN_MAX,
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
            r"Neutrino energy, "
            r"$E_\nu$ [GeV]"
        )

    if show_ylabel:

        ax.set_ylabel(
            r"$\cos\theta_z$"
        )

    if hide_xlabels:

        ax.tick_params(
            labelbottom=False
        )

    if hide_ylabels:

        ax.tick_params(
            labelleft=False
        )


def annotate_panel(
    ax,
    label,
):

    text = ax.text(
        0.955,
        0.905,
        label,

        transform=ax.transAxes,

        ha="right",
        va="top",

        multialignment="center",

        fontsize=PANEL_LABEL_FONTSIZE,

        color="black",

        zorder=40,
    )

    text.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=1.5,
                foreground="white",
            ),
            path_effects.Normal(),
        ]
    )


# =============================================================================
# FIGURE 1
#
# log10(Phi_osc / Phi_0)
# =============================================================================

def plot_log10_over_initial(
    std_data,
    lri_data,
    initial_data,
):

    check_common_axes(
        std_data,
        lri_data,
        "Standard",
        "LRI",
    )

    check_common_axes(
        std_data,
        initial_data,
        "Standard",
        "Initial MCEq",
    )

    energy_edges = std_data[
        "energy_edges_GeV"
    ]

    coszen_edges = std_data[
        "coszen_edges"
    ]


    # =========================================================================
    # RATIOS
    # =========================================================================

    ratios = {

        "std_nue":
            safe_log10_ratio(
                std_data[
                    "flux_std"
                ][
                    "nue"
                ],
                initial_data[
                    "flux_initial"
                ][
                    "nue"
                ],
            ),

        "std_antinue":
            safe_log10_ratio(
                std_data[
                    "flux_std"
                ][
                    "antinue"
                ],
                initial_data[
                    "flux_initial"
                ][
                    "antinue"
                ],
            ),

        "lri_nue":
            safe_log10_ratio(
                lri_data[
                    "flux_lri"
                ][
                    "nue"
                ],
                initial_data[
                    "flux_initial"
                ][
                    "nue"
                ],
            ),

        "lri_antinue":
            safe_log10_ratio(
                lri_data[
                    "flux_lri"
                ][
                    "antinue"
                ],
                initial_data[
                    "flux_initial"
                ][
                    "antinue"
                ],
            ),
    }


    vmax = robust_symmetric_limit(
        list(
            ratios.values()
        ),
        percentile=RATIO_COLOR_PERCENTILE,
        minimum=0.02,
    )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )


    # =========================================================================
    # FIGURE
    # =========================================================================

    fig = plt.figure(
        figsize=(9.2, 6.9)
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,

        width_ratios=[
            1.0,
            1.0,
            0.065,
        ],

        left=0.10,
        right=0.89,
        bottom=0.105,
        top=0.98,

        wspace=0.08,
        hspace=0.10,
    )

    axes = {

        "std_nue":
            fig.add_subplot(
                gs[0, 0]
            ),

        "std_antinue":
            fig.add_subplot(
                gs[0, 1]
            ),

        "lri_nue":
            fig.add_subplot(
                gs[1, 0]
            ),

        "lri_antinue":
            fig.add_subplot(
                gs[1, 1]
            ),
    }

    cax = fig.add_subplot(
        gs[:, 2]
    )


    panel_map = (

        (
            "std_nue",
            ratios[
                "std_nue"
            ],
            r"$\nu_e$"
            "\n(Standard Model)",
        ),

        (
            "std_antinue",
            ratios[
                "std_antinue"
            ],
            r"$\bar{\nu}_e$"
            "\n(Standard Model)",
        ),

        (
            "lri_nue",
            ratios[
                "lri_nue"
            ],
            r"$\nu_e$"
            "\n(Long-range interaction)",
        ),

        (
            "lri_antinue",
            ratios[
                "lri_antinue"
            ],
            r"$\bar{\nu}_e$"
            "\n(Long-range interaction)",
        ),
    )


    first_mesh = None


    for (
        key,
        array,
        panel_label,
    ) in panel_map:

        ax = axes[
            key
        ]

        mesh = ax.pcolormesh(
            energy_edges,
            coszen_edges,
            array,

            shading="auto",

            cmap=RATIO_CMAP,
            norm=norm,

            rasterized=True,
        )

        if first_mesh is None:

            first_mesh = mesh


        is_bottom = (
            key
            in (
                "lri_nue",
                "lri_antinue",
            )
        )

        is_left = (
            key
            in (
                "std_nue",
                "lri_nue",
            )
        )


        style_axis(
            ax,
            energy_edges,

            show_xlabel=is_bottom,
            show_ylabel=is_left,

            hide_xlabels=(
                not is_bottom
            ),

            hide_ylabels=(
                not is_left
            ),
        )


        annotate_panel(
            ax,
            panel_label,
        )


    # =========================================================================
    # COLORBAR
    # =========================================================================

    cbar = fig.colorbar(
        first_mesh,
        cax=cax,
        extend="both",
    )

    cbar.set_label(
        r"$\log_{10}\!\left("
        r"\Phi_{\alpha}^{\rm osc}"
        r"/"
        r"\Phi_{\alpha}^{0}"
        r"\right)$",

        labelpad=10,
        fontsize=14.5,
    )

    cbar.ax.tick_params(
        labelsize=11.5
    )


    # =========================================================================
    # SAVE
    # =========================================================================

    output_pdf = (
        OUTPUT_DIR
        /
        "flux_log10_osc_over_initial_Le-Lmu.pdf"
    )

    output_png = (
        OUTPUT_DIR
        /
        "flux_log10_osc_over_initial_Le-Lmu.png"
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
# FIGURE 2
#
# log10(Phi_LRI / Phi_SM)
# =============================================================================

def plot_log10_lri_over_sm(
    std_data,
    lri_data,
):

    check_common_axes(
        std_data,
        lri_data,
        "Standard",
        "LRI",
    )

    energy_edges = std_data[
        "energy_edges_GeV"
    ]

    coszen_edges = std_data[
        "coszen_edges"
    ]


    ratios = {

        "nue":
            safe_log10_ratio(
                lri_data[
                    "flux_lri"
                ][
                    "nue"
                ],
                std_data[
                    "flux_std"
                ][
                    "nue"
                ],
            ),

        "antinue":
            safe_log10_ratio(
                lri_data[
                    "flux_lri"
                ][
                    "antinue"
                ],
                std_data[
                    "flux_std"
                ][
                    "antinue"
                ],
            ),
    }


    vmax = robust_symmetric_limit(
        list(
            ratios.values()
        ),
        percentile=RATIO_COLOR_PERCENTILE,
        minimum=0.005,
    )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )


    # =========================================================================
    # FIGURE
    # =========================================================================

    fig = plt.figure(
        figsize=(9.2, 3.7)
    )

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,

        width_ratios=[
            1.0,
            1.0,
            0.065,
        ],

        left=0.10,
        right=0.89,
        bottom=0.18,
        top=0.96,

        wspace=0.08,
        hspace=0.0,
    )

    ax_nue = fig.add_subplot(
        gs[0, 0]
    )

    ax_antinue = fig.add_subplot(
        gs[0, 1]
    )

    cax = fig.add_subplot(
        gs[0, 2]
    )


    mesh = ax_nue.pcolormesh(
        energy_edges,
        coszen_edges,
        ratios[
            "nue"
        ],

        shading="auto",

        cmap=RATIO_CMAP,
        norm=norm,

        rasterized=True,
    )


    ax_antinue.pcolormesh(
        energy_edges,
        coszen_edges,
        ratios[
            "antinue"
        ],

        shading="auto",

        cmap=RATIO_CMAP,
        norm=norm,

        rasterized=True,
    )


    style_axis(
        ax_nue,
        energy_edges,

        show_xlabel=True,
        show_ylabel=True,
    )


    style_axis(
        ax_antinue,
        energy_edges,

        show_xlabel=True,
        show_ylabel=False,

        hide_ylabels=True,
    )


    annotate_panel(
        ax_nue,
        r"$\nu_e$",
    )

    annotate_panel(
        ax_antinue,
        r"$\bar{\nu}_e$",
    )


    # =========================================================================
    # COLORBAR
    # =========================================================================

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        extend="both",
    )

    cbar.set_label(
        r"$\log_{10}\!\left("
        r"\Phi_{\alpha}^{\rm LRI}"
        r"/"
        r"\Phi_{\alpha}^{\rm SM}"
        r"\right)$",

        labelpad=10,
        fontsize=14.5,
    )

    cbar.ax.tick_params(
        labelsize=11.5
    )


    # =========================================================================
    # SAVE
    # =========================================================================

    output_pdf = (
        OUTPUT_DIR
        /
        "flux_log10_lri_over_sm_Le-Lmu.pdf"
    )

    output_png = (
        OUTPUT_DIR
        /
        "flux_log10_lri_over_sm_Le-Lmu.png"
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

    initial_file = resolve_initial_file()


    print(
        "============================================================"
    )

    print(
        "Flux diagnostic plots"
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
        f"coszen      = "
        f"[{COSZEN_MIN}, {COSZEN_MAX}]"
    )

    print()


    print(
        "Reading initial MCEq flux:"
    )

    print(
        f"  {initial_file}"
    )

    initial_data = load_initial_mceq_flux(
        initial_file
    )


    print()

    print(
        "Reading Standard Model flux:"
    )

    print(
        f"  {STANDARD_FILE}"
    )

    std_data = load_standard_case(
        STANDARD_FILE
    )


    print()

    print(
        "Reading LRI flux:"
    )

    print(
        f"  {LRI_FILE}"
    )

    lri_data = load_lri_case(
        LRI_FILE
    )


    print()

    print(
        "Selected point: "
        f"log10(m/eV)="
        f"{lri_data['actual_log10_m']:.2f}, "
        f"log10(g)="
        f"{lri_data['actual_log10_g']:.2f}"
    )

    print()


    # =========================================================================
    # FIGURE 1:
    # log10(Phi_osc / Phi_0)
    # =========================================================================

    plot_log10_over_initial(
        std_data,
        lri_data,
        initial_data,
    )


    print()


    # =========================================================================
    # FIGURE 2:
    # log10(Phi_LRI / Phi_SM)
    # =========================================================================

    plot_log10_lri_over_sm(
        std_data,
        lri_data,
    )


    print()

    print(
        "Done."
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()

