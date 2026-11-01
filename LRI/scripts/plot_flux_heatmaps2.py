#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_flux_heatmaps_std_vs_lri.py

4x2 heatmap figure for atmospheric-neutrino fluxes.

Layout for each LRI model:

    row 1: Standard nu_e      | Standard anti-nu_e
    row 2: LRI      nu_e      | LRI      anti-nu_e
    row 3: Standard nu_mu     | Standard anti-nu_mu
    row 4: LRI      nu_mu     | LRI      anti-nu_mu

Notes:
    - No top titles
    - No left-side row/block labels
    - Only y-axis label on the left: cos(theta_z)
    - In-panel labels are white text, no box
    - LRI labels shown in 2 lines
    - One common linear color scale for all 8 panels
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
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
    / "flux2"
)

STANDARD_FILE = (
    ROOT
    / "RESULTS"
    / "standard"
    / "oscillated_flux_std.h5"
)


# =============================================================================
# PHYSICAL POINT
# =============================================================================

LOG10_M = -13.0
LOG10_G = -24.0

COSZEN_MIN = -1.0
COSZEN_MAX = 0.0


# =============================================================================
# MODELS
# =============================================================================

CASES = {

    "Le-Lmu": {

        "label":
            r"$L_e-L_\mu$",

        "file":
            ROOT
            / "RESULTS"
            / "Le-Lmu"
            / "oscillated_flux_lut.h5",
    },

    "Le-Ltau": {

        "label":
            r"$L_e-L_\tau$",

        "file":
            ROOT
            / "RESULTS"
            / "Le-Ltau"
            / "oscillated_flux_lut.h5",
    },

    "Lmu-Ltau": {

        "label":
            r"$L_\mu-L_\tau$",

        "file":
            ROOT
            / "RESULTS"
            / "Lmu-Ltau"
            / "oscillated_flux_lut.h5",
    },
}


# =============================================================================
# SPECIES TO SHOW
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

    (
        "numu",
        r"$\nu_\mu$",
    ),

    (
        "antinumu",
        r"$\bar{\nu}_\mu$",
    ),
)


# =============================================================================
# COLORMAP
# =============================================================================

FLUX_CMAP = "viridis"


# =============================================================================
# NORMALIZATION
# =============================================================================

FLUX_LOG_LOWER_PERCENTILE = 0.2
FLUX_LOG_UPPER_PERCENTILE = 99.8


# =============================================================================
# GRID
# =============================================================================

GRID_MAJOR_ALPHA = 0.14
GRID_MAJOR_LINEWIDTH = 0.45

GRID_MINOR_ALPHA = 0.035
GRID_MINOR_LINEWIDTH = 0.28


# =============================================================================
# PANEL LABEL
# =============================================================================

PANEL_LABEL_FONTSIZE = 9.0


# =============================================================================
# NUMERICAL HELPERS
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


def robust_positive_log_limits(
    arrays,
    lower_percentile=0.2,
    upper_percentile=99.8,
):

    positives = []

    for array in arrays:

        array = np.asarray(
            array,
            dtype=np.float64,
        )

        values = array[
            np.isfinite(array)
            &
            (array > 0.0)
        ]

        if values.size:
            positives.append(
                values.ravel()
            )

    if not positives:
        raise RuntimeError(
            "No hay valores positivos para LogNorm."
        )

    positives = np.concatenate(
        positives
    )

    vmin = np.percentile(
        positives,
        lower_percentile,
    )

    vmax = np.percentile(
        positives,
        upper_percentile,
    )

    vmin = max(
        float(vmin),
        float(np.min(positives)),
    )

    vmax = max(
        float(vmax),
        vmin * 1.000001,
    )

    return vmin, vmax


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
            h5["/axes/energy_GeV"][...],
            dtype=np.float64,
        )

        coszen = np.asarray(
            h5["/axes/coszen"][...],
            dtype=np.float64,
        )

        cz_mask = (
            (coszen >= COSZEN_MIN)
            &
            (coszen <= COSZEN_MAX)
        )

        coszen_plot = coszen[
            cz_mask
        ]

        flux_std = {}

        for (
            species,
            _
        ) in SPECIES:

            phi = np.asarray(
                h5[f"/flux/{species}"][...],
                dtype=np.float64,
            )[
                cz_mask,
                :
            ]

            flux_std[
                species
            ] = phi

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
            h5["/axes/log10_mass_eV"][...],
            dtype=np.float64,
        )

        log10_g = np.asarray(
            h5["/axes/log10_gprime"][...],
            dtype=np.float64,
        )

        energy = np.asarray(
            h5["/axes/energy_GeV"][...],
            dtype=np.float64,
        )

        coszen = np.asarray(
            h5["/axes/coszen"][...],
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
            (coszen >= COSZEN_MIN)
            &
            (coszen <= COSZEN_MAX)
        )

        coszen_plot = coszen[
            cz_mask
        ]

        flux_lri = {}

        for (
            species,
            _
        ) in SPECIES:

            phi = np.asarray(
                h5[f"/flux/{species}"][
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

            flux_lri[
                species
            ] = phi

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


def check_common_axes(
    std_data,
    lri_data,
    case_name,
):

    if not np.allclose(
        std_data["energy_GeV"],
        lri_data["energy_GeV"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"Energy grids do not match for {case_name}."
        )

    if not np.allclose(
        std_data["coszen"],
        lri_data["coszen"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            f"Coszen grids do not match for {case_name}."
        )


# =============================================================================
# STYLE HELPERS
# =============================================================================

def annotate_panel(
    ax,
    label,
):

    ax.text(
        0.955,
        0.905,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        multialignment="center",
        fontsize=PANEL_LABEL_FONTSIZE,
        color="white",
        zorder=40,
    )


def style_heatmap_axis(
    ax,
    energy_edges,
    show_xlabel,
    show_ylabel,
    hide_xlabels,
    hide_ylabels,
):

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        float(energy_edges[0]),
        float(energy_edges[-1]),
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
            r"$E_\nu\,[\mathrm{GeV}]$"
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


# =============================================================================
# FIGURE GEOMETRY
# =============================================================================

def create_figure():

    fig = plt.figure(
        figsize=(10.4, 9.45)
    )

    gs = fig.add_gridspec(
        nrows=7,
        ncols=3,

        width_ratios=[
            1.0,
            1.0,
            0.065,
        ],

        # rows:
        # 0 = std e
        # 1 = gap within e sector
        # 2 = lri e
        # 3 = gap between sectors
        # 4 = std mu
        # 5 = gap within mu sector
        # 6 = lri mu
        height_ratios=[
            1.0,
            0.085,
            1.0,
            0.11,
            1.0,
            0.085,
            1.0,
        ],

        left=0.10,
        right=0.89,
        bottom=0.075,
        top=0.985,

        wspace=0.08,
        hspace=0.0,
    )

    axes = {

        "std_nue":
            fig.add_subplot(gs[0, 0]),

        "std_antinue":
            fig.add_subplot(gs[0, 1]),

        "lri_nue":
            fig.add_subplot(gs[2, 0]),

        "lri_antinue":
            fig.add_subplot(gs[2, 1]),

        "std_numu":
            fig.add_subplot(gs[4, 0]),

        "std_antinumu":
            fig.add_subplot(gs[4, 1]),

        "lri_numu":
            fig.add_subplot(gs[6, 0]),

        "lri_antinumu":
            fig.add_subplot(gs[6, 1]),
    }

    cax = fig.add_subplot(
        gs[:, 2]
    )

    return fig, axes, cax


# =============================================================================
# MAIN PLOT
# =============================================================================

def plot_std_vs_lri(
    case_name,
    case,
    std_data,
    lri_data,
):

    check_common_axes(
        std_data,
        lri_data,
        case_name,
    )

    energy_edges = std_data[
        "energy_edges_GeV"
    ]

    coszen_edges = std_data[
        "coszen_edges"
    ]

    # =========================================================================
    # ONE SHARED LINEAR NORMALIZATION FOR ALL 8 PANELS
    # =========================================================================
    #
    # Important:
    #     This is intentionally LINEAR, not logarithmic.
    #
    #     vmin = 0
    #     vmax = global maximum across all Standard + LRI flux panels.
    #
    # This makes the color scale directly proportional to the physical flux.
    # =========================================================================

    all_flux_arrays = []

    for (
        species,
        _
    ) in SPECIES:

        all_flux_arrays.append(
            std_data["flux_std"][species]
        )

        all_flux_arrays.append(
            lri_data["flux_lri"][species]
        )

    finite_maxima = []

    for array in all_flux_arrays:

        finite = np.asarray(
            array,
            dtype=np.float64,
        )

        finite = finite[
            np.isfinite(
                finite
            )
        ]

        if finite.size > 0:

            finite_maxima.append(
                float(
                    np.max(
                        finite
                    )
                )
            )

    if not finite_maxima:

        raise RuntimeError(
            "No hay valores finitos para construir "
            "la normalización lineal del flujo."
        )

    flux_vmin = 0.0

    flux_vmax = max(
        finite_maxima
    )

    flux_norm = Normalize(
        vmin=flux_vmin,
        vmax=flux_vmax,
        clip=False,
    )

    print()
    print(f"{case_name}:")
    print("  Shared final-flux linear normalization:")
    print(f"    vmin = {flux_vmin:.6e}")
    print(f"    vmax = {flux_vmax:.6e}")

    fig, axes, cax = create_figure()

    # -------------------------------------------------------------------------
    # Panel definitions
    # -------------------------------------------------------------------------

    panel_map = [

        (
            "std_nue",
            std_data["flux_std"]["nue"],
            r"$\nu_e$ (Standard Model)",
        ),

        (
            "std_antinue",
            std_data["flux_std"]["antinue"],
            r"$\bar{\nu}_e$ (Standard Model)",
        ),

        (
            "lri_nue",
            lri_data["flux_lri"]["nue"],
            "$\\nu_e$\n(Long-range interaction)",
        ),

        (
            "lri_antinue",
            lri_data["flux_lri"]["antinue"],
            "$\\bar{\\nu}_e$\n(Long-range interaction)",
        ),

        (
            "std_numu",
            std_data["flux_std"]["numu"],
            r"$\nu_\mu$ (Standard Model)",
        ),

        (
            "std_antinumu",
            std_data["flux_std"]["antinumu"],
            r"$\bar{\nu}_\mu$ (Standard Model)",
        ),

        (
            "lri_numu",
            lri_data["flux_lri"]["numu"],
            "$\\nu_\\mu$\n(Long-range interaction)",
        ),

        (
            "lri_antinumu",
            lri_data["flux_lri"]["antinumu"],
            "$\\bar{\\nu}_\\mu$\n(Long-range interaction)",
        ),
    ]

    meshes = {}

    for (
        key,
        array,
        panel_label,
    ) in panel_map:

        ax = axes[key]

        mesh = ax.pcolormesh(
            energy_edges,
            coszen_edges,
            array,
            shading="auto",
            cmap=FLUX_CMAP,
            norm=flux_norm,
            rasterized=True,
        )

        meshes[key] = mesh

        is_bottom_row = (
            key in (
                "lri_numu",
                "lri_antinumu",
            )
        )

        is_left_column = (
            key in (
                "std_nue",
                "lri_nue",
                "std_numu",
                "lri_numu",
            )
        )

        style_heatmap_axis(
            ax,
            energy_edges,
            show_xlabel=is_bottom_row,
            show_ylabel=is_left_column,
            hide_xlabels=(not is_bottom_row),
            hide_ylabels=(not is_left_column),
        )

        annotate_panel(
            ax,
            panel_label,
        )

    # =========================================================================
    # COMMON COLORBAR
    # =========================================================================

    cbar = fig.colorbar(
        meshes["std_nue"],
        cax=cax,
    )

    cbar.set_label(
        r"Atmospheric neutrino flux at detector, "
        r"$\Phi_{\alpha}$ "
        r"[$\mathrm{GeV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}$]",
        labelpad=10,
        fontsize=14.5,
    )

    cbar.ax.tick_params(
        labelsize=11.5
    )

    cbar.ax.yaxis.set_ticks_position(
        "right"
    )

    cbar.ax.yaxis.set_label_position(
        "right"
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    output_pdf = (
        OUTPUT_DIR
        /
        f"flux_heatmap_std_vs_lri_{case_name}.pdf"
    )

    output_png = (
        OUTPUT_DIR
        /
        f"flux_heatmap_std_vs_lri_{case_name}.png"
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

    plt.close(fig)

    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================================")
    print("Atmospheric neutrino flux heatmaps: Standard vs LRI")
    print("============================================================")

    print(f"log10(m/eV) = {LOG10_M}")
    print(f"log10(g)    = {LOG10_G}")
    print(f"coszen      = [{COSZEN_MIN}, {COSZEN_MAX}]")
    print()

    print(f"Reading Standard file:\n  {STANDARD_FILE}")
    print()

    std_data = load_standard_case(
        STANDARD_FILE
    )

    for (
        case_name,
        case
    ) in CASES.items():

        print("------------------------------------------------------------")
        print(case_name)
        print("------------------------------------------------------------")
        print(f"Reading LRI file:\n  {case['file']}")

        lri_data = load_lri_case(
            case["file"]
        )

        print(
            "Selected point: "
            f"log10(m/eV)={lri_data['actual_log10_m']:.2f}, "
            f"log10(g)={lri_data['actual_log10_g']:.2f}"
        )

        plot_std_vs_lri(
            case_name,
            case,
            std_data,
            lri_data,
        )

        print()

    print("Done.")
    print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

