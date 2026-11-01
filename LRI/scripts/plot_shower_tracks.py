#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv

import numpy as np
import h5py
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, TwoSlopeNorm

import scienceplots


# =============================================================================
# SCIENCEPLOTS STYLE
# =============================================================================

plt.style.use("science")

plt.rcParams.update({

    # -------------------------------------------------------------------------
    # Typography
    # -------------------------------------------------------------------------

    "font.size": 13.5,

    "axes.labelsize": 15.0,
    "axes.titlesize": 15.0,

    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,

    "legend.fontsize": 12.0,

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

    "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8,

    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,

    "xtick.minor.width": 0.65,
    "ytick.minor.width": 0.65,

    # -------------------------------------------------------------------------
    # Vector output
    # -------------------------------------------------------------------------

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "detector"
)


CASE_LABELS = {

    "Le-Lmu":
        r"$L_e-L_\mu$",

    "Le-Ltau":
        r"$L_e-L_\tau$",

    "Lmu-Ltau":
        r"$L_\mu-L_\tau$",
}


CASES = {

    "Le-Lmu": {

        "file":
            ROOT
            / "RESULTS"
            / "Le-Lmu"
            / "event_templates_effective.h5",

        "log10_m":
            -13.0,

        "log10_g":
            -24.0,
    },

    "Le-Ltau": {

        "file":
            ROOT
            / "RESULTS"
            / "Le-Ltau"
            / "event_templates_effective.h5",

        "log10_m":
            -13.0,

        "log10_g":
            -24.0,
    },

    "Lmu-Ltau": {

        "file":
            ROOT
            / "RESULTS"
            / "Lmu-Ltau"
            / "event_templates_effective.h5",

        "log10_m":
            -13.0,

        "log10_g":
            -24.0,
    },
}


CATEGORIES = (
    "nu_nc",
    "nue_cc",
    "numu_cc",
    "nutau_cc",
)


PID_LABELS = (
    "Mixed",
    "Tracks",
)


APPLY_NOMINAL_DETECTOR_HYPERSURFACE = True

INCLUDE_ATMOSPHERIC_MUON = False


RELATIVE_MASK_FRACTION_OF_MAX = (
    1.0e-5
)

DELTA_PERCENTILE = (
    99.0
)


# =============================================================================
# VISUAL STYLE
# =============================================================================

GRID_ALPHA = 0.18

GRID_LINEWIDTH = 0.45


COLUMN_TITLE_FONTSIZE = 15.0

COLORBAR_LABELSIZE = 14.5


HIST_COLOR = "#8db5cc"

HIST_EDGE_COLOR = "black"

HIST_ALPHA = 0.85

HIST_LINEWIDTH = 0.90


# =============================================================================
# GROUPED FIGURE GEOMETRY
# =============================================================================

GROUPED_FIG_WIDTH = 17.8

GROUPED_FIG_HEIGHT = 8.6


# Width of the colorbars
COUNT_CBAR_WIDTH = 0.055

DELTA_CBAR_WIDTH = 0.055


# -----------------------------------------------------------------------------
# Space between event-count colorbar and Relative deviation column.
#
# Previous version:
#
#     0.34
#
# Reduced here because the third column was too far away.
# -----------------------------------------------------------------------------

COUNT_CBAR_TO_RELATIVE_GAP = 0.15


# Internal gap between main heatmap and side histogram
PANEL_SIDE_GAP = 0.035

# Internal gap between top histogram and heatmap
PANEL_TOP_GAP = 0.035


# =============================================================================
# UTILITIES
# =============================================================================

def exact_index(
    axis: np.ndarray,
    value: float,
    name: str,
    tol: float = 1.0e-10,
) -> int:

    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    idx = int(
        np.argmin(
            np.abs(
                axis
                -
                value
            )
        )
    )

    if (
        abs(
            float(
                axis[idx]
            )
            -
            float(
                value
            )
        )
        >
        tol
    ):

        raise ValueError(
            f"{name}={value} "
            f"no coincide exactamente "
            f"con un nodo.\n"
            f"Más cercano: "
            f"{axis[idx]}"
        )

    return idx


# =============================================================================
# ROBUST DELTA SCALE
# =============================================================================

def robust_abs_max(
    arrays,
    percentile=99.0,
):

    vals = []

    for arr in arrays:

        finite = arr[
            np.isfinite(
                arr
            )
        ]

        if finite.size > 0:

            vals.append(
                np.abs(
                    finite
                ).ravel()
            )

    if not vals:

        return 1.0

    vals = np.concatenate(
        vals
    )

    vmax = np.percentile(
        vals,
        percentile,
    )

    return max(
        float(
            vmax
        ),
        1.0e-12,
    )


# =============================================================================
# DYNAMIC HISTOGRAM LIMITS
# =============================================================================

def dynamic_axis_limits(
    values,
    pad_fraction=0.10,
):
    """
    Automatic limits for marginal histograms.

    Each histogram gets its own range instead of being forced
    to start at zero.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return (
            0.0,
            1.0,
        )

    vmin = float(
        np.min(
            values
        )
    )

    vmax = float(
        np.max(
            values
        )
    )

    span = (
        vmax
        -
        vmin
    )

    if span <= 0.0:

        reference = max(
            abs(
                vmax
            ),
            1.0,
        )

        padding = (
            pad_fraction
            *
            reference
        )

    else:

        padding = (
            pad_fraction
            *
            span
        )

    lower = (
        vmin
        -
        padding
    )

    upper = (
        vmax
        +
        padding
    )

    # Avoid a tiny negative lower edge when the actual minimum
    # is essentially zero.
    if (
        vmin >= 0.0
        and
        abs(
            lower
        )
        <
        0.02
        *
        max(
            abs(
                upper
            ),
            1.0,
        )
    ):

        lower = 0.0

    if upper <= lower:

        upper = (
            lower
            +
            1.0
        )

    return (
        lower,
        upper,
    )


# =============================================================================
# PREDICTION
# =============================================================================

def build_prediction(
    h5: h5py.File,
    im: int,
    ig: int,
) -> np.ndarray:
    """
    Detector-level histogram:

        (nE_reco, nCoszen_reco, nPID)
    """

    first = np.asarray(
        h5[
            f"templates/"
            f"{CATEGORIES[0]}"
        ][
            im,
            ig,
            :
        ],
        dtype=np.float64,
    )

    total = np.zeros_like(
        first,
        dtype=np.float64,
    )

    for cat in CATEGORIES:

        arr = np.asarray(
            h5[
                f"templates/{cat}"
            ][
                im,
                ig,
                :
            ],
            dtype=np.float64,
        )

        if (
            APPLY_NOMINAL_DETECTOR_HYPERSURFACE
        ):

            intercept = np.asarray(
                h5[
                    f"hypersurfaces/"
                    f"{cat}/"
                    f"intercept"
                ][...],
                dtype=np.float64,
            )

            arr = (
                arr
                *
                intercept
            )

        total += arr

    if INCLUDE_ATMOSPHERIC_MUON:

        mu = np.asarray(
            h5[
                "backgrounds/"
                "atmospheric_muon"
            ][...],
            dtype=np.float64,
        )

        total += mu

    hist_shape = tuple(
        int(
            x
        )
        for x in h5.attrs[
            "hist_shape"
        ]
    )

    return total.reshape(
        hist_shape
    )


# =============================================================================
# LOAD CASE
# =============================================================================

def load_case(
    case_name: str,
    info: dict,
):

    h5file = (
        info[
            "file"
        ]
    )

    if not h5file.is_file():

        raise FileNotFoundError(
            f"No existe:\n"
            f"{h5file}"
        )

    with h5py.File(
        h5file,
        "r",
    ) as h5:

        log10_mass = np.asarray(
            h5[
                "axes/"
                "log10_mass_eV"
            ][...],
            dtype=np.float64,
        )

        log10_g = np.asarray(
            h5[
                "axes/"
                "log10_gprime"
            ][...],
            dtype=np.float64,
        )

        reco_energy_edges = np.asarray(
            h5[
                "axes/"
                "reco_energy_edges_GeV"
            ][...],
            dtype=np.float64,
        )

        reco_coszen_edges = np.asarray(
            h5[
                "axes/"
                "reco_coszen_edges"
            ][...],
            dtype=np.float64,
        )

        im = exact_index(
            log10_mass,
            info[
                "log10_m"
            ],
            "log10(m/eV)",
        )

        ig_lri = exact_index(
            log10_g,
            info[
                "log10_g"
            ],
            "log10(g)",
        )

        ig_sm = int(
            np.argmin(
                log10_g
            )
        )

        actual_log10_m = float(
            log10_mass[
                im
            ]
        )

        actual_log10_g = float(
            log10_g[
                ig_lri
            ]
        )

        sm_log10_g = float(
            log10_g[
                ig_sm
            ]
        )

        standard = build_prediction(
            h5,
            im,
            ig_sm,
        )

        lri = build_prediction(
            h5,
            im,
            ig_lri,
        )


    # =========================================================================
    # RELATIVE DEVIATION
    # =========================================================================

    threshold = (
        RELATIVE_MASK_FRACTION_OF_MAX
        *
        np.nanmax(
            standard
        )
    )

    delta = np.full_like(
        standard,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            standard
        )
        &
        np.isfinite(
            lri
        )
        &
        (
            standard
            >
            threshold
        )
    )

    delta[
        valid
    ] = (
        100.0
        *
        (
            lri[
                valid
            ]
            -
            standard[
                valid
            ]
        )
        /
        standard[
            valid
        ]
    )


    return {

        "h5file":
            h5file,

        "reco_energy_edges":
            reco_energy_edges,

        "reco_coszen_edges":
            reco_coszen_edges,

        "standard":
            standard,

        "lri":
            lri,

        "delta":
            delta,

        "actual_log10_m":
            actual_log10_m,

        "actual_log10_g":
            actual_log10_g,

        "sm_log10_g":
            sm_log10_g,
    }


# =============================================================================
# MARGINAL RELATIVE DEVIATION
# =============================================================================

def relative_deviation_profile_energy(
    standard_2d,
    lri_2d,
):

    sm = np.sum(
        standard_2d,
        axis=1,
    )

    lr = np.sum(
        lri_2d,
        axis=1,
    )

    threshold = (
        RELATIVE_MASK_FRACTION_OF_MAX
        *
        np.nanmax(
            sm
        )
    )

    result = np.full_like(
        sm,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            sm
        )
        &
        np.isfinite(
            lr
        )
        &
        (
            sm
            >
            threshold
        )
    )

    result[
        valid
    ] = (
        100.0
        *
        (
            lr[
                valid
            ]
            -
            sm[
                valid
            ]
        )
        /
        sm[
            valid
        ]
    )

    return result


def relative_deviation_profile_coszen(
    standard_2d,
    lri_2d,
):

    sm = np.sum(
        standard_2d,
        axis=0,
    )

    lr = np.sum(
        lri_2d,
        axis=0,
    )

    threshold = (
        RELATIVE_MASK_FRACTION_OF_MAX
        *
        np.nanmax(
            sm
        )
    )

    result = np.full_like(
        sm,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            sm
        )
        &
        np.isfinite(
            lr
        )
        &
        (
            sm
            >
            threshold
        )
    )

    result[
        valid
    ] = (
        100.0
        *
        (
            lr[
                valid
            ]
            -
            sm[
                valid
            ]
        )
        /
        sm[
            valid
        ]
    )

    return result


# =============================================================================
# SIMPLE FIGURE
# =============================================================================

def make_single_figure(
    data3d,
    energy_edges,
    coszen_edges,
    outpath,
    cmap,
    norm,
    cbar_label=None,
):

    fig = plt.figure(
        figsize=(
            9.2,
            10.0,
        )
    )

    gs = fig.add_gridspec(
        2,
        2,

        width_ratios=[
            1.0,
            0.045,
        ],

        left=0.10,
        right=0.92,
        bottom=0.08,
        top=0.96,

        wspace=0.08,
        hspace=0.10,
    )

    ax0 = fig.add_subplot(
        gs[
            0,
            0
        ]
    )

    ax1 = fig.add_subplot(
        gs[
            1,
            0
        ],

        sharex=ax0,
        sharey=ax0,
    )

    axes = np.array(
        [
            ax0,
            ax1,
        ]
    )

    cax = fig.add_subplot(
        gs[
            :,
            1
        ]
    )

    meshes = []

    for pid_idx, ax in enumerate(
        axes
    ):

        z = (
            data3d[
                :,
                :,
                pid_idx
            ].T
        )

        mesh = ax.pcolormesh(
            energy_edges,
            coszen_edges,
            z,

            shading="auto",

            cmap=cmap,

            norm=norm,
        )

        meshes.append(
            mesh
        )

        ax.set_ylabel(
            PID_LABELS[
                pid_idx
            ]
            +
            "\n"
            +
            r"$\cos\theta_{\rm reco}$"
        )

        ax.set_xscale(
            "log"
        )

        ax.minorticks_on()

        ax.grid(
            alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH,
        )

    axes[
        -1
    ].set_xlabel(
        r"$E_{\rm reco}$ [GeV]"
    )

    plt.setp(
        axes[
            0
        ].get_xticklabels(),
        visible=False,
    )

    cbar = fig.colorbar(
        meshes[
            0
        ],
        cax=cax,
    )

    if cbar_label is not None:

        cbar.set_label(
            cbar_label,
            fontsize=COLORBAR_LABELSIZE,
        )

    cbar.ax.yaxis.set_label_position(
        "right"
    )

    cbar.ax.yaxis.set_ticks_position(
        "right"
    )

    # Save both raster and vector versions.
    png_outpath = outpath.with_suffix(".png")
    pdf_outpath = outpath.with_suffix(".pdf")

    fig.savefig(
        png_outpath,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.04,
    )

    fig.savefig(
        pdf_outpath,

        bbox_inches="tight",

        pad_inches=0.04,
    )

    plt.close(
        fig
    )


# =============================================================================
# PANEL WITH MARGINAL HISTOGRAMS
# =============================================================================

def create_marginal_panel(
    fig,
    subspec,
):

    sub = subspec.subgridspec(
        2,
        2,

        height_ratios=[
            0.28,
            1.0,
        ],

        width_ratios=[
            1.0,
            0.26,
        ],

        hspace=(
            PANEL_TOP_GAP
        ),

        wspace=(
            PANEL_SIDE_GAP
        ),
    )

    ax_top = fig.add_subplot(
        sub[
            0,
            0
        ]
    )

    ax_main = fig.add_subplot(
        sub[
            1,
            0
        ],

        sharex=ax_top,
    )

    ax_side = fig.add_subplot(
        sub[
            1,
            1
        ],

        sharey=ax_main,
    )

    return (
        ax_top,
        ax_main,
        ax_side,
    )


# =============================================================================
# GROUPED FIGURE
# =============================================================================

def make_grouped_figure(
    standard,
    lri,
    delta,
    energy_edges,
    coszen_edges,
    outpath,
):
    """
    Final layout:

        Standard Model
        Long-range interaction
        event colorbar
        smaller empty gap
        Relative deviation
        deviation colorbar

    The right-most colorbar intentionally has NO label.
    """

    # =========================================================================
    # EVENT NORMALIZATION
    # =========================================================================

    count_max = float(
        np.nanmax(
            [
                np.nanmax(
                    standard
                ),
                np.nanmax(
                    lri
                ),
            ]
        )
    )

    count_norm = Normalize(
        vmin=0.0,
        vmax=count_max,
    )


    # =========================================================================
    # RELATIVE-DEVIATION NORMALIZATION
    # =========================================================================

    delta_vmax = robust_abs_max(
        [
            delta
        ],
        percentile=(
            DELTA_PERCENTILE
        ),
    )

    delta_norm = TwoSlopeNorm(
        vmin=(
            -delta_vmax
        ),

        vcenter=0.0,

        vmax=(
            delta_vmax
        ),
    )


    # =========================================================================
    # FIGURE
    # =========================================================================

    fig = plt.figure(
        figsize=(
            GROUPED_FIG_WIDTH,
            GROUPED_FIG_HEIGHT,
        )
    )


    # =========================================================================
    # OUTER GRID
    #
    # 0  Standard Model
    # 1  Long-range interaction
    # 2  event colorbar
    # 3  small empty gap
    # 4  Relative deviation
    # 5  deviation colorbar
    # =========================================================================

    outer = fig.add_gridspec(
        2,
        6,

        width_ratios=[
            1.0,
            1.0,
            COUNT_CBAR_WIDTH,
            COUNT_CBAR_TO_RELATIVE_GAP,
            1.0,
            DELTA_CBAR_WIDTH,
        ],

        left=0.045,
        right=0.975,

        bottom=0.085,
        top=0.965,

        wspace=0.10,
        hspace=0.16,
    )


    # =========================================================================
    # PANELS
    # =========================================================================

    panels = {}


    # Mixed
    panels[
        (
            0,
            0
        )
    ] = create_marginal_panel(
        fig,
        outer[
            0,
            0
        ],
    )

    panels[
        (
            0,
            1
        )
    ] = create_marginal_panel(
        fig,
        outer[
            0,
            1
        ],
    )

    panels[
        (
            0,
            2
        )
    ] = create_marginal_panel(
        fig,
        outer[
            0,
            4
        ],
    )


    # Tracks
    panels[
        (
            1,
            0
        )
    ] = create_marginal_panel(
        fig,
        outer[
            1,
            0
        ],
    )

    panels[
        (
            1,
            1
        )
    ] = create_marginal_panel(
        fig,
        outer[
            1,
            1
        ],
    )

    panels[
        (
            1,
            2
        )
    ] = create_marginal_panel(
        fig,
        outer[
            1,
            4
        ],
    )


    # =========================================================================
    # COLORBARS
    # =========================================================================

    cax_counts = fig.add_subplot(
        outer[
            :,
            2
        ]
    )

    cax_delta = fig.add_subplot(
        outer[
            :,
            5
        ]
    )


    count_mesh = None
    delta_mesh = None


    # =========================================================================
    # BIN GEOMETRY
    # =========================================================================

    energy_left = (
        energy_edges[
            :-1
        ]
    )

    energy_width = np.diff(
        energy_edges
    )

    coszen_centers = (
        0.5
        *
        (
            coszen_edges[
                :-1
            ]
            +
            coszen_edges[
                1:
            ]
        )
    )

    coszen_height = np.diff(
        coszen_edges
    )


    # =========================================================================
    # LOOP OVER MIXED / TRACKS
    # =========================================================================

    for pid_idx in range(
        2
    ):

        sm2d = (
            standard[
                :,
                :,
                pid_idx
            ]
        )

        lri2d = (
            lri[
                :,
                :,
                pid_idx
            ]
        )

        delta2d = (
            delta[
                :,
                :,
                pid_idx
            ]
        )


        # =====================================================================
        # EVENT MARGINALS
        # =====================================================================

        sm_energy = np.sum(
            sm2d,
            axis=1,
        )

        sm_coszen = np.sum(
            sm2d,
            axis=0,
        )

        lri_energy = np.sum(
            lri2d,
            axis=1,
        )

        lri_coszen = np.sum(
            lri2d,
            axis=0,
        )


        # =====================================================================
        # DEVIATION MARGINALS
        # =====================================================================

        relative_energy = (
            relative_deviation_profile_energy(
                sm2d,
                lri2d,
            )
        )

        relative_coszen = (
            relative_deviation_profile_coszen(
                sm2d,
                lri2d,
            )
        )


        # =====================================================================
        # STANDARD MODEL
        # =====================================================================

        (
            ax_top_sm,
            ax_main_sm,
            ax_side_sm,
        ) = panels[
            (
                pid_idx,
                0
            )
        ]


        ax_top_sm.bar(
            energy_left,
            sm_energy,

            width=energy_width,
            align="edge",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        count_mesh = (
            ax_main_sm.pcolormesh(
                energy_edges,
                coszen_edges,
                sm2d.T,

                shading="auto",

                cmap="OrRd",

                norm=count_norm,
            )
        )


        ax_side_sm.barh(
            coszen_centers,
            sm_coszen,

            height=coszen_height,

            align="center",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        # =====================================================================
        # LONG-RANGE INTERACTION
        # =====================================================================

        (
            ax_top_lri,
            ax_main_lri,
            ax_side_lri,
        ) = panels[
            (
                pid_idx,
                1
            )
        ]


        ax_top_lri.bar(
            energy_left,
            lri_energy,

            width=energy_width,

            align="edge",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        ax_main_lri.pcolormesh(
            energy_edges,
            coszen_edges,
            lri2d.T,

            shading="auto",

            cmap="OrRd",

            norm=count_norm,
        )


        ax_side_lri.barh(
            coszen_centers,
            lri_coszen,

            height=coszen_height,

            align="center",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        # =====================================================================
        # RELATIVE DEVIATION
        # =====================================================================

        (
            ax_top_delta,
            ax_main_delta,
            ax_side_delta,
        ) = panels[
            (
                pid_idx,
                2
            )
        ]


        ax_top_delta.bar(
            energy_left,

            np.nan_to_num(
                relative_energy,
                nan=0.0,
            ),

            width=energy_width,

            align="edge",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        delta_mesh = (
            ax_main_delta.pcolormesh(
                energy_edges,
                coszen_edges,
                delta2d.T,

                shading="auto",

                cmap="RdBu_r",

                norm=delta_norm,
            )
        )


        ax_side_delta.barh(
            coszen_centers,

            np.nan_to_num(
                relative_coszen,
                nan=0.0,
            ),

            height=coszen_height,

            align="center",

            color=HIST_COLOR,

            edgecolor=(
                HIST_EDGE_COLOR
            ),

            linewidth=(
                HIST_LINEWIDTH
            ),

            alpha=(
                HIST_ALPHA
            ),
        )


        # =====================================================================
        # COMMON STYLE
        # =====================================================================

        row_panels = (

            (
                ax_top_sm,
                ax_main_sm,
                ax_side_sm,
            ),

            (
                ax_top_lri,
                ax_main_lri,
                ax_side_lri,
            ),

            (
                ax_top_delta,
                ax_main_delta,
                ax_side_delta,
            ),
        )


        for (
            ax_top,
            ax_main,
            ax_side,
        ) in row_panels:


            # -----------------------------------------------------------------
            # TOP HISTOGRAM
            # -----------------------------------------------------------------

            ax_top.set_xscale(
                "log"
            )

            ax_top.minorticks_on()

            ax_top.grid(
                alpha=(
                    GRID_ALPHA
                ),

                linewidth=(
                    GRID_LINEWIDTH
                ),
            )

            # No "Counts" or delta label
            ax_top.set_ylabel(
                ""
            )

            plt.setp(
                ax_top.get_xticklabels(),
                visible=False,
            )


            # -----------------------------------------------------------------
            # MAIN HEATMAP
            # -----------------------------------------------------------------

            ax_main.set_xscale(
                "log"
            )

            ax_main.minorticks_on()

            ax_main.grid(
                alpha=(
                    GRID_ALPHA
                ),

                linewidth=(
                    GRID_LINEWIDTH
                ),
            )


            # -----------------------------------------------------------------
            # SIDE HISTOGRAM
            # -----------------------------------------------------------------

            ax_side.minorticks_on()

            ax_side.grid(
                alpha=(
                    GRID_ALPHA
                ),

                linewidth=(
                    GRID_LINEWIDTH
                ),
            )

            ax_side.set_xlabel(
                ""
            )

            plt.setp(
                ax_side.get_yticklabels(),
                visible=False,
            )


        # =====================================================================
        # DYNAMIC SIDE LIMITS
        # =====================================================================

        sm_side_min, sm_side_max = (
            dynamic_axis_limits(
                sm_coszen,
                pad_fraction=0.08,
            )
        )

        ax_side_sm.set_xlim(
            sm_side_min,
            sm_side_max,
        )


        lri_side_min, lri_side_max = (
            dynamic_axis_limits(
                lri_coszen,
                pad_fraction=0.08,
            )
        )

        ax_side_lri.set_xlim(
            lri_side_min,
            lri_side_max,
        )


        finite_relative_coszen = (
            relative_coszen[
                np.isfinite(
                    relative_coszen
                )
            ]
        )

        if (
            finite_relative_coszen.size
            >
            0
        ):

            relative_side_min, relative_side_max = (
                dynamic_axis_limits(
                    finite_relative_coszen,
                    pad_fraction=0.10,
                )
            )

            ax_side_delta.set_xlim(
                relative_side_min,
                relative_side_max,
            )


        # =====================================================================
        # DYNAMIC TOP LIMITS
        # =====================================================================

        sm_top_min, sm_top_max = (
            dynamic_axis_limits(
                sm_energy,
                pad_fraction=0.08,
            )
        )

        ax_top_sm.set_ylim(
            sm_top_min,
            sm_top_max,
        )


        lri_top_min, lri_top_max = (
            dynamic_axis_limits(
                lri_energy,
                pad_fraction=0.08,
            )
        )

        ax_top_lri.set_ylim(
            lri_top_min,
            lri_top_max,
        )


        finite_relative_energy = (
            relative_energy[
                np.isfinite(
                    relative_energy
                )
            ]
        )

        if (
            finite_relative_energy.size
            >
            0
        ):

            relative_top_min, relative_top_max = (
                dynamic_axis_limits(
                    finite_relative_energy,
                    pad_fraction=0.10,
                )
            )

            ax_top_delta.set_ylim(
                relative_top_min,
                relative_top_max,
            )


        # =====================================================================
        # LEFT-SIDE LABELS
        # =====================================================================

        ax_main_sm.set_ylabel(
            PID_LABELS[
                pid_idx
            ]
            +
            "\n"
            +
            r"$\cos\theta_{\rm reco}$"
        )


    # =========================================================================
    # COLUMN TITLES
    # =========================================================================

    panels[
        (
            0,
            0
        )
    ][
        0
    ].set_title(
        "Standard Model",

        fontsize=(
            COLUMN_TITLE_FONTSIZE
        ),

        pad=5,
    )


    panels[
        (
            0,
            1
        )
    ][
        0
    ].set_title(
        "Long-range interaction",

        fontsize=(
            COLUMN_TITLE_FONTSIZE
        ),

        pad=5,
    )


    # No title above the Relative-deviation column.
    # Its meaning and units are stated on the colorbar.


    # =========================================================================
    # X LABELS — BOTTOM ROW ONLY
    # =========================================================================

    for col in range(
        3
    ):

        panels[
            (
                1,
                col
            )
        ][
            1
        ].set_xlabel(
            r"$E_{\rm reco}$ [GeV]"
        )


    # =========================================================================
    # REMOVE X TICK LABELS FROM TOP ROW HEATMAPS
    # =========================================================================

    for col in range(
        3
    ):

        plt.setp(
            panels[
                (
                    0,
                    col
                )
            ][
                1
            ].get_xticklabels(),

            visible=False,
        )


    # =========================================================================
    # REMOVE REPEATED Y TICK LABELS
    # =========================================================================

    for pid_idx in range(
        2
    ):

        panels[
            (
                pid_idx,
                1
            )
        ][
            1
        ].tick_params(
            labelleft=False
        )

        panels[
            (
                pid_idx,
                2
            )
        ][
            1
        ].tick_params(
            labelleft=False
        )


    # =========================================================================
    # EVENT COLORBAR
    # =========================================================================

    cbar_counts = fig.colorbar(
        count_mesh,
        cax=cax_counts,
    )

    cbar_counts.set_label(
        "Predicted neutrino events",

        fontsize=(
            COLORBAR_LABELSIZE
        ),

        labelpad=12,
    )

    cbar_counts.ax.yaxis.set_ticks_position(
        "right"
    )

    cbar_counts.ax.yaxis.set_label_position(
        "right"
    )


    # =========================================================================
    # RELATIVE-DEVIATION COLORBAR
    # =========================================================================

    cbar_delta = fig.colorbar(
        delta_mesh,

        cax=cax_delta,

        extend="both",
    )

    cbar_delta.set_label(
        r"Relative deviation [$\%$]",

        fontsize=(
            COLORBAR_LABELSIZE
        ),

        labelpad=12,
    )

    cbar_delta.ax.yaxis.set_ticks_position(
        "right"
    )

    cbar_delta.ax.yaxis.set_label_position(
        "right"
    )


    # =========================================================================
    # NO GLOBAL TITLE
    # =========================================================================


    # =========================================================================
    # SAVE
    # =========================================================================

    # Save both raster and vector versions.
    png_outpath = outpath.with_suffix(".png")
    pdf_outpath = outpath.with_suffix(".pdf")

    fig.savefig(
        png_outpath,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.04,
    )

    fig.savefig(
        pdf_outpath,

        bbox_inches="tight",

        pad_inches=0.04,
    )

    plt.close(
        fig
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary_rows(
    case_name: str,
    data: dict,
):

    rows = []

    for pid_idx, pid_label in enumerate(
        PID_LABELS
    ):

        sm = (
            data[
                "standard"
            ][
                :,
                :,
                pid_idx
            ]
        )

        lri = (
            data[
                "lri"
            ][
                :,
                :,
                pid_idx
            ]
        )

        difference = (
            lri
            -
            sm
        )


        n_sm = float(
            np.sum(
                sm
            )
        )

        n_lri = float(
            np.sum(
                lri
            )
        )


        signed_total_change_percent = (
            100.0
            *
            (
                n_lri
                -
                n_sm
            )
            /
            n_sm
        )


        absolute_redistribution_percent = (
            100.0
            *
            np.sum(
                np.abs(
                    difference
                )
            )
            /
            n_sm
        )


        rows.append(
            {

                "case":
                    case_name,

                "topology":
                    pid_label,

                "log10_m":
                    data[
                        "actual_log10_m"
                    ],

                "log10_g":
                    data[
                        "actual_log10_g"
                    ],

                "sm_reference_log10_g":
                    data[
                        "sm_log10_g"
                    ],

                "N_SM":
                    n_sm,

                "N_LRI":
                    n_lri,

                "signed_total_change_percent":
                    signed_total_change_percent,

                "absolute_redistribution_percent":
                    absolute_redistribution_percent,
            }
        )

    return rows


# =============================================================================
# WRITE SUMMARY
# =============================================================================

def write_summary_csv(
    rows,
    outpath,
):

    if not rows:

        return

    with outpath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,

            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "============================================================"
    )

    print(
        "Generating shower/tracks plots "
        "for all LRI models"
    )

    print(
        "============================================================"
    )


    all_rows = []


    for case_name, info in CASES.items():


        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            case_name
        )

        print(
            "------------------------------------------------------------"
        )


        data = load_case(
            case_name,
            info,
        )


        # =====================================================================
        # SHARED NORMALIZATION
        # =====================================================================

        count_max = float(
            np.nanmax(
                [
                    np.nanmax(
                        data[
                            "standard"
                        ]
                    ),

                    np.nanmax(
                        data[
                            "lri"
                        ]
                    ),
                ]
            )
        )


        count_norm = Normalize(
            vmin=0.0,
            vmax=count_max,
        )


        delta_vmax = robust_abs_max(
            [
                data[
                    "delta"
                ]
            ],

            percentile=(
                DELTA_PERCENTILE
            ),
        )


        delta_norm = TwoSlopeNorm(
            vmin=(
                -delta_vmax
            ),

            vcenter=0.0,

            vmax=(
                delta_vmax
            ),
        )


        # =====================================================================
        # STANDARD MODEL
        # =====================================================================

        out_standard = (
            OUTPUT_DIR
            /
            f"shower_tracks_standard_"
            f"{case_name}.png"
        )


        make_single_figure(
            data[
                "standard"
            ],

            data[
                "reco_energy_edges"
            ],

            data[
                "reco_coszen_edges"
            ],

            outpath=(
                out_standard
            ),

            cmap="OrRd",

            norm=count_norm,

            cbar_label=(
                "Predicted neutrino events"
            ),
        )


        # =====================================================================
        # LONG-RANGE INTERACTION
        # =====================================================================

        out_lri = (
            OUTPUT_DIR
            /
            f"shower_tracks_lri_"
            f"{case_name}.png"
        )


        make_single_figure(
            data[
                "lri"
            ],

            data[
                "reco_energy_edges"
            ],

            data[
                "reco_coszen_edges"
            ],

            outpath=(
                out_lri
            ),

            cmap="OrRd",

            norm=count_norm,

            cbar_label=(
                "Predicted neutrino events"
            ),
        )


        # =====================================================================
        # RELATIVE DEVIATION
        # =====================================================================

        out_delta = (
            OUTPUT_DIR
            /
            f"shower_tracks_delta_"
            f"{case_name}.png"
        )


        make_single_figure(
            data[
                "delta"
            ],

            data[
                "reco_energy_edges"
            ],

            data[
                "reco_coszen_edges"
            ],

            outpath=(
                out_delta
            ),

            cmap="RdBu_r",

            norm=delta_norm,

            cbar_label=(
                r"Relative deviation [$\%$]"
            ),
        )


        # =====================================================================
        # GROUPED FIGURE
        # =====================================================================

        out_grouped = (
            OUTPUT_DIR
            /
            f"shower_tracks_standard_vs_lri_"
            f"{case_name}.png"
        )


        make_grouped_figure(
            data[
                "standard"
            ],

            data[
                "lri"
            ],

            data[
                "delta"
            ],

            data[
                "reco_energy_edges"
            ],

            data[
                "reco_coszen_edges"
            ],

            outpath=(
                out_grouped
            ),
        )


        print(
            f"Saved: "
            f"{out_standard.with_suffix('.png').name}"
        )

        print(
            f"Saved: "
            f"{out_standard.with_suffix('.pdf').name}"
        )

        print(
            f"Saved: "
            f"{out_lri.with_suffix('.png').name}"
        )

        print(
            f"Saved: "
            f"{out_lri.with_suffix('.pdf').name}"
        )

        print(
            f"Saved: "
            f"{out_delta.with_suffix('.png').name}"
        )

        print(
            f"Saved: "
            f"{out_delta.with_suffix('.pdf').name}"
        )

        print(
            f"Saved: "
            f"{out_grouped.with_suffix('.png').name}"
        )

        print(
            f"Saved: "
            f"{out_grouped.with_suffix('.pdf').name}"
        )


        all_rows.extend(
            build_summary_rows(
                case_name,
                data,
            )
        )


    # =========================================================================
    # SUMMARY CSV
    # =========================================================================

    summary_file = (
        OUTPUT_DIR
        /
        "shower_tracks_summary.csv"
    )


    write_summary_csv(
        all_rows,
        summary_file,
    )


    print()

    print(
        "============================================================"
    )

    print(
        "Done."
    )

    print(
        "============================================================"
    )


    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Summary file    : "
        f"{summary_file}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

