#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_vlri_vs_L.py

Paper-style 3x2 figure for the L_e - L_mu terrestrial potential.

COLUMNS
-------
left  : cos(theta_z) = -1.0
right : cos(theta_z) = -0.4

ROWS
----
1 : m' = 10^-13 eV, g' = 10^-24
2 : automatically chosen g' such that V_LRI ~ V_CC in RMS scale
3 : m' = 10^-15 eV, g' = 10^-25

Each row uses exactly the same (m', g') in both columns.

Each panel:
    solid blue   : V_LRI
    dashed gray  : V_CC
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

import scienceplots


# =============================================================================
# SCIENCEPLOTS / LATEX STYLE
# =============================================================================

plt.style.use("science")

plt.rcParams.update({

    "font.size": 8.4,

    "axes.labelsize": 11.0,

    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,

    "legend.fontsize": 7.0,

    "axes.linewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 3.3,
    "ytick.major.size": 3.3,

    "xtick.minor.size": 1.7,
    "ytick.minor.size": 1.7,

    "xtick.major.width": 0.70,
    "ytick.major.width": 0.70,

    "xtick.minor.width": 0.50,
    "ytick.minor.width": 0.50,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

LUT_FILE = (
    ROOT
    / "LUT_VLRI"
    / "vlri_unit.h5"
)

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "potential"
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "VLRI_and_standard_vs_L_Le-Lmu.png"
)

OUTPUT_PDF = (
    OUTPUT_DIR
    / "VLRI_and_standard_vs_L_Le-Lmu.pdf"
)


# =============================================================================
# TRAJECTORIES = COLUMNS
# =============================================================================

COSZENS = (
    -1.0,
    -0.4,
)


# =============================================================================
# ROW CONFIGURATION
# =============================================================================

# Row 1
FIRST_ROW_POINT = (
    -13.0,
    -24.0,
)

# Row 3
THIRD_ROW_POINT = (
    -15.0,
    -25.0,
)


# =============================================================================
# ROW 2: AUTOMATIC MATCH
# =============================================================================

MATCH_ROW_LOG10_M = -13.0


# =============================================================================
# CONSTANTS
# =============================================================================

N_PATH_POINTS = 3500

# G_F = 1.1663787e-5 GeV^-2
#     = 1.1663787e-23 eV^-2

G_F_EV_MINUS2 = (
    1.1663787e-5
    * 1.0e-18
)

SQRT2 = np.sqrt(2.0)


# =============================================================================
# PLOT UNITS
# =============================================================================

# Display:
#
# L / 10^3 km
#
# V / 10^-14 eV

L_SCALE_KM = 1.0e3

POTENTIAL_SCALE_EV = 1.0e-14


# =============================================================================
# CURVE STYLE
# =============================================================================

LRI_COLOR = "C0"
STD_COLOR = "0.28"

LRI_LINESTYLE = "-"
STD_LINESTYLE = "--"

LRI_LINEWIDTH = 1.65
STD_LINEWIDTH = 1.50


# =============================================================================
# INTERNAL TEXT
# =============================================================================
#
# THESE TWO CONTROL:
#
# m', g'
# cos(theta_z)
#
# respectively.
#

PARAMETER_FONTSIZE = 9.2

COSZEN_FONTSIZE = 9.2


# Fixed offsets from the panel corners, in points.
# These replace the old PARAMETER_X/Y and COSZEN_X/Y coordinates.
#
# m', g' are anchored to the upper-left corner.
PARAMETER_OFFSET_X = 7
PARAMETER_OFFSET_Y = -7

# cos(theta_z) is anchored to the upper-right corner.
COSZEN_OFFSET_X = -1
COSZEN_OFFSET_Y = -7

# White halo around the internal labels so curves passing behind
# them do not reduce readability.
TEXT_HALO_LINEWIDTH = 1.35


# =============================================================================
# Y-LIMIT CONTROL
# =============================================================================

Y_PADDING_FRACTION = 0.055

ZERO_THRESHOLD_FRACTION = 0.08


# =============================================================================
# EXACT 3x2 GEOMETRY
# =============================================================================
#
# The panels themselves remain TRUE 1:1.
#

PANEL_SIZE_IN = 1.78


# -----------------------------------------------------------------------------
# SPACE BETWEEN PANELS
# -----------------------------------------------------------------------------
#
# COLUMN_GAP_IN:
#
#     horizontal distance between
#
#         left panel  <-->  right panel
#
# ROW_GAP_IN:
#
#     vertical distance between
#
#         row 1
#         row 2
#         row 3
#
# If you want them even closer:
#
#     COLUMN_GAP_IN = 0.14
#     ROW_GAP_IN    = 0.03
#
# -----------------------------------------------------------------------------

COLUMN_GAP_IN = 0.07

ROW_GAP_IN = 0.07


# -----------------------------------------------------------------------------
# OUTER MARGINS
# -----------------------------------------------------------------------------
#
# LEFT_MARGIN_IN partly controls the distance between the Y-axis title
# and the panels.
#
# BOTTOM_MARGIN_IN partly controls the distance between the X-axis title
# and the bottom panels.
#

LEFT_MARGIN_IN = 0.52

RIGHT_MARGIN_IN = 0.12

BOTTOM_MARGIN_IN = 0.43

TOP_MARGIN_IN = 0.10


# -----------------------------------------------------------------------------
# GLOBAL AXIS LABEL POSITIONS
# -----------------------------------------------------------------------------
#
# These are normalized figure coordinates.
#
# Increase X_AXIS_LABEL_Y -> x label moves UP, closer to plots.
#
# Increase Y_AXIS_LABEL_X -> y label moves RIGHT, closer to plots.
#

X_AXIS_LABEL_Y = 0.008

Y_AXIS_LABEL_X = 0.037


# -----------------------------------------------------------------------------
# FIGURE SIZE
# -----------------------------------------------------------------------------
#
# Calculated directly from physical panel dimensions, gaps and margins.
# This prevents extra unused whitespace from appearing.
#

FIG_WIDTH_IN = (
    LEFT_MARGIN_IN
    +
    PANEL_SIZE_IN
    +
    COLUMN_GAP_IN
    +
    PANEL_SIZE_IN
    +
    RIGHT_MARGIN_IN
)

FIG_HEIGHT_IN = (
    BOTTOM_MARGIN_IN
    +
    PANEL_SIZE_IN
    +
    ROW_GAP_IN
    +
    PANEL_SIZE_IN
    +
    ROW_GAP_IN
    +
    PANEL_SIZE_IN
    +
    TOP_MARGIN_IN
)


# =============================================================================
# LOAD LUT
# =============================================================================

def load_lut(filename):

    if not filename.is_file():

        raise FileNotFoundError(
            f"No existe el LUT:\n{filename}"
        )

    with h5py.File(
        filename,
        "r",
    ) as h5:

        log10_mass_eV = np.asarray(
            h5[
                "/axes/log10_mass_eV"
            ][...],
            dtype=np.float64,
        )

        radius_km = np.asarray(
            h5[
                "/axes/radius_km"
            ][...],
            dtype=np.float64,
        )

        v_unit_eV = np.asarray(
            h5[
                "/potential/V_unit_eV"
            ][...],
            dtype=np.float64,
        )

        ne_eV3 = np.asarray(
            h5[
                "/earth/ne_eV3"
            ][...],
            dtype=np.float64,
        )

        earth_radius_km = float(
            h5.attrs.get(
                "earth_radius_km",
                radius_km[-1],
            )
        )

        detector_radius_km = float(
            h5.attrs.get(
                "detector_radius_km",
                earth_radius_km - 1.5,
            )
        )

    if v_unit_eV.shape != (
        log10_mass_eV.size,
        radius_km.size,
    ):

        raise RuntimeError(
            "Shape inesperado para V_unit:\n"
            f"  encontrado = {v_unit_eV.shape}\n"
            f"  esperado   = "
            f"({log10_mass_eV.size}, "
            f"{radius_km.size})"
        )

    if ne_eV3.shape != radius_km.shape:

        raise RuntimeError(
            "ne_eV3 y radius_km "
            "no tienen la misma shape."
        )

    return {

        "log10_mass_eV":
            log10_mass_eV,

        "radius_km":
            radius_km,

        "v_unit_eV":
            v_unit_eV,

        "ne_eV3":
            ne_eV3,

        "earth_radius_km":
            earth_radius_km,

        "detector_radius_km":
            detector_radius_km,
    }


# =============================================================================
# MASS INTERPOLATION
# =============================================================================

def interpolate_vunit_in_mass(
    lut,
    requested_log10_mass,
):

    mass_axis = (
        lut[
            "log10_mass_eV"
        ]
    )

    table = (
        lut[
            "v_unit_eV"
        ]
    )

    if (
        requested_log10_mass
        <
        mass_axis[0]
        or
        requested_log10_mass
        >
        mass_axis[-1]
    ):

        raise ValueError(
            "La masa pedida está fuera del LUT:\n"
            f"  requested = {requested_log10_mass}\n"
            f"  rango LUT = "
            f"[{mass_axis[0]}, {mass_axis[-1]}]"
        )

    exact = np.where(
        np.isclose(
            mass_axis,
            requested_log10_mass,
            rtol=0.0,
            atol=1.0e-12,
        )
    )[0]

    if exact.size:

        return np.asarray(
            table[
                exact[0],
                :
            ],
            dtype=np.float64,
        )

    upper = int(
        np.searchsorted(
            mass_axis,
            requested_log10_mass,
        )
    )

    lower = (
        upper
        -
        1
    )

    x1 = mass_axis[lower]
    x2 = mass_axis[upper]

    fraction = (
        (
            requested_log10_mass
            -
            x1
        )
        /
        (
            x2
            -
            x1
        )
    )

    return (
        (1.0 - fraction)
        *
        table[
            lower,
            :
        ]
        +
        fraction
        *
        table[
            upper,
            :
        ]
    )


# =============================================================================
# TRAJECTORY
# =============================================================================

def trajectory_radius_vs_L(
    coszen,
    earth_radius_km,
    detector_radius_km,
    n_points=N_PATH_POINTS,
):

    if not (
        -1.0
        <=
        coszen
        <=
        0.0
    ):

        raise ValueError(
            "Se espera coszen entre -1 y 0."
        )

    Re = float(
        earth_radius_km
    )

    Rd = float(
        detector_radius_km
    )

    discriminant = (
        Re * Re
        -
        Rd * Rd
        *
        (
            1.0
            -
            coszen * coszen
        )
    )

    if discriminant < 0.0:

        if abs(
            discriminant
        ) < 1.0e-8:

            discriminant = 0.0

        else:

            raise RuntimeError(
                "La trayectoria no intersecta la Tierra."
            )

    L_total_km = (
        -Rd
        *
        coszen
        +
        np.sqrt(
            discriminant
        )
    )

    L_km = np.linspace(
        0.0,
        L_total_km,
        n_points,
    )

    t_km = (
        L_total_km
        -
        L_km
    )

    R_km = np.sqrt(
        Rd * Rd
        +
        t_km * t_km
        +
        2.0
        *
        Rd
        *
        coszen
        *
        t_km
    )

    R_km = np.clip(
        R_km,
        0.0,
        Re,
    )

    return (
        L_km,
        R_km,
    )


# =============================================================================
# STANDARD MATTER POTENTIAL
# =============================================================================

def standard_potential_radial(
    lut,
):

    return (
        SQRT2
        *
        G_F_EV_MINUS2
        *
        lut[
            "ne_eV3"
        ]
    )


# =============================================================================
# POTENTIALS ALONG PATH
# =============================================================================

def potentials_along_path(
    lut,
    coszen,
    log10_m,
    log10_g,
):

    (
        L_km,
        R_km,
    ) = trajectory_radius_vs_L(
        coszen=coszen,
        earth_radius_km=(
            lut[
                "earth_radius_km"
            ]
        ),
        detector_radius_km=(
            lut[
                "detector_radius_km"
            ]
        ),
    )

    radial_vunit = (
        interpolate_vunit_in_mass(
            lut,
            log10_m,
        )
    )

    gprime = (
        10.0
        **
        log10_g
    )

    radial_vlri_eV = (
        gprime
        *
        gprime
        *
        radial_vunit
    )

    radial_vcc_eV = (
        standard_potential_radial(
            lut
        )
    )

    V_LRI_eV = np.interp(
        R_km,
        lut[
            "radius_km"
        ],
        radial_vlri_eV,
    )

    V_CC_eV = np.interp(
        R_km,
        lut[
            "radius_km"
        ],
        radial_vcc_eV,
    )

    return (
        L_km,
        V_LRI_eV,
        V_CC_eV,
    )


# =============================================================================
# UNIT POTENTIAL ALONG PATH
# =============================================================================

def vunit_along_path(
    lut,
    coszen,
    log10_m,
):

    (
        L_km,
        R_km,
    ) = trajectory_radius_vs_L(
        coszen=coszen,
        earth_radius_km=(
            lut[
                "earth_radius_km"
            ]
        ),
        detector_radius_km=(
            lut[
                "detector_radius_km"
            ]
        ),
    )

    radial_vunit = (
        interpolate_vunit_in_mass(
            lut,
            log10_m,
        )
    )

    radial_vcc = (
        standard_potential_radial(
            lut
        )
    )

    V_unit = np.interp(
        R_km,
        lut[
            "radius_km"
        ],
        radial_vunit,
    )

    V_CC = np.interp(
        R_km,
        lut[
            "radius_km"
        ],
        radial_vcc,
    )

    return (
        L_km,
        V_unit,
        V_CC,
    )


# =============================================================================
# RMS
# =============================================================================

def rms(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:

        raise RuntimeError(
            "No finite values "
            "available for RMS."
        )

    return float(
        np.sqrt(
            np.mean(
                finite
                *
                finite
            )
        )
    )


# =============================================================================
# AUTOMATIC MATCHING COUPLING
# =============================================================================

def find_matching_log10_g(
    lut,
    log10_m,
):

    """
    Determine one g' such that the RMS magnitude of V_LRI
    approximately matches V_CC using BOTH trajectories.

        V_LRI = g'^2 V_unit

    Therefore

        g'^2 = RMS(V_CC) / RMS(V_unit)
    """

    all_vunit = []
    all_vcc = []

    for coszen in COSZENS:

        (
            _,
            V_unit,
            V_CC,
        ) = vunit_along_path(
            lut=lut,
            coszen=coszen,
            log10_m=log10_m,
        )

        all_vunit.append(
            V_unit
        )

        all_vcc.append(
            V_CC
        )

    all_vunit = np.concatenate(
        all_vunit
    )

    all_vcc = np.concatenate(
        all_vcc
    )

    rms_unit = rms(
        all_vunit
    )

    rms_cc = rms(
        all_vcc
    )

    if rms_unit <= 0.0:

        raise RuntimeError(
            "RMS(V_unit) <= 0."
        )

    g_squared = (
        rms_cc
        /
        rms_unit
    )

    if g_squared <= 0.0:

        raise RuntimeError(
            "Computed g'^2 <= 0."
        )

    gprime = np.sqrt(
        g_squared
    )

    return float(
        np.log10(
            gprime
        )
    )


# =============================================================================
# LABEL HELPERS
# =============================================================================

def ten_power_string(
    log10_value,
):

    nearest_integer = round(
        log10_value
    )

    if np.isclose(
        log10_value,
        nearest_integer,
        atol=5.0e-3,
    ):

        exponent = (
            f"{int(nearest_integer)}"
        )

    else:

        exponent = (
            f"{log10_value:.2f}"
        )

    return (
        rf"10^{{{exponent}}}"
    )


def panel_parameter_text(
    log10_m,
    log10_g,
):

    return (
        rf"$m'="
        rf"{ten_power_string(log10_m)}"
        rf"\,\mathrm{{eV}}$"
        "\n"
        rf"$g'="
        rf"{ten_power_string(log10_g)}$"
    )


# =============================================================================
# PREPARE PANEL DATA
# =============================================================================

def prepare_panel_data(
    lut,
    coszen,
    log10_m,
    log10_g,
):

    (
        L_km,
        V_LRI_eV,
        V_CC_eV,
    ) = potentials_along_path(
        lut=lut,
        coszen=coszen,
        log10_m=log10_m,
        log10_g=log10_g,
    )

    return {

        "L_plot":
            (
                L_km
                /
                L_SCALE_KM
            ),

        "V_LRI_plot":
            (
                V_LRI_eV
                /
                POTENTIAL_SCALE_EV
            ),

        "V_CC_plot":
            (
                V_CC_eV
                /
                POTENTIAL_SCALE_EV
            ),
    }


# =============================================================================
# ROW Y LIMIT
# =============================================================================

def get_row_ylim(
    row_results,
):

    arrays = []

    for result in row_results:

        arrays.append(
            result[
                "V_LRI_plot"
            ]
        )

        arrays.append(
            result[
                "V_CC_plot"
            ]
        )

    values = np.concatenate(
        arrays
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    ymin = float(
        np.min(
            values
        )
    )

    ymax = float(
        np.max(
            values
        )
    )

    span = (
        ymax
        -
        ymin
    )

    if span <= 0.0:

        span = max(
            abs(
                ymax
            ),
            1.0,
        )

    if (
        ymin >= 0.0
        and
        ymin
        <=
        ZERO_THRESHOLD_FRACTION
        *
        ymax
    ):

        lower = 0.0

    else:

        lower = (
            ymin
            -
            Y_PADDING_FRACTION
            *
            span
        )

    upper = (
        ymax
        +
        Y_PADDING_FRACTION
        *
        span
    )

    return (
        lower,
        upper,
    )


# =============================================================================
# DRAW PANEL
# =============================================================================

def draw_panel(
    ax,
    result,
    log10_m,
    log10_g,
    coszen,
    ylim,
):

    L_plot = (
        result[
            "L_plot"
        ]
    )

    V_LRI_plot = (
        result[
            "V_LRI_plot"
        ]
    )

    V_CC_plot = (
        result[
            "V_CC_plot"
        ]
    )


    # =========================================================================
    # CURVES
    # =========================================================================

    ax.plot(
        L_plot,
        V_LRI_plot,

        color=LRI_COLOR,

        linestyle=LRI_LINESTYLE,

        linewidth=LRI_LINEWIDTH,

        zorder=3,
    )


    ax.plot(
        L_plot,
        V_CC_plot,

        color=STD_COLOR,

        linestyle=STD_LINESTYLE,

        linewidth=STD_LINEWIDTH,

        zorder=2,
    )


    # =========================================================================
    # LIMITS
    # =========================================================================

    ax.set_xlim(
        float(
            L_plot[
                0
            ]
        ),
        float(
            L_plot[
                -1
            ]
        ),
    )

    ax.set_ylim(
        *ylim
    )


    # =========================================================================
    # STYLE
    # =========================================================================

    ax.minorticks_on()

    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    ax.grid(
        True,
        which="major",

        alpha=0.14,

        linewidth=0.42,
    )


    # =========================================================================
    # m', g' — UPPER LEFT
    #
    # Anchored exactly to the upper-left corner and moved inward by a fixed
    # number of points. This makes the margin independent of panel size.
    # =========================================================================

    parameter_text = ax.annotate(
        panel_parameter_text(
            log10_m,
            log10_g,
        ),

        xy=(0.0, 1.0),
        xycoords="axes fraction",

        xytext=(
            PARAMETER_OFFSET_X,
            PARAMETER_OFFSET_Y,
        ),
        textcoords="offset points",

        ha="left",
        va="top",

        fontsize=PARAMETER_FONTSIZE,
        linespacing=0.95,

        zorder=20,
    )

    parameter_text.set_path_effects([
        path_effects.Stroke(
            linewidth=TEXT_HALO_LINEWIDTH,
            foreground="white",
        ),
        path_effects.Normal(),
    ])


    # =========================================================================
    # cos(theta_z) — UPPER RIGHT
    #
    # Anchored exactly to the upper-right corner and moved inward by a fixed
    # number of points.
    # =========================================================================

    coszen_text = ax.annotate(
        rf"$\cos\theta_z={coszen:g}$",

        xy=(1.0, 1.0),
        xycoords="axes fraction",

        xytext=(
            COSZEN_OFFSET_X,
            COSZEN_OFFSET_Y,
        ),
        textcoords="offset points",

        ha="right",
        va="top",

        fontsize=COSZEN_FONTSIZE,

        zorder=20,
    )

    coszen_text.set_path_effects([
        path_effects.Stroke(
            linewidth=TEXT_HALO_LINEWIDTH,
            foreground="white",
        ),
        path_effects.Normal(),
    ])


# =============================================================================
# CREATE 3x2 AXES
# =============================================================================

def create_axes():

    fig = plt.figure(
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        )
    )


    # =========================================================================
    # CONVERT PHYSICAL INCHES TO FIGURE COORDINATES
    # =========================================================================

    panel_w = (
        PANEL_SIZE_IN
        /
        FIG_WIDTH_IN
    )

    panel_h = (
        PANEL_SIZE_IN
        /
        FIG_HEIGHT_IN
    )

    left0 = (
        LEFT_MARGIN_IN
        /
        FIG_WIDTH_IN
    )

    bottom0 = (
        BOTTOM_MARGIN_IN
        /
        FIG_HEIGHT_IN
    )

    gap_x = (
        COLUMN_GAP_IN
        /
        FIG_WIDTH_IN
    )

    gap_y = (
        ROW_GAP_IN
        /
        FIG_HEIGHT_IN
    )


    axes = np.empty(
        (
            3,
            2,
        ),
        dtype=object,
    )


    # =========================================================================
    # CREATE PANELS
    # =========================================================================

    for row in range(3):

        y = (
            bottom0
            +
            (
                2
                -
                row
            )
            *
            (
                panel_h
                +
                gap_y
            )
        )

        for col in range(2):

            x = (
                left0
                +
                col
                *
                (
                    panel_w
                    +
                    gap_x
                )
            )

            axes[
                row,
                col
            ] = fig.add_axes(
                [
                    x,
                    y,
                    panel_w,
                    panel_h,
                ]
            )

    return (
        fig,
        axes,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =========================================================================
    # LOAD LUT
    # =========================================================================

    lut = load_lut(
        LUT_FILE
    )


    # =========================================================================
    # AUTOMATIC SECOND ROW
    # =========================================================================

    matching_log10_g = (
        find_matching_log10_g(
            lut=lut,
            log10_m=(
                MATCH_ROW_LOG10_M
            ),
        )
    )


    # =========================================================================
    # FINAL ROW ORDER
    # =========================================================================

    row_points = [

        # Row 1
        FIRST_ROW_POINT,

        # Row 2
        (
            MATCH_ROW_LOG10_M,
            matching_log10_g,
        ),

        # Row 3
        THIRD_ROW_POINT,
    ]


    # =========================================================================
    # INFO
    # =========================================================================

    print(
        "============================================================"
    )

    print(
        "V_LRI(L) vs V_CC(L)"
    )

    print(
        "============================================================"
    )

    print(
        f"LUT: {LUT_FILE}"
    )

    print()

    print(
        "Final row ordering:"
    )

    for i, (
        log10_m,
        log10_g,
    ) in enumerate(
        row_points,
        start=1,
    ):

        print(
            f"  row {i}: "
            f"log10(m/eV)="
            f"{log10_m:.4f}, "
            f"log10(g)="
            f"{log10_g:.6f}"
        )

    print()

    print(
        "Geometry:"
    )

    print(
        f"  panel size       = "
        f"{PANEL_SIZE_IN:.2f} in"
    )

    print(
        f"  column gap       = "
        f"{COLUMN_GAP_IN:.2f} in"
    )

    print(
        f"  row gap          = "
        f"{ROW_GAP_IN:.2f} in"
    )

    print(
        f"  figure size      = "
        f"{FIG_WIDTH_IN:.2f} x "
        f"{FIG_HEIGHT_IN:.2f} in"
    )

    print()


    # =========================================================================
    # CREATE FIGURE
    # =========================================================================

    fig, axes = (
        create_axes()
    )


    # =========================================================================
    # DRAW ROWS
    # =========================================================================

    for row, (
        log10_m,
        log10_g,
    ) in enumerate(
        row_points
    ):

        row_results = []

        for coszen in COSZENS:

            result = (
                prepare_panel_data(
                    lut=lut,
                    coszen=coszen,
                    log10_m=log10_m,
                    log10_g=log10_g,
                )
            )

            row_results.append(
                result
            )


        # Same Y range for both columns in each row
        row_ylim = (
            get_row_ylim(
                row_results
            )
        )


        for col, coszen in enumerate(
            COSZENS
        ):

            draw_panel(

                ax=(
                    axes[
                        row,
                        col
                    ]
                ),

                result=(
                    row_results[
                        col
                    ]
                ),

                log10_m=(
                    log10_m
                ),

                log10_g=(
                    log10_g
                ),

                coszen=(
                    coszen
                ),

                ylim=(
                    row_ylim
                ),
            )


    # =========================================================================
    # REMOVE REPEATED TICK LABELS
    # =========================================================================

    # Only bottom row shows x tick labels.
    for row in range(2):

        for col in range(2):

            axes[
                row,
                col
            ].tick_params(
                labelbottom=False
            )


    # Only left column shows y tick labels.
    for row in range(3):

        axes[
            row,
            1
        ].tick_params(
            labelleft=False
        )


    # =========================================================================
    # GLOBAL X LABEL
    # =========================================================================
    #
    # DISTANCE FROM PANELS:
    #
    # controlled mainly by
    #
    #     X_AXIS_LABEL_Y
    #
    # Increase it -> closer to the bottom plots.
    # Decrease it -> farther away.
    #
    # BOTTOM_MARGIN_IN also contributes.
    # =========================================================================

    fig.text(
        0.515,
        X_AXIS_LABEL_Y,

        r"Neutrino path length, $L$ "
        r"[$10^3\,\mathrm{km}$]",

        ha="center",
        va="bottom",

        fontsize=11.0,
    )


    # =========================================================================
    # GLOBAL Y LABEL
    # =========================================================================
    #
    # DISTANCE FROM PANELS:
    #
    # controlled mainly by
    #
    #     Y_AXIS_LABEL_X
    #
    # Increase it -> closer to the left panels.
    # Decrease it -> farther left.
    #
    # LEFT_MARGIN_IN also contributes.
    # =========================================================================

    fig.text(
        Y_AXIS_LABEL_X,
        0.50,

        r"Matter potential, $V(L)$ "
        r"[$10^{-14}\,\mathrm{eV}$]",

        ha="center",
        va="center",

        rotation="vertical",

        fontsize=11.0,
    )


    # =========================================================================
    # LEGEND
    # =========================================================================

    legend_handles = [

        Line2D(
            [0],
            [0],

            color=LRI_COLOR,

            linestyle=LRI_LINESTYLE,

            linewidth=LRI_LINEWIDTH,

            label=r"$V_{\rm LRI}$",
        ),

        Line2D(
            [0],
            [0],

            color=STD_COLOR,

            linestyle=STD_LINESTYLE,

            linewidth=STD_LINEWIDTH,

            label=r"$V_{\mathrm{CC}}$",
        ),
    ]


    legend = axes[
        0,
        1
    ].legend(

        handles=(
            legend_handles
        ),

        loc="upper right",

        # Below cos(theta_z)
        bbox_to_anchor=(
            0.965,
            0.80,
        ),

        frameon=True,

        facecolor="white",

        framealpha=0.94,

        edgecolor="black",

        fontsize=6.9,

        handlelength=2.5,

        borderpad=0.40,

        labelspacing=0.30,

        borderaxespad=0.0,
    )


    legend.get_frame().set_linewidth(
        0.65
    )


    # =========================================================================
    # SAVE
    # =========================================================================

    fig.savefig(
        OUTPUT_PNG,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.025,
    )


    fig.savefig(
        OUTPUT_PDF,

        bbox_inches="tight",

        pad_inches=0.025,
    )


    plt.close(
        fig
    )


    print(
        f"Saved: {OUTPUT_PNG}"
    )

    print(
        f"Saved: {OUTPUT_PDF}"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()

