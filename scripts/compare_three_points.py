#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "three_point_all_LRI"
)

FILES = {
    "Le-Lmu": (
        ROOT
        / "RESULTS"
        / "Le-Lmu"
        / "event_templates_effective.h5"
    ),

    "Le-Ltau": (
        ROOT
        / "RESULTS"
        / "Le-Ltau"
        / "event_templates_effective.h5"
    ),

    "Lmu-Ltau": (
        ROOT
        / "RESULTS"
        / "Lmu-Ltau"
        / "event_templates_effective.h5"
    ),
}


LABELS = {
    "Le-Lmu": r"$L_e-L_\mu$",
    "Le-Ltau": r"$L_e-L_\tau$",
    "Lmu-Ltau": r"$L_\mu-L_\tau$",
}


# =============================================================================
# SAME THREE POINTS FOR ALL MODELS
# =============================================================================

POINTS = [
    {
        "name": "A",
        "log10_m": -15.0,
        "log10_g": -25.4,
    },

    {
        "name": "B",
        "log10_m": -13.0,
        "log10_g": -25.0,
    },

    {
        "name": "C",
        "log10_m": -12.0,
        "log10_g": -24.2,
    },
]


# =============================================================================
# DETECTOR CONFIG
# =============================================================================

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

# Mejor dejarlo False para aislar el cambio debido a neutrinos/LRI.
INCLUDE_ATMOSPHERIC_MUON = False

FLOOR_FRACTION = 1.0e-12

DELTA_VISUAL_PERCENTILE = 99.0


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
                axis - requested
            )
        )
    )

    difference = abs(
        float(axis[index])
        -
        float(requested)
    )

    if difference > tolerance:
        raise ValueError(
            f"{name}={requested} no es un nodo exacto.\n"
            f"Nodo más cercano = {axis[index]}"
        )

    return index


def robust_abs_max(
    arrays,
    percentile=99.0,
):
    values = []

    for array in arrays:

        finite = np.asarray(
            array,
            dtype=np.float64,
        )

        finite = finite[
            np.isfinite(
                finite
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

    result = float(
        np.percentile(
            values,
            percentile,
        )
    )

    if result <= 0.0:
        result = 1.0

    return result


# =============================================================================
# DETECTOR-LEVEL PREDICTION
# =============================================================================

def build_prediction(
    h5,
    mass_index,
    g_index,
):

    first = np.asarray(
        h5[
            f"templates/{CATEGORIES[0]}"
        ][
            mass_index,
            g_index,
            :
        ],
        dtype=np.float64,
    )

    total = np.zeros_like(
        first,
        dtype=np.float64,
    )

    for category in CATEGORIES:

        template = np.asarray(
            h5[
                f"templates/{category}"
            ][
                mass_index,
                g_index,
                :
            ],
            dtype=np.float64,
        )

        if APPLY_NOMINAL_DETECTOR_HYPERSURFACE:

            intercept = np.asarray(
                h5[
                    f"hypersurfaces/"
                    f"{category}/intercept"
                ][...],
                dtype=np.float64,
            )

            template = (
                template
                * intercept
            )

        total += template

    if INCLUDE_ATMOSPHERIC_MUON:

        total += np.asarray(
            h5[
                "backgrounds/atmospheric_muon"
            ][...],
            dtype=np.float64,
        )

    hist_shape = tuple(
        int(x)
        for x in h5.attrs[
            "hist_shape"
        ]
    )

    return total.reshape(
        hist_shape
    )


# =============================================================================
# LOAD ONE MODEL AT ONE POINT
# =============================================================================

def load_model_point(
    model,
    point,
):

    filename = FILES[
        model
    ]

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
                "axes/log10_mass_eV"
            ][...],
            dtype=np.float64,
        )

        log10_g = np.asarray(
            h5[
                "axes/log10_gprime"
            ][...],
            dtype=np.float64,
        )

        energy_edges = np.asarray(
            h5[
                "axes/reco_energy_edges_GeV"
            ][...],
            dtype=np.float64,
        )

        coszen_edges = np.asarray(
            h5[
                "axes/reco_coszen_edges"
            ][...],
            dtype=np.float64,
        )

        im = exact_index(
            log10_mass,
            point[
                "log10_m"
            ],
            "log10(m/eV)",
        )

        ig = exact_index(
            log10_g,
            point[
                "log10_g"
            ],
            "log10(g)",
        )

        # Standard efectivo:
        ig_sm = int(
            np.argmin(
                log10_g
            )
        )

        standard = build_prediction(
            h5,
            im,
            ig_sm,
        )

        lri = build_prediction(
            h5,
            im,
            ig,
        )

    return {
        "standard": standard,
        "lri": lri,

        "delta": (
            lri
            -
            standard
        ),

        "energy_edges": energy_edges,
        "coszen_edges": coszen_edges,

        "log10_m": float(
            log10_mass[
                im
            ]
        ),

        "log10_g": float(
            log10_g[
                ig
            ]
        ),

        "log10_g_sm": float(
            log10_g[
                ig_sm
            ]
        ),
    }


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    model,
    point_name,
    data,
    pid_index,
):

    topology = PID_LABELS[
        pid_index
    ]

    sm = data[
        "standard"
    ][
        :,
        :,
        pid_index
    ]

    lri = data[
        "lri"
    ][
        :,
        :,
        pid_index
    ]

    delta = (
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

    # ---------------------------------------------------------
    # Net change
    # ---------------------------------------------------------

    signed_total_percent = (
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

    # ---------------------------------------------------------
    # x or y:
    #
    # sum |Delta N| / sum N_SM
    # ---------------------------------------------------------

    absolute_redistribution_percent = (
        100.0
        *
        np.sum(
            np.abs(
                delta
            )
        )
        /
        n_sm
    )

    # ---------------------------------------------------------
    # Avoid zero bins
    # ---------------------------------------------------------

    floor = (
        FLOOR_FRACTION
        *
        np.max(
            sm
        )
    )

    valid = (
        sm
        >
        floor
    )

    # ---------------------------------------------------------
    # Mean absolute fractional change
    # ---------------------------------------------------------

    mean_abs_fractional_change_percent = (
        100.0
        *
        np.mean(
            np.abs(
                delta[
                    valid
                ]
                /
                sm[
                    valid
                ]
            )
        )
    )

    # ---------------------------------------------------------
    # Pearson-like diagnostic
    #
    # NOT the real likelihood
    # ---------------------------------------------------------

    S = float(
        np.sum(
            delta[
                valid
            ] ** 2
            /
            sm[
                valid
            ]
        )
    )

    return {
        "point": point_name,
        "model": model,
        "topology": topology,

        "log10_m": data[
            "log10_m"
        ],

        "log10_g": data[
            "log10_g"
        ],

        "N_SM": n_sm,
        "N_LRI": n_lri,

        "signed_total_percent": (
            signed_total_percent
        ),

        "absolute_redistribution_percent": (
            absolute_redistribution_percent
        ),

        "mean_abs_fractional_change_percent": (
            mean_abs_fractional_change_percent
        ),

        "S": S,
    }


# =============================================================================
# DIRECT 3-MODEL DELTA COMPARISON
# =============================================================================

def plot_three_model_delta(
    point_name,
    results,
):

    arrays = [
        results[
            model
        ][
            "delta"
        ]
        for model in FILES
    ]

    vmax = robust_abs_max(
        arrays,
        percentile=DELTA_VISUAL_PERCENTILE,
    )

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 11.0),
        sharex=True,
        sharey=True,
    )

    mesh = None

    for row, model in enumerate(
        (
            "Le-Lmu",
            "Le-Ltau",
            "Lmu-Ltau",
        )
    ):

        data = results[
            model
        ]

        for pid_index in range(
            2
        ):

            ax = axes[
                row,
                pid_index
            ]

            mesh = ax.pcolormesh(
                data[
                    "energy_edges"
                ],
                data[
                    "coszen_edges"
                ],
                data[
                    "delta"
                ][
                    :,
                    :,
                    pid_index
                ].T,
                shading="auto",
                cmap="RdBu_r",
                norm=norm,
            )

            ax.set_xscale(
                "log"
            )

            ax.grid(
                alpha=0.12
            )

        axes[
            row,
            0
        ].set_ylabel(
            LABELS[
                model
            ]
            +
            "\n"
            +
            r"$\cos\theta_{\rm reco}$"
        )

    axes[
        0,
        0
    ].set_title(
        "Mixed"
    )

    axes[
        0,
        1
    ].set_title(
        "Tracks"
    )

    axes[
        2,
        0
    ].set_xlabel(
        r"$E_{\rm reco}$ [GeV]"
    )

    axes[
        2,
        1
    ].set_xlabel(
        r"$E_{\rm reco}$ [GeV]"
    )

    cbar = fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        fraction=0.028,
        pad=0.03,
    )

    cbar.set_label(
        r"$\Delta N=N_{\rm LRI}-N_{\rm SM}$"
    )

    first = results[
        "Le-Lmu"
    ]

    fig.suptitle(
        f"Point {point_name}\n"
        +
        rf"$\log_{{10}}(m'/\mathrm{{eV}})"
        +
        f"={first['log10_m']:.2f}$, "
        +
        rf"$\log_{{10}}g'"
        +
        f"={first['log10_g']:.2f}$",
        fontsize=16,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.88,
        bottom=0.07,
        top=0.91,
        hspace=0.14,
        wspace=0.12,
    )

    output = (
        OUTPUT_DIR
        /
        f"point_{point_name}_"
        "delta_all_models.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# X, Y, X+Y SUMMARY
# =============================================================================

def plot_xy_summary(
    compact_rows,
):

    lookup = {
        (
            row[
                "point"
            ],
            row[
                "model"
            ],
        ):
        row
        for row in compact_rows
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        sharey=True,
    )

    models = (
        "Le-Lmu",
        "Le-Ltau",
        "Lmu-Ltau",
    )

    positions = np.arange(
        3
    )

    width = 0.24

    offsets = {
        "Le-Lmu": -width,
        "Le-Ltau": 0.0,
        "Lmu-Ltau": width,
    }

    for ax, point in zip(
        axes,
        POINTS,
    ):

        point_name = point[
            "name"
        ]

        for model in models:

            row = lookup[
                (
                    point_name,
                    model,
                )
            ]

            values = [
                row[
                    "x_Mixed_percent"
                ],

                row[
                    "y_Tracks_percent"
                ],

                row[
                    "x_plus_y_percent"
                ],
            ]

            ax.bar(
                positions
                +
                offsets[
                    model
                ],
                values,
                width=width,
                label=LABELS[
                    model
                ],
            )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            (
                "Mixed\n$x$",
                "Tracks\n$y$",
                "$x+y$",
            )
        )

        ax.set_title(
            f"Point {point_name}\n"
            +
            rf"$({point['log10_m']:.1f}, "
            +
            rf"{point['log10_g']:.1f})$"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    axes[
        0
    ].set_ylabel(
        "Absolute redistribution [%]"
    )

    axes[
        -1
    ].legend(
        fontsize=9,
    )

    fig.suptitle(
        "Topology-change size for the three LRI models",
        fontsize=15,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "summary_x_y_xplusy_all_models.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# S_TOTAL SUMMARY
# =============================================================================

def plot_S_summary(
    compact_rows,
):

    lookup = {
        (
            row[
                "point"
            ],
            row[
                "model"
            ],
        ):
        row
        for row in compact_rows
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    models = (
        "Le-Lmu",
        "Le-Ltau",
        "Lmu-Ltau",
    )

    positions = np.arange(
        3
    )

    width = 0.24

    offsets = {
        "Le-Lmu": -width,
        "Le-Ltau": 0.0,
        "Lmu-Ltau": width,
    }

    for ax, point in zip(
        axes,
        POINTS,
    ):

        point_name = point[
            "name"
        ]

        for model in models:

            row = lookup[
                (
                    point_name,
                    model,
                )
            ]

            values = [
                row[
                    "S_Mixed"
                ],

                row[
                    "S_Tracks"
                ],

                row[
                    "S_total"
                ],
            ]

            ax.bar(
                positions
                +
                offsets[
                    model
                ],
                values,
                width=width,
                label=LABELS[
                    model
                ],
            )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            (
                r"$S_{\rm Mixed}$",
                r"$S_{\rm Tracks}$",
                r"$S_{\rm total}$",
            )
        )

        ax.set_title(
            f"Point {point_name}"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    axes[
        0
    ].set_ylabel(
        r"Pearson-like size "
        r"$\sum(\Delta N)^2/N_{\rm SM}$"
    )

    axes[
        -1
    ].legend(
        fontsize=9,
    )

    fig.suptitle(
        "Approximate statistical-size comparison",
        fontsize=15,
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        /
        "summary_S_all_models.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    filename,
    rows,
):

    with filename.open(
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

    detailed_rows = []

    results_all_points = {}

    # -------------------------------------------------------------------------
    # Load and analyze
    # -------------------------------------------------------------------------

    for point in POINTS:

        point_name = point[
            "name"
        ]

        results_all_points[
            point_name
        ] = {}

        for model in (
            "Le-Lmu",
            "Le-Ltau",
            "Lmu-Ltau",
        ):

            print(
                f"Loading Point {point_name}, {model}"
            )

            data = load_model_point(
                model,
                point,
            )

            results_all_points[
                point_name
            ][
                model
            ] = data

            for pid_index in range(
                2
            ):

                detailed_rows.append(
                    calculate_metrics(
                        model,
                        point_name,
                        data,
                        pid_index,
                    )
                )

        plot_three_model_delta(
            point_name,
            results_all_points[
                point_name
            ],
        )

    # -------------------------------------------------------------------------
    # Compact table
    # -------------------------------------------------------------------------

    lookup = {
        (
            row[
                "point"
            ],
            row[
                "model"
            ],
            row[
                "topology"
            ],
        ):
        row
        for row in detailed_rows
    }

    compact_rows = []

    for point in POINTS:

        point_name = point[
            "name"
        ]

        for model in (
            "Le-Lmu",
            "Le-Ltau",
            "Lmu-Ltau",
        ):

            mixed = lookup[
                (
                    point_name,
                    model,
                    "Mixed",
                )
            ]

            tracks = lookup[
                (
                    point_name,
                    model,
                    "Tracks",
                )
            ]

            x = mixed[
                "absolute_redistribution_percent"
            ]

            y = tracks[
                "absolute_redistribution_percent"
            ]

            S_mixed = mixed[
                "S"
            ]

            S_tracks = tracks[
                "S"
            ]

            compact_rows.append(
                {
                    "point": point_name,

                    "model": model,

                    "log10_m": point[
                        "log10_m"
                    ],

                    "log10_g": point[
                        "log10_g"
                    ],

                    "x_Mixed_percent": x,

                    "y_Tracks_percent": y,

                    "x_plus_y_percent": (
                        x
                        +
                        y
                    ),

                    "signed_Mixed_percent": mixed[
                        "signed_total_percent"
                    ],

                    "signed_Tracks_percent": tracks[
                        "signed_total_percent"
                    ],

                    "S_Mixed": S_mixed,

                    "S_Tracks": S_tracks,

                    "S_total": (
                        S_mixed
                        +
                        S_tracks
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------

    write_csv(
        OUTPUT_DIR
        /
        "three_point_all_models_detailed.csv",
        detailed_rows,
    )

    write_csv(
        OUTPUT_DIR
        /
        "three_point_all_models_summary.csv",
        compact_rows,
    )

    # -------------------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------------------

    plot_xy_summary(
        compact_rows
    )

    plot_S_summary(
        compact_rows
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()
    print(
        "Point   Model       "
        "x Mixed[%]   "
        "y Tracks[%]   "
        "x+y[%]       "
        "S Mixed      "
        "S Tracks     "
        "S total"
    )

    print(
        "-"
        *
        100
    )

    for row in compact_rows:

        print(
            f"{row['point']:<7}"
            f"{row['model']:<12}"

            f"{row['x_Mixed_percent']:>11.5f}"
            f"{row['y_Tracks_percent']:>14.5f}"
            f"{row['x_plus_y_percent']:>12.5f}"

            f"{row['S_Mixed']:>13.5f}"
            f"{row['S_Tracks']:>13.5f}"
            f"{row['S_total']:>13.5f}"
        )

    print()
    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
