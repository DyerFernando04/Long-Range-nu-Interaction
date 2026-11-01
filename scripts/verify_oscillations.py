#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import shlex
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

CASE = "Le-Lmu"   # "Le-Lmu", "Le-Ltau", "Lmu-Ltau"

CPP_SOURCE = SCRIPT_DIR / "oscillogram_pee.cpp"
BUILD_DIR = SCRIPT_DIR / ".oscillogram_build"

OUTPUT_DIR = (
    ROOT
    / "FIGURES"
    / "oscillation_verification"
    / CASE
)

DATA_DIR = OUTPUT_DIR / "data"

N_ENERGY = 180
N_COSZEN = 101

E_MIN_GEV = 1.0
E_MAX_GEV = 200.0

# Test 1: turn LRI down by shortening its range.
MASS_SCAN_LOG10_G = -25.0

MASS_SCAN = (
    -15.0,
    -14.0,
    -13.0,
    -12.0,
    -11.0,
)

# Test 2: clean control, because V_LRI ~ g'^2.
G_SCAN_LOG10_M = -13.0

G_SCAN = (
    -25.0,
    -26.0,
    -27.0,
    -28.0,
    -30.0,
)

SLICE_COSZENS = (
    -1.0,
    -0.4,
)

MODELS = {
    "Le-Lmu": {
        "include_dir": ROOT / "LUT_FLUX",
        "lut": ROOT / "LUT_VLRI" / "vlri_unit.h5",
        "label": r"$L_e-L_\mu$",
    },

    "Le-Ltau": {
        "include_dir": ROOT / "LUT_FLUX" / "Le-Ltau",
        "lut": ROOT / "LUT_VLRI" / "vlri_unit.h5",
        "label": r"$L_e-L_\tau$",
    },

    "Lmu-Ltau": {
        "include_dir": ROOT / "LUT_FLUX" / "Lmu-Ltau",
        "lut": ROOT / "LUT_VLRI" / "Lmu-Ltau" / "vlri_unit.h5",
        "label": r"$L_\mu-L_\tau$",
    },
}


def pkg_config(flag):
    text = subprocess.check_output(
        ["pkg-config", flag, "nusquids"],
        text=True,
    ).strip()

    return shlex.split(text)


def compile_helper():
    model = MODELS[CASE]

    if not CPP_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing C++ source: {CPP_SOURCE}"
        )

    if not (model["include_dir"] / "LRI.h").is_file():
        raise FileNotFoundError(
            f"Missing LRI.h: {model['include_dir'] / 'LRI.h'}"
        )

    if not model["lut"].is_file():
        raise FileNotFoundError(
            f"Missing LUT1: {model['lut']}"
        )

    BUILD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exe = BUILD_DIR / f"oscillogram_pee_{CASE}"

    command = (
        [
            "g++",
            "-O3",
            "-std=c++17",
            str(CPP_SOURCE),
            f"-I{model['include_dir']}",
            f"-I{ROOT / 'LUT_FLUX'}",
        ]
        + pkg_config("--cflags")
        + pkg_config("--libs")
        + [
            "-lhdf5_hl",
            "-o",
            str(exe),
        ]
    )

    print("Compiling:")
    print(" ".join(shlex.quote(x) for x in command))

    subprocess.run(
        command,
        check=True,
    )

    return exe


def safe_tag(value):
    return (
        f"{value:.2f}"
        .replace("-", "m")
        .replace(".", "p")
    )


def scenario_filename(mode, log10_m):
    if mode == "SM":
        name = f"SM_m_{safe_tag(log10_m)}"
    else:
        name = (
            f"g_{safe_tag(float(mode))}"
            f"_m_{safe_tag(log10_m)}"
        )

    return DATA_DIR / f"{name}.csv"


def run_scenario(exe, mode, log10_m):
    filename = scenario_filename(
        mode,
        log10_m,
    )

    if filename.is_file():
        print(f"Using cached: {filename.name}")
        return filename

    model = MODELS[CASE]

    command = [
        str(exe),
        str(model["lut"]),
        str(filename),
        str(mode),
        str(log10_m),
        str(N_ENERGY),
        str(N_COSZEN),
        str(E_MIN_GEV),
        str(E_MAX_GEV),
    ]

    print()
    print("Running:")
    print(" ".join(shlex.quote(x) for x in command))

    subprocess.run(
        command,
        check=True,
    )

    return filename


def load_grid(filename, probability_column):
    df = pd.read_csv(filename)

    coszen = np.sort(
        df["coszen"].unique()
    )

    energy = np.sort(
        df["energy_GeV"].unique()
    )

    pivot = df.pivot(
        index="coszen",
        columns="energy_GeV",
        values=probability_column,
    )

    pivot = pivot.reindex(
        index=coszen,
        columns=energy,
    )

    values = pivot.to_numpy(
        dtype=float
    )

    if np.any(~np.isfinite(values)):
        raise RuntimeError(
            f"Non-finite values in {filename}"
        )

    return coszen, energy, values


def make_scan_oscillograms(
    sm_file,
    scenarios,
    probability_column,
    particle_label,
    scan_name,
):
    sm_cz, sm_e, sm = load_grid(
        sm_file,
        probability_column,
    )

    all_items = [("Standard", sm)]

    for label, filename in scenarios:
        cz, energy, values = load_grid(
            filename,
            probability_column,
        )

        if not (
            np.array_equal(cz, sm_cz)
            and np.array_equal(energy, sm_e)
        ):
            raise RuntimeError("Grid mismatch.")

        all_items.append(
            (label, values)
        )

    n = len(all_items)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.0, 4.0 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).reshape(-1)

    norm = Normalize(
        vmin=0.0,
        vmax=1.0,
    )

    mesh = None

    for ax, (label, values) in zip(
        axes,
        all_items,
    ):
        mesh = ax.pcolormesh(
            sm_e,
            sm_cz,
            values,
            shading="auto",
            cmap="plasma",
            norm=norm,
        )

        ax.set_xscale("log")
        ax.set_ylim(-1.0, 0.0)
        ax.set_xlim(E_MIN_GEV, E_MAX_GEV)
        ax.set_title(label)
        ax.grid(alpha=0.12)

    for ax in axes[n:]:
        ax.axis("off")

    for row in range(nrows):
        axes[row * ncols].set_ylabel(
            r"$\cos\theta_z$"
        )

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel(
                r"$E_\nu$ [GeV]"
            )

    cbar = fig.colorbar(
        mesh,
        ax=[ax for ax in axes if ax.get_visible()],
        fraction=0.025,
        pad=0.025,
    )

    cbar.set_label(
        particle_label
    )

    fig.suptitle(
        MODELS[CASE]["label"]
        + " — "
        + scan_name,
        fontsize=16,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.08,
        top=0.91,
        wspace=0.14,
        hspace=0.18,
    )

    tag = (
        "nu"
        if probability_column == "Pee_nu"
        else
        "antinu"
    )

    output = (
        OUTPUT_DIR
        / f"Pee_{scan_name}_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def make_difference_oscillograms(
    sm_file,
    scenarios,
    probability_column,
    scan_name,
):
    sm_cz, sm_e, sm = load_grid(
        sm_file,
        probability_column,
    )

    differences = []

    for label, filename in scenarios:
        _, _, values = load_grid(
            filename,
            probability_column,
        )

        differences.append(
            (
                label,
                values - sm,
            )
        )

    vmax = max(
        float(np.max(np.abs(delta)))
        for _, delta in differences
    )

    if vmax <= 0.0:
        vmax = 1.0

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0.0,
        vmax=vmax,
    )

    n = len(differences)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.0, 4.0 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(axes).reshape(-1)

    mesh = None

    for ax, (label, delta) in zip(
        axes,
        differences,
    ):
        mesh = ax.pcolormesh(
            sm_e,
            sm_cz,
            delta,
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )

        ax.set_xscale("log")
        ax.set_ylim(-1.0, 0.0)
        ax.set_xlim(E_MIN_GEV, E_MAX_GEV)
        ax.set_title(label)
        ax.grid(alpha=0.12)

    for ax in axes[n:]:
        ax.axis("off")

    for row in range(nrows):
        axes[row * ncols].set_ylabel(
            r"$\cos\theta_z$"
        )

    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel(
                r"$E_\nu$ [GeV]"
            )

    cbar = fig.colorbar(
        mesh,
        ax=[ax for ax in axes if ax.get_visible()],
        fraction=0.025,
        pad=0.025,
    )

    cbar.set_label(
        r"$\Delta P_{ee}"
        r"=P_{ee}^{\rm LRI}"
        r"-P_{ee}^{\rm SM}$"
    )

    particle = (
        r"$\nu_e$"
        if probability_column == "Pee_nu"
        else
        r"$\bar{\nu}_e$"
    )

    fig.suptitle(
        MODELS[CASE]["label"]
        + " — "
        + scan_name
        + " — "
        + particle,
        fontsize=16,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.08,
        top=0.91,
        wspace=0.14,
        hspace=0.18,
    )

    tag = (
        "nu"
        if probability_column == "Pee_nu"
        else
        "antinu"
    )

    output = (
        OUTPUT_DIR
        / f"deltaPee_{scan_name}_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def convergence_metrics(
    sm_file,
    scenarios,
    probability_column,
    scan_name,
):
    _, _, sm = load_grid(
        sm_file,
        probability_column,
    )

    rows = []

    for label, filename in scenarios:
        _, _, values = load_grid(
            filename,
            probability_column,
        )

        delta = values - sm

        rows.append(
            {
                "scan": scan_name,
                "particle": (
                    "nu"
                    if probability_column == "Pee_nu"
                    else
                    "antinu"
                ),
                "scenario": label,
                "max_abs_delta_Pee": float(
                    np.max(np.abs(delta))
                ),
                "mean_abs_delta_Pee": float(
                    np.mean(np.abs(delta))
                ),
                "rms_delta_Pee": float(
                    np.sqrt(
                        np.mean(delta * delta)
                    )
                ),
            }
        )

    return rows


def nearest_coszen_index(
    coszen,
    requested,
):
    return int(
        np.argmin(
            np.abs(coszen - requested)
        )
    )


def make_slice_plot(
    sm_file,
    scenarios,
    probability_column,
    scan_name,
):
    coszen, energy, sm = load_grid(
        sm_file,
        probability_column,
    )

    fig, axes = plt.subplots(
        1,
        len(SLICE_COSZENS),
        figsize=(12.5, 4.8),
        sharey=True,
    )

    if len(SLICE_COSZENS) == 1:
        axes = [axes]

    for ax, requested_cz in zip(
        axes,
        SLICE_COSZENS,
    ):
        iz = nearest_coszen_index(
            coszen,
            requested_cz,
        )

        actual_cz = coszen[iz]

        ax.plot(
            energy,
            sm[iz, :],
            linewidth=2.4,
            label="Standard",
        )

        for label, filename in scenarios:
            _, _, values = load_grid(
                filename,
                probability_column,
            )

            ax.plot(
                energy,
                values[iz, :],
                linewidth=1.5,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(
            r"$E_\nu$ [GeV]"
        )
        ax.set_title(
            rf"$\cos\theta_z={actual_cz:.2f}$"
        )
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(
        (
            r"$P_{ee}$"
            if probability_column == "Pee_nu"
            else
            r"$\bar P_{ee}$"
        )
    )

    axes[-1].legend(
        fontsize=8,
    )

    fig.suptitle(
        MODELS[CASE]["label"]
        + " — "
        + scan_name,
        fontsize=15,
    )

    fig.tight_layout()

    tag = (
        "nu"
        if probability_column == "Pee_nu"
        else
        "antinu"
    )

    output = (
        OUTPUT_DIR
        / f"Pee_slices_{scan_name}_{tag}.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


def write_metrics(rows):
    filename = (
        OUTPUT_DIR
        / "convergence_to_standard.csv"
    )

    with filename.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {filename}")


def print_metrics(rows):
    print()
    print("=" * 100)
    print("CONVERGENCE TO STANDARD")
    print("=" * 100)

    print(
        f"{'scan':<14}"
        f"{'particle':<10}"
        f"{'scenario':<28}"
        f"{'max|dP|':>14}"
        f"{'mean|dP|':>14}"
        f"{'RMS(dP)':>14}"
    )

    print("-" * 100)

    for row in rows:
        print(
            f"{row['scan']:<14}"
            f"{row['particle']:<10}"
            f"{row['scenario']:<28}"
            f"{row['max_abs_delta_Pee']:>14.6e}"
            f"{row['mean_abs_delta_Pee']:>14.6e}"
            f"{row['rms_delta_Pee']:>14.6e}"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("P_ee oscillation verification")
    print("=" * 72)

    print(f"Model = {CASE}")
    print(f"E     = {E_MIN_GEV} -> {E_MAX_GEV} GeV")
    print(
        f"Grid  = {N_COSZEN} coszen x {N_ENERGY} energies"
    )

    exe = compile_helper()

    # Exact SM: g'=0.
    sm_file = run_scenario(
        exe,
        "SM",
        G_SCAN_LOG10_M,
    )

    mass_scenarios = []

    for logm in MASS_SCAN:
        filename = run_scenario(
            exe,
            MASS_SCAN_LOG10_G,
            logm,
        )

        mass_scenarios.append(
            (
                rf"$\log_{{10}}m'={logm:.0f}$",
                filename,
            )
        )

    g_scenarios = []

    for logg in G_SCAN:
        filename = run_scenario(
            exe,
            logg,
            G_SCAN_LOG10_M,
        )

        g_scenarios.append(
            (
                rf"$\log_{{10}}g'={logg:.0f}$",
                filename,
            )
        )

    all_metrics = []

    for probability_column, particle_label in (
        (
            "Pee_nu",
            r"$P(\nu_e\to\nu_e)$",
        ),
        (
            "Pee_antinu",
            r"$P(\bar\nu_e\to\bar\nu_e)$",
        ),
    ):
        # Mass scan
        make_scan_oscillograms(
            sm_file,
            mass_scenarios,
            probability_column,
            particle_label,
            "mass_scan",
        )

        make_difference_oscillograms(
            sm_file,
            mass_scenarios,
            probability_column,
            "mass_scan",
        )

        make_slice_plot(
            sm_file,
            mass_scenarios,
            probability_column,
            "mass_scan",
        )

        all_metrics.extend(
            convergence_metrics(
                sm_file,
                mass_scenarios,
                probability_column,
                "mass_scan",
            )
        )

        # Coupling scan
        make_scan_oscillograms(
            sm_file,
            g_scenarios,
            probability_column,
            particle_label,
            "g_scan",
        )

        make_difference_oscillograms(
            sm_file,
            g_scenarios,
            probability_column,
            "g_scan",
        )

        make_slice_plot(
            sm_file,
            g_scenarios,
            probability_column,
            "g_scan",
        )

        all_metrics.extend(
            convergence_metrics(
                sm_file,
                g_scenarios,
                probability_column,
                "g_scan",
            )
        )

    write_metrics(all_metrics)
    print_metrics(all_metrics)

    print()
    print("Expected consistency checks:")
    print(
        "  1) log10(g)=-30 should be nearly indistinguishable from exact SM."
    )
    print(
        "  2) max|DeltaPee| and RMS(DeltaPee) should fall strongly as g decreases."
    )
    print(
        "  3) At sufficiently large m, the mass scan should move toward SM as "
        "the Yukawa range becomes shorter."
    )
    print(
        "  4) The 1D slices at coszen=-1 and -0.4 should show the SM minima "
        "reappearing continuously as LRI is switched off."
    )
    print(
        "  5) The C++ helper checks Pee+Pemu+Petau=1 at every point."
    )

    print()
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

