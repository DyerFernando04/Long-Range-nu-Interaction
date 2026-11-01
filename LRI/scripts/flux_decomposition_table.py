#!/usr/bin/env python3

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

CASE = "Le-Lmu"

LOG10_M = -13.0
LOG10_G = -25.0

COSZEN = -1.0

ENERGIES_GEV = [
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    100.0,
    200.0,
]


INITIAL_FLUX_FILE = (
    ROOT
    / "INITIAL_FLUX"
    / "mceq_initial_fluxes1.h5"
)

FINAL_FLUX_FILE = (
    ROOT
    / "RESULTS"
    / CASE
    / "oscillated_flux_lut.h5"
)

PROBABILITY_FILE = (
    ROOT
    / "FIGURES"
    / "probability"
    / "data"
    / "probabilities_Le-Lmu_cz_m1p00.csv"
)


# =============================================================================
# Probability names
# =============================================================================

# PROBABILITY_NAME[final][initial]

PROBABILITY_NAME = {
    "e": {
        "e": "Pee",
        "mu": "Pmue",
    },
    "mu": {
        "e": "Pemu",
        "mu": "Pmumu",
    },
}


# =============================================================================
# Helpers
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
        - requested
    )

    if difference > tolerance:
        raise ValueError(
            f"{name}={requested} not found. "
            f"Nearest value = {axis[index]}"
        )

    return index


def interpolate_flux(
    energy_axis,
    coszen_axis,
    flux,
    energy_GeV,
    coszen,
):
    energy_axis = np.asarray(
        energy_axis,
        dtype=np.float64,
    )

    coszen_axis = np.asarray(
        coszen_axis,
        dtype=np.float64,
    )

    flux = np.asarray(
        flux,
        dtype=np.float64,
    )

    positive = flux[
        flux > 0.0
    ]

    if positive.size == 0:
        return 0.0

    floor = max(
        np.finfo(np.float64).tiny,
        float(np.min(positive))
        * 1.0e-12,
    )

    log_flux = np.log(
        np.maximum(
            flux,
            floor,
        )
    )

    log_energy_axis = np.log10(
        energy_axis
    )

    log_energy = np.log10(
        energy_GeV
    )

    values_at_energy = np.array(
        [
            np.interp(
                log_energy,
                log_energy_axis,
                row,
            )
            for row in log_flux
        ]
    )

    value = np.interp(
        coszen,
        coszen_axis,
        values_at_energy,
    )

    return float(
        np.exp(value)
    )


def interpolate_probability(
    table,
    column,
    energy_GeV,
):
    energy = np.asarray(
        table["energy_GeV"],
        dtype=np.float64,
    )

    probability = np.asarray(
        table[column],
        dtype=np.float64,
    )

    return float(
        np.interp(
            np.log10(energy_GeV),
            np.log10(energy),
            probability,
        )
    )


# =============================================================================
# Load initial flux
# =============================================================================

def load_initial_flux():

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

        flux = {
            "e": np.asarray(
                h5[
                    "/flux/nue"
                ][...],
                dtype=np.float64,
            ),

            "mu": np.asarray(
                h5[
                    "/flux/numu"
                ][...],
                dtype=np.float64,
            ),
        }

    return (
        energy,
        coszen,
        flux,
    )


# =============================================================================
# Load final LUT
# =============================================================================

def load_final_flux():

    with h5py.File(
        FINAL_FLUX_FILE,
        "r",
    ) as h5:

        masses = np.asarray(
            h5[
                "/axes/log10_mass_eV"
            ][...],
            dtype=np.float64,
        )

        couplings = np.asarray(
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
            masses,
            LOG10_M,
            "log10(m/eV)",
        )

        lri_index = exact_index(
            couplings,
            LOG10_G,
            "log10(g)",
        )

        sm_index = int(
            np.argmin(couplings)
        )

        output = {
            "SM": {},
            "LRI": {},
        }

        for flavor, name in (
            ("e", "nue"),
            ("mu", "numu"),
        ):

            dataset = h5[
                f"/flux/{name}"
            ]

            output[
                "SM"
            ][flavor] = np.asarray(
                dataset[
                    mass_index,
                    sm_index,
                    :,
                    :,
                ],
                dtype=np.float64,
            )

            output[
                "LRI"
            ][flavor] = np.asarray(
                dataset[
                    mass_index,
                    lri_index,
                    :,
                    :,
                ],
                dtype=np.float64,
            )

    return (
        energy,
        coszen,
        output,
    )


# =============================================================================
# Build table
# =============================================================================

def build_table(
    scenario,
    final_flavor,
    probability_table,
    initial_energy,
    initial_coszen,
    initial_flux,
    final_energy,
    final_coszen,
    final_flux,
):

    prefix = (
        "sm_"
        if scenario == "SM"
        else "lri_"
    )

    other_flavor = (
        "mu"
        if final_flavor == "e"
        else "e"
    )

    rows = []

    for energy in ENERGIES_GEV:

        # -----------------------------------------------------
        # Initial fluxes
        # -----------------------------------------------------

        phi0_final = interpolate_flux(
            initial_energy,
            initial_coszen,
            initial_flux[
                final_flavor
            ],
            energy,
            COSZEN,
        )

        phi0_other = interpolate_flux(
            initial_energy,
            initial_coszen,
            initial_flux[
                other_flavor
            ],
            energy,
            COSZEN,
        )

        # -----------------------------------------------------
        # Probabilities
        # -----------------------------------------------------

        survival_column = (
            prefix
            + PROBABILITY_NAME[
                final_flavor
            ][
                final_flavor
            ]
        )

        appearance_column = (
            prefix
            + PROBABILITY_NAME[
                final_flavor
            ][
                other_flavor
            ]
        )

        p_survival = (
            interpolate_probability(
                probability_table,
                survival_column,
                energy,
            )
        )

        p_appearance = (
            interpolate_probability(
                probability_table,
                appearance_column,
                energy,
            )
        )

        # -----------------------------------------------------
        # Survival / loss / appearance
        # -----------------------------------------------------

        surviving_flux = (
            phi0_final
            * p_survival
        )

        loss = (
            phi0_final
            - surviving_flux
        )

        appearance = (
            phi0_other
            * p_appearance
        )

        # -----------------------------------------------------
        # Actual final flux from LUT
        # -----------------------------------------------------

        phi_final = interpolate_flux(
            final_energy,
            final_coszen,
            final_flux[
                scenario
            ][
                final_flavor
            ],
            energy,
            COSZEN,
        )

        flux_ratio = (
            phi_final
            / phi0_final
        )

        # -----------------------------------------------------
        # Compensation
        # -----------------------------------------------------

        if loss > 0.0:
            compensation = (
                appearance
                / loss
            )
        else:
            compensation = np.nan

        # Net change relative to original flux.
        flux_change_percent = (
            100.0
            * (
                phi_final
                / phi0_final
                - 1.0
            )
        )

        rows.append(
            {
                "E [GeV]":
                    energy,

                "P_surv":
                    p_survival,

                "Phi_final/Phi0":
                    flux_ratio,

                "loss":
                    loss,

                "appearance":
                    appearance,

                "appearance/loss":
                    compensation,

                "flux change [%]":
                    flux_change_percent,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Print
# =============================================================================

def print_table(
    scenario,
    flavor,
    table,
):

    label = (
        "nu_e"
        if flavor == "e"
        else "nu_mu"
    )

    print()
    print("=" * 110)

    print(
        f"{scenario} | "
        f"{label} | "
        f"coszen = {COSZEN:+.2f}"
    )

    print("=" * 110)

    formatters = {

        "E [GeV]":
            lambda x:
            f"{x:7.1f}",

        "P_surv":
            lambda x:
            f"{x:8.4f}",

        "Phi_final/Phi0":
            lambda x:
            f"{x:10.4f}",

        "loss":
            lambda x:
            f"{x:11.3e}",

        "appearance":
            lambda x:
            f"{x:11.3e}",

        "appearance/loss":
            lambda x:
            f"{x:10.4f}",

        "flux change [%]":
            lambda x:
            f"{x:+9.2f}",
    }

    print(
        table.to_string(
            index=False,
            formatters=formatters,
        )
    )


# =============================================================================
# Main
# =============================================================================

def main():

    probability_table = pd.read_csv(
        PROBABILITY_FILE
    )

    (
        initial_energy,
        initial_coszen,
        initial_flux,
    ) = load_initial_flux()

    (
        final_energy,
        final_coszen,
        final_flux,
    ) = load_final_flux()

    print()
    print(
        f"Model: {CASE} | "
        f"log10(m/eV)={LOG10_M} | "
        f"log10(g)={LOG10_G}"
    )

    for scenario in (
        "SM",
        "LRI",
    ):

        for flavor in (
            "e",
            "mu",
        ):

            table = build_table(
                scenario,
                flavor,
                probability_table,
                initial_energy,
                initial_coszen,
                initial_flux,
                final_energy,
                final_coszen,
                final_flux,
            )

            print_table(
                scenario,
                flavor,
                table,
            )


if __name__ == "__main__":
    main()
