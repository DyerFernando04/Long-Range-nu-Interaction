#!/usr/bin/env python3

import os
import subprocess

import h5py
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/simulation/LRI"

INITIAL_FILE = os.path.join(
    BASE,
    "INITIAL_FLUX",
    "mceq_initial_fluxes1.h5",
)

SM_FILE = os.path.join(
    BASE,
    "RESULTS",
    "standard",
    "oscillated_flux_std.h5",
)

PROBABILITY_FILE = os.path.join(
    BASE,
    "FIGURES",
    "probability",
    "data",
    "probabilities_Le-Lmu_cz_m1p00_gm24.csv",
)

PROBABILITY_EXECUTABLE = os.path.join(
    BASE,
    "scripts",
    ".probability_build",
    "transition_probabilities_Le-Lmu",
)

VLRI_FILE = os.path.join(
    BASE,
    "LUT_VLRI",
    "vlri_unit.h5",
)

# Benchmark actual de tus figuras
LOG10_M = -13.0
LOG10_G = -24.0
COSZEN = -1.0

# Rango usado por tu plot de probabilidades
N_E_PROB = 240
E_MIN_PROB = 5.0
E_MAX_PROB = 200.0

# Regenerar el CSV para garantizar que LRI use g'=1e-24
REGENERATE_PROBABILITIES = True

# Energías que queremos inspeccionar manualmente
SAMPLE_ENERGIES = np.array(
    [5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0]
)


# ============================================================
# GENERATE CURRENT-BENCHMARK PROBABILITIES
# ============================================================

def generate_probability_file():
    if not REGENERATE_PROBABILITIES:
        return

    if not os.path.isfile(PROBABILITY_EXECUTABLE):
        raise FileNotFoundError(
            "\nNo encuentro el ejecutable:\n"
            f"  {PROBABILITY_EXECUTABLE}\n\n"
            "Primero ejecuta una vez tu script de probabilidades para "
            "compilar transition_probabilities_Le-Lmu."
        )

    cmd = [
        PROBABILITY_EXECUTABLE,
        VLRI_FILE,
        PROBABILITY_FILE,
        str(LOG10_G),
        str(LOG10_M),
        str(COSZEN),
        str(N_E_PROB),
        str(E_MIN_PROB),
        str(E_MAX_PROB),
    ]

    print("=" * 80)
    print("GENERATING PROBABILITIES")
    print("=" * 80)
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


# ============================================================
# GENERIC HDF5 FLUX INTERPOLATION
# ============================================================

def load_flux_grid(filename, species):
    with h5py.File(filename, "r") as f:
        energy = np.asarray(f["/axes/energy_GeV"][...], dtype=float)
        coszen = np.asarray(f["/axes/coszen"][...], dtype=float)
        flux = np.asarray(f[f"/flux/{species}"][...], dtype=float)

    # Expected convention in this project:
    # flux[coszen, energy]
    if flux.shape == (len(energy), len(coszen)):
        flux = flux.T

    expected = (len(coszen), len(energy))
    if flux.shape != expected:
        raise ValueError(
            f"Unexpected shape for /flux/{species} in {filename}: "
            f"{flux.shape}, expected {expected}"
        )

    return energy, coszen, flux


def interp_flux(filename, species, energy_query, coszen_query):
    """
    Bilinear interpolation in
        log10(E), coszen
    performed on
        log(flux).

    This follows the convention used elsewhere in the LRI analysis.
    """

    energy, coszen, flux = load_flux_grid(filename, species)

    energy_query = np.atleast_1d(energy_query).astype(float)

    tiny = np.finfo(float).tiny
    log_flux = np.log(np.maximum(flux, tiny))

    logE_grid = np.log10(energy)
    logE_query = np.log10(energy_query)

    # Find zenith bracketing nodes
    if coszen_query <= coszen[0]:
        i0 = i1 = 0
        t = 0.0

    elif coszen_query >= coszen[-1]:
        i0 = i1 = len(coszen) - 1
        t = 0.0

    else:
        i1 = np.searchsorted(coszen, coszen_query)
        i0 = i1 - 1

        z0 = coszen[i0]
        z1 = coszen[i1]

        t = (coszen_query - z0) / (z1 - z0)

    # Interpolate in log10(E) at the two zenith nodes
    y0 = np.interp(
        logE_query,
        logE_grid,
        log_flux[i0],
    )

    y1 = np.interp(
        logE_query,
        logE_grid,
        log_flux[i1],
    )

    # Linear interpolation in coszen
    y = (1.0 - t) * y0 + t * y1

    return np.exp(y)


# ============================================================
# PROBABILITY INTERPOLATION
# ============================================================

def load_probabilities():
    df = pd.read_csv(PROBABILITY_FILE)

    required = [
        "energy_GeV",

        "sm_Pee",
        "sm_Pemu",
        "sm_Petau",
        "sm_Pmue",
        "sm_Pmumu",
        "sm_Pmutau",
        "sm_Ptaue",
        "sm_Ptaumu",
        "sm_Ptautau",

        "lri_Pee",
        "lri_Pemu",
        "lri_Petau",
        "lri_Pmue",
        "lri_Pmumu",
        "lri_Pmutau",
        "lri_Ptaue",
        "lri_Ptaumu",
        "lri_Ptautau",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            "Missing columns in probability CSV:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(df.columns)
        )

    return df


def interp_probability(df, column, energies):
    return np.interp(
        np.log10(energies),
        np.log10(df["energy_GeV"].to_numpy()),
        df[column].to_numpy(),
    )


# ============================================================
# ELECTRON-FLUX DECOMPOSITION
# ============================================================

def build_electron_table(prob_df, mode, energies):
    """
    Reconstruct final nu_e flux:

      Phi_e =
          Phi_e^0   P(e -> e)
        + Phi_mu^0  P(mu -> e)
        + Phi_tau^0 P(tau -> e)
    """

    prefix = "sm" if mode == "SM" else "lri"

    # Initial atmospheric fluxes
    phi_e0 = interp_flux(
        INITIAL_FILE,
        "nue",
        energies,
        COSZEN,
    )

    phi_mu0 = interp_flux(
        INITIAL_FILE,
        "numu",
        energies,
        COSZEN,
    )

    phi_tau0 = interp_flux(
        INITIAL_FILE,
        "nutau",
        energies,
        COSZEN,
    )

    # Probabilities INTO final electron flavor
    Pee = interp_probability(
        prob_df,
        f"{prefix}_Pee",
        energies,
    )

    Pmue = interp_probability(
        prob_df,
        f"{prefix}_Pmue",
        energies,
    )

    Ptaue = interp_probability(
        prob_df,
        f"{prefix}_Ptaue",
        energies,
    )

    # Individual contributions
    e_to_e = phi_e0 * Pee
    mu_to_e = phi_mu0 * Pmue
    tau_to_e = phi_tau0 * Ptaue

    appearance = mu_to_e + tau_to_e

    phi_e_sum = (
        e_to_e
        + mu_to_e
        + tau_to_e
    )

    # How much same-flavor electron flux disappeared?
    disappearance_loss = phi_e0 - e_to_e

    # Net effect after appearance compensation
    net_change = appearance - disappearance_loss

    # How efficiently does appearance compensate disappearance?
    compensation = np.divide(
        appearance,
        disappearance_loss,
        out=np.full_like(appearance, np.nan),
        where=np.abs(disappearance_loss) > 0.0,
    )

    # Fraction of final nu_e flux that came from OTHER flavors
    appearance_fraction = np.divide(
        appearance,
        phi_e_sum,
        out=np.zeros_like(appearance),
        where=phi_e_sum > 0.0,
    )

    # Initial mu/e flux hierarchy
    mu_over_e = np.divide(
        phi_mu0,
        phi_e0,
        out=np.full_like(phi_mu0, np.nan),
        where=phi_e0 > 0.0,
    )

    result = pd.DataFrame(
        {
            "E_GeV": energies,
            "Phi_e0": phi_e0,
            "Phi_mu0/Phi_e0": mu_over_e,

            "Pee": Pee,
            "Pmue": Pmue,

            "Phi_e0*Pee": e_to_e,
            "Phi_mu0*Pmue": mu_to_e,

            "e_loss": disappearance_loss,
            "appearance": appearance,

            "app/loss": compensation,
            "appearance_%": 100.0 * appearance_fraction,

            "Phi_e_sum": phi_e_sum,
            "Phi_e_sum/Phi_e0": phi_e_sum / phi_e0,

            "net_change": net_change,
        }
    )

    return result


# ============================================================
# FULL NUSQUIDS SM CHECK
# ============================================================

def add_full_sm_check(table):
    """
    Compare oscillation-only reconstruction with the full propagated
    Standard Model flux stored in RESULTS/standard.
    """

    full_sm = interp_flux(
        SM_FILE,
        "nue",
        table["E_GeV"].to_numpy(),
        COSZEN,
    )

    table = table.copy()

    table["Phi_e_full_SM"] = full_sm

    table["full/sum"] = (
        full_sm
        / table["Phi_e_sum"].to_numpy()
    )

    return table


# ============================================================
# FIND STRONG Pee DIPS
# ============================================================

def find_interesting_sm_dips(prob_df, n=6, min_spacing_logE=0.08):
    """
    Find low-Pee points, avoiding many nearly identical neighboring
    energy nodes.

    These are useful because they are exactly where one might expect
    strong 'stripes' from Pee alone.
    """

    E = prob_df["energy_GeV"].to_numpy()
    Pee = prob_df["sm_Pee"].to_numpy()

    order = np.argsort(Pee)

    selected = []

    for idx in order:
        logE = np.log10(E[idx])

        if all(
            abs(logE - np.log10(E[j])) >= min_spacing_logE
            for j in selected
        ):
            selected.append(idx)

        if len(selected) == n:
            break

    selected = sorted(selected, key=lambda i: E[i])

    return E[selected]


# ============================================================
# PRETTY PRINTING
# ============================================================

def print_table(title, df):
    print()
    print("=" * 145)
    print(title)
    print("=" * 145)

    columns = [
        "E_GeV",
        "Phi_e0",
        "Phi_mu0/Phi_e0",
        "Pee",
        "Pmue",
        "e_loss",
        "appearance",
        "app/loss",
        "appearance_%",
        "Phi_e_sum/Phi_e0",
    ]

    if "full/sum" in df.columns:
        columns.append("full/sum")

    out = df[columns].copy()

    formatters = {
        "E_GeV": lambda x: f"{x:7.2f}",

        "Phi_e0": lambda x: f"{x:10.3e}",
        "Phi_mu0/Phi_e0": lambda x: f"{x:8.3f}",

        "Pee": lambda x: f"{x:8.5f}",
        "Pmue": lambda x: f"{x:8.5f}",

        "e_loss": lambda x: f"{x:10.3e}",
        "appearance": lambda x: f"{x:10.3e}",

        "app/loss": lambda x: f"{x:9.3f}",
        "appearance_%": lambda x: f"{x:10.2f}",

        "Phi_e_sum/Phi_e0": lambda x: f"{x:10.4f}",
    }

    if "full/sum" in out.columns:
        formatters["full/sum"] = lambda x: f"{x:9.5f}"

    print(
        out.to_string(
            index=False,
            formatters=formatters,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 80)
    print("ELECTRON-FLUX COMPENSATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Model        : Le-Lmu")
    print(f"log10(m/eV) : {LOG10_M}")
    print(f"log10(g')    : {LOG10_G}")
    print(f"coszen       : {COSZEN}")
    print()

    generate_probability_file()

    prob_df = load_probabilities()

    # --------------------------------------------------------
    # 1. Standard Model at fixed representative energies
    # --------------------------------------------------------

    sm_table = build_electron_table(
        prob_df,
        "SM",
        SAMPLE_ENERGIES,
    )

    sm_table = add_full_sm_check(sm_table)

    print_table(
        "STANDARD MODEL: final nu_e flux decomposition",
        sm_table,
    )

    # --------------------------------------------------------
    # 2. Look specifically at strong Pee minima
    # --------------------------------------------------------

    dip_energies = find_interesting_sm_dips(
        prob_df,
        n=6,
    )

    sm_dips = build_electron_table(
        prob_df,
        "SM",
        dip_energies,
    )

    sm_dips = add_full_sm_check(sm_dips)

    print_table(
        "STANDARD MODEL: energies where Pee is especially small",
        sm_dips,
    )

    # --------------------------------------------------------
    # 3. Same current LRI benchmark, for comparison
    # --------------------------------------------------------

    lri_table = build_electron_table(
        prob_df,
        "LRI",
        SAMPLE_ENERGIES,
    )

    print_table(
        "LRI BENCHMARK: final nu_e flux decomposition",
        lri_table,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("HOW TO INTERPRET app/loss")
    print("=" * 80)

    print(
        """
For final nu_e:

    Phi_e =
        Phi_e^0  Pee
      + Phi_mu^0 P(mu -> e)
      + Phi_tau^0 P(tau -> e)

The quantity

    e_loss =
        Phi_e^0 (1 - Pee)

is how much of the original electron-neutrino flux is lost.

The quantity

    appearance =
        Phi_mu^0 P(mu -> e)
      + Phi_tau^0 P(tau -> e)

is how much flux arrives INTO the electron sector from other flavors.

Therefore

    app/loss ~ 1

means that appearance almost exactly compensates disappearance.

    app/loss > 1

means appearance is even larger than the same-flavor loss.

    app/loss << 1

means disappearance survives strongly in the final flux.

This is precisely why a deep minimum in Pee does NOT necessarily produce
a similarly deep stripe in the final nu_e atmospheric flux.

The column

    appearance_%

is the fraction of the FINAL nu_e flux that originated in another flavor.

The column

    full/sum

compares the full Standard-Model nuSQuIDS flux, which includes transport
effects, against the simple coherent sum Phi0*P.

If full/sum is close to 1 in this energy range, the table is directly
showing that the disappearance/appearance compensation is the dominant
reason for the smoothing.
"""
    )


if __name__ == "__main__":
    main()
