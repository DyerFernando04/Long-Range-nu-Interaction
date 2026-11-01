
#!/usr/bin/env python3

from pathlib import Path
import csv
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import daemonflux


# ============================================================
# 1. CONFIGURACION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "LRI" / "INITIAL_FLUX"
    / "mceq_gsf2019_ddm.h5"
)

OUTPUT_DIR = ROOT / "results" / "mceq_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = ("nue", "numu", "antinue", "antinumu")

# Direcciones para los espectros individuales.
PLOT_COSZEN = (0.05, 0.4, 1.0)


# ============================================================
# 2. LEER LOS FLUJOS DE MCEq
# ============================================================

print("Loading MCEq fluxes...")

with h5py.File(INPUT_FILE, "r") as h5:

    energy = h5["axes/energy_GeV"][:]
    coszen = h5["axes/coszen"][:]

    # El archivo contiene angulos negativos y positivos.
    # Como el flujo inicial es simetrico, usamos solo
    # los angulos positivos para evitar duplicaciones.
    mask = coszen >= 0.0

    abs_coszen = coszen[mask]

    flux_mceq = {
        sp: h5[f"flux/{sp}"][:][mask]
        for sp in SPECIES
    }

# Usar la misma convencion angular del generador.
theta = np.degrees(np.arccos(abs_coszen))
theta = np.clip(theta, 1e-4, 90.0 - 1e-4)

print(f"Energy points : {len(energy)}")
print(f"Angles        : {len(abs_coszen)}")
print(f"Energy range  : {energy[0]:.2f} - {energy[-1]:.2f} GeV")


# ============================================================
# 3. GENERAR FLUJOS CON DAEMONFLUX
# ============================================================

print("\nInitializing Daemonflux...")

daemon = daemonflux.Flux(location="generic")

flux_daemon = {
    sp: np.zeros_like(flux_mceq[sp])
    for sp in SPECIES
}

start = time.perf_counter()

for i, angle in enumerate(theta):

    for sp in SPECIES:

        # Daemonflux devuelve E^3 * Phi.
        flux_daemon[sp][i] = (
            daemon.flux(energy, float(angle), sp)
            / energy**3
        )

    if (i + 1) % 20 == 0 or i == len(theta) - 1:
        print(f"Daemonflux: {i + 1}/{len(theta)} angles")

elapsed = time.perf_counter() - start

print(f"\nDaemonflux generation time: {elapsed:.2f} s")


# ============================================================
# 4. CALCULAR COCIENTES
# ============================================================

ratio = {}

for sp in SPECIES:

    ratio[sp] = flux_mceq[sp] / flux_daemon[sp]

    if not np.all(np.isfinite(ratio[sp])):
        raise ValueError(f"Non-finite ratio for {sp}")


# ============================================================
# 5. GRAFICOS DE ESPECTROS Y COCIENTES
# ============================================================

print("\nGenerating comparison plots...")

for sp in SPECIES:

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    for cz in PLOT_COSZEN:

        i = np.argmin(np.abs(abs_coszen - cz))

        mceq_flux = flux_mceq[sp][i]
        daemon_flux = flux_daemon[sp][i]

        # Espectros E^3 Phi.
        line, = axes[0].loglog(
            energy,
            energy**3 * mceq_flux,
            lw=2,
            label=fr"MCEq, $|\cos\theta_z|={cz}$",
        )

        axes[0].loglog(
            energy,
            energy**3 * daemon_flux,
            color=line.get_color(),
            ls="--",
            lw=2,
            label=fr"Daemonflux, $|\cos\theta_z|={cz}$",
        )

        # Cociente MCEq / Daemonflux.
        axes[1].semilogx(
            energy,
            ratio[sp][i],
            color=line.get_color(),
            lw=2,
            label=fr"$|\cos\theta_z|={cz}$",
        )

    axes[0].set_ylabel(
        r"$E^3\Phi$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
    )

    axes[1].set_xlabel("Neutrino energy [GeV]")
    axes[1].set_ylabel(
        r"$\Phi_{\mathrm{MCEq}}/\Phi_{\mathrm{Daemonflux}}$"
    )

    axes[1].axhline(
        1.0,
        color="black",
        ls=":",
        lw=1,
    )

    axes[1].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"comparison_{sp}.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 6. MAPAS DE COCIENTES
# ============================================================

for sp in SPECIES:

    fig, ax = plt.subplots(figsize=(8, 5))

    mesh = ax.pcolormesh(
        energy,
        abs_coszen,
        np.log10(ratio[sp]),
        shading="auto",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )

    ax.set_xscale("log")

    ax.set_xlabel("Neutrino energy [GeV]")
    ax.set_ylabel(r"$|\cos\theta_z|$")

    cbar = fig.colorbar(mesh, ax=ax)

    cbar.set_label(
        r"$\log_{10}(\Phi_{\mathrm{MCEq}}/"
        r"\Phi_{\mathrm{Daemonflux}})$"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"ratio_map_{sp}.png",
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# 7. RESUMEN NUMERICO
# ============================================================

# Analizamos especialmente el rango de DeepCore.
energy_mask = (energy >= 5.0) & (energy <= 200.0)

rows = []

for sp in SPECIES:

    print(f"\n{sp}")
    print("-" * 55)

    for cz in PLOT_COSZEN:

        i = np.argmin(np.abs(abs_coszen - cz))

        r = ratio[sp][i]

        median = np.median(r[energy_mask])

        r10 = np.interp(10.0, energy, r)
        r100 = np.interp(100.0, energy, r)
        r1000 = np.interp(1000.0, energy, r)

        rows.append({
            "species": sp,
            "abs_coszen": float(abs_coszen[i]),
            "median_ratio_5_200GeV": float(median),
            "ratio_at_10GeV": float(r10),
            "ratio_at_100GeV": float(r100),
            "ratio_at_1000GeV": float(r1000),
        })

        print(
            f"|coszen|={abs_coszen[i]:.2f}  "
            f"median={median:.3f}  "
            f"10GeV={r10:.3f}  "
            f"100GeV={r100:.3f}  "
            f"1000GeV={r1000:.3f}"
        )

with open(
    OUTPUT_DIR / "summary.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# 8. GUARDAR LOS DATOS
# ============================================================

data = {
    "energy_GeV": energy,
    "abs_coszen": abs_coszen,
}

for sp in SPECIES:

    data[f"mceq_{sp}"] = flux_mceq[sp]
    data[f"daemon_{sp}"] = flux_daemon[sp]
    data[f"ratio_{sp}"] = ratio[sp]

np.savez_compressed(
    OUTPUT_DIR / "comparison.npz",
    **data,
)

print("\nComparison completed.")
print(f"Results saved in: {OUTPUT_DIR.resolve()}")
