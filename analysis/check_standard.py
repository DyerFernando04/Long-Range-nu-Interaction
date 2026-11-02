#!/usr/bin/env python3
"""Auditoria independiente del pipeline standard (propagacion -> templates -> IceCube).

Ejecutar desde cualquier directorio:
  python3 analysis/check_standard.py

No modifica ningun HDF5; guarda, si matplotlib esta disponible,
  results/figures/standard_validation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("nu_nc", "nue_cc", "numu_cc", "nutau_cc")
SPECIES = ("nue", "numu", "nutau", "antinue", "antinumu", "antinutau")


def check(condition, msg):
    if not condition:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def get_string(x):
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def finite_nonnegative(a, name):
    check(np.all(np.isfinite(a)) and np.all(a >= 0), f"{name}: finito y no negativo")


def interpolate_log_bilinear(flux, energies, angles, true_e, true_c):
    """Reproduce la interpolacion de build_templates.py de forma independiente."""
    positive = flux[flux > 0]
    if not positive.size:
        return np.zeros_like(true_e)
    floor = max(np.finfo(np.float64).tiny, float(np.min(positive)) * 1e-12)
    e_grid = np.log10(energies)
    e = np.clip(np.log10(true_e), e_grid[0], e_grid[-1])
    c = np.clip(true_c, angles[0], angles[-1])
    ie = np.clip(np.searchsorted(e_grid, e, side="right") - 1, 0, len(energies) - 2)
    ic = np.clip(np.searchsorted(angles, c, side="right") - 1, 0, len(angles) - 2)
    te = (e - e_grid[ie]) / (e_grid[ie + 1] - e_grid[ie])
    tc = (c - angles[ic]) / (angles[ic + 1] - angles[ic])
    f00 = np.log(np.maximum(flux[ic, ie], floor))
    f10 = np.log(np.maximum(flux[ic, ie + 1], floor))
    f01 = np.log(np.maximum(flux[ic + 1, ie], floor))
    f11 = np.log(np.maximum(flux[ic + 1, ie + 1], floor))
    return np.exp((1-te)*(1-tc)*f00 + te*(1-tc)*f10 + (1-te)*tc*f01 + te*tc*f11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux", type=Path, default=ROOT/"results/oscillated_flux/standard.h5")
    parser.add_argument("--templates", type=Path, default=ROOT/"results/templates/standard.h5")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n==== I. FLUJOS OSCILADOS (standard) ====")
    if not args.flux.is_file():
        raise FileNotFoundError(f"No existe {args.flux}")
    with h5py.File(args.flux, "r") as f:
        for key in ("energy_GeV", "coszen", "flux", "done"):
            check(key in f, f"Existe /{key}")
        energies = f["energy_GeV"][:]
        angles = f["coszen"][:]
        check(np.all(np.diff(energies) > 0) and np.all(energies > 0), "Eje de energia creciente y positivo")
        check(np.all(np.diff(angles) > 0) and np.all(np.abs(angles) <= 1.0 + 1e-12), "Eje de coszen creciente en [-1,1]")
        check(f["flux"].shape == (len(angles), len(energies), 6), "Forma [coszen, energy, species=6]")
        check(f["done"].shape == (len(angles),) and np.all(f["done"][:]), "Todos los angulos propagados")
        print(f"  Malla: {len(angles)} coszen x {len(energies)} energias")
        for ic in range(len(angles)):
            piece = f["flux"][ic, :, :]
            if not (np.all(np.isfinite(piece)) and np.all(piece >= 0)):
                raise AssertionError(f"Flujo invalido en indice coszen={ic}")
        print("  [OK] Los seis flujos son finitos y no negativos en toda la malla")

    if not args.templates.is_file():
        print(f"\n[INCOMPLETO] No existe {args.templates}")
        print("Generalo primero: python3 analysis/build_templates.py standard")
        return 2

    print("\n==== II. TEMPLATES Y PESOS MC ====")
    with h5py.File(args.templates, "r") as h, h5py.File(args.flux, "r") as f:
        check(get_string(h.attrs.get("model", "")) == "standard", "Modelo standard")
        check(get_string(h.attrs.get("format_version", "")) == "3.1", "Formato de templates 3.1")
        check(bool(h.attrs.get("build_completed", False)), "Construccion completada")
        check(h["status/completed_nodes"].shape == (1, 1) and bool(h["status/completed_nodes"][0,0]), "Checkpoint [1,1] completo")
        check(get_string(h.attrs.get("source_flux_signature", "")) == get_string(f.attrs.get("config_signature", "")), "Identificador de propagacion coincide")

        weights_ds = h["event_weights/nominal"]
        check(weights_ds.ndim == 3 and weights_ds.shape[:2] == (1, 1), "Pesos standard [1,1,event]")
        weights = weights_ds[0,0,:].astype(np.float64)
        finite_nonnegative(weights, "Pesos de eventos")
        category_ids = h["event_cache/category_id"][:]
        reco_bins = h["event_cache/reco_bin_id"][:].astype(np.int64)
        n_event = len(weights)
        n_bin = len(h["observed/data"])
        check(len(category_ids) == len(reco_bins) == n_event and n_event > 0, "Indices y pesos alineados por evento")
        check(np.all((category_ids >= 0) & (category_ids < 4)), "Categorias de evento validas")
        check(np.all((reco_bins >= 0) & (reco_bins < n_bin)), "Eventos dentro de los bins reconstruidos")
        shape = tuple(int(x) for x in h.attrs["hist_shape"])
        check(int(np.prod(shape)) == n_bin, f"Binning reconstruido {shape} = {n_bin} bins")
        print(f"  Eventos MC usados: {n_event:,}")

        raw_total = np.zeros(n_bin, dtype=np.float64)
        corrected_total = np.zeros(n_bin, dtype=np.float64)
        for cid, cat in enumerate(CATEGORIES):
            mask = category_ids == cid
            expected = np.bincount(reco_bins[mask], weights=weights[mask], minlength=n_bin)
            expected_w2 = np.bincount(reco_bins[mask], weights=weights[mask]**2, minlength=n_bin)
            actual = h[f"templates/{cat}"][0,0,:]
            actual_w2 = h[f"sumw2/{cat}"][0,0,:]
            check(np.allclose(expected, actual, rtol=1e-12, atol=1e-14), f"{cat}: histograma reconstruido desde los pesos")
            check(np.allclose(expected_w2, actual_w2, rtol=1e-12, atol=1e-18), f"{cat}: sumw2 correcto")
            raw_total += actual
            # A valores nominales de las 5 hypersurfaces, el factor es intercept.
            intercept = h[f"hypersurfaces/{cat}/intercept"][:]
            check(intercept.shape == (n_bin,) and np.all(np.isfinite(intercept)), f"{cat}: hypersurface nominal disponible")
            corrected_total += actual * intercept
            print(f"    {cat:9s} | MC={int(np.count_nonzero(mask)):7d} | eventos nominales={actual.sum():.6g}")
        check(np.isclose(raw_total.sum(), weights.sum(), rtol=1e-12, atol=1e-10), "Suma de pesos coincide con suma de templates sin hypersurface")

        print("\n==== III. REWEIGHTING DESDE FLUJO ====")
        indices = np.unique(np.linspace(0, n_event-1, min(n_event, 512), dtype=np.int64))
        ev_e = h["event_cache/true_energy_GeV"][indices]
        ev_c = h["event_cache/true_coszen"][indices]
        ev_w = h["event_cache/base_mc_weight"][indices]
        ev_species = h["event_cache/species_id"][indices]
        reweighted = np.zeros(indices.size)
        for sid, species in enumerate(SPECIES):
            m = ev_species == sid
            if not np.any(m):
                continue
            fl = np.asarray(f["flux"][:,:,sid], dtype=np.float64)
            reweighted[m] = interpolate_log_bilinear(fl, energies, angles, ev_e[m], ev_c[m])
        factor = float(h.attrs["flux_to_icecube_weight_factor"])
        nucount = float(h.attrs["nucount"])
        predicted = (nucount * ev_w * factor * reweighted).astype(np.float32).astype(np.float64)
        check(np.allclose(predicted, weights[indices], rtol=3e-6, atol=1e-25), f"Reweighting en {len(indices)} eventos independiente del constructor")
        print(f"  Factor de conversion en builder: {factor:.6g}")
        if not np.isclose(factor, 1.0):
            print("  [ALERTA] Revisar unidades: MCEq ya entrega flujos por cm^2. Un factor 1e4 puede estar duplicado.")
        else:
            print("  [OK] No se aplica conversion de area adicional al flujo por cm^2")

        print("\n==== IV. DATOS OBSERVADOS Y PREDICCION NOMINAL ====")
        observed = h["observed/data"][:]
        muon = h["backgrounds/atmospheric_muon"][:]
        mu_var = h["backgrounds/atmospheric_muon_sumw2"][:]
        for k, v in (("observed", observed), ("atmospheric_muon", muon), ("muon variance", mu_var)):
            check(v.shape == (n_bin,), f"{k}: dimension correcta")
            finite_nonnegative(v, k)
        check(np.isclose(observed.sum(), nucount, rtol=0, atol=1e-8), "Suma observada coincide con NUCOUNT")
        prediction = corrected_total + muon
        print(f"  Observados                       : {observed.sum():.6g}")
        print(f"  Neutrinos (sin hypersurfaces)     : {raw_total.sum():.6g}")
        print(f"  Neutrinos (hypersurfaces nominales): {corrected_total.sum():.6g}")
        print(f"  Muones atmosfericos nominales     : {muon.sum():.6g}")
        print(f"  Total predicho nominal           : {prediction.sum():.6g}")
        print(f"  Predicho / observado             : {prediction.sum()/observed.sum():.6g}")
        print("  NOTA: No es un ajuste: neutrino_norm=muon_norm=1 y nuisances detectorales nominales.")

        if not args.no_plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                out_dir = ROOT / "results/figures"
                out_dir.mkdir(parents=True, exist_ok=True)
                fig, axes = plt.subplots(1,3, figsize=(15,4.5))
                for ix, (axis, label) in enumerate(zip(axes, ("Reco energy", "Reco coszen", "PID"))):
                    ax_obs = observed.reshape(shape).sum(axis=tuple(j for j in range(3) if j != ix))
                    ax_pred = prediction.reshape(shape).sum(axis=tuple(j for j in range(3) if j != ix))
                    ax_mu = muon.reshape(shape).sum(axis=tuple(j for j in range(3) if j != ix))
                    xs = np.arange(len(ax_obs))
                    axis.step(xs, ax_pred, where="mid", label="Standard + atmospheric muons")
                    axis.step(xs, ax_mu, where="mid", linestyle="--", label="Atmospheric muons")
                    axis.errorbar(xs, ax_obs, yerr=np.sqrt(ax_obs), fmt="o", markersize=4, label="Observed")
                    axis.set_xlabel(label+" bin")
                    axis.set_ylabel("Events")
                    axis.grid(alpha=.2)
                axes[0].legend(fontsize=8)
                fig.suptitle("DeepCore standard: prediction at nominal nuisance parameters (not fitted)")
                fig.tight_layout()
                outfile = out_dir / "standard_validation.png"
                fig.savefig(outfile, dpi=180)
                plt.close(fig)
                print(f"\n  Grafica: {outfile}")
            except ImportError as e:
                print(f"  [AVISO] No se genero grafica (matplotlib no disponible): {e}")

    print("\n[FIN] Validaciones internas completadas. La concordancia con datos/referencia requiere una prueba fisica adicional.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, RuntimeError, ValueError, OSError, KeyError) as e:
        print(f"\n[ERROR DE VALIDACION] {e}", file=sys.stderr)
        sys.exit(1)
