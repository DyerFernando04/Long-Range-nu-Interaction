from MCEq.core import MCEqRun
import crflux.models as crf
import numpy as np

# ============================================================
# 1) Parámetros desde tus estadísticas (SIN cargar MC)
# ============================================================

# Energía verdadera (de tu salida)
#E_min = 1.31747
#E_max = 5917.2

# Alternativa "notebook + margen" (si luego decides recortar cola):
E_min = 1.31747
E_max = 2000 #2.0 * 290.141   # 2*p99 ≈ 580.282

# Theta (0..180) de tu salida
theta_full_min = 0.0001 #0.669213
theta_full_max = 179.9999 #179.436

# Evitar extremos exactos por estabilidad numérica
E_min = max(E_min, 1e-6)
theta_full_min = max(theta_full_min, 0.0001)
theta_full_max = min(theta_full_max, 179.9999)

# ============================================================
# 2) Grillas
# ============================================================

N_E = 2000
E_grid = np.logspace(np.log10(E_min), np.log10(E_max), N_E)

N_theta_full = 2000
theta_full = np.linspace(theta_full_min, theta_full_max, N_theta_full)

# Fold como notebook: evalúa en 0..90, pero luego guardas en 0..180
theta_ref = np.where(theta_full > 90.0, 180.0 - theta_full, theta_full)
theta_ref = np.clip(theta_ref, 0.0001, 89.9999)

# Resolver solo en thetas únicos del rango 0..90 (eficiencia)
theta_ref_unique, inv = np.unique(np.round(theta_ref, 6), return_inverse=True)
N_theta_ref = len(theta_ref_unique)

print("=== Grid summary ===")
print(f"E_true grid     : [{E_min:.6g}, {E_max:.6g}] GeV   | N_E = {N_E}")
print(f"theta_full grid : [{theta_full_min:.6g}, {theta_full_max:.6g}] deg | N_theta_full = {N_theta_full}")
print(f"theta_ref (fold): [0, 90] deg (unique = {N_theta_ref})")

mag = 0  # E^mag * Phi (mag=0 => Phi)

# ============================================================
# 3) Interpolación log-log (tu función, robustecida)
# ============================================================

def interp_loglog(E_old, phi_old, E_new):
    """
    Interpola phi(E) en espacio log-log. Fuera de rango o phi<=0 => 0.
    """
    phi_new = np.zeros_like(E_new, dtype=float)

    E_old = np.asarray(E_old, dtype=float)
    phi_old = np.asarray(phi_old, dtype=float)

    mask = (E_old > 0) & np.isfinite(E_old) & np.isfinite(phi_old) & (phi_old > 0)
    if np.count_nonzero(mask) < 2:
        return phi_new

    logE_old = np.log(E_old[mask])
    logphi_old = np.log(phi_old[mask])

    logE_new = np.log(E_new)
    in_range = (logE_new >= logE_old.min()) & (logE_new <= logE_old.max())

    phi_new[in_range] = np.exp(
        np.interp(logE_new[in_range], logE_old, logphi_old)
    )
    return phi_new

# ============================================================
# 4) Contenedores de flujo
# ============================================================

species_list = ["numu", "nue", "antinumu", "antinue"]

# Primero guardamos en la grilla eficiente 0..90 (theta_ref_unique)
flux_ref = {sp: np.zeros((N_E, N_theta_ref), dtype=float) for sp in species_list}

# ============================================================
# 5) Loop angular: MCEq solve por theta_ref_unique
# ============================================================

for j, th in enumerate(theta_ref_unique):
    if (j % 25) == 0 or (j == N_theta_ref - 1):
        print(f"Theta_ref = {th:.4f} deg ({j+1}/{N_theta_ref})")

    mceq = MCEqRun(
        interaction_model="SIBYLL23C",
        primary_model=(crf.HillasGaisser2012, "H3a"),
        theta_deg=float(th),
        density_model=("MSIS00_IC", ("SouthPole", "January"))
    )
    mceq.solve()

    E_old = np.asarray(mceq.e_grid, dtype=float)

    for sp in species_list:
        phi = np.asarray(mceq.get_solution(sp, mag), dtype=float)

        # Conversión de unidades pedida: m^2 -> cm^2
        # phi = phi * 1e4

        flux_ref[sp][:, j] = interp_loglog(E_old, phi, E_grid)

# ============================================================
# 6) Expandir a 0..180 (theta_full) y guardar
# ============================================================

# Expandimos usando inv: para cada theta_full, usa su theta_ref correspondiente
flux_full = {sp: flux_ref[sp][:, inv] for sp in species_list}  # (N_E, N_theta_full)

header = "E_GeV " + " ".join([f"theta_{t:.4f}deg" for t in theta_full])

for sp in species_list:
    data = np.column_stack([E_grid, flux_full[sp]])
    np.savetxt(f"{sp}_flux_E_theta_MCEq_theta0_180.txt", data, header=header)
    print(f"Guardado: {sp}_flux_E_theta_MCEq_theta0_180.txt")

