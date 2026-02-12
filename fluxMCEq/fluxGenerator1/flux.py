from MCEq.core import MCEqRun
import crflux.models as crf
import numpy as np

# ============================
# Grids
# ============================
theta_array = np.linspace(0.0001, 89.9999, 2000)
N_theta = len(theta_array)

N_E = 2000
E_min, E_max = 1e0, 2.5e3   # GeV
E_grid = np.logspace(np.log10(E_min), np.log10(E_max), N_E)

mag = 0  # si quieres graficar E^mag * Phi, cambia esto

# ============================
# Contenedores para los 4 flujos
# ============================
species_list = ["numu", "nue", "antinumu", "antinue"]

flux = {sp: np.zeros((N_E, N_theta), dtype=float) for sp in species_list}

# ============================
# Interpolación log-log (misma para todos)
# ============================
def interp_loglog(E_old, phi_old, E_new):
    """
    Interpola phi(E) en espacio log-log, devolviendo valores solo dentro del rango
    cubierto por E_old donde phi_old > 0. Fuera de rango: 0.
    """
    phi_new = np.zeros_like(E_new, dtype=float)

    mask = (phi_old > 0) & np.isfinite(phi_old) & np.isfinite(E_old)
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

# ============================
# Loop angular
# ============================
for j, theta in enumerate(theta_array):
    print(f"Theta = {theta:.4f} deg ({j+1}/{N_theta})")

    mceq = MCEqRun(
        interaction_model='SIBYLL23C',
        primary_model=(crf.HillasGaisser2012, 'H3a'),
        theta_deg=theta,
        density_model=('MSIS00_IC', ('SouthPole', 'January'))
    )
    mceq.solve()

    E_old = mceq.e_grid

    # sacar, interpolar y almacenar los 4 flujos
    for sp in species_list:
        phi = mceq.get_solution(sp, mag)
        flux[sp][:, j] = interp_loglog(E_old, phi, E_grid)

# ============================
# Guardar (4 archivos)
# ============================
# Recomendación: evita nombres de columnas duplicados.
# Incluyo índice + theta con 4 decimales.
#header = "E_GeV " + " ".join([f"theta_{t:.1f}deg" for t in theta_array])
theta_labels = [
    np.format_float_positional(t, precision=15, unique=True, trim='k')
    for t in theta_array
]

header = "E_GeV " + " ".join([f"theta_{lab}deg" for lab in theta_labels])

for sp in species_list:
    data = np.column_stack([E_grid, flux[sp]])
    np.savetxt(f"{sp}_flux_E_theta.txt", data, header=header)
    print(f"Guardado: {sp}_flux_E_theta.txt")
