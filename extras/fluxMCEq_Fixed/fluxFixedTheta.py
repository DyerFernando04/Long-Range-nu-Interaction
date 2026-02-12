from MCEq.core import config, MCEqRun
import crflux.models as crf
import numpy as np

# Ángulos cenitales en grados (MCEq)
angles = [90.0, 60.0, 30.0, 0.0001]

# Potencia para multiplicar el flujo (E**mag)
mag = 0

# Número de puntos deseados en el grid interpolado completo
N_new = 2000

# Corte máximo en energía (GeV) para guardar a archivo
E_max_cut = 1e4  # 10^4 GeV

results_interp = {}
E_new = None

for theta in angles:
    # --- Correr MCEq para este ángulo ---
    mceq = MCEqRun(
        interaction_model='SIBYLL23C',
        primary_model=(crf.HillasGaisser2012, 'H3a'),
        theta_deg=theta,
        density_model=('MSIS00_IC', ('SouthPole', 'January'))
    )
    mceq.solve()

    # Flujo de numu + antinumu, multiplicado por E**mag
    numu_flux = (mceq.get_solution('numu', mag) +
                 mceq.get_solution('antinumu', mag))

    E_old = mceq.e_grid  # grid original de MCEq (121 puntos)

    # Definir el grid común de energía (logspace) usando el primer ángulo
    if E_new is None:
        E_new = np.logspace(np.log10(E_old[0]),
                            np.log10(E_old[-1]),
                            N_new)

    # ============================
    # Interpolación log–log robusta
    # ============================
    # Nos quedamos solo con puntos donde el flujo es > 0
    mask = numu_flux > 0
    E_valid = E_old[mask]
    phi_valid = numu_flux[mask]

    if E_valid.size == 0:
        # Caso patológico: no hay puntos positivos
        numu_flux_new = np.zeros_like(E_new)
    else:
        logE_valid = np.log(E_valid)
        logphi_valid = np.log(phi_valid)
        logE_new = np.log(E_new)

        numu_flux_new = np.zeros_like(E_new)

        # Sólo interpolamos dentro del rango [E_valid.min(), E_valid.max()]
        in_range = (logE_new >= logE_valid[0]) & (logE_new <= logE_valid[-1])
        numu_flux_new[in_range] = np.exp(
            np.interp(logE_new[in_range], logE_valid, logphi_valid)
        )

    # Reemplazar cualquier nan/inf residual por 0 (por si acaso)
    numu_flux_new = np.nan_to_num(numu_flux_new,
                                  nan=0.0, posinf=0.0, neginf=0.0)

    # Guardamos el flujo interpolado completo para este ángulo
    results_interp[theta] = numu_flux_new

# ============================
# Corte en energía: E <= 1e4 GeV
# ============================
mask_cut = E_new <= E_max_cut
E_cut = E_new[mask_cut]

flux_90   = results_interp[90.0][mask_cut]
flux_60   = results_interp[60.0][mask_cut]
flux_30   = results_interp[30.0][mask_cut]
flux_0001 = results_interp[0.0001][mask_cut]

print("E_new total:   ", len(E_new), "puntos")
print("E_cut (<=1e4): ", len(E_cut), "puntos")
print("E_min, E_max (cortado):", E_cut[0], E_cut[-1])

# ============================
# Construir matriz y guardar
# ============================
data = np.column_stack([
    E_cut,
    flux_90,
    flux_60,
    flux_30,
    flux_0001
])

header = "E_GeV numu_90deg numu_60deg numu_30deg numu_0.0001deg (E<=1e4 GeV)"

np.savetxt(
    "numu_flux_icecube_angles.txt",
    data,
    header=header,
    comments="# "
)

print("Archivo 'numu_flux_icecube_angles.txt' generado con éxito.")

