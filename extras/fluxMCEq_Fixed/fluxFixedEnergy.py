from MCEq.core import config, MCEqRun
import crflux.models as crf
import numpy as np

# ---------------------------
# Configuración
# ---------------------------

# Ángulos cenitales en grados (MCEq)
# Ajusta el rango según lo que quieras muestrear
theta_array = np.linspace(0.0001, 90.0, 400)  # 100 ángulos

# Energías que te interesan (GeV)
E_target = np.array([1e-1, 1e0, 1e1, 1e2, 1e3])

# mag=0 → Φ(E); mag=3 → E^3 Φ(E)
mag = 0

# Matriz de flujos: shape = (nE, nTheta)
nE = len(E_target)
nTheta = len(theta_array)
flux_matrix = np.zeros((nE, nTheta))

# ---------------------------
# Loop sobre ángulos
# ---------------------------
for i_theta, theta in enumerate(theta_array):
    print(f"Corriendo MCEq para θ = {theta:.4f}° ({i_theta+1}/{nTheta}) ...")

    # Correr MCEq para este ángulo
    mceq = MCEqRun(
        interaction_model='SIBYLL23C',
        primary_model=(crf.HillasGaisser2012, 'H3a'),
        theta_deg=theta,
        density_model=('MSIS00_IC', ('SouthPole', 'January'))
    )
    mceq.solve()

    # Flujo νμ + \barνμ (mag=0 → flujo físico)
    phi_numu = (mceq.get_solution('numu', mag) +
                mceq.get_solution('antinumu', mag))

    E_old = mceq.e_grid  # grid original de MCEq

    # ---------------------------------
    # Interpolación log–log robusta
    # ---------------------------------
    mask = (phi_numu > 0) & (E_old > 0)
    E_valid = E_old[mask]
    phi_valid = phi_numu[mask]

    if E_valid.size == 0:
        # caso patológico: no hay flujo >0
        phi_target = np.zeros_like(E_target)
    else:
        logE_valid = np.log(E_valid)
        logphi_valid = np.log(phi_valid)

        logE_target = np.log(E_target)

        phi_target = np.zeros_like(E_target)

        in_range = (logE_target >= logE_valid[0]) & (logE_target <= logE_valid[-1])

        # interpolación log–log solo dentro de rango
        phi_target[in_range] = np.exp(
            np.interp(logE_target[in_range], logE_valid, logphi_valid)
        )

    # limpiar posibles nan/inf
    phi_target = np.nan_to_num(phi_target, nan=0.0, posinf=0.0, neginf=0.0)

    # guardar columna correspondiente a este ángulo
    flux_matrix[:, i_theta] = phi_target

# ---------------------------
# Construir matriz final y guardar
# Formato: primera columna E, luego una columna por ángulo
# ---------------------------
data = np.column_stack([E_target, flux_matrix])

# Header con nombres simbólicos (nuSQuIDS los ignora, solo sirven para ti)
cols = ["E_GeV"] + [f"numu_{th:.4f}deg" for th in theta_array]
header = " ".join(cols)

outfile = "numu_flux_manyangles_Efixed.txt"
np.savetxt(
    outfile,
    data,
    header=header,
    comments="# "
)

print(f"Archivo '{outfile}' generado.")
print(f"NE (energías) = {nE}, Nθ (ángulos) = {nTheta}")
