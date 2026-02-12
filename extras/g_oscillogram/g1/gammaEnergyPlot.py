import sys
import time
import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# ===================== Constantes y PMNS =====================
G_F = 1.1663787e-5 * 1e-18  # eV^-2
GeV_to_eV = 1e9
E_GeV = np.logspace(-1, 2, 850)
E = E_GeV * GeV_to_eV

a_km = 6371.0
a = a_km * 5.067731e9  # eV^-1

delta_m21 = 7.49e-5
delta_m31 = 2.513e-3

theta12 = np.radians(33.68)
theta13 = np.radians(8.56)
theta23 = np.radians(43.3)
delta_cp = np.radians(212)

c12, s12 = np.cos(theta12), np.sin(theta12)
c13, s13 = np.cos(theta13), np.sin(theta13)
c23, s23 = np.cos(theta23), np.sin(theta23)

U1 = np.array([[1,0,0],[0,c23,s23],[0,-s23,c23]], dtype=np.complex128)
U2 = np.array([[c13,0, s13*np.exp(-1j*delta_cp)],[0,1,0],[-s13*np.exp(1j*delta_cp),0,c13]], dtype=np.complex128)
U3 = np.array([[c12,s12,0],[-s12,c12,0],[0,0,1]], dtype=np.complex128)
U = U1 @ U2 @ U3

M2 = np.diag([0.0, delta_m21, delta_m31]).astype(np.complex128)  # eV^2

# ===================== Conversión densidad =====================
m_p, m_n, m_e = 938.27e6, 939.57e6, 0.511e6  # eV
mass_total = m_p + m_n + m_e
rho_to_eV4 = 4.3625e18  # g/cm^3 -> eV^4

def N_e_from_rho(rho_mass):
    return (rho_mass * rho_to_eV4) / mass_total  # número de e^-

# ===================== Geometría =====================
def l_max(theta):
    return a*np.sqrt(2*(1-np.cos(2*theta-np.pi)))

def radio(l, theta):
    return np.sqrt(l*l + a*a - 2*a*l*np.cos(np.pi-theta))

# ===================== PREM por tramos (rho(x)=c0+c1 x+c2 x^2+c3 x^3) =====================
prem_segments_km = [
    (0.0,     1221.5,  [13.0885,           0.0,      -8.8381,       0.0]),
    (1221.5,  3480.0,  [12.5815,     -1.2638,      -3.6426,     -5.5281]),
    (3480.0,  3630.0,  [7.9565,      -6.4761,       5.5283,     -3.0807]),
    (3630.0,  5600.0,  [7.9565,      -6.4761,       5.5283,     -3.0807]),
    (5600.0,  5701.0,  [7.9565,      -6.4761,       5.5283,     -3.0807]),
    (5701.0,  5771.0,  [5.3197,      -1.4836,       0.0,         0.0   ]),
    (5771.0,  5971.0,  [11.2494,     -8.0298,       0.0,         0.0   ]),
    (5971.0,  6151.0,  [7.1089,      -3.8045,       0.0,         0.0   ]),
    (6151.0,  6291.0,  [2.6910,       0.6924,       0.0,         0.0   ]),
    (6291.0,  6346.6,  [2.6910,       0.6924,       0.0,         0.0   ]),
    (6346.6,  6356.0,  [2.9,          0.0,          0.0,         0.0   ]),
    (6356.0,  6368.0,  [2.6,          0.0,          0.0,         0.0   ]),
    (6368.0,  6371.0,  [1.02,         0.0,          0.0,         0.0   ]),
]
prem_segments = [(r0*5.067731e9, r1*5.067731e9, np.array(c,dtype=np.float64))
                 for (r0,r1,c) in prem_segments_km]

# ===================== Integrales cerradas de r^n e^{± m r} =====================
def I_exp_minus(n, t, m):
    if t == 0.0: return 0.0
    mt = m*t
    fact = math.factorial(n) / (m**(n+1))
    s = 0.0; term = 1.0
    for k in range(0, n+1):
        if k>0: term *= mt / k
        s += term
    return fact * (1.0 - np.exp(-mt) * s)

def I_exp_plus(n, t, m):
    if t == 0.0: return 0.0
    mt = m*t
    poly = 0.0
    for k in range(0, n+1):
        coeff = ((-1)**(n-k)) * math.factorial(n) / (math.factorial(k) * (m**(n-k+1)))
        poly += coeff * (t**k)
    return np.exp(mt) * poly - math.factorial(n) / (m**(n+1))

def AB_on_interval(r_lo, r_hi, coeffs, m):
    c0, c1, c2, c3 = coeffs
    scale = [1.0, 1.0/a, 1.0/(a*a), 1.0/(a*a*a)]
    def J_plus(n):   return I_exp_plus(n,  r_hi, m) - I_exp_plus(n,  r_lo, m)
    def J_minus(n):  return I_exp_minus(n, r_hi, m) - I_exp_minus(n, r_lo, m)
    dA = 0.0; dB = 0.0
    for k, ck in enumerate([c0, c1, c2, c3]):
        if ck == 0.0: continue
        n = k + 1
        s = scale[k]
        dB += ck * s * J_minus(n)
        dA += ck * s * (J_plus(n) - J_minus(n)) * 0.5
    conv = (rho_to_eV4 / mass_total)
    return dA*conv, dB*conv

def A_B_upto_r(r, m):
    A = 0.0; B = 0.0
    for (r0, r1, coeffs) in prem_segments:
        if r <= r0: break
        hi = min(r, r1)
        dA, dB = AB_on_interval(r0, hi, coeffs, m)
        A += dA; B += dB
        if r <= r1: break
    return A, B

# ===================== Parámetros LRI (m fijo; g' variable luego) =====================
m_e_beta       = 1.0e-13    # [eV] (ajusta)
mLR = float(m_e_beta)

# Precompute B_total = B(a) (depende de m, no de g')
A_tot, B_tot = A_B_upto_r(a, mLR)

def Phi_core_at_r(r):
    """Phi(r) tal que V_LRI(r; g') = (g'^2/(4π)) * Phi(r)."""
    r_eff = r if r > 0.0 else 1e-16
    A_r, B_r = A_B_upto_r(r_eff, mLR)
    return (4.0*np.pi / (mLR * r_eff)) * (np.exp(-mLR*r_eff)*A_r + np.sinh(mLR*r_eff)*(B_tot - B_r))

# ===================== H_vac para todas las E =====================
H_vac = np.array([(U @ M2 @ U.conj().T) / (2.0*Ei) for Ei in E], dtype=np.complex128)  # (nE,3,3)

# ===================== Estados sabor =====================
nu_e  = np.array([1+0j,0+0j,0+0j], dtype=np.complex128)
nu_mu = np.array([0+0j,1+0j,0+0j], dtype=np.complex128)
nu_tau= np.array([0+0j,0+0j,1+0j], dtype=np.complex128)

# ===================== Propagación (batch en energías) =====================
def propagate_allE_for_angle_with_g(theta, psi0_flavor, out_index, gprime):
    """Devuelve P(E; g', theta) con batch en energías."""
    nE = E.size
    L = l_max(theta)
    cosphi = np.cos(np.pi - theta)

    Psi0 = np.tile(psi0_flavor, (nE, 1))
    y0 = Psi0.ravel()

    # Pre-referencias para matvec sin copias
    H00 = H_vac[:, 0, 0]; H01 = H_vac[:, 0, 1]; H02 = H_vac[:, 0, 2]
    H10 = H_vac[:, 1, 0]; H11 = H_vac[:, 1, 1]; H12 = H_vac[:, 1, 2]
    H20 = H_vac[:, 2, 0]; H21 = H_vac[:, 2, 1]; H22 = H_vac[:, 2, 2]

    g2_over_4pi = (gprime**2) / (4.0*np.pi)

    def rhs(l, y):
        Y0 = y[0::3]; Y1 = y[1::3]; Y2 = y[2::3]
        r = np.sqrt(l*l + a*a - 2*a*l*cosphi)

        # MSW local (misma forma que vienes usando)
        rho_mass_here = 0.0
        for (r0, r1, coeffs) in prem_segments:
            if (r >= r0) and (r <= r1):
                x = r / a
                c0, c1, c2, c3 = coeffs
                rho_mass_here = c0 + c1*x + c2*(x**2) + c3*(x**3)
                break
        Ve = np.sqrt(2.0) * G_F * N_e_from_rho(rho_mass_here)

        # LRI: V = (g'^2/4π) * Phi(r)
        Vbsm = g2_over_4pi * Phi_core_at_r(r)

        # H_vac @ Y
        Z0 = H00*Y0 + H01*Y1 + H02*Y2
        Z1 = H10*Y0 + H11*Y1 + H12*Y2
        Z2 = H20*Y0 + H21*Y1 + H22*Y2

        # Diagonal extra
        Z0 = Z0 + (Ve + Vbsm) * Y0
        Z1 = Z1 + (-Vbsm)      * Y1

        dY0 = -1j * Z0
        dY1 = -1j * Z1
        dY2 = -1j * Z2

        out = np.empty_like(y)
        out[0::3] = dY0; out[1::3] = dY1; out[2::3] = dY2
        return out

    sol = spi.solve_ivp(rhs, (0.0, L), y0, t_eval=[L],
                        method='DOP853', rtol=1e-9, atol=1e-12)
    PsiL = sol.y[:, -1].reshape(nE, 3)
    return np.abs(PsiL[:, out_index])**2  # (nE,)

# ===================== (1) Tu barrido por ángulos original (si lo quieres mantener) =====================
angulos = np.radians(np.linspace(90.0001, 179.9999, 100))
psi0 = nu_mu
out_idx = 1

# (Opcional) descomenta para volver a producir matriz_mumu_LRI(E,θ)
# matriz_mumu_LRI = []
# Ntheta = len(angulos)
# Emin, Emax, nE = E_GeV.min(), E_GeV.max(), E.size
# for idx_theta, theta in enumerate(angulos, start=1):
#     L_km = l_max(theta) / 5.067731e9
#     sys.stdout.write(
#         f"\rÁngulo {idx_theta:02d}/{Ntheta} | θ={np.degrees(theta):6.2f}° | "
#         f"E [{Emin:.2f}, {Emax:.2f}] GeV, nE={nE} | L={L_km:8.2f} km  "
#     ); sys.stdout.flush()
#     P = propagate_allE_for_angle_with_g(theta, psi0, out_idx, gprime=0.0)  # ejemplo g'=0
#     matriz_mumu_LRI.append(P.tolist())
# sys.stdout.write("\nÁngulos listos.\n")
# matriz_mumu_LRI = np.array(matriz_mumu_LRI).T  # (nE, nTheta)

# ===================== (2) Barrido en g' para un ángulo fijo =====================
theta_fixed = np.radians(120.0)   # <-- cambia aquí tu ángulo fijo
g_list = np.logspace(-31, -22, num=850)  # 1e-27 ... 1e-22 (51 puntos)

P_gE = np.empty((g_list.size, E.size), dtype=float)

print(f"Barrido g' en θ={np.degrees(theta_fixed):.2f}°, n_g={g_list.size}, nE={E.size}")
for i, gprime in enumerate(g_list, start=1):
    sys.stdout.write(f"\r  g' {i:02d}/{g_list.size} = {gprime:.3e}"); sys.stdout.flush()
    P_gE[i-1, :] = propagate_allE_for_angle_with_g(theta_fixed, nu_mu, out_index=1, gprime=gprime)
sys.stdout.write("\nListo barrido g'.\n")

# Guardar la matriz P_gE en un archivo .npy
np.save('P_gE.npy', P_gE)
print("Matriz P_gE guardada en 'P_gE.npy'.")
