#pragma once
#include <nuSQuIDS/nuSQuIDS.h>
#include <cmath>
#include <vector>
#include <iostream>

using namespace nusquids;

class nuSQUIDSLRI : public nuSQUIDS {
public:
  // Parámetros de la nueva interacción
  double g_prime_e_beta = 1e-26;
  double m_e_beta       = 1e-13;   // [eV] masa del mediador

  // Conversión de densidad de masa a número de electrones
  double rho_to_eV4 = 4.3625e18;   // g/cm^3 -> eV^4
  double m_p = 938.27e6;           // eV
  double m_n = 939.57e6;           // eV
  double m_e = 0.511e6;            // eV
  double mass_total;               // m_p + m_n + m_e

  // Radio de la Tierra en eV^{-1}
  double a;

  // Ángulo cenital (convención del usuario)
  double theta_z = M_PI;

  using nuSQUIDS::nuSQUIDS;

  // Debe llamarse después de construir el objeto
  void InitializeLRI(){
    mass_total = m_p + m_n + m_e;
    a = 6371.0 * 5.067731e9; // km -> eV^{-1}

    // Construir los segmentos PREM (en r [eV^{-1}] y coeficientes en g/cm^3)
    build_prem_segments();

    // Preparar parámetros de LRI
    mLR = m_e_beta;
    g2_over_4pi = (g_prime_e_beta * g_prime_e_beta) / (4.0 * M_PI);

    // Precomputar A(a), B(a) para toda la Tierra
    A_B_upto_r(a, mLR, A_tot, B_tot);
  }

  void SetZenithAngle(double theta_z_rad){ theta_z = theta_z_rad; }

  // Hamiltoniano de interacción: H_total = H_std + H_LRI
  squids::SU_vector HI(unsigned int ie, unsigned int irho, double x) const override {
    // H estándar (vacío + materia) desde nuSQUIDS
    squids::SU_vector H = nuSQUIDS::HI(ie, irho);

    // x lo interpretamos como coordenada a lo largo de la trayectoria, L
    const double L = x;
    const double r = radius_at(L);       // radio desde el centro
    const double V_e_beta = compute_VLRI_from_radius(r);

    // Termino diag(V, -V, 0) en base de sabor:
    // V * (|e><e| - |mu><mu|)
    squids::SU_vector term = GetFlavorProj(0); // |e>
    term -= GetFlavorProj(1);                  // -|mu>
    term *= V_e_beta;

    H += term;
    return H;
  }

private:
  // ==== Estructura para los tramos PREM ====
  struct PremSegment {
    double r0;   // [eV^{-1}]
    double r1;   // [eV^{-1}]
    double c0;   // coeficientes rho(x) = c0 + c1 x + c2 x^2 + c3 x^3
    double c1;
    double c2;
    double c3;
  };

  std::vector<PremSegment> prem_segments;

  // Parámetros LRI internos
  double mLR = 0.0;       // masa del mediador
  double g2_over_4pi = 0.0;
  double A_tot = 0.0;     // A(a)
  double B_tot = 0.0;     // B(a)

  // ==== Construcción de los tramos PREM (igual que en Python) ====
  void build_prem_segments(){
    prem_segments.clear();

    auto K = [](double km){ return km * 5.067731e9; }; // km -> eV^{-1}

    // (r0_km, r1_km, [c0, c1, c2, c3]) en g/cm^3
    prem_segments.push_back({K(   0.0), K(1221.5), 13.0885,      0.0,      -8.8381,      0.0});
    prem_segments.push_back({K(1221.5), K(3480.0), 12.5815, -1.2638,      -3.6426,  -5.5281});
    prem_segments.push_back({K(3480.0), K(3630.0),  7.9565, -6.4761,       5.5283,  -3.0807});
    prem_segments.push_back({K(3630.0), K(5600.0),  7.9565, -6.4761,       5.5283,  -3.0807});
    prem_segments.push_back({K(5600.0), K(5701.0),  7.9565, -6.4761,       5.5283,  -3.0807});
    prem_segments.push_back({K(5701.0), K(5771.0),  5.3197, -1.4836,       0.0,      0.0});
    prem_segments.push_back({K(5771.0), K(5971.0), 11.2494, -8.0298,       0.0,      0.0});
    prem_segments.push_back({K(5971.0), K(6151.0),  7.1089, -3.8045,       0.0,      0.0});
    prem_segments.push_back({K(6151.0), K(6291.0),  2.6910,  0.6924,       0.0,      0.0});
    prem_segments.push_back({K(6291.0), K(6346.6),  2.6910,  0.6924,       0.0,      0.0});
    prem_segments.push_back({K(6346.6), K(6356.0),  2.9,     0.0,          0.0,      0.0});
    prem_segments.push_back({K(6356.0), K(6368.0),  2.6,     0.0,          0.0,      0.0});
    prem_segments.push_back({K(6368.0), K(6371.0),  1.02,    0.0,          0.0,      0.0});
  }

  // ==== Geometría: L -> r(L) mediante la cuerda a través de la Tierra ====
  double l_max() const {
    const double c = std::cos(theta_z);
    double Lmax = 2.0 * a * (-c);
    if (Lmax < 0)    Lmax = 0;
    if (Lmax > 2.0*a) Lmax = 2.0 * a;
    return Lmax;
  }

  // Radio desde el centro para una posición L a lo largo de la trayectoria
  double radius_at(double L) const {
    const double Lmax = l_max();
    // parametrización centrada en el punto medio de la cuerda
    const double s = L - 0.5 * Lmax;
    const double b = a * std::sin(theta_z);
    double r = std::sqrt(b*b + s*s);
    if (r > a) r = a;
    if (r < 0.0) r = 0.0;
    return r;
  }

  // ==== Factorial pequeño (n<=4) ====
  static double factorial_int(int n){
    double f = 1.0;
    for(int k=2; k<=n; ++k) f *= k;
    return f;
  }

  // ==== I_exp_minus: ∫_0^t r^n e^{-m r} dr ====
  double I_exp_minus(int n, double t, double m) const {
    if (t == 0.0 || m <= 0.0) return 0.0;
    const double mt = m * t;
    const double fact = factorial_int(n) / std::pow(m, n+1);
    double s = 0.0;
    double term = 1.0;
    for(int k=0; k<=n; ++k){
      if (k > 0) term *= mt / double(k);
      s += term;
    }
    return fact * (1.0 - std::exp(-mt) * s);
  }

  // ==== I_exp_plus: ∫_0^t r^n e^{+m r} dr ====
  double I_exp_plus(int n, double t, double m) const {
    if (t == 0.0 || m <= 0.0) return 0.0;
    const double mt = m * t;
    const double n_fact = factorial_int(n);

    double poly = 0.0;
    for(int k=0; k<=n; ++k){
      // coeff = ((-1)^{n-k} * n! / (k! * m^{n-k+1}))
      double k_fact = factorial_int(k);
      double coeff = std::pow(-1.0, double(n-k)) * n_fact / (k_fact * std::pow(m, n-k+1));
      poly += coeff * std::pow(t, k);
    }
    return std::exp(mt) * poly - n_fact / std::pow(m, n+1);
  }

  // ==== AB_on_interval: contribuciones ΔA, ΔB en [r_lo, r_hi] ====
  void AB_on_interval(double r_lo,
                      double r_hi,
                      const PremSegment& seg,
                      double m,
                      double& dA,
                      double& dB) const
  {
    dA = 0.0;
    dB = 0.0;
    if (r_hi <= r_lo) return;

    const double c0 = seg.c0;
    const double c1 = seg.c1;
    const double c2 = seg.c2;
    const double c3 = seg.c3;

    // Escalas para (r/a)^k
    const double scale[4] = {
      1.0,
      1.0 / a,
      1.0 / (a*a),
      1.0 / (a*a*a)
    };

    auto J_plus  = [&](int n)->double  {
      return I_exp_plus(n,  r_hi, m) - I_exp_plus(n,  r_lo, m);
    };
    auto J_minus = [&](int n)->double  {
      return I_exp_minus(n, r_hi, m) - I_exp_minus(n, r_lo, m);
    };

    const double coeffs[4] = {c0, c1, c2, c3};

    for(int k=0; k<4; ++k){
      double ck = coeffs[k];
      if (ck == 0.0) continue;
      int n = k + 1; // r^{k+1} por el factor r

      double s = scale[k];
      double Jm = J_minus(n);
      double Jp = J_plus(n);

      dB += ck * s * Jm;
      dA += ck * s * 0.5 * (Jp - Jm);
    }

    // Conversión rho -> N_e
    const double conv = (rho_to_eV4 / mass_total);
    dA *= conv;
    dB *= conv;
  }

  // ==== A_B_upto_r: A(r), B(r) sumando tramos hasta r ====
  void A_B_upto_r(double r, double m, double& A, double& B) const {
    A = 0.0;
    B = 0.0;
    if (r <= 0.0) return;

    for(const auto& seg : prem_segments){
      if (r <= seg.r0) break;
      double hi = (r < seg.r1) ? r : seg.r1;

      double dA = 0.0, dB = 0.0;
      AB_on_interval(seg.r0, hi, seg, m, dA, dB);
      A += dA;
      B += dB;

      if (r <= seg.r1) break;
    }
  }

  // ==== Potencial LRI radial V_LRI(r) ====
  double V_LRI_at_r(double r) const {
    double r_eff = (r > 0.0) ? r : 1e-16;
    double A_r = 0.0, B_r = 0.0;
    A_B_upto_r(r_eff, mLR, A_r, B_r);

    // Phi(r) = (4π / (m r)) [ e^{-m r} A(r) + sinh(m r) (B(a) - B(r)) ]
    const double mr = mLR * r_eff;
    const double exp_mr = std::exp(-mr);
    const double sinh_mr = std::sinh(mr);

    double Phi = (4.0 * M_PI / (mLR * r_eff)) * (exp_mr * A_r + sinh_mr * (B_tot - B_r));

    // V_LRI = (g'^2 / 4π) * Phi
    return g2_over_4pi * Phi;
  }

  // ==== Wrapper: desde L (coordenada a lo largo de la trayectoria) ====
  double compute_VLRI_from_radius(double r) const {
    if (r <= 0.0) return V_LRI_at_r(1e-16);
    if (r > a)   return V_LRI_at_r(a);
    return V_LRI_at_r(r);
  }
};

