#pragma once
#include <nuSQuIDS/nuSQuIDS.h>
#include <cmath>
#include <vector>
#include <iostream>

using namespace nusquids;

class nuSQUIDSLRI : public nuSQUIDS {
public:
  double g_prime_e_beta = 1e-26;
  double m_e_beta       = 1e-13;

  double rho_to_eV4 = 4.3625e18;
  double m_p = 938.27e6;  // eV
  double m_n = 939.57e6;  // eV
  double m_e = 0.511e6;   // eV
  double mass_total;

  double a; // radio Tierra en eV^{-1}

  int NR  = 20;
  int NTH = 20;
  int NPH = 20;

  double theta_z = M_PI;

  using nuSQUIDS::nuSQUIDS;

  void InitializeLRI(){
    mass_total = m_p + m_n + m_e;
    a = 6371.0 * 5.067731e9;
  }

  void SetZenithAngle(double theta_z_rad){ theta_z = theta_z_rad; }

  squids::SU_vector HI(unsigned int ie, unsigned int irho, double x) const override {
    squids::SU_vector H = nuSQUIDS::HI(ie, irho);

    const double L = x;
    const double V_e_beta = compute_VLRI(L);

    // diag(V, -V, 0) = V*(|e><e| - |mu><mu|)
    squids::SU_vector term = GetFlavorProj(0); // |e>
    term -= GetFlavorProj(1);                  // -|mu>
    term *= V_e_beta;
    H += term;

    return H;
  }

private:
  double l_max() const {
    const double c = std::cos(theta_z);
    double Lmax = 2.0 * a * (-c);
    if (Lmax < 0) Lmax = 0;
    if (Lmax > 2.0 * a) Lmax = 2.0 * a;
    return Lmax;
  }

  double radius_at(double L) const {
    const double Lmax = l_max();
    const double s = L - 0.5 * Lmax;
    const double b = a * std::sin(theta_z);
    double r = std::sqrt(b*b + s*s);
    if (r > a) r = a;
    if (r < 0.0) r = 0.0;
    return r;
  }

  double alpha_at(double L, double r) const {
    const double Lmax = l_max();
    if (r <= 0.0) return M_PI;
    const double arg = (r*r + a*a - (Lmax - L)*(Lmax - L)) / (2.0 * a * r);
    if (arg < -1.0) return M_PI;
    if (arg >  1.0) return 0.0;
    return std::acos(arg);
  }

  double electron_density_prem(double r) const {
    const double x = r / a;
    auto K = [](double km){ return km * 5.067731e9; };

    if (0.0 <= r && r < K(1221.5))  return 13.0885 - 8.8381 * x*x;
    else if (r < K(3480))           return 12.5815 - 1.2638*x - 3.6426*x*x - 5.5281*x*x*x;
    else if (r < K(3630))           return 7.9565 - 6.4761*x + 5.5283*x*x - 3.0807*x*x*x;
    else if (r < K(5600))           return 7.9565 - 6.4761*x + 5.5283*x*x - 3.0807*x*x*x;
    else if (r < K(5701))           return 7.9565 - 6.4761*x + 5.5283*x*x - 3.0807*x*x*x;
    else if (r < K(5771))           return 5.3197 - 1.4836*x;
    else if (r < K(5971))           return 11.2494 - 8.0298*x;
    else if (r < K(6151))           return 7.1089 - 3.8045 * x;
    else if (r < K(6291))           return 2.6910 + 0.6924*x;
    else if (r < K(6346.6))         return 2.6910 + 0.6924*x;
    else if (r < K(6356))           return 2.9;
    else if (r < K(6368))           return 2.6;
    else if (r <= K(6371))          return 1.02;
    else                            return 0.0;
  }

  double number_density_from_rho(double rho_g_cm3) const {
    return (rho_g_cm3 * rho_to_eV4) / mass_total;
  }

  static double simpson_1d(const std::vector<double>& f, double a0, double a1){
    const size_t n = f.size();
    if (n < 2) return 0.0;
    const double h = (a1 - a0) / (n - 1);
    double s = f[0] + f[n-1];
    for (size_t i=1; i+1<n; ++i) s += ((i & 1) ? 4.0 : 2.0) * f[i];
    return s * (h/3.0);
  }

  double compute_VLRI(double L) const {
    const double r  = radius_at(L);
    const double al = alpha_at(L, r);

    const double rho_max = 2.0 * a;
    const int NRloc = NR, NTHloc = NTH, NPHloc = NPH;

    std::vector<double> rho_vals(NRloc), th_vals(NTHloc), ph_vals(NPHloc);
    for (int i=0;i<NRloc; ++i)  rho_vals[i] = (rho_max * i) / (NRloc - 1);
    for (int j=0;j<NTHloc;++j)  th_vals [j] = (M_PI * j)    / (NTHloc - 1);
    for (int k=0;k<NPHloc;++k)  ph_vals [k] = (2.0*M_PI*k)  / (NPHloc - 1);

    const double cA = std::cos(al - M_PI*0.5);
    const double sA = std::sin(al - M_PI*0.5);

    auto integrate_over_phi = [&](double rho, double th)->double{
      std::vector<double> fphi(NPHloc);
      const double sin_th = std::sin(th);
      const double cos_th = std::cos(th);
      for (int k=0;k<NPHloc;++k){
        const double phi = ph_vals[k];
        const double cphi = std::cos(phi);
        const double sphi = std::sin(phi);

        const double xs = r * cA + rho * sin_th * cphi;
        const double ys =              rho * sin_th * sphi;
        const double zs = r * sA + rho * cos_th;
        const double rp = std::sqrt(xs*xs + ys*ys + zs*zs);

        double ne_val = 0.0;
        if (rp <= a){
          const double rho_g_cm3 = electron_density_prem(rp);
          ne_val = number_density_from_rho(rho_g_cm3);
        }
        fphi[k] = rho * ne_val * sin_th * std::exp(-m_e_beta * rho);
      }
      return simpson_1d(fphi, ph_vals.front(), ph_vals.back());
    };

    auto integrate_over_theta = [&](double rho)->double{
      std::vector<double> fth(NTHloc);
      for (int j=0;j<NTHloc;++j) fth[j] = integrate_over_phi(rho, th_vals[j]);
      return simpson_1d(fth, th_vals.front(), th_vals.back());
    };

    std::vector<double> frho(NRloc);
    for (int i=0;i<NRloc;++i) frho[i] = integrate_over_theta(rho_vals[i]);
    const double integral = simpson_1d(frho, rho_vals.front(), rho_vals.back());

    const double gp2_over_4pi = (g_prime_e_beta * g_prime_e_beta) / (4.0 * M_PI);
    return gp2_over_4pi * integral;
  }
};


