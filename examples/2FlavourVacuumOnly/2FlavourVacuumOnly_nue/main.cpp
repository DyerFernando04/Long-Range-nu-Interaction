// =====================================================
// vac_osc_2flavor_nue.cpp
//
// nuSQuIDS vacuum oscillation in the 2-flavor-like limit
// matching notebook osc_flux (μ↔τ only):
//   - θ12=0, θ13=0, Δm21^2=0
//   - θ23=45° , Δm31^2=2.41e-3 eV^2
//   - Vacuum body + Vacuum track with notebook geometry L(coszen)
//
// INPUT FILE format (same as your MCEq tables):
//   # E_GeV theta_0.0deg theta_0.9deg ...
//   E  f(theta0) f(theta1) ...
//
// Here we read ONLY νe(E,theta) and propagate.
// In μ↔τ-only mixing, νe is unchanged (sanity check).
//
// Outputs (NA x NE):
//   out_E_GeV.txt
//   out_theta_file_deg.txt
//   out_theta_det_deg.txt
//   out_flux_nue.txt
//   out_flux_numu.txt
//   out_flux_nutau.txt
// =====================================================

#include <nuSQuIDS/nuSQuIDS.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using namespace nusquids;

// =====================================================
// Input structure: store only νe(E,theta) from file
// =====================================================
struct SimpleFlux {
  size_t NE = 0;
  size_t NA = 0;
  std::vector<double> E_GeV;                 // size NE
  std::vector<double> theta_deg;             // size NA
  std::vector<std::vector<double>> flux_nue; // [NA][NE]
};

static bool try_parse_theta_token(const std::string& tok, double& out_deg) {
  const std::string prefix = "theta_";
  const std::string suffix = "deg";

  auto p = tok.find(prefix);
  auto q = tok.rfind(suffix);
  if (p == std::string::npos || q == std::string::npos) return false;
  if (q <= p + prefix.size()) return false;

  std::string num = tok.substr(p + prefix.size(), q - (p + prefix.size()));

  while (!num.empty() && (num.back() == ',' || num.back() == ';')) num.pop_back();
  while (!num.empty() && (num.front() == '"' || num.front() == '\'')) num.erase(num.begin());

  try {
    out_deg = std::stod(num);
    return true;
  } catch (...) {
    return false;
  }
}

// Read file:
// # E_GeV theta_...deg theta_...deg ...
// E  f1 f2 ...
static SimpleFlux read_nue_flux_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("No pude abrir archivo de flujo: " + path);

  SimpleFlux F;

  std::string line;
  bool first_data = true;
  size_t NA_local = 0;
  size_t line_num = 0;

  std::vector<double> header_thetas;

  while (std::getline(in, line)) {
    line_num++;
    if (line.empty()) continue;

    if (line[0] == '#') {
      std::istringstream hs(line);
      std::string tok;
      std::vector<std::string> toks;
      while (hs >> tok) toks.push_back(tok);

      std::vector<double> thetas_found;
      for (const auto& t : toks) {
        double th = 0.0;
        if (try_parse_theta_token(t, th)) thetas_found.push_back(th);
      }
      if (!thetas_found.empty()) header_thetas = thetas_found;
      continue;
    }

    std::istringstream ss(line);
    std::vector<double> vals;
    double x;
    while (ss >> x) vals.push_back(x);
    if (vals.empty()) continue;

    if (first_data) {
      if (vals.size() < 2) {
        std::cerr << "Advertencia: primera linea de datos en " << path
                  << " tiene <2 columnas. Linea " << line_num << "\n";
        continue;
      }
      NA_local = vals.size() - 1;
      F.NA = NA_local;
      F.flux_nue.assign(NA_local, std::vector<double>{});
      first_data = false;

      std::cout << "read_nue_flux_file: primera linea valida: "
                << vals.size() << " cols (1E + " << NA_local << " flujos)\n";
    } else {
      if (vals.size() != 1 + NA_local) {
        std::ostringstream msg;
        msg << "Inconsistencia de columnas en " << path
            << " linea " << line_num
            << ": esperaba " << (1 + NA_local)
            << " y encontro " << vals.size()
            << "\nContenido: " << line;
        throw std::runtime_error(msg.str());
      }
    }

    F.E_GeV.push_back(vals[0]);
    for (size_t ia = 0; ia < NA_local; ++ia) {
      F.flux_nue[ia].push_back(vals[1 + ia]);
    }
  }

  if (first_data) throw std::runtime_error("No se encontro ninguna linea de datos valida en " + path);

  F.NE = F.E_GeV.size();
  if (F.NE == 0) throw std::runtime_error("Archivo vacio o sin datos validos.");

  if (!header_thetas.empty()) {
    if (header_thetas.size() == F.NA) {
      F.theta_deg = header_thetas;
    } else {
      std::cerr << "Advertencia: header tiene " << header_thetas.size()
                << " thetas pero hay " << F.NA
                << " columnas de flujo. Ignoro header.\n";
    }
  }

  if (F.theta_deg.empty()) {
    throw std::runtime_error(
      "El archivo no trae header con theta_*deg.\n"
      "Agrega una linea tipo:\n"
      "# E_GeV theta_0.0deg theta_0.9deg ...\n"
      "para mapear correctamente los angulos."
    );
  }

  std::cout << "read_nue_flux_file: leido NE=" << F.NE << " y NA=" << F.NA << "\n";
  return F;
}

// Build initial state marray (NE x 3) in flavor basis
static marray<double,2> build_init_from_flux(
  const std::vector<double>& fe,
  const std::vector<double>& fmu,
  const std::vector<double>& ftau
){
  const size_t NE = fe.size();
  if (fmu.size()!=NE || ftau.size()!=NE)
    throw std::runtime_error("Sizes de flujo no coinciden");
  marray<double,2> init({NE,(size_t)3});
  for (size_t i=0;i<NE;i++){
    init[{i,0}] = fe[i];
    init[{i,1}] = fmu[i];
    init[{i,2}] = ftau[i];
  }
  return init;
}

// 2-flavor-like setup matching osc_flux: μ↔τ only
static void set_pmns_2flavor_like_osc_flux(nuSQUIDS& nus) {
  nus.Set_MixingAngle(0,1, 0.0);           // theta12
  nus.Set_MixingAngle(0,2, 0.0);           // theta13
  nus.Set_MixingAngle(1,2, M_PI/4.0);      // theta23 = 45 deg

  nus.Set_SquareMassDifference(1, 0.0);     // dm21^2
  nus.Set_SquareMassDifference(2, 2.41e-3); // dm31^2

  nus.Set_CPPhase(0,2, 0.0);
}

// Notebook-style baseline geometry
static constexpr double L1_km = 19.0;
static constexpr double R_km  = 6378.2 + L1_km;

static double zen_to_d_km(double coszen){
  coszen = std::clamp(coszen, -1.0, 1.0);
  double zen = std::acos(coszen);
  double phi = std::asin((1.0 - L1_km / R_km) * std::sin(zen));
  double psi = zen - phi;
  return std::sqrt((R_km - L1_km)*(R_km - L1_km) + R_km*R_km
                   - 2.0*(R_km - L1_km)*R_km*std::cos(psi));
}

int main(int argc, char** argv) {
  try {
    // Usage:
    //   ./vac_osc_2flavor_nue nue_flux_icecube_angles.txt
    //   ./vac_osc_2flavor_nue nue_flux_icecube_angles.txt --map=identity
    //   ./vac_osc_2flavor_nue nue_flux_icecube_angles.txt --map=flip180   (default)

    std::string flux_file = "nue_flux_icecube_angles.txt";
    if (argc >= 2) flux_file = argv[1];

    std::string map_mode = "--map=flip180";
    if (argc >= 3) map_mode = argv[2];

    // 1) Read input νe(E,theta_file)
    SimpleFlux SF = read_nue_flux_file(flux_file);
    const size_t NE = SF.NE;
    const size_t NA = SF.NA;

    std::vector<double> theta_file_deg = SF.theta_deg;

    // 2) Map theta_file -> theta_det
    std::vector<double> theta_det_deg(NA, 0.0);
    bool identity = (map_mode == "--map=identity");
    bool flip180  = (map_mode == "--map=flip180" || map_mode == "flip180");
    if (!identity && !flip180) {
      std::cerr << "Advertencia: modo '" << map_mode
                << "' no reconocido. Uso '--map=flip180'.\n";
      flip180 = true;
    }
    for (size_t ia = 0; ia < NA; ++ia) {
      theta_det_deg[ia] = identity ? theta_file_deg[ia] : (180.0 - theta_file_deg[ia]);
    }

    // 3) Energy axis in nuSQuIDS units
    squids::Const units;
    marray<double,1> E_internal({NE});
    for (size_t i=0; i<NE; ++i) E_internal[i] = SF.E_GeV[i] * units.GeV;

    // 4) Vacuum body
    auto body = std::make_shared<Vacuum>();

    // 5) nuSQuIDS neutrinos in vacuum
    nuSQUIDS nus_nu(E_internal, 3, NeutrinoType::neutrino);
    nus_nu.Set_Body(body);
    set_pmns_2flavor_like_osc_flux(nus_nu);
    nus_nu.Set_rel_error(1e-7);
    nus_nu.Set_abs_error(1e-12);

    // 6) Outputs NA x NE
    std::vector<std::vector<double>> Out_nue(NA,   std::vector<double>(NE, 0.0));
    std::vector<std::vector<double>> Out_numu(NA,  std::vector<double>(NE, 0.0));
    std::vector<std::vector<double>> Out_nutau(NA, std::vector<double>(NE, 0.0));

    // 7) Loop over angles
    for (size_t ia = 0; ia < NA; ++ia) {
      const double th_file_deg = theta_file_deg[ia];
      const double th_det_deg  = theta_det_deg[ia];

      const double theta_z_rad = th_det_deg * M_PI / 180.0;
      const double coszen = std::cos(theta_z_rad);

      const double L_km = zen_to_d_km(coszen);

      std::shared_ptr<nusquids::Body::Track> track =
        std::make_shared<nusquids::Vacuum::Track>(L_km * units.km);

      nus_nu.Set_Track(track);

      // Initial state: pure νe, no νμ, no ντ
      std::vector<double> fe(NE, 0.0), fmu(NE, 0.0), ftau(NE, 0.0);
      for (size_t i=0; i<NE; ++i) {
        fe[i] = SF.flux_nue[ia][i];
      }

      auto init_state_nu = build_init_from_flux(fe, fmu, ftau);
      nus_nu.Set_initial_state(init_state_nu, Basis::flavor);
      nus_nu.EvolveState();

      for (size_t iE=0; iE<NE; ++iE) {
        double Ei = E_internal[iE];
        Out_nue[ia][iE]   = nus_nu.EvalFlavor(0, Ei, 0);
        Out_numu[ia][iE]  = nus_nu.EvalFlavor(1, Ei, 0);
        Out_nutau[ia][iE] = nus_nu.EvalFlavor(2, Ei, 0);
      }

      std::cout << "Listo ia=" << ia
                << " theta_file=" << th_file_deg
                << " deg -> theta_z_det=" << th_det_deg
                << " deg | L=" << L_km << " km\n";
    }

    // 8) Save outputs
    auto save_axis = [](const std::string& path, const std::vector<double>& v){
      std::ofstream out(path);
      if (!out) throw std::runtime_error("No pude escribir: " + path);
      out << std::setprecision(16);
      for (double x : v) out << x << "\n";
    };

    auto save_matrix = [](const std::string& path, const std::vector<std::vector<double>>& M){
      std::ofstream out(path);
      if (!out) throw std::runtime_error("No pude escribir: " + path);
      out << std::setprecision(16);
      for (size_t r=0; r<M.size(); ++r) {
        for (size_t c=0; c<M[r].size(); ++c) {
          if (c) out << " ";
          out << M[r][c];
        }
        out << "\n";
      }
    };

    save_axis("out_E_GeV.txt", SF.E_GeV);
    save_axis("out_theta_file_deg.txt", theta_file_deg);
    save_axis("out_theta_det_deg.txt",  theta_det_deg);

    save_matrix("out_flux_nue.txt",   Out_nue);
    save_matrix("out_flux_numu.txt",  Out_numu);
    save_matrix("out_flux_nutau.txt", Out_nutau);

    std::cout << "OK: VACUUM 2-flavor-like propagation (νe input) complete.\n"
              << "Outputs NA x NE in out_flux_*.txt (NA=" << NA << ", NE=" << NE << ")\n";
    return 0;

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}

