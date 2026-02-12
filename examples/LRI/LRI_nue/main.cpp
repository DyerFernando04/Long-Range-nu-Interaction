#include <nuSQuIDS/nuSQuIDS.h>
#include "LRI.h"

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
// Estructura de flujo leída del archivo:
// - E_GeV: eje de energías (NE)
// - theta_deg: eje de ángulos (NA) extraído del header si existe
// - flux_numu[ia][iE]
// =====================================================
struct SimpleFlux {
  size_t NE = 0;
  size_t NA = 0;
  std::vector<double> E_GeV;                 // size NE
  std::vector<double> theta_deg;             // size NA (si hay header)
  std::vector<std::vector<double>> flux_numu; // [NA][NE]
};

// Intenta extraer ángulo en grados desde un token tipo "theta_0.9deg"
static bool try_parse_theta_token(const std::string& tok, double& out_deg) {
  // Espera algo como: theta_<number>deg
  // Ejemplos válidos: theta_0.0deg, theta_10deg, theta_1.05deg
  const std::string prefix = "theta_";
  const std::string suffix = "deg";

  auto p = tok.find(prefix);
  auto q = tok.rfind(suffix);
  if (p == std::string::npos || q == std::string::npos) return false;
  if (q <= p + prefix.size()) return false;

  std::string num = tok.substr(p + prefix.size(), q - (p + prefix.size()));

  // Por si vienen cosas raras pegadas (coma, etc.), limpiamos extremos
  while (!num.empty() && (num.back() == ',' || num.back() == ';')) num.pop_back();
  while (!num.empty() && (num.front() == '"' || num.front() == '\'')) num.erase(num.begin());

  try {
    out_deg = std::stod(num);
    return true;
  } catch (...) {
    return false;
  }
}

// Lee archivo tipo:
// # E_GeV theta_0.0deg theta_0.9deg ...
// E  f1(E) f2(E) ...
static SimpleFlux read_numu_flux_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("No pude abrir archivo de flujo: " + path);

  SimpleFlux F;

  std::string line;
  bool first_data = true;
  size_t NA_local = 0;
  size_t line_num = 0;

  // Si hay header, aquí guardamos los thetas
  std::vector<double> header_thetas;

  while (std::getline(in, line)) {
    line_num++;

    if (line.empty()) continue;

    // Header/comment
    if (line[0] == '#') {
      // Intentamos parsear ángulos del header
      // Ejemplo: "# E_GeV theta_0.0deg theta_0.9deg ..."

      std::istringstream hs(line);
      std::string tok;
      std::vector<std::string> toks;
      while (hs >> tok) toks.push_back(tok);

      // Buscamos tokens theta_*deg
      std::vector<double> thetas_found;
      for (const auto& t : toks) {
        double th = 0.0;
        if (try_parse_theta_token(t, th)) thetas_found.push_back(th);
      }
      if (!thetas_found.empty()) {
        header_thetas = thetas_found;
      }
      continue;
    }

    // Parse numéricos en la línea
    std::istringstream ss(line);
    std::vector<double> vals;
    double x;
    while (ss >> x) vals.push_back(x);

    if (vals.empty()) continue;

    if (first_data) {
      if (vals.size() < 2) {
        std::cerr << "Advertencia: primera línea de datos en " << path
                  << " tiene menos de 2 columnas numéricas. "
                  << "Línea " << line_num << ": " << line << "\n";
        continue;
      }

      NA_local = vals.size() - 1; // 1 energía + NA flujos
      F.NA = NA_local;
      F.flux_numu.assign(NA_local, std::vector<double>{});
      first_data = false;

      std::cout << "read_numu_flux_file: primera línea válida en " << path
                << " tiene " << vals.size()
                << " columnas numéricas (1 energía + " << NA_local << " flujos)\n";
    } else {
      if (vals.size() != 1 + NA_local) {
        std::ostringstream msg;
        msg << "Inconsistencia de columnas en archivo de flujo " << path
            << " en la línea " << line_num
            << ": esperaba " << (1 + NA_local)
            << " números, pero encontré " << vals.size()
            << "\nContenido: " << line;
        throw std::runtime_error(msg.str());
      }
    }

    // Columna 0 = energía
    F.E_GeV.push_back(vals[0]);

    // Columnas 1..NA_local = flujos para cada ángulo
    for (size_t ia = 0; ia < NA_local; ++ia) {
      F.flux_numu[ia].push_back(vals[1 + ia]);
    }
  }

  if (first_data) {
    throw std::runtime_error("No se encontró ninguna línea de datos válida en " + path);
  }

  F.NE = F.E_GeV.size();
  if (F.NE == 0) throw std::runtime_error("Archivo de flujo vacío o sin datos válidos.");

  // Si header tiene thetas, verificamos tamaño
  if (!header_thetas.empty()) {
    if (header_thetas.size() == F.NA) {
      F.theta_deg = header_thetas;
    } else {
      std::cerr << "Advertencia: el header tiene " << header_thetas.size()
                << " ángulos pero el archivo tiene " << F.NA
                << " columnas de flujo. Ignoro header y continuo.\n";
    }
  }

  std::cout << "read_numu_flux_file: leído " << F.NE << " puntos de energía y "
            << F.NA << " ángulos.\n";

  return F;
}

// ============================
// Utilidades nuSQuIDS
// ============================
static void set_pmns_standard(nuSQUIDSLRI& nus) {
  nus.Set_MixingAngle(0,1, 0.563942);
  nus.Set_MixingAngle(0,2, 0.154085);
  nus.Set_MixingAngle(1,2, 0.785398);
  nus.Set_SquareMassDifference(1, 7.65e-5);
  nus.Set_SquareMassDifference(2, 2.47e-3);
  nus.Set_CPPhase(0,2, 0.0);
}

static marray<double,2> build_init_from_flux(
  const std::vector<double>& fe,
  const std::vector<double>& fmu,
  const std::vector<double>& ftau)
{
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

// ============================
// Programa principal
// ============================
int main(int argc, char** argv) {
  try {
    // Archivo de entrada opcional: ./lri miarchivo.txt
    std::string flux_file = "numu_flux_icecube_angles.txt";
    if (argc >= 2) flux_file = argv[1];

    // Mapeo angular:
    // por defecto: theta_det_deg = 180 - theta_file_deg (como tu código original)
    // si pones "--map=identity", usa theta_det_deg = theta_file_deg
    std::string map_mode = "flip180";
    if (argc >= 3) map_mode = argv[2]; // ej: "--map=identity"

    // 1) Leer flujo
    SimpleFlux SF = read_numu_flux_file(flux_file);

    // 2) Definir eje angular desde header (recomendado)
    std::vector<double> theta_file_deg;
    if (!SF.theta_deg.empty()) {
      theta_file_deg = SF.theta_deg; // tamaño NA
      std::cout << "Usando " << theta_file_deg.size()
                << " ángulos desde el header del archivo.\n";
    } else {
      // Si no hay header, aún puedes correr, pero no sabes qué ángulo es cada columna.
      // Por seguridad, aquí lanzamos error con explicación.
      throw std::runtime_error(
        "El archivo no trae header con theta_*deg. "
        "Agrega una línea tipo '# E_GeV theta_0.0deg theta_0.9deg ...' "
        "para poder mapear correctamente los ángulos."
      );
    }

    const size_t NE = SF.NE;
    const size_t NA = SF.NA;

    if (theta_file_deg.size() != NA) {
      throw std::runtime_error("Inconsistencia: número de thetas != número de columnas de flujo.");
    }

    // 3) Mapear ángulo del archivo al zenith del detector (en grados)
    std::vector<double> theta_det_deg(NA, 0.0);

    bool identity = (map_mode == "--map=identity");
    bool flip180  = (map_mode == "--map=flip180" || map_mode == "flip180");

    if (!identity && !flip180) {
      std::cerr << "Advertencia: modo '" << map_mode
                << "' no reconocido. Uso por defecto '--map=flip180'.\n";
      flip180 = true;
    }

    for (size_t ia=0; ia<NA; ++ia) {
      if (identity) theta_det_deg[ia] = theta_file_deg[ia];
      else          theta_det_deg[ia] = 180.0 - theta_file_deg[ia];
    }

    // 4) Preparar eje interno de energías
    squids::Const units;
    marray<double,1> E_internal({NE});
    for (size_t i=0;i<NE;i++) E_internal[i] = SF.E_GeV[i] * units.GeV;

    // 5) Construir Tierra + objetos nuSQuIDS (nu y anu)
    auto body = std::make_shared<EarthAtm>();

    nuSQUIDSLRI nus_nu(E_internal, 3, NeutrinoType::neutrino);
    nus_nu.InitializeLRI();
    nus_nu.Set_Body(body);
    set_pmns_standard(nus_nu);
    nus_nu.Set_rel_error(1e-6);
    nus_nu.Set_abs_error(1e-9);

    nuSQUIDSLRI nus_anu(E_internal, 3, NeutrinoType::antineutrino);
    nus_anu.InitializeLRI();
    nus_anu.Set_Body(body);
    set_pmns_standard(nus_anu);
    nus_anu.Set_rel_error(1e-6);
    nus_anu.Set_abs_error(1e-9);

    // 6) Salidas NA x NE
    auto zeros = std::vector<std::vector<double>>(NA, std::vector<double>(NE, 0.0));
    auto Out_nue        = zeros, Out_numu        = zeros, Out_nutau        = zeros;
    auto Out_antinue    = zeros, Out_antinumu    = zeros, Out_antinutau    = zeros;

    // 7) Loop por ángulos
    for (size_t ia=0; ia<NA; ++ia) {
      const double th_file_deg = theta_file_deg[ia];
      const double th_det_deg  = theta_det_deg[ia];
      const double theta_z_rad = th_det_deg * M_PI / 180.0;

      auto track_derived = std::make_shared<nusquids::EarthAtm::Track>( body->MakeTrack(theta_z_rad) );
      std::shared_ptr<nusquids::Body::Track> track = track_derived;

      // ----- NEUTRINOS -----
      nus_nu.Set_Track(track);
      nus_nu.SetZenithAngle(theta_z_rad);

      std::vector<double> fe(NE,0.0), fmu(NE,0.0), ftau(NE,0.0);
      for (size_t i=0;i<NE;i++){
        fe[i] = SF.flux_numu[ia][i]; // numu(E_i, ángulo ia)
      }

      auto init_state_nu = build_init_from_flux(fe, fmu, ftau);
      nus_nu.Set_initial_state(init_state_nu, Basis::flavor);
      nus_nu.EvolveState();

      for (size_t iE=0;iE<NE;iE++){
        double Ei = E_internal[iE];
        Out_nue[ia][iE]   = nus_nu.EvalFlavor(0, Ei, 0);
        Out_numu[ia][iE]  = nus_nu.EvalFlavor(1, Ei, 0);
        Out_nutau[ia][iE] = nus_nu.EvalFlavor(2, Ei, 0);
      }

      // ----- ANTINEUTRINOS -----
      nus_anu.Set_Track(track);
      nus_anu.SetZenithAngle(theta_z_rad);

      std::fill(fe.begin(), fe.end(), 0.0);
      std::fill(fmu.begin(), fmu.end(), 0.0);
      std::fill(ftau.begin(), ftau.end(), 0.0);

      for (size_t i=0;i<NE;i++){
        fmu[i] = SF.flux_numu[ia][i]; // por ahora igual que neutrinos
      }

      auto init_state_anu = build_init_from_flux(fe, fmu, ftau);
      nus_anu.Set_initial_state(init_state_anu, Basis::flavor);
      nus_anu.EvolveState();

      for (size_t iE=0;iE<NE;iE++){
        double Ei = E_internal[iE];
        Out_antinue[ia][iE]   = nus_anu.EvalFlavor(0, Ei, 0);
        Out_antinumu[ia][iE]  = nus_anu.EvalFlavor(1, Ei, 0);
        Out_antinutau[ia][iE] = nus_anu.EvalFlavor(2, Ei, 0);
      }

      std::cout << "Listo ia=" << ia
                << " theta_file=" << th_file_deg
                << " deg -> theta_z_det=" << th_det_deg
                << " deg\n";
    }

    // 8) Guardar resultados
    auto save_axis = [](const std::string& path, const std::vector<double>& v){
      std::ofstream out(path);
      out << std::setprecision(16);
      for (double x : v) out << x << "\n";
    };
    auto save_matrix = [](const std::string& path, const std::vector<std::vector<double>>& M){
      std::ofstream out(path);
      out << std::setprecision(16);
      for (size_t r=0; r<M.size(); ++r){
        for (size_t c=0; c<M[r].size(); ++c){
          if (c) out << " ";
          out << M[r][c];
        }
        out << "\n";
      }
    };

    save_axis("out_E_GeV.txt", SF.E_GeV);
    save_axis("out_theta_file_deg.txt", theta_file_deg);
    save_axis("out_theta_det_deg.txt",  theta_det_deg);

    save_matrix("out_flux_nue.txt",        Out_nue);
    save_matrix("out_flux_numu.txt",       Out_numu);
    save_matrix("out_flux_nutau.txt",      Out_nutau);
    save_matrix("out_flux_antinue.txt",    Out_antinue);
    save_matrix("out_flux_antinumu.txt",   Out_antinumu);
    save_matrix("out_flux_antinutau.txt",  Out_antinutau);

    std::cout << "OK: propagación (flujo_in -> flujo_out) completada.\n"
              << "Salidas NA x NE en out_flux_*.txt (NA=" << NA << ", NE=" << NE << ")\n";
    return 0;

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}

