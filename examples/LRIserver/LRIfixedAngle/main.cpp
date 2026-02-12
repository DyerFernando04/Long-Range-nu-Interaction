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

using namespace nusquids;

// =====================================================
// Lector simple de archivo: E_GeV  numu_theta1  numu_theta2 ...
// =====================================================
struct SimpleFlux {
  size_t NE = 0;             // número de energías
  size_t NA = 0;             // número de ángulos
  std::vector<double> E_GeV; // tamaño NE
  // flux_numu[ia][iE] = flujo numu(E_i, ángulo ia)
  std::vector<std::vector<double>> flux_numu;
};

// Lee archivo tipo:
// # comentario
// E  f1(E) f2(E) f3(E) f4(E)
// ...
static SimpleFlux read_numu_flux_file(const std::string& path){
  std::ifstream in(path);
  if(!in) throw std::runtime_error("No pude abrir archivo de flujo: " + path);

  SimpleFlux F;
  std::string line;
  bool first_data = true;
  size_t NA_local = 0;
  size_t line_num = 0;

  while(std::getline(in, line)){
    line_num++;
    if(line.empty()) continue;
    if(line[0] == '#') continue;  // ignorar comentarios

    std::istringstream ss(line);
    std::vector<double> vals;
    double x;
    while(ss >> x) vals.push_back(x);

    // Línea sin números → la saltamos (por si hubiera algo raro)
    if(vals.empty()) continue;

    if(first_data){
      if(vals.size() < 2){
        std::cerr << "Advertencia: primera línea de datos en "
                  << path << " tiene menos de 2 columnas numéricas. "
                  << "Línea " << line_num << ": " << line << "\n";
        continue; // intenta con la siguiente línea
      }
      NA_local = vals.size() - 1;   // una columna es E, el resto flujos
      F.NA     = NA_local;
      F.flux_numu.assign(NA_local, std::vector<double>{});

      first_data = false;
      std::cout << "read_numu_flux_file: primera línea válida en "
                << path << " tiene " << vals.size()
                << " columnas numéricas (" << 1 << " energía + "
                << NA_local << " flujos)\n";
    } else {
      if(vals.size() != 1 + NA_local){
        std::ostringstream msg;
        msg << "Inconsistencia de columnas en archivo de flujo "
            << path << " en la línea " << line_num
            << ": esperaba " << (1 + NA_local)
            << " números, pero encontré " << vals.size()
            << "\nContenido de la línea: " << line;
        throw std::runtime_error(msg.str());
      }
    }

    // columna 0 = energía
    F.E_GeV.push_back(vals[0]);

    // columnas 1..NA_local = flujos numu
    for(size_t ia=0; ia<NA_local; ++ia){
      F.flux_numu[ia].push_back(vals[1 + ia]);
    }
  }

  if(first_data){
    throw std::runtime_error("No se encontró ninguna línea de datos válida en " + path);
  }

  F.NE = F.E_GeV.size();
  if(F.NE == 0) throw std::runtime_error("Archivo de flujo vacío o sin datos válidos.");

  std::cout << "read_numu_flux_file: leído " << F.NE
            << " puntos de energía y " << F.NA
            << " ángulos.\n";

  return F;
}

// ============================
// Utilidades nuSQuIDS
// ============================
static void set_pmns_standard(nuSQUIDSLRI& nus){
  // Mismos valores que en tu código viejo
  nus.Set_MixingAngle(0,1, 0.563942);
  nus.Set_MixingAngle(0,2, 0.154085);
  nus.Set_MixingAngle(1,2, 0.785398);
  nus.Set_SquareMassDifference(1, 7.65e-5);
  nus.Set_SquareMassDifference(2, 2.47e-3);
  nus.Set_CPPhase(0,2, 0.0);
}

// Construye init_state (NE x 3) desde flujos por sabor (e, mu, tau)
static marray<double,2> build_init_from_flux(
  const std::vector<double>& fe,
  const std::vector<double>& fmu,
  const std::vector<double>& ftau)
{
  const size_t NE = fe.size();
  if(fmu.size()!=NE || ftau.size()!=NE) throw std::runtime_error("Sizes de flujo no coinciden");
  marray<double,2> init({NE,(size_t)3});
  for(size_t i=0;i<NE;i++){
    init[{i,0}] = fe[i];
    init[{i,1}] = fmu[i];
    init[{i,2}] = ftau[i];
  }
  return init;
}

// ============================
// Programa principal
// ============================
int main(){
  try{
    // 1) Cargar archivo con flujo numu(E, ángulos_MCEq)
    const std::string flux_file = "numu_flux_icecube_angles.txt";
    SimpleFlux SF = read_numu_flux_file(flux_file);

    // Esperamos 4 ángulos: 90, 60, 30, 0 grados en MCEq
    if(SF.NA != 4){
      std::cerr << "Advertencia: se esperaban 4 ángulos (90,60,30,0 deg). NA="<<SF.NA<<"\n";
    }

    // Definimos explícitamente los ángulos MCEq (en grados) y el mapeo a zenit detector
    std::vector<double> theta_MCEq_deg = {90.0, 60.0, 30.0, 0.0001};
    if(SF.NA < theta_MCEq_deg.size()){
      throw std::runtime_error("El archivo tiene menos columnas de ángulos que las esperadas.");
    }

    // Ángulos de zenit en el detector: theta_z = 180° - theta_MCEq
    std::vector<double> theta_det_deg(theta_MCEq_deg.size());
    for(size_t ia=0; ia<theta_MCEq_deg.size(); ++ia){
      theta_det_deg[ia] = 180.0 - theta_MCEq_deg[ia];
    }

    // 2) Preparar eje interno de energías
    squids::Const units;
    const size_t NE = SF.NE;
    const size_t NA = theta_MCEq_deg.size(); // trabajamos sólo con estos 4
    marray<double,1> E_internal({NE});
    for(size_t i=0;i<NE;i++) E_internal[i] = SF.E_GeV[i] * units.GeV;

    // 3) Construir objetos físicos (Tierra + LRI) para ν y ν̄
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

    // 4) Salidas (NA x NE) de flujos en el detector
    auto zeros = std::vector<std::vector<double>>(NA, std::vector<double>(NE,0.0));
    auto Out_nue        = zeros, Out_numu        = zeros, Out_nutau        = zeros;
    auto Out_antinue    = zeros, Out_antinumu    = zeros, Out_antinutau    = zeros;

    // 5) Loop por ángulos MCEq (mapeados a zenit detector)
    for(size_t ia=0; ia<NA; ++ia){
      const double th_MCEq_deg  = theta_MCEq_deg[ia];    // 90,60,30,0
      const double th_det_deg   = theta_det_deg[ia];     // 90,120,150,180
      const double theta_z_rad  = th_det_deg * M_PI / 180.0;

      auto track_derived = std::make_shared<nusquids::EarthAtm::Track>( body->MakeTrack(theta_z_rad) );
      std::shared_ptr<nusquids::Body::Track> track = track_derived;

      // ----- NEUTRINOS -----
      nus_nu.Set_Track(track);
      nus_nu.SetZenithAngle(theta_z_rad);

      // Sólo flujo numu. nue = nutau = 0
      std::vector<double> fe(NE,0.0), fmu(NE,0.0), ftau(NE,0.0);
      for(size_t i=0;i<NE;i++){
        fmu[i] = SF.flux_numu[ia][i];  // numu(E_i, ángulo ia)
      }

      auto init_state_nu = build_init_from_flux(fe, fmu, ftau);
      nus_nu.Set_initial_state(init_state_nu, Basis::flavor);
      nus_nu.EvolveState();

      for(size_t iE=0;iE<NE;iE++){
        double Ei = E_internal[iE];
        Out_nue[ia][iE]   = nus_nu.EvalFlavor(0, Ei, 0);
        Out_numu[ia][iE]  = nus_nu.EvalFlavor(1, Ei, 0);
        Out_nutau[ia][iE] = nus_nu.EvalFlavor(2, Ei, 0);
      }

      // ----- ANTINEUTRINOS -----
      // Por ahora usamos el mismo flujo numu como entrada para ν̄μ.
      nus_anu.Set_Track(track);
      nus_anu.SetZenithAngle(theta_z_rad);

      std::fill(fe.begin(), fe.end(), 0.0);
      std::fill(fmu.begin(), fmu.end(), 0.0);
      std::fill(ftau.begin(), ftau.end(), 0.0);
      for(size_t i=0;i<NE;i++){
        fmu[i] = SF.flux_numu[ia][i];  // usar mismo flujo para ν̄μ (puedes cambiar esto luego)
      }

      auto init_state_anu = build_init_from_flux(fe, fmu, ftau);
      nus_anu.Set_initial_state(init_state_anu, Basis::flavor);
      nus_anu.EvolveState();

      for(size_t iE=0;iE<NE;iE++){
        double Ei = E_internal[iE];
        Out_antinue[ia][iE]   = nus_anu.EvalFlavor(0, Ei, 0);
        Out_antinumu[ia][iE]  = nus_anu.EvalFlavor(1, Ei, 0);
        Out_antinutau[ia][iE] = nus_anu.EvalFlavor(2, Ei, 0);
      }

      std::cout<<"Listo ángulo MCEq "<<th_MCEq_deg<<" deg -> theta_z_det="<<th_det_deg<<" deg\n";
    }

    // 6) Guardar resultados
    auto save_axis = [](const std::string& path, const std::vector<double>& v){
      std::ofstream out(path); out<<std::setprecision(16);
      for(double x: v) out<<x<<"\n";
    };
    auto save_matrix = [](const std::string& path, const std::vector<std::vector<double>>& M){
      std::ofstream out(path); out<<std::setprecision(16);
      for(size_t r=0;r<M.size();++r){
        for(size_t c=0;c<M[r].size();++c){
          if(c) out<<" ";
          out<<M[r][c];
        }
        out<<"\n";
      }
    };

    save_axis("out_E_GeV.txt", SF.E_GeV);
    save_axis("out_theta_MCEq_deg.txt", theta_MCEq_deg);
    save_axis("out_theta_det_deg.txt",  theta_det_deg);

    save_matrix("out_flux_nue.txt",        Out_nue);
    save_matrix("out_flux_numu.txt",       Out_numu);
    save_matrix("out_flux_nutau.txt",      Out_nutau);
    save_matrix("out_flux_antinue.txt",    Out_antinue);
    save_matrix("out_flux_antinumu.txt",   Out_antinumu);
    save_matrix("out_flux_antinutau.txt",  Out_antinutau);

    std::cout<<"OK: propagación (flujo_in → flujo_out) completada. Salidas NA x NE en out_flux_*.txt\n";
    return 0;

  }catch(const std::exception& ex){
    std::cerr<<"Error: "<<ex.what()<<"\n";
    return 1;
  }
}

