#include <nuSQuIDS/nuSQuIDS.h>
#include "LRI.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <fstream>
#include <iomanip>

using namespace nusquids;

// linspace -> marray<double,1>
marray<double,1> linspace(double a, double b, int n){
  marray<double,1> arr({(size_t)n});
  if(n==1){ arr[0]=a; return arr; }
  const double h=(b-a)/(n-1);
  for(int i=0;i<n;i++) arr[i]=a + h*i;
  return arr;
}

int main(){
  try{
    // Energías
    squids::Const units;
    const int NE = 120;
    auto E_GeV = linspace(1.25, 10.0, NE);
    marray<double,1> E({(size_t)NE});
    for(int i=0;i<NE;i++) E[i] = E_GeV[i] * units.GeV;

    // Objeto con LRI
    nuSQUIDSLRI nus_LRI(E, 3, NeutrinoType::neutrino);
    nus_LRI.InitializeLRI();

    // Body Tierra+atm
    auto body = std::make_shared<EarthAtm>();
    nus_LRI.Set_Body(body);

    // Mezcla y Δm2
    nus_LRI.Set_MixingAngle(0,1, 0.563942);
    nus_LRI.Set_MixingAngle(0,2, 0.154085);
    nus_LRI.Set_MixingAngle(1,2, 0.785398);
    nus_LRI.Set_SquareMassDifference(1, 7.65e-5);
    nus_LRI.Set_SquareMassDifference(2, 2.47e-3);
    nus_LRI.Set_CPPhase(0,2, 0.0);

    // estado inicial: nu_mu
    marray<double,2> init_state({(size_t)NE, (size_t)3});
    for(int i=0;i<NE;i++){
      init_state[{(size_t)i, (size_t)0}] = 0.0;
      init_state[{(size_t)i, (size_t)1}] = 1.0;
      init_state[{(size_t)i, (size_t)2}] = 0.0;
    }

    nus_LRI.Set_rel_error(1e-6);
    nus_LRI.Set_abs_error(1e-9);

    // Barrido angular (deg)
    const int NANG = 120;
    std::vector<double> angles_deg(NANG);
    for(int i=0;i<NANG;i++){
      angles_deg[i] = 90.001 + (179.999 - 90.001) * i / (NANG - 1);
    }

    // Matriz P_mumu[ángulo][energía]
    std::vector<std::vector<double>> P_mumu(NANG, std::vector<double>(NE, 0.0));

    for(int ia=0; ia<NANG; ++ia){
      const double ang_deg = angles_deg[ia];
      const double theta_z = ang_deg * M_PI / 180.0;

      // Track (derivado -> base)
      auto track_derived = std::make_shared<nusquids::EarthAtm::Track>( body->MakeTrack(theta_z) );
      std::shared_ptr<nusquids::Body::Track> track = track_derived; // upcast implícito
      nus_LRI.Set_Track(track);

      nus_LRI.SetZenithAngle(theta_z);
      nus_LRI.Set_initial_state(init_state, Basis::flavor);

      nus_LRI.EvolveState();

      for(int iE=0;iE<NE;iE++){
        const double Ei = E[iE];
        // Tu función válida en esta versión:
        P_mumu[ia][iE] = nus_LRI.EvalFlavor(1, Ei, 0); // P(nu_mu -> nu_mu)
      }

      std::cout << "Ángulo " << ang_deg << "° listo.\n";
    }

    // ======= Guardado de resultados =======
    // Archivo principal: matriz con cabecera
    {
      std::ofstream out("Pmumu_LRI_matrix.txt");
      out << std::setprecision(16);

      // Cabecera con dimensiones y ejes
      out << "# P_mumu con LRI\n";
      out << "# filas=NANG=" << NANG << "  columnas=NE=" << NE << "\n";
      out << "# columnas: angle_deg  seguido de NE probabilidades (en el mismo orden de E_GeV)\n";
      out << "# E_GeV:";
      for(int iE=0;iE<NE;iE++) out << " " << (double)E_GeV[iE];
      out << "\n";

      // Cada fila: ángulo (deg) + NE valores P_mumu
      for(int ia=0; ia<NANG; ++ia){
        out << angles_deg[ia];
        for(int iE=0;iE<NE;iE++) out << " " << P_mumu[ia][iE];
        out << "\n";
      }
    }

    // (Opcional) guardar ejes por separado, útil para Python/ROOT
    {
      std::ofstream fx("E_GeV_axis.txt");
      fx << std::setprecision(16);
      for(int iE=0;iE<NE;iE++) fx << (double)E_GeV[iE] << "\n";
    }
    {
      std::ofstream fa("angles_deg_axis.txt");
      fa << std::setprecision(16);
      for(int ia=0; ia<NANG; ++ia) fa << angles_deg[ia] << "\n";
    }

    std::cout << "OK: evolución con V_LRI completada.\n";
    std::cout << "Archivos:\n";
    std::cout << "  - Pmumu_LRI_matrix.txt\n";
    std::cout << "  - E_GeV_axis.txt\n";
    std::cout << "  - angles_deg_axis.txt\n";

    return 0;
  }catch(const std::exception& ex){
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}




