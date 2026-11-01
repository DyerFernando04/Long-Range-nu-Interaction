// oscillogram_pee.cpp
//
// Compute 2D electron-neutrino survival oscillograms using the same
// nuSQUIDSLRI + EarthDepthBody machinery as the LUT2 production.
//
// Outputs, for every (coszen,E):
//
//   Pee_nu, Pemu_nu, Petau_nu,
//   Pee_antinu, Pemu_antinu, Petau_antinu
//
// Non-coherent interactions are DISABLED: these are genuine flavor
// probabilities, not propagated atmospheric fluxes.
//
// Usage:
//
//   oscillogram_pee LUT1.h5 output.csv MODE log10_m \
//                   [N_E] [N_CZ] [E_MIN_GEV] [E_MAX_GEV]
//
// MODE:
//   SM        -> g'=0 exactly
//   -25.0     -> g'=10^-25, etc.
//
// coszen is always sampled from -1 to 0.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <nuSQuIDS/nuSQuIDS.h>

#include "LRI.h"

using namespace nusquids;

namespace {

constexpr int N_FLAVORS = 3;

constexpr double THETA_01 = 0.563942;
constexpr double THETA_02 = 0.154085;
constexpr double THETA_12 = 0.785398;
constexpr double DELTA_CP = 0.0;

constexpr double DM2_10_EV2 = 7.65e-5;
constexpr double DM2_20_EV2 = 2.47e-3;

constexpr double DETECTOR_DEPTH_KM = 1.5;

constexpr double H_MAX_KM = 200.0;
constexpr double REL_ERROR = 1.0e-9;
constexpr double ABS_ERROR = 1.0e-9;

void Require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

marray<double, 1>
BuildEnergyRange(
    std::size_t number_energy,
    double energy_min_GeV,
    double energy_max_GeV)
{
    Require(number_energy >= 2, "Need at least two energy points.");
    Require(
        energy_min_GeV > 0.0 && energy_max_GeV > energy_min_GeV,
        "Invalid energy range."
    );

    squids::Const units;
    marray<double, 1> energy_range({number_energy});

    const double log_min = std::log10(energy_min_GeV);
    const double log_max = std::log10(energy_max_GeV);

    for (std::size_t i = 0; i < number_energy; ++i) {
        const double f =
            static_cast<double>(i) /
            static_cast<double>(number_energy - 1);

        const double energy_GeV =
            std::pow(10.0, log_min + f * (log_max - log_min));

        energy_range[i] = energy_GeV * units.GeV;
    }

    return energy_range;
}

std::vector<std::array<double, 3>>
ComputeOneCoszen(
    double coupling,
    double coszen,
    nusquids::NeutrinoType neutrino_type,
    const std::shared_ptr<const VLRIRadialProfile>& radial_profile,
    const marray<double, 1>& energy_range,
    const std::shared_ptr<EarthDepthBody>& body)
{
    const std::size_t nE = energy_range.extent(0);

    auto track = body->MakeTrackWithCosineToDetector(coszen);

    nuSQUIDSLRI nus(
        coupling,
        radial_profile,
        energy_range,
        N_FLAVORS,
        neutrino_type,
        false,  // interactions OFF
        THETA_01,
        THETA_02,
        THETA_12
    );

    nus.Set_Body(body);
    nus.Set_Track(track);

    nus.Set_SquareMassDifference(1, DM2_10_EV2);
    nus.Set_SquareMassDifference(2, DM2_20_EV2);

    if (DELTA_CP != 0.0) {
        nus.Set_CPPhase(0, 2, DELTA_CP);
    }

    squids::Const units;

    nus.Set_h_max(H_MAX_KM * units.km);
    nus.Set_rel_error(REL_ERROR);
    nus.Set_abs_error(ABS_ERROR);
    nus.Set_GSL_step(gsl_odeiv2_step_rk4);
    nus.Set_ProgressBar(false);

    // Pure electron flavor at production.
    marray<double, 2> initial_state({
        nE,
        static_cast<std::size_t>(N_FLAVORS)
    });

    std::fill(
        initial_state.begin(),
        initial_state.end(),
        0.0
    );

    for (std::size_t e = 0; e < nE; ++e) {
        initial_state[e][0] = 1.0;
    }

    nus.Set_initial_state(initial_state, flavor);
    nus.EvolveState();

    std::vector<std::array<double, 3>> result(nE);

    for (std::size_t e = 0; e < nE; ++e) {
        double sum = 0.0;

        for (int beta = 0; beta < N_FLAVORS; ++beta) {
            double value = nus.EvalFlavorAtNode(
                static_cast<unsigned int>(beta),
                static_cast<unsigned int>(e),
                0
            );

            if (value < 0.0 && std::abs(value) < 1.0e-11) {
                value = 0.0;
            }

            if (value > 1.0 && std::abs(value - 1.0) < 1.0e-11) {
                value = 1.0;
            }

            Require(std::isfinite(value), "Non-finite probability.");

            result[e][static_cast<std::size_t>(beta)] = value;
            sum += value;
        }

        Require(
            std::abs(sum - 1.0) < 5.0e-5,
            "Probability conservation failed."
        );
    }

    return result;
}

}  // namespace

int main(int argc, char* argv[])
{
    try {
        gsl_set_error_handler_off();

        if (argc < 5 || argc > 9) {
            std::cerr
                << "Usage:\n  "
                << argv[0]
                << " LUT1.h5 output.csv MODE log10_m "
                << "[N_E] [N_CZ] [E_MIN_GEV] [E_MAX_GEV]\n\n"
                << "MODE = SM or numeric log10(g), e.g. -25\n";
            return 1;
        }

        const std::string lut_filename = argv[1];
        const std::string output_filename = argv[2];
        const std::string mode = argv[3];
        const double log10_m = std::stod(argv[4]);

        const std::size_t nE =
            (argc >= 6)
            ? static_cast<std::size_t>(std::stoul(argv[5]))
            : 180;

        const std::size_t nCZ =
            (argc >= 7)
            ? static_cast<std::size_t>(std::stoul(argv[6]))
            : 101;

        const double e_min_GeV =
            (argc >= 8)
            ? std::stod(argv[7])
            : 1.0;

        const double e_max_GeV =
            (argc >= 9)
            ? std::stod(argv[8])
            : 200.0;

        Require(nCZ >= 2, "Need at least two coszen points.");

        const bool is_sm = (mode == "SM" || mode == "sm");

        const double coupling =
            is_sm
            ? 0.0
            : std::pow(10.0, std::stod(mode));

        const double mediator_mass_eV =
            std::pow(10.0, log10_m);

        const auto energy_range =
            BuildEnergyRange(nE, e_min_GeV, e_max_GeV);

        std::cout << "Loading LUT1: " << lut_filename << '\n';

        const auto library =
            LRIPotentialLibrary::LoadFromHDF5(lut_filename);

        const auto radial_profile =
            library->BuildRadialProfile(mediator_mass_eV);

        const auto body =
            std::make_shared<EarthDepthBody>(DETECTOR_DEPTH_KM);

        std::ofstream output(output_filename);

        Require(output.good(), "Cannot open output CSV.");

        output
            << "coszen,energy_GeV,"
            << "Pee_nu,Pemu_nu,Petau_nu,"
            << "Pee_antinu,Pemu_antinu,Petau_antinu\n";

        output << std::scientific << std::setprecision(12);

        squids::Const units;

        double max_conservation_error = 0.0;

        for (std::size_t iz = 0; iz < nCZ; ++iz) {
            const double coszen =
                -1.0
                +
                static_cast<double>(iz)
                /
                static_cast<double>(nCZ - 1);

            std::cout
                << "\r"
                << "coszen "
                << (iz + 1)
                << "/"
                << nCZ
                << " = "
                << std::fixed
                << std::setprecision(3)
                << coszen
                << std::flush;

            const auto nu =
                ComputeOneCoszen(
                    coupling,
                    coszen,
                    neutrino,
                    radial_profile,
                    energy_range,
                    body
                );

            const auto antinu =
                ComputeOneCoszen(
                    coupling,
                    coszen,
                    antineutrino,
                    radial_profile,
                    energy_range,
                    body
                );

            for (std::size_t ie = 0; ie < nE; ++ie) {
                const double energy_GeV =
                    energy_range[ie] / units.GeV;

                const double sum_nu =
                    nu[ie][0] + nu[ie][1] + nu[ie][2];

                const double sum_antinu =
                    antinu[ie][0] + antinu[ie][1] + antinu[ie][2];

                max_conservation_error =
                    std::max(
                        max_conservation_error,
                        std::abs(sum_nu - 1.0)
                    );

                max_conservation_error =
                    std::max(
                        max_conservation_error,
                        std::abs(sum_antinu - 1.0)
                    );

                output
                    << std::scientific
                    << std::setprecision(12)
                    << coszen << ","
                    << energy_GeV << ","
                    << nu[ie][0] << ","
                    << nu[ie][1] << ","
                    << nu[ie][2] << ","
                    << antinu[ie][0] << ","
                    << antinu[ie][1] << ","
                    << antinu[ie][2] << "\n";
            }
        }

        std::cout
            << "\nSaved: "
            << output_filename
            << "\n";

        std::cout
            << "Max probability-conservation error = "
            << std::scientific
            << max_conservation_error
            << "\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "\nERROR: "
            << error.what()
            << "\n";
        return 1;
    }
}

