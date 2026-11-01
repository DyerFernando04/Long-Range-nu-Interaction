// transition_probabilities.cpp
//
// Computes Standard and LRI flavor-transition probabilities using the same
// nuSQUIDSLRI + EarthDepthBody implementation used for LUT2.
//
// Compile this source against the desired model-specific LRI.h:
//   LUT_FLUX/                  -> L_e-L_mu
//   LUT_FLUX/Le-Ltau/          -> L_e-L_tau
//   LUT_FLUX/Lmu-Ltau/         -> L_mu-L_tau
//
// IMPORTANT:
// Non-coherent interactions are DISABLED here. Therefore the propagated
// flavor content is an oscillation probability, not a flux including
// absorption, NC redistribution, or tau regeneration.
//
// Usage:
//   transition_probabilities \
//       LUT1.h5 output.csv log10_g log10_m coszen \
//       [N_E] [E_MIN_GEV] [E_MAX_GEV]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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


struct ProbabilitySet
{
    // channel index = alpha*3 + beta
    std::array<std::vector<double>, 9> values;
};


void Require(
    bool condition,
    const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(
            message);
    }
}


marray<double, 1>
BuildEnergyRange(
    std::size_t number_energy,
    double energy_min_GeV,
    double energy_max_GeV)
{
    Require(
        number_energy >= 2,
        "Need at least two energy points.");

    Require(
        std::isfinite(energy_min_GeV)
        &&
        std::isfinite(energy_max_GeV)
        &&
        energy_min_GeV > 0.0
        &&
        energy_max_GeV > energy_min_GeV,
        "Invalid energy range.");

    squids::Const units;

    marray<double, 1>
        energy_range(
        {
            number_energy
        });

    const double log_min =
        std::log10(
            energy_min_GeV);

    const double log_max =
        std::log10(
            energy_max_GeV);

    for (std::size_t i = 0;
         i < number_energy;
         ++i)
    {
        const double fraction =
            static_cast<double>(i)
            /
            static_cast<double>(
                number_energy - 1);

        const double energy_GeV =
            std::pow(
                10.0,
                log_min
                +
                fraction
                * (log_max - log_min));

        energy_range[
            i]
            =
            energy_GeV
            *
            units.GeV;
    }

    return
        energy_range;
}


ProbabilitySet
ComputeProbabilities(
    double coupling,
    double coszen,
    const std::shared_ptr<
        const VLRIRadialProfile>& radial_profile,
    const marray<double, 1>& energy_range,
    const std::shared_ptr<
        EarthDepthBody>& body)
{
    const std::size_t number_energy =
        energy_range.extent(
            0);

    ProbabilitySet result;

    for (auto& channel :
         result.values)
    {
        channel.assign(
            number_energy,
            0.0);
    }

    // One pure initial flavor at a time.
    for (int alpha = 0;
         alpha < N_FLAVORS;
         ++alpha)
    {
        auto track =
            body->
                MakeTrackWithCosineToDetector(
                    coszen);

        // interactions=false on purpose:
        // these are coherent flavor-transition probabilities.
        nuSQUIDSLRI nus(
            coupling,
            radial_profile,
            energy_range,
            N_FLAVORS,
            neutrino,
            false,
            THETA_01,
            THETA_02,
            THETA_12
        );

        nus.Set_Body(
            body);

        nus.Set_Track(
            track);

        nus.Set_SquareMassDifference(
            1,
            DM2_10_EV2);

        nus.Set_SquareMassDifference(
            2,
            DM2_20_EV2);

        if (DELTA_CP != 0.0)
        {
            nus.Set_CPPhase(
                0,
                2,
                DELTA_CP);
        }

        squids::Const units;

        nus.Set_h_max(
            H_MAX_KM
            *
            units.km);

        nus.Set_rel_error(
            REL_ERROR);

        nus.Set_abs_error(
            ABS_ERROR);

        nus.Set_GSL_step(
            gsl_odeiv2_step_rk4);

        nus.Set_ProgressBar(
            false);

        marray<double, 2>
            initial_state(
            {
                number_energy,
                static_cast<std::size_t>(
                    N_FLAVORS)
            });

        std::fill(
            initial_state.begin(),
            initial_state.end(),
            0.0);

        for (std::size_t e = 0;
             e < number_energy;
             ++e)
        {
            initial_state[
                e][
                alpha]
                =
                1.0;
        }

        nus.Set_initial_state(
            initial_state,
            flavor);

        nus.EvolveState();

        for (std::size_t e = 0;
             e < number_energy;
             ++e)
        {
            double probability_sum =
                0.0;

            for (int beta = 0;
                 beta < N_FLAVORS;
                 ++beta)
            {
                double value =
                    nus.EvalFlavorAtNode(
                        static_cast<unsigned int>(
                            beta),
                        static_cast<unsigned int>(
                            e),
                        0);

                // Only trim tiny numerical excursions.
                if (
                    value < 0.0
                    &&
                    std::abs(value)
                    < 1.0e-11
                )
                {
                    value =
                        0.0;
                }

                if (
                    value > 1.0
                    &&
                    std::abs(
                        value - 1.0)
                    < 1.0e-11
                )
                {
                    value =
                        1.0;
                }

                Require(
                    std::isfinite(
                        value),
                    "Non-finite probability.");

                result.values[
                    alpha * 3
                    +
                    beta][
                    e]
                    =
                    value;

                probability_sum +=
                    value;
            }

            Require(
                std::abs(
                    probability_sum
                    -
                    1.0)
                    <
                    5.0e-5,
                "Probability sum differs from one by more than tolerance."
            );
        }
    }

    return
        result;
}


}  // namespace


int main(
    int argc,
    char* argv[])
{
    try
    {
        gsl_set_error_handler_off();

        if (
            argc < 6
            ||
            argc > 9
        )
        {
            std::cerr
                << "Usage:\n  "
                << argv[0]
                << " LUT1.h5 output.csv "
                << "log10_g log10_m coszen "
                << "[N_E] [E_MIN_GEV] [E_MAX_GEV]\n";

            return
                1;
        }

        const std::string lut_filename =
            argv[
                1];

        const std::string output_filename =
            argv[
                2];

        const double log10_g =
            std::stod(
                argv[
                    3]);

        const double log10_m =
            std::stod(
                argv[
                    4]);

        const double coszen =
            std::stod(
                argv[
                    5]);

        const std::size_t number_energy =
            (
                argc >= 7
            )
            ?
            static_cast<std::size_t>(
                std::stoul(
                    argv[
                        6]))
            :
            240;

        const double energy_min_GeV =
            (
                argc >= 8
            )
            ?
            std::stod(
                argv[
                    7])
            :
            5.0;

        const double energy_max_GeV =
            (
                argc >= 9
            )
            ?
            std::stod(
                argv[
                    8])
            :
            200.0;

        Require(
            std::isfinite(
                log10_g),
            "Invalid log10_g.");

        Require(
            std::isfinite(
                log10_m),
            "Invalid log10_m.");

        Require(
            std::isfinite(
                coszen)
            &&
            coszen >= -1.0
            &&
            coszen <= 1.0,
            "coszen must be in [-1,1].");

        const double coupling =
            std::pow(
                10.0,
                log10_g);

        const double mediator_mass_eV =
            std::pow(
                10.0,
                log10_m);

        const auto energy_range =
            BuildEnergyRange(
                number_energy,
                energy_min_GeV,
                energy_max_GeV);

        std::cout
            << "Loading LUT1: "
            << lut_filename
            << '\n';

        const auto lri_library =
            LRIPotentialLibrary::
                LoadFromHDF5(
                    lut_filename);

        const auto radial_profile =
            lri_library->
                BuildRadialProfile(
                    mediator_mass_eV);

        const auto body =
            std::make_shared<
                EarthDepthBody>(
                    DETECTOR_DEPTH_KM);

        std::cout
            << "coszen = "
            << coszen
            << '\n'
            << "log10(g) = "
            << log10_g
            << '\n'
            << "log10(m/eV) = "
            << log10_m
            << '\n'
            << "E range = "
            << energy_min_GeV
            << " -> "
            << energy_max_GeV
            << " GeV\n";

        // Standard = same propagation with g'=0.
        std::cout
            << "Computing Standard probabilities...\n";

        const auto standard =
            ComputeProbabilities(
                0.0,
                coszen,
                radial_profile,
                energy_range,
                body);

        std::cout
            << "Computing LRI probabilities...\n";

        const auto lri =
            ComputeProbabilities(
                coupling,
                coszen,
                radial_profile,
                energy_range,
                body);

        std::ofstream output(
            output_filename);

        Require(
            output.good(),
            "Could not open output CSV.");

        output
            << "energy_GeV";

        const std::array<
            std::string,
            9>
            channel_names =
        {
            "Pee",
            "Pemu",
            "Petau",
            "Pmue",
            "Pmumu",
            "Pmutau",
            "Ptaue",
            "Ptaumu",
            "Ptautau"
        };

        for (
            const auto& name :
            channel_names
        )
        {
            output
                << ",sm_"
                << name;
        }

        for (
            const auto& name :
            channel_names
        )
        {
            output
                << ",lri_"
                << name;
        }

        output
            << '\n';

        squids::Const units;

        output
            << std::scientific
            << std::setprecision(
                12);

        for (std::size_t e = 0;
             e < number_energy;
             ++e)
        {
            const double energy_GeV =
                energy_range[
                    e]
                /
                units.GeV;

            output
                << energy_GeV;

            for (std::size_t c = 0;
                 c < 9;
                 ++c)
            {
                output
                    << ","
                    << standard.values[
                        c][
                        e];
            }

            for (std::size_t c = 0;
                 c < 9;
                 ++c)
            {
                output
                    << ","
                    << lri.values[
                        c][
                        e];
            }

            output
                << '\n';
        }

        output.close();

        std::cout
            << "Saved: "
            << output_filename
            << '\n';

        return
            0;
    }
    catch (
        const std::exception& error)
    {
        std::cerr
            << "ERROR: "
            << error.what()
            << '\n';

        return
            1;
    }
}

