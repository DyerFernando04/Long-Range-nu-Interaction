#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/gsl_odeiv2.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hdf5.h>

#include <nuSQuIDS/nuSQuIDS.h>

#include "EarthDepth.h"


using namespace nusquids;


namespace
{

// ============================================================================
// PHYSICS / GRID CONFIGURATION
// ============================================================================

constexpr unsigned int N_FLAVORS = 3;

constexpr unsigned int RHO_NEUTRINO =
    0;

constexpr unsigned int RHO_ANTINEUTRINO =
    1;

constexpr std::size_t N_SPECIES =
    6;


const std::array<std::string, N_SPECIES>
SPECIES_NAMES =
{
    "nue",
    "numu",
    "nutau",
    "antinue",
    "antinumu",
    "antinutau"
};


// ============================================================================
// OSCILLATION PARAMETERS
// ============================================================================
//
// IMPORTANT:
//
// These values are intentionally identical to those used in LUT_FLUX/main.cpp.
// The only physics difference between this program and the LRI propagation
// must be the absence of H_LRI.
//
// ============================================================================

constexpr double THETA_01 =
    0.563942;

constexpr double THETA_02 =
    0.154085;

constexpr double THETA_12 =
    0.785398;

constexpr double DELTA_CP =
    0.0;

constexpr double DM2_10_EV2 =
    7.65e-5;

constexpr double DM2_20_EV2 =
    2.47e-3;


// ============================================================================
// PROPAGATION CONFIGURATION
// ============================================================================

constexpr double DEFAULT_DETECTOR_DEPTH_KM =
    1.5;


// Same numerical configuration as current LRI propagation.
constexpr double DEFAULT_H_MAX_KM =
    200.0;

constexpr double DEFAULT_REL_ERROR =
    1.0e-9;

constexpr double DEFAULT_ABS_ERROR =
    1.0e-9;


// Same interaction physics as LRI run.
constexpr bool ENABLE_INTERACTIONS =
    true;

constexpr bool ENABLE_TAU_REGENERATION =
    true;

constexpr bool ENABLE_GLASHOW_RESONANCE =
    false;


// Print one progress message every N completed zenith angles.
constexpr std::size_t PROGRESS_EVERY =
    20;


// ============================================================================
// SMALL HDF5 RAII HANDLE
// ============================================================================

class LocalH5Handle
{
public:

    using Closer =
        herr_t (*)(hid_t);


private:

    hid_t id_ =
        -1;

    Closer closer_ =
        nullptr;


public:

    LocalH5Handle() = default;


    LocalH5Handle(
        hid_t id,
        Closer closer)
        :
        id_(id),
        closer_(closer)
    {}


    LocalH5Handle(
        const LocalH5Handle&) = delete;


    LocalH5Handle&
    operator=(
        const LocalH5Handle&) = delete;


    LocalH5Handle(
        LocalH5Handle&& other) noexcept
        :
        id_(other.id_),
        closer_(other.closer_)
    {
        other.id_ =
            -1;

        other.closer_ =
            nullptr;
    }


    LocalH5Handle&
    operator=(
        LocalH5Handle&& other) noexcept
    {
        if (this != &other)
        {
            Reset();

            id_ =
                other.id_;

            closer_ =
                other.closer_;

            other.id_ =
                -1;

            other.closer_ =
                nullptr;
        }

        return *this;
    }


    ~LocalH5Handle()
    {
        Reset();
    }


    hid_t Get() const
    {
        return id_;
    }


    explicit operator bool() const
    {
        return id_ >= 0;
    }


    void Reset()
    {
        if (
            id_ >= 0
            &&
            closer_ != nullptr
        )
        {
            closer_(id_);
        }

        id_ =
            -1;

        closer_ =
            nullptr;
    }
};


// ============================================================================
// BASIC HELPERS
// ============================================================================

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


bool FileExists(
    const std::string& filename)
{
    std::ifstream file(
        filename);

    return file.good();
}


void ValidateIncreasing(
    const std::vector<double>& values,
    const std::string& name)
{
    Require(
        values.size() >= 2,
        name
        + " requires at least two points."
    );


    Require(
        std::isfinite(
            values.front()),
        name
        + " contains a non-finite value."
    );


    for (std::size_t i = 1;
         i < values.size();
         ++i)
    {
        Require(
            std::isfinite(
                values[i])
            &&
            values[i]
                >
            values[i - 1],
            name
            + " is not strictly increasing."
        );
    }
}


void ValidateFlux(
    const std::vector<double>& values,
    const std::string& name)
{
    for (std::size_t i = 0;
         i < values.size();
         ++i)
    {
        Require(
            std::isfinite(
                values[i]),
            name
            + " contains a non-finite value."
        );


        Require(
            values[i] >= 0.0,
            name
            + " contains a negative flux."
        );
    }
}


// ============================================================================
// HDF5 READ HELPERS
// ============================================================================

std::vector<double>
ReadVector(
    hid_t file_id,
    const std::string& path)
{
    LocalH5Handle dataset(
        H5Dopen2(
            file_id,
            path.c_str(),
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not open HDF5 dataset "
        + path
    );


    LocalH5Handle dataspace(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not get dataspace for "
        + path
    );


    Require(
        H5Sget_simple_extent_ndims(
            dataspace.Get()) == 1,
        path
        + " is not one-dimensional."
    );


    hsize_t dimensions[1] =
        {0};


    Require(
        H5Sget_simple_extent_dims(
            dataspace.Get(),
            dimensions,
            nullptr) >= 0,
        "Could not read dimensions of "
        + path
    );


    std::vector<double> data(
        static_cast<std::size_t>(
            dimensions[0]));


    Require(
        H5Dread(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data.data()) >= 0,
        "Could not read "
        + path
    );


    return data;
}


std::vector<double>
ReadMatrix(
    hid_t file_id,
    const std::string& path,
    std::size_t expected_rows,
    std::size_t expected_columns)
{
    LocalH5Handle dataset(
        H5Dopen2(
            file_id,
            path.c_str(),
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not open "
        + path
    );


    LocalH5Handle dataspace(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not get dataspace for "
        + path
    );


    Require(
        H5Sget_simple_extent_ndims(
            dataspace.Get()) == 2,
        path
        + " is not two-dimensional."
    );


    hsize_t dimensions[2] =
        {0, 0};


    Require(
        H5Sget_simple_extent_dims(
            dataspace.Get(),
            dimensions,
            nullptr) >= 0,
        "Could not read dimensions of "
        + path
    );


    Require(
        dimensions[0]
            ==
        expected_rows
        &&
        dimensions[1]
            ==
        expected_columns,
        path
        + " has an unexpected shape."
    );


    std::vector<double> data(
        expected_rows
        *
        expected_columns);


    Require(
        H5Dread(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            data.data()) >= 0,
        "Could not read "
        + path
    );


    return data;
}


// ============================================================================
// INPUT FLUX
// ============================================================================
//
// Expected:
//
// /axes/energy_GeV
// /axes/coszen
// /axes/theta_true_deg       optional
//
// /flux/nue                  [coszen, energy]
// /flux/numu
// /flux/nutau
// /flux/antinue
// /flux/antinumu
// /flux/antinutau
//
// ============================================================================

struct InputFluxData
{
    std::vector<double>
        energy_GeV;

    std::vector<double>
        coszen;

    std::vector<double>
        theta_true_deg;

    std::array<
        std::vector<double>,
        N_SPECIES>
        flux;
};


InputFluxData
LoadInitialFluxFile(
    const std::string& filename)
{
    LocalH5Handle file(
        H5Fopen(
            filename.c_str(),
            H5F_ACC_RDONLY,
            H5P_DEFAULT),
        H5Fclose
    );


    Require(
        static_cast<bool>(
            file),
        "Could not open initial flux file: "
        + filename
    );


    InputFluxData input;


    input.energy_GeV =
        ReadVector(
            file.Get(),
            "/axes/energy_GeV");


    input.coszen =
        ReadVector(
            file.Get(),
            "/axes/coszen");


    ValidateIncreasing(
        input.energy_GeV,
        "energy_GeV");


    ValidateIncreasing(
        input.coszen,
        "coszen");


    Require(
        input.energy_GeV.front()
            >
        0.0,
        "Energy grid contains non-positive values."
    );


    Require(
        input.coszen.front() >= -1.0
        &&
        input.coszen.back() <= 1.0,
        "coszen lies outside [-1,1]."
    );


    const std::size_t number_energy =
        input.energy_GeV.size();

    const std::size_t number_coszen =
        input.coszen.size();


    // ------------------------------------------------------------------------
    // theta_true_deg is optional.
    // ------------------------------------------------------------------------

    if (
        H5Lexists(
            file.Get(),
            "/axes/theta_true_deg",
            H5P_DEFAULT) > 0
    )
    {
        input.theta_true_deg =
            ReadVector(
                file.Get(),
                "/axes/theta_true_deg");


        Require(
            input.theta_true_deg.size()
                ==
            number_coszen,
            "theta_true_deg has wrong length."
        );
    }
    else
    {
        input.theta_true_deg.resize(
            number_coszen);


        const double pi =
            std::acos(-1.0);


        for (std::size_t i = 0;
             i < number_coszen;
             ++i)
        {
            input.theta_true_deg[i] =
                std::acos(
                    std::max(
                        -1.0,
                        std::min(
                            1.0,
                            input.coszen[i])))
                *
                180.0
                /
                pi;
        }
    }


    // ------------------------------------------------------------------------
    // Check theta/coszen consistency.
    // ------------------------------------------------------------------------

    const double pi =
        std::acos(-1.0);


    for (std::size_t i = 0;
         i < number_coszen;
         ++i)
    {
        const double expected =
            std::acos(
                std::max(
                    -1.0,
                    std::min(
                        1.0,
                        input.coszen[i])))
            *
            180.0
            /
            pi;


        Require(
            std::isfinite(
                input.theta_true_deg[i]),
            "theta_true_deg contains a non-finite value."
        );


        Require(
            std::abs(
                expected
                -
                input.theta_true_deg[i])
                <
            1.0e-8,
            "theta_true_deg is inconsistent "
            "with coszen at index "
            +
            std::to_string(i)
        );
    }


    // ------------------------------------------------------------------------
    // Flux datasets.
    // ------------------------------------------------------------------------

    for (std::size_t species_index = 0;
         species_index < N_SPECIES;
         ++species_index)
    {
        const std::string path =
            "/flux/"
            +
            SPECIES_NAMES[
                species_index];


        input.flux[
            species_index]
            =
            ReadMatrix(
                file.Get(),
                path,
                number_coszen,
                number_energy);


        ValidateFlux(
            input.flux[
                species_index],
            path);
    }


    return input;
}


// ============================================================================
// HDF5 WRITE HELPERS
// ============================================================================

void WriteStringAttribute(
    hid_t object_id,
    const std::string& name,
    const std::string& value)
{
    LocalH5Handle datatype(
        H5Tcopy(
            H5T_C_S1),
        H5Tclose
    );


    Require(
        static_cast<bool>(
            datatype),
        "Could not create string datatype."
    );


    Require(
        H5Tset_size(
            datatype.Get(),
            value.size() + 1) >= 0,
        "Could not set string size."
    );


    Require(
        H5Tset_strpad(
            datatype.Get(),
            H5T_STR_NULLTERM) >= 0,
        "Could not configure string datatype."
    );


    LocalH5Handle dataspace(
        H5Screate(
            H5S_SCALAR),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not create scalar dataspace."
    );


    LocalH5Handle attribute(
        H5Acreate2(
            object_id,
            name.c_str(),
            datatype.Get(),
            dataspace.Get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Aclose
    );


    Require(
        static_cast<bool>(
            attribute),
        "Could not create string attribute "
        + name
    );


    Require(
        H5Awrite(
            attribute.Get(),
            datatype.Get(),
            value.c_str()) >= 0,
        "Could not write string attribute "
        + name
    );
}


void WriteDoubleAttribute(
    hid_t object_id,
    const std::string& name,
    double value)
{
    LocalH5Handle dataspace(
        H5Screate(
            H5S_SCALAR),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not create scalar dataspace."
    );


    LocalH5Handle attribute(
        H5Acreate2(
            object_id,
            name.c_str(),
            H5T_IEEE_F64LE,
            dataspace.Get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Aclose
    );


    Require(
        static_cast<bool>(
            attribute),
        "Could not create double attribute "
        + name
    );


    Require(
        H5Awrite(
            attribute.Get(),
            H5T_NATIVE_DOUBLE,
            &value) >= 0,
        "Could not write double attribute "
        + name
    );
}


void WriteUInt64Attribute(
    hid_t object_id,
    const std::string& name,
    std::uint64_t value)
{
    LocalH5Handle dataspace(
        H5Screate(
            H5S_SCALAR),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not create scalar dataspace."
    );


    LocalH5Handle attribute(
        H5Acreate2(
            object_id,
            name.c_str(),
            H5T_STD_U64LE,
            dataspace.Get(),
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Aclose
    );


    Require(
        static_cast<bool>(
            attribute),
        "Could not create integer attribute "
        + name
    );


    unsigned long long stored =
        static_cast<unsigned long long>(
            value);


    Require(
        H5Awrite(
            attribute.Get(),
            H5T_NATIVE_ULLONG,
            &stored) >= 0,
        "Could not write integer attribute "
        + name
    );
}


void WriteVector(
    hid_t group_id,
    const std::string& name,
    const std::vector<double>& values)
{
    const hsize_t dimensions[1] =
    {
        static_cast<hsize_t>(
            values.size())
    };


    LocalH5Handle dataspace(
        H5Screate_simple(
            1,
            dimensions,
            nullptr),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not create vector dataspace."
    );


    LocalH5Handle dataset(
        H5Dcreate2(
            group_id,
            name.c_str(),
            H5T_IEEE_F64LE,
            dataspace.Get(),
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not create vector "
        + name
    );


    Require(
        H5Dwrite(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()) >= 0,
        "Could not write vector "
        + name
    );
}


void WriteMatrix(
    hid_t group_id,
    const std::string& name,
    const std::vector<double>& values,
    std::size_t number_rows,
    std::size_t number_columns)
{
    Require(
        values.size()
            ==
        number_rows
            *
        number_columns,
        "WriteMatrix: inconsistent matrix size for "
        + name
    );


    const hsize_t dimensions[2] =
    {
        static_cast<hsize_t>(
            number_rows),

        static_cast<hsize_t>(
            number_columns)
    };


    LocalH5Handle dataspace(
        H5Screate_simple(
            2,
            dimensions,
            nullptr),
        H5Sclose
    );


    Require(
        static_cast<bool>(
            dataspace),
        "Could not create matrix dataspace for "
        + name
    );


    // One complete zenith row per chunk.
    LocalH5Handle property_list(
        H5Pcreate(
            H5P_DATASET_CREATE),
        H5Pclose
    );


    Require(
        static_cast<bool>(
            property_list),
        "Could not create dataset property list."
    );


    const hsize_t chunk[2] =
    {
        1,
        static_cast<hsize_t>(
            number_columns)
    };


    Require(
        H5Pset_chunk(
            property_list.Get(),
            2,
            chunk) >= 0,
        "Could not configure matrix chunking."
    );


    LocalH5Handle dataset(
        H5Dcreate2(
            group_id,
            name.c_str(),
            H5T_IEEE_F64LE,
            dataspace.Get(),
            H5P_DEFAULT,
            property_list.Get(),
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not create matrix "
        + name
    );


    Require(
        H5Dwrite(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()) >= 0,
        "Could not write matrix "
        + name
    );
}


// ============================================================================
// STANDARD MODEL PROPAGATION FOR ONE ZENITH ANGLE
// ============================================================================
//
// This is intentionally the same propagation logic used by the LRI code,
// except:
//
//     nuSQUIDSLRI  -->  nuSQUIDS
//
// and there is no mediator mass, gprime, radial LRI profile, or H_LRI.
//
// ============================================================================

void PropagateAngle(
    std::size_t angular_index,
    const InputFluxData& input,
    const std::shared_ptr<
        EarthDepthBody>& body,
    const marray<double, 1>& energy_range,
    std::array<
        std::vector<double>,
        N_SPECIES>& output)
{
    const std::size_t number_energy =
        input.energy_GeV.size();

    const std::size_t number_coszen =
        input.coszen.size();


    const double coszen =
        input.coszen[
            angular_index];


    // ------------------------------------------------------------------------
    // Exactly the same trajectory construction as the LRI propagation.
    // ------------------------------------------------------------------------

    auto track =
        body->
            MakeTrackWithCosineToDetector(
                coszen);


    // ------------------------------------------------------------------------
    // Standard nuSQuIDS.
    //
    // Hamiltonian:
    //
    //     H = H_vac + H_standard_matter
    //
    // No LRI class and no extra potential.
    // ------------------------------------------------------------------------

    nuSQUIDS nus(
        energy_range,
        N_FLAVORS,
        both,
        ENABLE_INTERACTIONS
    );


    // ------------------------------------------------------------------------
    // Mixing parameters.
    // ------------------------------------------------------------------------

    nus.Set_MixingAngle(
        0,
        1,
        THETA_01);


    nus.Set_MixingAngle(
        0,
        2,
        THETA_02);


    nus.Set_MixingAngle(
        1,
        2,
        THETA_12);


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


    // ------------------------------------------------------------------------
    // Same PREM / atmosphere / detector-depth geometry.
    // ------------------------------------------------------------------------

    nus.Set_Body(
        body);


    nus.Set_Track(
        track);


    // ------------------------------------------------------------------------
    // Same interaction physics.
    // ------------------------------------------------------------------------

    nus.Set_TauRegeneration(
        ENABLE_TAU_REGENERATION);


    nus.Set_GlashowResonance(
        ENABLE_GLASHOW_RESONANCE);


    // ------------------------------------------------------------------------
    // Same numerical settings as current LRI code.
    // ------------------------------------------------------------------------

    squids::Const units;


    nus.Set_h_max(
        DEFAULT_H_MAX_KM
        *
        units.km);


    nus.Set_rel_error(
        DEFAULT_REL_ERROR);


    nus.Set_abs_error(
        DEFAULT_ABS_ERROR);


    nus.Set_GSL_step(
        gsl_odeiv2_step_rk4);


    nus.Set_ProgressBar(
        false);


    // ========================================================================
    // INITIAL STATE
    // ========================================================================
    //
    // state[energy][rho][flavor]
    //
    // rho = 0 -> neutrino
    // rho = 1 -> antineutrino
    //
    // Exactly as in LRI:
    //
    // all six species at this zenith angle are divided by ONE common
    // normalization. This preserves the relative atmospheric-flux
    // normalization between flavors and neutrino/antineutrino states.
    //
    // ========================================================================

    marray<double, 3>
        initial_state(
        {
            number_energy,
            static_cast<std::size_t>(2),
            static_cast<std::size_t>(
                N_FLAVORS)
        });


    const std::size_t input_offset =
        angular_index
        *
        number_energy;


    double normalization =
        0.0;


    for (std::size_t species_index = 0;
         species_index < N_SPECIES;
         ++species_index)
    {
        for (std::size_t energy_index = 0;
             energy_index < number_energy;
             ++energy_index)
        {
            normalization =
                std::max(
                    normalization,
                    input.flux[
                        species_index][
                        input_offset
                        +
                        energy_index]);
        }
    }


    Require(
        std::isfinite(
            normalization)
        &&
        normalization > 0.0,
        "Initial spectrum is zero at angular index "
        +
        std::to_string(
            angular_index)
    );


    // ------------------------------------------------------------------------
    // Neutrinos.
    // ------------------------------------------------------------------------

    for (std::size_t energy_index = 0;
         energy_index < number_energy;
         ++energy_index)
    {
        const std::size_t index =
            input_offset
            +
            energy_index;


        initial_state[
            energy_index][
            RHO_NEUTRINO][0]
            =
            input.flux[0][index]
            /
            normalization;


        initial_state[
            energy_index][
            RHO_NEUTRINO][1]
            =
            input.flux[1][index]
            /
            normalization;


        initial_state[
            energy_index][
            RHO_NEUTRINO][2]
            =
            input.flux[2][index]
            /
            normalization;


        // --------------------------------------------------------------------
        // Antineutrinos.
        // --------------------------------------------------------------------

        initial_state[
            energy_index][
            RHO_ANTINEUTRINO][0]
            =
            input.flux[3][index]
            /
            normalization;


        initial_state[
            energy_index][
            RHO_ANTINEUTRINO][1]
            =
            input.flux[4][index]
            /
            normalization;


        initial_state[
            energy_index][
            RHO_ANTINEUTRINO][2]
            =
            input.flux[5][index]
            /
            normalization;
    }


    nus.Set_initial_state(
        initial_state,
        flavor);


    // ========================================================================
    // PROPAGATE
    // ========================================================================

    nus.EvolveState();


    // ========================================================================
    // READ FINAL STATE
    // ========================================================================

    for (std::size_t energy_index = 0;
         energy_index < number_energy;
         ++energy_index)
    {
        const std::array<
            double,
            N_SPECIES>
            propagated =
        {
            nus.EvalFlavorAtNode(
                0,
                static_cast<unsigned int>(
                    energy_index),
                RHO_NEUTRINO),

            nus.EvalFlavorAtNode(
                1,
                static_cast<unsigned int>(
                    energy_index),
                RHO_NEUTRINO),

            nus.EvalFlavorAtNode(
                2,
                static_cast<unsigned int>(
                    energy_index),
                RHO_NEUTRINO),

            nus.EvalFlavorAtNode(
                0,
                static_cast<unsigned int>(
                    energy_index),
                RHO_ANTINEUTRINO),

            nus.EvalFlavorAtNode(
                1,
                static_cast<unsigned int>(
                    energy_index),
                RHO_ANTINEUTRINO),

            nus.EvalFlavorAtNode(
                2,
                static_cast<unsigned int>(
                    energy_index),
                RHO_ANTINEUTRINO)
        };


        const std::size_t output_index =
            angular_index
            *
            number_energy
            +
            energy_index;


        for (std::size_t species_index = 0;
             species_index < N_SPECIES;
             ++species_index)
        {
            double value =
                propagated[
                    species_index]
                *
                normalization;


            // ---------------------------------------------------------------
            // Floating-point guard against tiny negative values.
            // Same philosophy as in the LRI propagation.
            // ---------------------------------------------------------------

            if (value < 0.0)
            {
                const double scale =
                    std::max(
                        normalization,
                        1.0);


                if (
                    std::abs(value)
                        <=
                    1.0e-13
                        *
                    scale
                )
                {
                    value =
                        0.0;
                }
            }


            Require(
                std::isfinite(
                    value),
                "Non-finite propagated flux."
            );


            Require(
                value >= 0.0,
                "Negative propagated flux."
            );


            output[
                species_index][
                output_index]
                =
                value;
        }
    }
}


// ============================================================================
// OUTPUT FILE
// ============================================================================
//
// Output:
//
// /axes/energy_GeV
// /axes/coszen
// /axes/theta_true_deg
//
// /flux/nue          [coszen, energy]
// /flux/numu
// /flux/nutau
// /flux/antinue
// /flux/antinumu
// /flux/antinutau
//
// ============================================================================

void WriteOutputFile(
    const std::string& output_filename,
    const std::string& input_filename,
    const InputFluxData& input,
    const std::array<
        std::vector<double>,
        N_SPECIES>& output,
    const EarthDepthBody& earth)
{
    const std::size_t number_energy =
        input.energy_GeV.size();

    const std::size_t number_coszen =
        input.coszen.size();


    LocalH5Handle file(
        H5Fcreate(
            output_filename.c_str(),
            H5F_ACC_EXCL,
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Fclose
    );


    Require(
        static_cast<bool>(
            file),
        "Could not create output file: "
        + output_filename
    );


    // ========================================================================
    // ROOT METADATA
    // ========================================================================

    WriteStringAttribute(
        file.Get(),
        "format_name",
        "nuSQuIDS Standard Model oscillated atmospheric flux"
    );


    WriteStringAttribute(
        file.Get(),
        "format_version",
        "1.0"
    );


    WriteStringAttribute(
        file.Get(),
        "input_flux_file",
        input_filename
    );


    WriteStringAttribute(
        file.Get(),
        "flux_dimension_order",
        "coszen, energy"
    );


    WriteStringAttribute(
        file.Get(),
        "hamiltonian",
        "H_vac + H_standard_matter"
    );


    WriteStringAttribute(
        file.Get(),
        "new_physics",
        "none"
    );


    WriteStringAttribute(
        file.Get(),
        "neutrino_type",
        "both"
    );


    WriteStringAttribute(
        file.Get(),
        "interactions",
        ENABLE_INTERACTIONS
            ?
        "true"
            :
        "false"
    );


    WriteStringAttribute(
        file.Get(),
        "tau_regeneration",
        ENABLE_TAU_REGENERATION
            ?
        "true"
            :
        "false"
    );


    WriteStringAttribute(
        file.Get(),
        "glashow_resonance",
        ENABLE_GLASHOW_RESONANCE
            ?
        "true"
            :
        "false"
    );


    WriteStringAttribute(
        file.Get(),
        "coszen_convention",
        "-1 upgoing, 0 horizontal, +1 downgoing"
    );


    WriteStringAttribute(
        file.Get(),
        "flux_quantity",
        "differential atmospheric neutrino flux; units inherited from input"
    );


    WriteStringAttribute(
        file.Get(),
        "earth_model",
        "nuSQuIDS EarthAtm / PREM through EarthDepthBody"
    );


    // ========================================================================
    // NUMERICAL / PHYSICAL METADATA
    // ========================================================================

    WriteDoubleAttribute(
        file.Get(),
        "detector_depth_km",
        earth.GetDetectorDepthKm()
    );


    WriteDoubleAttribute(
        file.Get(),
        "earth_radius_km",
        earth.GetEarthRadiusKm()
    );


    WriteDoubleAttribute(
        file.Get(),
        "detector_radius_km",
        earth.GetDetectorRadiusKm()
    );


    WriteDoubleAttribute(
        file.Get(),
        "atmosphere_height_km",
        earth.GetAtmosphereHeightKm()
    );


    WriteDoubleAttribute(
        file.Get(),
        "theta01",
        THETA_01
    );


    WriteDoubleAttribute(
        file.Get(),
        "theta02",
        THETA_02
    );


    WriteDoubleAttribute(
        file.Get(),
        "theta12",
        THETA_12
    );


    WriteDoubleAttribute(
        file.Get(),
        "delta_cp",
        DELTA_CP
    );


    WriteDoubleAttribute(
        file.Get(),
        "dm2_10_eV2",
        DM2_10_EV2
    );


    WriteDoubleAttribute(
        file.Get(),
        "dm2_20_eV2",
        DM2_20_EV2
    );


    WriteDoubleAttribute(
        file.Get(),
        "h_max_km",
        DEFAULT_H_MAX_KM
    );


    WriteDoubleAttribute(
        file.Get(),
        "relative_error",
        DEFAULT_REL_ERROR
    );


    WriteDoubleAttribute(
        file.Get(),
        "absolute_error",
        DEFAULT_ABS_ERROR
    );


    WriteUInt64Attribute(
        file.Get(),
        "number_coszen_points",
        static_cast<std::uint64_t>(
            number_coszen)
    );


    WriteUInt64Attribute(
        file.Get(),
        "number_energy_points",
        static_cast<std::uint64_t>(
            number_energy)
    );


    // ========================================================================
    // AXES
    // ========================================================================

    LocalH5Handle axes_group(
        H5Gcreate2(
            file.Get(),
            "axes",
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Gclose
    );


    Require(
        static_cast<bool>(
            axes_group),
        "Could not create /axes."
    );


    WriteVector(
        axes_group.Get(),
        "energy_GeV",
        input.energy_GeV
    );


    WriteVector(
        axes_group.Get(),
        "coszen",
        input.coszen
    );


    WriteVector(
        axes_group.Get(),
        "theta_true_deg",
        input.theta_true_deg
    );


    // ========================================================================
    // FLUX
    // ========================================================================

    LocalH5Handle flux_group(
        H5Gcreate2(
            file.Get(),
            "flux",
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Gclose
    );


    Require(
        static_cast<bool>(
            flux_group),
        "Could not create /flux."
    );


    WriteStringAttribute(
        flux_group.Get(),
        "dimension_order",
        "coszen, energy"
    );


    for (std::size_t species_index = 0;
         species_index < N_SPECIES;
         ++species_index)
    {
        WriteMatrix(
            flux_group.Get(),
            SPECIES_NAMES[
                species_index],
            output[
                species_index],
            number_coszen,
            number_energy
        );
    }


    Require(
        H5Fflush(
            file.Get(),
            H5F_SCOPE_GLOBAL) >= 0,
        "Could not flush output HDF5 file."
    );
}


// ============================================================================
// USAGE
// ============================================================================

void PrintUsage(
    const char* executable)
{
    std::cerr
        << "\nUsage:\n\n"
        << "  "
        << executable
        << " INITIAL_FLUX.h5 OUTPUT.h5"
        << " [detector_depth_km]\n\n"

        << "Example:\n\n"
        << "  "
        << executable
        << " ../INITIAL_FLUX/mceq_initial_fluxes1.h5"
        << " ../RESULTS/standard/oscillated_flux_std.h5"
        << " 1.5\n\n";
}


} // namespace



// ============================================================================
// MAIN
// ============================================================================

int main(
    int argc,
    char* argv[])
{
    try
    {
        // Disable giant automatic HDF5 error dumps.
        H5Eset_auto2(
            H5E_DEFAULT,
            nullptr,
            nullptr);


        // --------------------------------------------------------------------
        // Arguments.
        // --------------------------------------------------------------------

        if (
            argc < 3
            ||
            argc > 4
        )
        {
            PrintUsage(
                argv[0]);

            return 1;
        }


        const std::string
            initial_flux_filename =
                argv[1];


        const std::string
            output_filename =
                argv[2];


        double detector_depth_km =
            DEFAULT_DETECTOR_DEPTH_KM;


        if (argc == 4)
        {
            detector_depth_km =
                std::stod(
                    argv[3]);
        }


        Require(
            std::isfinite(
                detector_depth_km)
            &&
            detector_depth_km >= 0.0
            &&
            detector_depth_km < 6371.0,
            "Invalid detector depth."
        );


        Require(
            FileExists(
                initial_flux_filename),
            "Initial flux file does not exist: "
            +
            initial_flux_filename
        );


        Require(
            !FileExists(
                output_filename),
            "Output file already exists: "
            +
            output_filename
            +
            "\nRemove it explicitly if you want to regenerate it."
        );


        // ====================================================================
        // HEADER
        // ====================================================================

        std::cout
            << "============================================================\n"
            << "   nuSQuIDS Standard Model atmospheric-flux propagation\n"
            << "============================================================\n\n";


        std::cout
            << "Initial flux file : "
            << initial_flux_filename
            << '\n';


        std::cout
            << "Output file       : "
            << output_filename
            << '\n';


        std::cout
            << "Detector depth    : "
            << detector_depth_km
            << " km\n\n";


        std::cout
            << "Hamiltonian       : "
            << "H_vac + H_standard_matter\n";


        std::cout
            << "Interactions      : "
            << (
                ENABLE_INTERACTIONS
                    ?
                "ON"
                    :
                "OFF"
            )
            << '\n';


        std::cout
            << "Tau regeneration  : "
            << (
                ENABLE_TAU_REGENERATION
                    ?
                "ON"
                    :
                "OFF"
            )
            << '\n';


        std::cout
            << "Glashow resonance : "
            << (
                ENABLE_GLASHOW_RESONANCE
                    ?
                "ON"
                    :
                "OFF"
            )
            << "\n\n";


#ifdef _OPENMP

        std::cout
            << "OpenMP max threads: "
            << omp_get_max_threads()
            << "\n\n";

#else

        std::cout
            << "OpenMP            : disabled\n\n";

#endif


        // ====================================================================
        // INITIAL FLUX
        // ====================================================================

        std::cout
            << "Loading initial flux...\n";


        const InputFluxData input =
            LoadInitialFluxFile(
                initial_flux_filename);


        const std::size_t number_energy =
            input.energy_GeV.size();


        const std::size_t number_coszen =
            input.coszen.size();


        std::cout
            << "Energy points     : "
            << number_energy
            << '\n';


        std::cout
            << "Energy range      : "
            << std::scientific
            << std::setprecision(6)
            << input.energy_GeV.front()
            << " -> "
            << input.energy_GeV.back()
            << " GeV\n";


        std::cout
            << "coszen points     : "
            << number_coszen
            << '\n';


        std::cout
            << "coszen range      : "
            << std::fixed
            << std::setprecision(6)
            << input.coszen.front()
            << " -> "
            << input.coszen.back()
            << "\n\n";


        // ====================================================================
        // ENERGY ARRAY FOR nuSQuIDS
        // ====================================================================

        squids::Const units;


        marray<double, 1>
            energy_range(
            {
                number_energy
            });


        for (std::size_t energy_index = 0;
             energy_index < number_energy;
             ++energy_index)
        {
            energy_range[
                energy_index]
                =
                input.energy_GeV[
                    energy_index]
                *
                units.GeV;
        }


        // ====================================================================
        // EARTH MODEL
        // ====================================================================
        //
        // Exactly the same EarthDepthBody used by the LRI propagation.
        //
        // ====================================================================

        auto earth =
            std::make_shared<
                EarthDepthBody>(
                    detector_depth_km);


        std::cout
            << "Earth radius       : "
            << std::fixed
            << std::setprecision(3)
            << earth->
                GetEarthRadiusKm()
            << " km\n";


        std::cout
            << "Detector radius    : "
            << earth->
                GetDetectorRadiusKm()
            << " km\n";


        std::cout
            << "Atmosphere height  : "
            << earth->
                GetAtmosphereHeightKm()
            << " km\n\n";


        // ====================================================================
        // OUTPUT MEMORY
        // ====================================================================

        const std::size_t matrix_size =
            number_coszen
            *
            number_energy;


        std::array<
            std::vector<double>,
            N_SPECIES>
            output;


        for (std::size_t species_index = 0;
             species_index < N_SPECIES;
             ++species_index)
        {
            output[
                species_index]
                .assign(
                    matrix_size,
                    std::numeric_limits<double>::
                        quiet_NaN());
        }


        // ====================================================================
        // PROPAGATION
        // ====================================================================

        std::cout
            << "Starting Standard Model propagation...\n"
            << "Number of zenith propagations: "
            << number_coszen
            << "\n\n";


        std::atomic<std::size_t>
            completed_angles(
                0);


        std::atomic<bool>
            failed(
                false);


        std::exception_ptr
            first_exception =
                nullptr;


#ifdef _OPENMP

#pragma omp parallel for schedule(dynamic, 1)

#endif

        for (long long angular_index_signed = 0;
             angular_index_signed
                 <
             static_cast<long long>(
                 number_coszen);
             ++angular_index_signed)
        {
            if (failed.load())
            {
                continue;
            }


            const std::size_t angular_index =
                static_cast<std::size_t>(
                    angular_index_signed);


            try
            {
                PropagateAngle(
                    angular_index,
                    input,
                    earth,
                    energy_range,
                    output
                );


                const std::size_t done =
                    ++completed_angles;


                if (
                    done % PROGRESS_EVERY == 0
                    ||
                    done == number_coszen
                )
                {

#ifdef _OPENMP

#pragma omp critical(progress_output)

#endif

                    {
                        std::cout
                            << "  Progress: "
                            << done
                            << "/"
                            << number_coszen
                            << " angles completed"
                            << std::endl;
                    }
                }
            }
            catch (...)
            {
                failed.store(
                    true);


#ifdef _OPENMP

#pragma omp critical(exception_capture)

#endif

                {
                    if (
                        first_exception
                            ==
                        nullptr
                    )
                    {
                        first_exception =
                            std::current_exception();
                    }
                }
            }
        }


        if (
            first_exception
                !=
            nullptr
        )
        {
            std::rethrow_exception(
                first_exception);
        }


        Require(
            completed_angles.load()
                ==
            number_coszen,
            "Not all zenith angles were propagated."
        );


        // ====================================================================
        // FINAL VALIDATION
        // ====================================================================

        std::cout
            << "\nValidating propagated fluxes...\n";


        for (std::size_t species_index = 0;
             species_index < N_SPECIES;
             ++species_index)
        {
            ValidateFlux(
                output[
                    species_index],
                "propagated "
                +
                SPECIES_NAMES[
                    species_index]
            );
        }


        // ====================================================================
        // WRITE HDF5
        // ====================================================================

        std::cout
            << "Writing output file...\n";


        WriteOutputFile(
            output_filename,
            initial_flux_filename,
            input,
            output,
            *earth
        );


        // ====================================================================
        // DONE
        // ====================================================================

        std::cout
            << "\n============================================================\n"
            << "Standard Model propagation completed successfully.\n"
            << "============================================================\n\n";


        std::cout
            << "Output:\n"
            << "  "
            << output_filename
            << "\n\n";


        std::cout
            << "Datasets:\n"
            << "  /axes/energy_GeV\n"
            << "  /axes/coszen\n"
            << "  /axes/theta_true_deg\n";


        for (const std::string& species :
             SPECIES_NAMES)
        {
            std::cout
                << "  /flux/"
                << species
                << '\n';
        }


        std::cout
            << "\nFlux shape: [coszen, energy]\n";


        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\nERROR:\n"
            << error.what()
            << '\n';


        return 1;
    }
}
