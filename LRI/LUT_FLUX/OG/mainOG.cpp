#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <hdf5.h>

#include <nuSQuIDS/nuSQuIDS.h>

#include "EarthDepth.h"
#include "LRI.h"


using namespace nusquids;


namespace
{

// ============================================================================
// Physics / grid configuration
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
// Oscillation parameters
// ============================================================================
//
// These are the same values we had been using in the previous propagation
// code. They are kept here explicitly so the LUT metadata can identify
// exactly what physics was used.
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
// Propagation configuration
// ============================================================================

constexpr double DEFAULT_DETECTOR_DEPTH_KM =
    1.5;

constexpr double DEFAULT_H_MAX_KM =
    20.0;

constexpr double DEFAULT_REL_ERROR =
    1.0e-12;

constexpr double DEFAULT_ABS_ERROR =
    1.0e-24;

constexpr bool ENABLE_INTERACTIONS =
    true;

constexpr bool ENABLE_TAU_REGENERATION =
    true;

constexpr bool ENABLE_GLASHOW_RESONANCE =
    false;


// ============================================================================
// LUT2 parameter grid
// ============================================================================
//
// LUT1:
//
//   log10(m/eV) = -15 ... -11
//   161 points
//   Delta log10(m) = 0.025
//
// LUT2 starts less densely:
//
//   log10(m/eV) = -15 ... -11
//   41 points
//   Delta log10(m) = 0.1
//
// These nodes are a subset of LUT1, so no mass interpolation is needed
// while GENERATING LUT2.
//
// Coupling:
//
//   log10(g') = -30 ... -24
//   61 points
//   Delta log10(g') = 0.1
//
// Later we will perform convergence tests before declaring this the final
// analysis grid.
//
// ============================================================================

constexpr double LOG10_MASS_MIN =
    -15.0;

constexpr double LOG10_MASS_MAX =
    -11.0;

constexpr std::size_t N_MASS =
    41;


constexpr double LOG10_GPRIME_MIN =
    -30.0;

constexpr double LOG10_GPRIME_MAX =
    -24.0;

constexpr std::size_t N_GPRIME =
    61;


// ============================================================================
// HDF5 helper
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
        if (id_ >= 0 &&
            closer_ != nullptr)
        {
            closer_(id_);
        }

        id_ =
            -1;

        closer_ =
            nullptr;
    }
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


// ============================================================================
// Misc helpers
// ============================================================================

bool FileExists(
    const std::string& filename)
{
    std::ifstream file(
        filename);

    return
        file.good();
}


std::vector<double> Linspace(
    double minimum,
    double maximum,
    std::size_t number_points)
{
    Require(
        number_points >= 2,
        "Linspace requires at least two points."
    );

    std::vector<double> result(
        number_points);

    const double step =
        (maximum - minimum)
        /
        static_cast<double>(
            number_points - 1);

    for (std::size_t i = 0;
         i < number_points;
         ++i)
    {
        result[i] =
            minimum
            +
            static_cast<double>(i)
            * step;
    }

    result.front() =
        minimum;

    result.back() =
        maximum;

    return result;
}


std::vector<double> Pow10Vector(
    const std::vector<double>& logarithms)
{
    std::vector<double> result(
        logarithms.size());

    for (std::size_t i = 0;
         i < logarithms.size();
         ++i)
    {
        result[i] =
            std::pow(
                10.0,
                logarithms[i]);
    }

    return result;
}


bool AxesEqual(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double relative_tolerance = 1.0e-12)
{
    if (a.size() !=
        b.size())
    {
        return false;
    }

    for (std::size_t i = 0;
         i < a.size();
         ++i)
    {
        if (a[i] == b[i])
        {
            continue;
        }

        const double scale =
            std::max(
                std::abs(a[i]),
                std::abs(b[i]));

        if (scale == 0.0)
        {
            continue;
        }

        if (std::abs(
                a[i] - b[i])
            >
            relative_tolerance
            * scale)
        {
            return false;
        }
    }

    return true;
}


// ============================================================================
// HDF5 read utilities
// ============================================================================

std::vector<double> ReadVector(
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


std::vector<double> ReadMatrix(
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
        dimensions[0] ==
            expected_rows
        &&
        dimensions[1] ==
            expected_columns,
        path
        + " has an unexpected shape."
    );

    std::vector<double> data(
        expected_rows
        * expected_columns);

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
// HDF5 write utilities
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
        "Could not create attribute "
        + name
    );

    Require(
        H5Awrite(
            attribute.Get(),
            datatype.Get(),
            value.c_str()) >= 0,
        "Could not write attribute "
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
        "Could not create attribute "
        + name
    );

    Require(
        H5Awrite(
            attribute.Get(),
            H5T_NATIVE_DOUBLE,
            &value) >= 0,
        "Could not write attribute "
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
        "Could not create attribute "
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
        "Could not write attribute "
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


// ============================================================================
// Input flux file
// ============================================================================
//
// Expected format:
//
// /axes/energy_GeV
// /axes/coszen
// /axes/theta_true_deg        optional
//
// /flux/nue                   [coszen, energy]
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


void ValidateIncreasing(
    const std::vector<double>& values,
    const std::string& name)
{
    Require(
        values.size() >= 2,
        name
        + " requires at least two points."
    );

    for (std::size_t i = 1;
         i < values.size();
         ++i)
    {
        Require(
            std::isfinite(
                values[i])
            &&
            values[i] >
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


InputFluxData LoadInitialFluxFile(
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
        input.energy_GeV.front() > 0.0,
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


    // theta_true_deg is optional.
    if (H5Lexists(
            file.Get(),
            "/axes/theta_true_deg",
            H5P_DEFAULT) > 0)
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
                * 180.0
                / pi;
        }
    }


    // Check theta/coszen consistency.
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
            * 180.0
            / pi;

        Require(
            std::abs(
                expected
                -
                input.theta_true_deg[i])
            < 1.0e-8,
            "theta_true_deg is inconsistent "
            "with coszen at index "
            + std::to_string(i)
        );
    }


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
// Output LUT2
// ============================================================================
//
// Axes:
//
// /axes/mass_eV
// /axes/log10_mass_eV
// /axes/gprime
// /axes/log10_gprime
// /axes/energy_GeV
// /axes/coszen
// /axes/theta_true_deg
//
// Flux:
//
// /flux/nue          [mass, gprime, coszen, energy]
// /flux/numu
// /flux/nutau
// /flux/antinue
// /flux/antinumu
// /flux/antinutau
//
// Status:
//
// /status/completed  [mass, gprime]
//
// completed = 1 means all zenith/energy/flavor values for this point
// were propagated and safely written.
//
// This makes the LUT resumable.
//
// ============================================================================

void CreateFluxDataset(
    hid_t flux_group,
    const std::string& name,
    std::size_t number_mass,
    std::size_t number_gprime,
    std::size_t number_coszen,
    std::size_t number_energy)
{
    const hsize_t dimensions[4] =
    {
        static_cast<hsize_t>(
            number_mass),

        static_cast<hsize_t>(
            number_gprime),

        static_cast<hsize_t>(
            number_coszen),

        static_cast<hsize_t>(
            number_energy)
    };


    LocalH5Handle dataspace(
        H5Screate_simple(
            4,
            dimensions,
            nullptr),
        H5Sclose
    );

    Require(
        static_cast<bool>(
            dataspace),
        "Could not create flux dataspace."
    );


    LocalH5Handle property_list(
        H5Pcreate(
            H5P_DATASET_CREATE),
        H5Pclose
    );

    Require(
        static_cast<bool>(
            property_list),
        "Could not create dataset properties."
    );


    // One complete (mass, gprime) slice per chunk.
    const hsize_t chunk[4] =
    {
        1,
        1,
        static_cast<hsize_t>(
            number_coszen),
        static_cast<hsize_t>(
            number_energy)
    };


    Require(
        H5Pset_chunk(
            property_list.Get(),
            4,
            chunk) >= 0,
        "Could not set HDF5 chunking."
    );


    const double fill_value =
        std::numeric_limits<double>::
            quiet_NaN();


    Require(
        H5Pset_fill_value(
            property_list.Get(),
            H5T_NATIVE_DOUBLE,
            &fill_value) >= 0,
        "Could not set HDF5 fill value."
    );


    LocalH5Handle dataset(
        H5Dcreate2(
            flux_group,
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
        "Could not create flux dataset "
        + name
    );
}


void CreateStatusDataset(
    hid_t status_group,
    std::size_t number_mass,
    std::size_t number_gprime)
{
    const hsize_t dimensions[2] =
    {
        static_cast<hsize_t>(
            number_mass),

        static_cast<hsize_t>(
            number_gprime)
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
        "Could not create status dataspace."
    );


    LocalH5Handle property_list(
        H5Pcreate(
            H5P_DATASET_CREATE),
        H5Pclose
    );


    const unsigned char zero =
        0;


    Require(
        H5Pset_fill_value(
            property_list.Get(),
            H5T_NATIVE_UCHAR,
            &zero) >= 0,
        "Could not configure status fill value."
    );


    LocalH5Handle dataset(
        H5Dcreate2(
            status_group,
            "completed",
            H5T_STD_U8LE,
            dataspace.Get(),
            H5P_DEFAULT,
            property_list.Get(),
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not create status dataset."
    );
}


void CreateOutputFile(
    const std::string& output_filename,
    const std::string& initial_flux_filename,
    const std::string& potential_filename,
    const std::vector<double>& mass_eV,
    const std::vector<double>& log_mass,
    const std::vector<double>& gprime,
    const std::vector<double>& log_gprime,
    const InputFluxData& input,
    double detector_depth_km)
{
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
        "Could not create output file "
        + output_filename
    );


    WriteStringAttribute(
        file.Get(),
        "format_name",
        "nuSQuIDS LRI oscillated atmospheric flux LUT"
    );


    WriteStringAttribute(
        file.Get(),
        "format_version",
        "2.0"
    );


    WriteStringAttribute(
        file.Get(),
        "input_flux_file",
        initial_flux_filename
    );


    WriteStringAttribute(
        file.Get(),
        "lri_potential_library",
        potential_filename
    );


    WriteStringAttribute(
        file.Get(),
        "flux_dimension_order",
        "mass, gprime, coszen, energy"
    );


    WriteStringAttribute(
        file.Get(),
        "potential_definition",
        "V_LRI(R,m,gprime) = gprime^2 * V_unit(R,m)"
    );


    WriteStringAttribute(
        file.Get(),
        "hamiltonian",
        "H_vac + H_standard_matter + H_LRI"
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
            ? "true"
            : "false"
    );


    WriteStringAttribute(
        file.Get(),
        "tau_regeneration",
        ENABLE_TAU_REGENERATION
            ? "true"
            : "false"
    );


    WriteStringAttribute(
        file.Get(),
        "glashow_resonance",
        ENABLE_GLASHOW_RESONANCE
            ? "true"
            : "false"
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
        "resume_semantics",
        "/status/completed=1 only after all six flux datasets for a parameter point are written"
    );


    WriteDoubleAttribute(
        file.Get(),
        "detector_depth_km",
        detector_depth_km
    );


    WriteDoubleAttribute(
        file.Get(),
        "detector_radius_km",
        6371.0
        -
        detector_depth_km
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
        "number_mass_points",
        mass_eV.size()
    );


    WriteUInt64Attribute(
        file.Get(),
        "number_gprime_points",
        gprime.size()
    );


    WriteUInt64Attribute(
        file.Get(),
        "number_coszen_points",
        input.coszen.size()
    );


    WriteUInt64Attribute(
        file.Get(),
        "number_energy_points",
        input.energy_GeV.size()
    );


    // ------------------------------------------------------------------------
    // Axes
    // ------------------------------------------------------------------------

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
        "mass_eV",
        mass_eV);


    WriteVector(
        axes_group.Get(),
        "log10_mass_eV",
        log_mass);


    WriteVector(
        axes_group.Get(),
        "gprime",
        gprime);


    WriteVector(
        axes_group.Get(),
        "log10_gprime",
        log_gprime);


    WriteVector(
        axes_group.Get(),
        "energy_GeV",
        input.energy_GeV);


    WriteVector(
        axes_group.Get(),
        "coszen",
        input.coszen);


    WriteVector(
        axes_group.Get(),
        "theta_true_deg",
        input.theta_true_deg);


    // ------------------------------------------------------------------------
    // Flux
    // ------------------------------------------------------------------------

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
        "mass, gprime, coszen, energy"
    );


    for (const std::string& species :
         SPECIES_NAMES)
    {
        CreateFluxDataset(
            flux_group.Get(),
            species,
            mass_eV.size(),
            gprime.size(),
            input.coszen.size(),
            input.energy_GeV.size());
    }


    // ------------------------------------------------------------------------
    // Status
    // ------------------------------------------------------------------------

    LocalH5Handle status_group(
        H5Gcreate2(
            file.Get(),
            "status",
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT),
        H5Gclose
    );


    Require(
        static_cast<bool>(
            status_group),
        "Could not create /status."
    );


    CreateStatusDataset(
        status_group.Get(),
        mass_eV.size(),
        gprime.size());


    Require(
        H5Fflush(
            file.Get(),
            H5F_SCOPE_GLOBAL) >= 0,
        "Could not flush new output file."
    );
}


// ============================================================================
// Resume validation
// ============================================================================

void ValidateExistingOutput(
    hid_t file_id,
    const std::vector<double>& mass_eV,
    const std::vector<double>& log_mass,
    const std::vector<double>& gprime,
    const std::vector<double>& log_gprime,
    const InputFluxData& input)
{
    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/mass_eV"),
            mass_eV),
        "Existing output has a different mass grid."
    );


    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/log10_mass_eV"),
            log_mass),
        "Existing output has a different log-mass grid."
    );


    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/gprime"),
            gprime),
        "Existing output has a different gprime grid."
    );


    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/log10_gprime"),
            log_gprime),
        "Existing output has a different log-gprime grid."
    );


    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/energy_GeV"),
            input.energy_GeV),
        "Existing output has a different energy grid."
    );


    Require(
        AxesEqual(
            ReadVector(
                file_id,
                "/axes/coszen"),
            input.coszen),
        "Existing output has a different coszen grid."
    );
}


// ============================================================================
// Status functions
// ============================================================================

std::vector<unsigned char>
ReadCompletedStatus(
    hid_t file_id,
    std::size_t number_mass,
    std::size_t number_gprime)
{
    LocalH5Handle dataset(
        H5Dopen2(
            file_id,
            "/status/completed",
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not open /status/completed."
    );


    LocalH5Handle dataspace(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );


    hsize_t dimensions[2] =
        {0, 0};


    Require(
        H5Sget_simple_extent_dims(
            dataspace.Get(),
            dimensions,
            nullptr) >= 0,
        "Could not read status dimensions."
    );


    Require(
        dimensions[0] ==
            number_mass
        &&
        dimensions[1] ==
            number_gprime,
        "Status dataset has wrong shape."
    );


    std::vector<unsigned char> status(
        number_mass
        *
        number_gprime,
        0);


    Require(
        H5Dread(
            dataset.Get(),
            H5T_NATIVE_UCHAR,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            status.data()) >= 0,
        "Could not read status dataset."
    );


    return status;
}


void MarkCompleted(
    hid_t file_id,
    std::size_t mass_index,
    std::size_t gprime_index)
{
    LocalH5Handle dataset(
        H5Dopen2(
            file_id,
            "/status/completed",
            H5P_DEFAULT),
        H5Dclose
    );


    Require(
        static_cast<bool>(
            dataset),
        "Could not open status dataset."
    );


    LocalH5Handle file_space(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );


    const hsize_t offset[2] =
    {
        static_cast<hsize_t>(
            mass_index),

        static_cast<hsize_t>(
            gprime_index)
    };


    const hsize_t count[2] =
    {
        1,
        1
    };


    Require(
        H5Sselect_hyperslab(
            file_space.Get(),
            H5S_SELECT_SET,
            offset,
            nullptr,
            count,
            nullptr) >= 0,
        "Could not select status hyperslab."
    );


    LocalH5Handle memory_space(
        H5Screate_simple(
            2,
            count,
            nullptr),
        H5Sclose
    );


    const unsigned char completed =
        1;


    Require(
        H5Dwrite(
            dataset.Get(),
            H5T_NATIVE_UCHAR,
            memory_space.Get(),
            file_space.Get(),
            H5P_DEFAULT,
            &completed) >= 0,
        "Could not update status dataset."
    );
}


// ============================================================================
// Write one complete (mass,gprime) flux slice
// ============================================================================

void WriteFluxSlice(
    hid_t file_id,
    const std::string& species,
    std::size_t mass_index,
    std::size_t gprime_index,
    std::size_t number_coszen,
    std::size_t number_energy,
    const double* values)
{
    const std::string path =
        "/flux/"
        +
        species;


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


    LocalH5Handle file_space(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );


    const hsize_t offset[4] =
    {
        static_cast<hsize_t>(
            mass_index),

        static_cast<hsize_t>(
            gprime_index),

        0,

        0
    };


    const hsize_t count[4] =
    {
        1,
        1,

        static_cast<hsize_t>(
            number_coszen),

        static_cast<hsize_t>(
            number_energy)
    };


    Require(
        H5Sselect_hyperslab(
            file_space.Get(),
            H5S_SELECT_SET,
            offset,
            nullptr,
            count,
            nullptr) >= 0,
        "Could not select flux hyperslab."
    );


    LocalH5Handle memory_space(
        H5Screate_simple(
            4,
            count,
            nullptr),
        H5Sclose
    );


    Require(
        H5Dwrite(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            memory_space.Get(),
            file_space.Get(),
            H5P_DEFAULT,
            values) >= 0,
        "Could not write "
        + path
    );
}


// ============================================================================
// Propagate one (gprime, angle) task
// ============================================================================

void PropagateAngle(
    double gprime,
    std::size_t angular_index,
    const InputFluxData& input,
    const std::shared_ptr<
        EarthDepthBody>& body,
    const std::shared_ptr<
        const VLRIRadialProfile>& radial_profile,
    const marray<double, 1>& energy_range,
    std::array<
        std::vector<double>,
        N_SPECIES>& mass_output,
    std::size_t gprime_index)
{
    const std::size_t number_energy =
        input.energy_GeV.size();

    const std::size_t number_coszen =
        input.coszen.size();


    const double coszen =
        input.coszen[
            angular_index];


    auto track =
        body->
            MakeTrackWithCosineToDetector(
                coszen);


    nuSQUIDSLRI nus(
        gprime,
        radial_profile,
        energy_range,
        N_FLAVORS,
        both,
        ENABLE_INTERACTIONS,
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


    nus.Set_TauRegeneration(
        ENABLE_TAU_REGENERATION);


    nus.Set_GlashowResonance(
        ENABLE_GLASHOW_RESONANCE);


    squids::Const units;


    nus.Set_h_max(
        DEFAULT_H_MAX_KM
        *
        units.km);


    nus.Set_rel_error(
        DEFAULT_REL_ERROR);


    nus.Set_abs_error(
        DEFAULT_ABS_ERROR);


    nus.Set_ProgressBar(
        false);


    // ========================================================================
    // Initial state
    // ========================================================================
    //
    // state[energy][rho][flavor]
    //
    // rho = 0 neutrino
    // rho = 1 antineutrino
    //
    // IMPORTANT:
    //
    // All six species at one zenith angle are divided by ONE common
    // normalization. This preserves their relative normalization while
    // keeping the numerical state O(1).
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


    nus.EvolveState();


    // ========================================================================
    // Read propagated state
    // ========================================================================

    for (std::size_t energy_index = 0;
         energy_index < number_energy;
         ++energy_index)
    {
        const std::array<double, N_SPECIES>
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
            (
                gprime_index
                *
                number_coszen
                +
                angular_index
            )
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


            // Small numerical negative values only.
            if (value < 0.0)
            {
                const double scale =
                    std::max(
                        normalization,
                        1.0);

                if (std::abs(value)
                    <=
                    1.0e-13
                    * scale)
                {
                    value =
                        0.0;
                }
            }


            Require(
                std::isfinite(value),
                "Non-finite propagated flux."
            );


            Require(
                value >= 0.0,
                "Negative propagated flux."
            );


            mass_output[
                species_index][
                output_index]
                =
                value;
        }
    }
}


// ============================================================================
// Usage
// ============================================================================

void PrintUsage(
    const char* executable)
{
    std::cerr
        << "\nUsage:\n\n"
        << "  "
        << executable
        << " INITIAL_FLUX.h5 V_LRI_UNIT.h5 OUTPUT_LUT.h5"
        << " [detector_depth_km]\n\n"
        << "Example:\n\n"
        << "  "
        << executable
        << " ../INITIAL_FLUX/initial_flux.h5"
        << " ../LUT_VLRI/vlri_unit.h5"
        << " oscillated_flux_lut.h5"
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
        H5Eset_auto2(
            H5E_DEFAULT,
            nullptr,
            nullptr);


        if (argc < 4 ||
            argc > 5)
        {
            PrintUsage(
                argv[0]);

            return 1;
        }


        const std::string
            initial_flux_filename =
                argv[1];


        const std::string
            potential_filename =
                argv[2];


        const std::string
            output_filename =
                argv[3];


        double detector_depth_km =
            DEFAULT_DETECTOR_DEPTH_KM;


        if (argc == 5)
        {
            detector_depth_km =
                std::stod(
                    argv[4]);
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


        // ====================================================================
        // Parameter grids
        // ====================================================================

        const std::vector<double>
            log_mass =
                Linspace(
                    LOG10_MASS_MIN,
                    LOG10_MASS_MAX,
                    N_MASS);


        const std::vector<double>
            mass_eV =
                Pow10Vector(
                    log_mass);


        const std::vector<double>
            log_gprime =
                Linspace(
                    LOG10_GPRIME_MIN,
                    LOG10_GPRIME_MAX,
                    N_GPRIME);


        const std::vector<double>
            gprime =
                Pow10Vector(
                    log_gprime);


        // ====================================================================
        // Header
        // ====================================================================

        std::cout
            << "============================================================\n"
            << "   nuSQuIDS LRI oscillated-flux LUT generator\n"
            << "============================================================\n\n";


        std::cout
            << "Initial flux file : "
            << initial_flux_filename
            << '\n';


        std::cout
            << "V_LRI LUT1        : "
            << potential_filename
            << '\n';


        std::cout
            << "Output LUT2       : "
            << output_filename
            << '\n';


        std::cout
            << "Detector depth    : "
            << detector_depth_km
            << " km\n\n";


        std::cout
            << "Mass grid         : "
            << mass_eV.size()
            << " points\n";


        std::cout
            << "log10(m/eV)       : "
            << log_mass.front()
            << " -> "
            << log_mass.back()
            << '\n';


        std::cout
            << "gprime grid       : "
            << gprime.size()
            << " points\n";


        std::cout
            << "log10(gprime)     : "
            << log_gprime.front()
            << " -> "
            << log_gprime.back()
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
        // Initial flux
        // ====================================================================

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
            << input.coszen.front()
            << " -> "
            << input.coszen.back()
            << "\n\n";


        // ====================================================================
        // Energy array for nuSQuIDS
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
        // Load LUT1 once
        // ====================================================================

        std::cout
            << "Loading LUT1...\n";


        const auto lri_library =
            LRIPotentialLibrary::
                LoadFromHDF5(
                    potential_filename);


        std::cout
            << "  masses : "
            << lri_library->
                GetNumberMassPoints()
            << '\n';


        std::cout
            << "  radii  : "
            << lri_library->
                GetNumberRadiusPoints()
            << '\n';


        std::cout
            << "  m range: "
            << std::scientific
            << lri_library->
                GetMinimumMassEV()
            << " -> "
            << lri_library->
                GetMaximumMassEV()
            << " eV\n\n";


        Require(
            mass_eV.front()
                >=
            lri_library->
                GetMinimumMassEV()
            *
            (1.0 - 1.0e-12)
            &&
            mass_eV.back()
                <=
            lri_library->
                GetMaximumMassEV()
            *
            (1.0 + 1.0e-12),
            "LUT2 mass grid is outside LUT1."
        );


        // ====================================================================
        // Shared Earth model
        // ====================================================================
        //
        // One immutable EarthDepthBody is shared between threads.
        // Each propagation gets its own Track and its own nuSQUIDSLRI object.
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
        // Create or resume LUT2
        // ====================================================================

        const bool resume =
            FileExists(
                output_filename);


        if (!resume)
        {
            std::cout
                << "Creating new LUT2...\n";


            CreateOutputFile(
                output_filename,
                initial_flux_filename,
                potential_filename,
                mass_eV,
                log_mass,
                gprime,
                log_gprime,
                input,
                detector_depth_km);
        }
        else
        {
            std::cout
                << "Existing LUT2 found. "
                << "Resume mode enabled.\n";
        }


        LocalH5Handle output_file(
            H5Fopen(
                output_filename.c_str(),
                H5F_ACC_RDWR,
                H5P_DEFAULT),
            H5Fclose
        );


        Require(
            static_cast<bool>(
                output_file),
            "Could not open output LUT2."
        );


        ValidateExistingOutput(
            output_file.Get(),
            mass_eV,
            log_mass,
            gprime,
            log_gprime,
            input);


        std::vector<unsigned char>
            completed =
                ReadCompletedStatus(
                    output_file.Get(),
                    mass_eV.size(),
                    gprime.size());


        std::size_t completed_pairs =
            0;


        for (const unsigned char value :
             completed)
        {
            if (value != 0)
            {
                ++completed_pairs;
            }
        }


        const std::size_t total_pairs =
            mass_eV.size()
            *
            gprime.size();


        const std::size_t total_propagations =
            total_pairs
            *
            number_coszen;


        const long double estimated_bytes =
            static_cast<long double>(
                total_pairs)
            *
            static_cast<long double>(
                number_coszen)
            *
            static_cast<long double>(
                number_energy)
            *
            static_cast<long double>(
                N_SPECIES)
            *
            sizeof(double);


        const long double estimated_gib =
            estimated_bytes
            /
            (
                1024.0L
                *
                1024.0L
                *
                1024.0L
            );


        std::cout
            << "\nParameter pairs    : "
            << total_pairs
            << '\n';


        std::cout
            << "Angle propagations : "
            << total_propagations
            << '\n';


        std::cout
            << "Already completed  : "
            << completed_pairs
            << "/"
            << total_pairs
            << '\n';


        std::cout
            << "Approx flux size   : "
            << std::fixed
            << std::setprecision(3)
            << estimated_gib
            << " GiB\n\n";


        // ====================================================================
        // Main mass loop
        // ====================================================================

        for (std::size_t mass_index = 0;
             mass_index < mass_eV.size();
             ++mass_index)
        {
            // ---------------------------------------------------------------
            // Is there anything left for this mass?
            // ---------------------------------------------------------------

            bool mass_needed =
                false;


            for (std::size_t g_index = 0;
                 g_index < gprime.size();
                 ++g_index)
            {
                if (completed[
                        mass_index
                        * gprime.size()
                        +
                        g_index]
                    == 0)
                {
                    mass_needed =
                        true;

                    break;
                }
            }


            if (!mass_needed)
            {
                std::cout
                    << "[mass "
                    << (mass_index + 1)
                    << "/"
                    << mass_eV.size()
                    << "] already complete; skipping.\n";

                continue;
            }


            const double mediator_mass_eV =
                mass_eV[
                    mass_index];


            std::cout
                << "\n============================================================\n";


            std::cout
                << "Mass "
                << (mass_index + 1)
                << "/"
                << mass_eV.size()
                << "   m = "
                << std::scientific
                << std::setprecision(6)
                << mediator_mass_eV
                << " eV\n";


            std::cout
                << "============================================================\n";


            // ---------------------------------------------------------------
            // Build V_unit(R,m) ONCE for this mass.
            // ---------------------------------------------------------------

            const auto radial_profile =
                lri_library->
                    BuildRadialProfile(
                        mediator_mass_eV);


            // ---------------------------------------------------------------
            // Temporary buffer for all g values at this mass.
            //
            // Shape per species:
            //
            //   [gprime, coszen, energy]
            //
            // ---------------------------------------------------------------

            const std::size_t values_per_species =
                gprime.size()
                *
                number_coszen
                *
                number_energy;


            std::array<
                std::vector<double>,
                N_SPECIES>
                mass_output;


            for (auto& species :
                 mass_output)
            {
                species.assign(
                    values_per_species,
                    std::numeric_limits<double>::
                        quiet_NaN());
            }


            std::atomic<bool>
                failed(false);


            std::mutex
                error_mutex;


            std::string
                first_error;


            // ---------------------------------------------------------------
            // Parallelize over:
            //
            //   gprime x coszen
            //
            // This gives much better CPU occupancy than parallelizing only
            // over zenith.
            // ---------------------------------------------------------------

            const std::size_t tasks =
                gprime.size()
                *
                number_coszen;


#pragma omp parallel for schedule(dynamic)

            for (long long task_signed = 0;
                 task_signed <
                    static_cast<long long>(
                        tasks);
                 ++task_signed)
            {
                if (failed.load())
                {
                    continue;
                }


                const std::size_t task =
                    static_cast<std::size_t>(
                        task_signed);


                const std::size_t g_index =
                    task
                    /
                    number_coszen;


                const std::size_t angular_index =
                    task
                    %
                    number_coszen;


                // Already present in an earlier run.
                if (completed[
                        mass_index
                        * gprime.size()
                        +
                        g_index]
                    != 0)
                {
                    continue;
                }


                try
                {
                    PropagateAngle(
                        gprime[
                            g_index],
                        angular_index,
                        input,
                        earth,
                        radial_profile,
                        energy_range,
                        mass_output,
                        g_index);
                }
                catch (
                    const std::exception&
                        error)
                {
                    failed.store(
                        true);


                    std::lock_guard<
                        std::mutex>
                        lock(
                            error_mutex);


                    if (first_error.empty())
                    {
                        first_error =
                            "m index "
                            +
                            std::to_string(
                                mass_index)
                            +
                            ", g index "
                            +
                            std::to_string(
                                g_index)
                            +
                            ", angle index "
                            +
                            std::to_string(
                                angular_index)
                            +
                            ": "
                            +
                            error.what();
                    }
                }
                catch (...)
                {
                    failed.store(
                        true);


                    std::lock_guard<
                        std::mutex>
                        lock(
                            error_mutex);


                    if (first_error.empty())
                    {
                        first_error =
                            "Unknown exception at "
                            "mass/g/angle task.";
                    }
                }
            }


            if (failed.load())
            {
                throw std::runtime_error(
                    "Propagation failed: "
                    +
                    first_error);
            }


            // ================================================================
            // Write each completed g slice serially
            // ================================================================

            for (std::size_t g_index = 0;
                 g_index < gprime.size();
                 ++g_index)
            {
                const std::size_t status_index =
                    mass_index
                    *
                    gprime.size()
                    +
                    g_index;


                if (completed[
                        status_index]
                    != 0)
                {
                    continue;
                }


                const std::size_t slice_offset =
                    g_index
                    *
                    number_coszen
                    *
                    number_energy;


                const std::size_t slice_size =
                    number_coszen
                    *
                    number_energy;


                // -----------------------------------------------------------
                // Validate complete slice before touching HDF5.
                // -----------------------------------------------------------

                for (std::size_t species_index = 0;
                     species_index < N_SPECIES;
                     ++species_index)
                {
                    for (std::size_t i = 0;
                         i < slice_size;
                         ++i)
                    {
                        const double value =
                            mass_output[
                                species_index][
                                slice_offset
                                +
                                i];


                        Require(
                            std::isfinite(
                                value)
                            &&
                            value >= 0.0,
                            "Incomplete/invalid mass-output slice."
                        );
                    }
                }


                // -----------------------------------------------------------
                // Write all six species.
                // -----------------------------------------------------------

                for (std::size_t species_index = 0;
                     species_index < N_SPECIES;
                     ++species_index)
                {
                    WriteFluxSlice(
                        output_file.Get(),
                        SPECIES_NAMES[
                            species_index],
                        mass_index,
                        g_index,
                        number_coszen,
                        number_energy,
                        mass_output[
                            species_index]
                            .data()
                        +
                        slice_offset);
                }


                // Only after all six writes succeeded:
                MarkCompleted(
                    output_file.Get(),
                    mass_index,
                    g_index);


                completed[
                    status_index]
                    =
                    1;


                ++completed_pairs;


                // Checkpoint to disk.
                Require(
                    H5Fflush(
                        output_file.Get(),
                        H5F_SCOPE_GLOBAL) >= 0,
                    "Could not flush checkpoint."
                );


                std::cout
                    << "  completed "
                    << completed_pairs
                    << "/"
                    << total_pairs
                    << "   log10(m/eV)="
                    << std::fixed
                    << std::setprecision(2)
                    << log_mass[
                        mass_index]
                    << "   log10(g)="
                    << log_gprime[
                        g_index]
                    << '\n';
            }
        }


        // ====================================================================
        // Final status
        // ====================================================================

        completed =
            ReadCompletedStatus(
                output_file.Get(),
                mass_eV.size(),
                gprime.size());


        std::size_t final_completed =
            0;


        for (const unsigned char value :
             completed)
        {
            if (value != 0)
            {
                ++final_completed;
            }
        }


        Require(
            final_completed ==
                total_pairs,
            "LUT finished but not all parameter points are marked complete."
        );


        Require(
            H5Fflush(
                output_file.Get(),
                H5F_SCOPE_GLOBAL) >= 0,
            "Final HDF5 flush failed."
        );


        std::cout
            << "\n============================================================\n"
            << "LUT2 generated successfully\n"
            << "============================================================\n";


        std::cout
            << "File            : "
            << output_filename
            << '\n';


        std::cout
            << "Shape/species   : ["
            << mass_eV.size()
            << ", "
            << gprime.size()
            << ", "
            << number_coszen
            << ", "
            << number_energy
            << "]\n";


        std::cout
            << "Parameter points: "
            << final_completed
            << '\n';


        std::cout
            << "Interactions    : "
            << (
                ENABLE_INTERACTIONS
                ?
                "ON"
                :
                "OFF")
            << '\n';


        std::cout
            << "Tau regeneration: "
            << (
                ENABLE_TAU_REGENERATION
                ?
                "ON"
                :
                "OFF")
            << '\n';


        std::cout
            << "Formula         : "
            << "V_LRI = gprime^2 V_unit\n";


        std::cout
            << "============================================================\n";


        return 0;
    }
    catch (
        const std::exception& error)
    {
        std::cerr
            << "\n[FATAL ERROR] "
            << error.what()
            << '\n';

        return 1;
    }
}
