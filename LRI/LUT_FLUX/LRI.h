#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>

#include <hdf5.h>

#include <nuSQuIDS/nuSQuIDS.h>

#include "EarthDepth.h"


using namespace nusquids;


// ============================================================================
// Internal utilities
// ============================================================================

namespace lri_detail
{

// ============================================================================
// HDF5 RAII handle
// ============================================================================

class H5Handle
{
public:

    using Closer =
        herr_t (*)(hid_t);


private:

    hid_t id_;
    Closer closer_;


public:

    H5Handle()
        :
        id_(-1),
        closer_(nullptr)
    {}


    H5Handle(
        hid_t id,
        Closer closer)
        :
        id_(id),
        closer_(closer)
    {}


    H5Handle(
        const H5Handle&) = delete;


    H5Handle&
    operator=(
        const H5Handle&) = delete;


    H5Handle(
        H5Handle&& other) noexcept
        :
        id_(other.id_),
        closer_(other.closer_)
    {
        other.id_ = -1;
        other.closer_ = nullptr;
    }


    H5Handle&
    operator=(
        H5Handle&& other) noexcept
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


    ~H5Handle()
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


// ============================================================================
// Small utilities
// ============================================================================

inline void Require(
    bool condition,
    const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(
            message);
    }
}


inline bool StrictlyIncreasing(
    const std::vector<double>& values)
{
    if (values.size() < 2)
    {
        return false;
    }

    for (std::size_t i = 1;
         i < values.size();
         ++i)
    {
        if (!(values[i] >
              values[i - 1]))
        {
            return false;
        }
    }

    return true;
}


inline bool NearlyEqualRelative(
    double a,
    double b,
    double relative_tolerance =
        64.0
        * std::numeric_limits<double>::epsilon())
{
    if (a == b)
    {
        return true;
    }

    const double scale =
        std::max(
            std::abs(a),
            std::abs(b));

    if (scale == 0.0)
    {
        return true;
    }

    return
        std::abs(a - b)
        <= relative_tolerance
           * scale;
}

inline int Sign(
    double value)
{
    return
        (value > 0.0)
        -
        (value < 0.0);
}


// ============================================================================
// HDF5 readers
// ============================================================================

inline std::vector<double>
ReadVector(
    hid_t file_id,
    const std::string& dataset_path)
{
    H5Handle dataset(
        H5Dopen2(
            file_id,
            dataset_path.c_str(),
            H5P_DEFAULT),
        H5Dclose
    );

    Require(
        static_cast<bool>(dataset),
        "No pude abrir el dataset HDF5: "
        + dataset_path
    );

    H5Handle dataspace(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );

    Require(
        static_cast<bool>(dataspace),
        "No pude obtener el dataspace de "
        + dataset_path
    );

    const int rank =
        H5Sget_simple_extent_ndims(
            dataspace.Get());

    Require(
        rank == 1,
        dataset_path
        + " no es un dataset 1D."
    );

    hsize_t dimensions[1] = {0};

    Require(
        H5Sget_simple_extent_dims(
            dataspace.Get(),
            dimensions,
            nullptr) >= 0,
        "No pude leer las dimensiones de "
        + dataset_path
    );

    std::vector<double> values(
        static_cast<std::size_t>(
            dimensions[0]));

    Require(
        H5Dread(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()) >= 0,
        "No pude leer "
        + dataset_path
    );

    return values;
}


inline std::vector<double>
ReadMatrix(
    hid_t file_id,
    const std::string& dataset_path,
    std::size_t expected_rows,
    std::size_t expected_columns)
{
    H5Handle dataset(
        H5Dopen2(
            file_id,
            dataset_path.c_str(),
            H5P_DEFAULT),
        H5Dclose
    );

    Require(
        static_cast<bool>(dataset),
        "No pude abrir "
        + dataset_path
    );

    H5Handle dataspace(
        H5Dget_space(
            dataset.Get()),
        H5Sclose
    );

    Require(
        static_cast<bool>(dataspace),
        "No pude obtener el dataspace de "
        + dataset_path
    );

    const int rank =
        H5Sget_simple_extent_ndims(
            dataspace.Get());

    Require(
        rank == 2,
        dataset_path
        + " no es un dataset 2D."
    );

    hsize_t dimensions[2] =
        {0, 0};

    Require(
        H5Sget_simple_extent_dims(
            dataspace.Get(),
            dimensions,
            nullptr) >= 0,
        "No pude obtener las dimensiones de "
        + dataset_path
    );

    Require(
        dimensions[0] ==
            expected_rows &&
        dimensions[1] ==
            expected_columns,
        dataset_path
        + " tiene dimensiones inesperadas."
    );

    std::vector<double> values(
        expected_rows
        * expected_columns);

    Require(
        H5Dread(
            dataset.Get(),
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()) >= 0,
        "No pude leer "
        + dataset_path
    );

    return values;
}


inline double ReadOptionalDoubleAttribute(
    hid_t object_id,
    const std::string& attribute_name,
    double default_value)
{
    const htri_t exists =
        H5Aexists(
            object_id,
            attribute_name.c_str());

    if (exists <= 0)
    {
        return default_value;
    }

    H5Handle attribute(
        H5Aopen(
            object_id,
            attribute_name.c_str(),
            H5P_DEFAULT),
        H5Aclose
    );

    if (!attribute)
    {
        return default_value;
    }

    double value =
        default_value;

    const herr_t status =
        H5Aread(
            attribute.Get(),
            H5T_NATIVE_DOUBLE,
            &value);

    if (status < 0 ||
        !std::isfinite(value))
    {
        return default_value;
    }

    return value;
}


// ============================================================================
// PCHIP
// ============================================================================
//
// Fritsch-Carlson shape-preserving derivatives.
//
// Used twice:
//
//   1. log(V) vs log(m)
//   2. V(R) vs R
//
// ============================================================================

inline std::vector<double>
ComputePchipDerivatives(
    const std::vector<double>& x,
    const std::vector<double>& y)
{
    Require(
        x.size() == y.size(),
        "PCHIP: x e y tienen tamaños diferentes."
    );

    const std::size_t n =
        x.size();

    Require(
        n >= 2,
        "PCHIP requiere al menos dos puntos."
    );

    Require(
        StrictlyIncreasing(x),
        "PCHIP requiere x estrictamente creciente."
    );

    std::vector<double> h(
        n - 1);

    std::vector<double> delta(
        n - 1);

    std::vector<double> derivative(
        n,
        0.0);

    for (std::size_t i = 0;
         i + 1 < n;
         ++i)
    {
        h[i] =
            x[i + 1]
            - x[i];

        delta[i] =
            (
                y[i + 1]
                - y[i]
            )
            / h[i];
    }


    if (n == 2)
    {
        derivative[0] =
            delta[0];

        derivative[1] =
            delta[0];

        return derivative;
    }


    // ------------------------------------------------------------------------
    // Left endpoint
    // ------------------------------------------------------------------------

    derivative[0] =
        (
            (2.0 * h[0] + h[1])
                * delta[0]
            -
            h[0]
                * delta[1]
        )
        /
        (
            h[0]
            + h[1]
        );


    if (Sign(derivative[0]) !=
        Sign(delta[0]))
    {
        derivative[0] =
            0.0;
    }
    else if (
        Sign(delta[0]) !=
            Sign(delta[1])
        &&
        std::abs(
            derivative[0])
            >
            3.0
            * std::abs(
                delta[0]))
    {
        derivative[0] =
            3.0
            * delta[0];
    }


    // ------------------------------------------------------------------------
    // Interior nodes
    // ------------------------------------------------------------------------

    for (std::size_t i = 1;
         i + 1 < n;
         ++i)
    {
        const double left_slope =
            delta[i - 1];

        const double right_slope =
            delta[i];

        if (left_slope == 0.0 ||
            right_slope == 0.0 ||
            Sign(left_slope) !=
                Sign(right_slope))
        {
            derivative[i] =
                0.0;
        }
        else
        {
            const double w1 =
                2.0 * h[i]
                + h[i - 1];

            const double w2 =
                h[i]
                + 2.0 * h[i - 1];

            derivative[i] =
                (w1 + w2)
                /
                (
                    w1 / left_slope
                    +
                    w2 / right_slope
                );
        }
    }


    // ------------------------------------------------------------------------
    // Right endpoint
    // ------------------------------------------------------------------------

    const std::size_t last =
        n - 2;

    derivative[n - 1] =
        (
            (
                2.0 * h[last]
                + h[last - 1]
            )
            * delta[last]
            -
            h[last]
            * delta[last - 1]
        )
        /
        (
            h[last]
            + h[last - 1]
        );


    if (Sign(derivative[n - 1]) !=
        Sign(delta[last]))
    {
        derivative[n - 1] =
            0.0;
    }
    else if (
        Sign(delta[last]) !=
            Sign(delta[last - 1])
        &&
        std::abs(
            derivative[n - 1])
            >
            3.0
            * std::abs(
                delta[last]))
    {
        derivative[n - 1] =
            3.0
            * delta[last];
    }


    return derivative;
}


// ============================================================================
// Cubic Hermite evaluation
// ============================================================================

inline double EvaluateHermite(
    double x0,
    double x1,
    double y0,
    double y1,
    double d0,
    double d1,
    double x)
{
    const double h =
        x1 - x0;

    Require(
        h > 0.0,
        "EvaluateHermite recibio intervalo no positivo."
    );

    double t =
        (x - x0)
        / h;

    t =
        std::max(
            0.0,
            std::min(
                1.0,
                t));

    const double t2 =
        t * t;

    const double t3 =
        t2 * t;

    const double h00 =
        2.0 * t3
        - 3.0 * t2
        + 1.0;

    const double h10 =
        t3
        - 2.0 * t2
        + t;

    const double h01 =
        -2.0 * t3
        + 3.0 * t2;

    const double h11 =
        t3
        - t2;

    return
        h00 * y0
        +
        h10 * h * d0
        +
        h01 * y1
        +
        h11 * h * d1;
}


// ============================================================================
// Find interval
// ============================================================================

inline std::size_t FindInterval(
    const std::vector<double>& axis,
    double value)
{
    Require(
        axis.size() >= 2,
        "FindInterval necesita al menos dos puntos."
    );

    if (value <= axis.front())
    {
        return 0;
    }

    if (value >= axis.back())
    {
        return
            axis.size()
            - 2;
    }

    const auto upper =
        std::upper_bound(
            axis.begin(),
            axis.end(),
            value);

    return
        static_cast<std::size_t>(
            std::distance(
                axis.begin(),
                upper)
            - 1);
}

} // namespace lri_detail



// ============================================================================
// VLRIRadialProfile
// ============================================================================
//
// One mediator mass.
//
// Stores:
//
//     V_unit(R, m)
//
// after interpolation in mediator mass.
//
// During EvolveState() only the radial interpolation remains.
//
// ============================================================================

class VLRIRadialProfile
{
private:

    std::string source_filename_;

    double mediator_mass_eV_;

    double km_to_eV_inverse_;

    std::vector<double>
        radius_km_;

    std::vector<double>
        V_unit_eV_;

    std::vector<double>
        radial_derivative_;


private:

    VLRIRadialProfile(
        std::string source_filename,
        double mediator_mass_eV,
        double km_to_eV_inverse,
        std::vector<double> radius_km,
        std::vector<double> V_unit_eV)
        :
        source_filename_(
            std::move(
                source_filename)),
        mediator_mass_eV_(
            mediator_mass_eV),
        km_to_eV_inverse_(
            km_to_eV_inverse),
        radius_km_(
            std::move(
                radius_km)),
        V_unit_eV_(
            std::move(
                V_unit_eV))
    {
        Validate();

        radial_derivative_ =
            lri_detail::
                ComputePchipDerivatives(
                    radius_km_,
                    V_unit_eV_);
    }


    void Validate() const
    {
        lri_detail::Require(
            std::isfinite(
                mediator_mass_eV_)
            &&
            mediator_mass_eV_ > 0.0,
            "VLRIRadialProfile: masa invalida."
        );

        lri_detail::Require(
            std::isfinite(
                km_to_eV_inverse_)
            &&
            km_to_eV_inverse_ > 0.0,
            "VLRIRadialProfile: conversion km invalida."
        );

        lri_detail::Require(
            radius_km_.size()
                ==
            V_unit_eV_.size(),
            "VLRIRadialProfile: tamaños inconsistentes."
        );

        lri_detail::Require(
            lri_detail::
                StrictlyIncreasing(
                    radius_km_),
            "VLRIRadialProfile: eje radial invalido."
        );

        for (std::size_t i = 0;
             i < V_unit_eV_.size();
             ++i)
        {
            lri_detail::Require(
                std::isfinite(
                    V_unit_eV_[i])
                &&
                V_unit_eV_[i] > 0.0,
                "VLRIRadialProfile: "
                "V_unit invalido en indice "
                + std::to_string(i)
            );
        }
    }


    friend class LRIPotentialLibrary;


public:

    // ========================================================================
    // Evaluate V_unit(R)
    // ========================================================================

    double EvaluateAtRadiusKm(
        double radius_km) const
    {
        lri_detail::Require(
            std::isfinite(
                radius_km)
            &&
            radius_km >= 0.0,
            "VLRIRadialProfile::EvaluateAtRadiusKm: "
            "radio invalido."
        );


        const double earth_radius_km =
            radius_km_.back();


        // --------------------------------------------------------------------
        // Center
        // --------------------------------------------------------------------

        if (radius_km <=
            radius_km_.front())
        {
            return
                V_unit_eV_.front();
        }


        // --------------------------------------------------------------------
        // Outside the Earth
        // --------------------------------------------------------------------
        //
        // For a spherically symmetric Yukawa source:
        //
        //     V(r) propto exp(-m r) / r
        //
        // so relative to the surface:
        //
        //     V(r)
        //       = V(Re)
        //         (Re/r)
        //         exp[-m(r-Re)].
        //
        // Our LUT source contains terrestrial electrons only.
        // Atmospheric electrons are not part of LUT1.
        // --------------------------------------------------------------------

        if (radius_km >=
            earth_radius_km)
        {
            if (lri_detail::
                NearlyEqualRelative(
                    radius_km,
                    earth_radius_km))
            {
                return
                    V_unit_eV_.back();
            }

            const double separation_eV_inverse =
                (
                    radius_km
                    -
                    earth_radius_km
                )
                *
                km_to_eV_inverse_;

            const double value =
                V_unit_eV_.back()
                *
                (
                    earth_radius_km
                    /
                    radius_km
                )
                *
                std::exp(
                    -mediator_mass_eV_
                    *
                    separation_eV_inverse);

            lri_detail::Require(
                std::isfinite(value)
                &&
                value >= 0.0,
                "VLRIRadialProfile: "
                "extrapolacion exterior invalida."
            );

            return value;
        }


        // --------------------------------------------------------------------
        // Inside Earth: radial PCHIP
        // --------------------------------------------------------------------

        const std::size_t interval =
            lri_detail::FindInterval(
                radius_km_,
                radius_km);

        double value =
            lri_detail::
                EvaluateHermite(
                    radius_km_[
                        interval],
                    radius_km_[
                        interval + 1],
                    V_unit_eV_[
                        interval],
                    V_unit_eV_[
                        interval + 1],
                    radial_derivative_[
                        interval],
                    radial_derivative_[
                        interval + 1],
                    radius_km);


        // Floating-point guard only.
        if (value < 0.0)
        {
            const double scale =
                std::max(
                    std::abs(
                        V_unit_eV_[
                            interval]),
                    std::abs(
                        V_unit_eV_[
                            interval + 1]));

            if (std::abs(value) <=
                128.0
                *
                std::numeric_limits<double>::
                    epsilon()
                *
                std::max(
                    scale,
                    std::numeric_limits<double>::
                        min()))
            {
                value =
                    0.0;
            }
        }


        lri_detail::Require(
            std::isfinite(value)
            &&
            value >= 0.0,
            "VLRIRadialProfile: "
            "interpolacion radial invalida."
        );

        return value;
    }


    // ========================================================================
    // Getters
    // ========================================================================

    double GetMediatorMassEV() const
    {
        return
            mediator_mass_eV_;
    }


    double GetEarthRadiusKm() const
    {
        return
            radius_km_.back();
    }


    double GetKmToEVInverse() const
    {
        return
            km_to_eV_inverse_;
    }


    const std::string&
    GetSourceFilename() const
    {
        return
            source_filename_;
    }


    const std::vector<double>&
    GetRadiusGridKm() const
    {
        return
            radius_km_;
    }


    const std::vector<double>&
    GetVUnitGridEV() const
    {
        return
            V_unit_eV_;
    }
};



// ============================================================================
// LRIPotentialLibrary
// ============================================================================
//
// Loads LUT1 completely ONCE.
//
// File:
//
//     vlri_unit.h5
//
// Required datasets:
//
//     /axes/mass_eV
//     /axes/radius_km
//     /potential/V_unit_eV
//
// V_unit dimensions:
//
//     [mass, radius]
//
// ============================================================================

class LRIPotentialLibrary
{
private:

    std::string filename_;

    double km_to_eV_inverse_;

    std::vector<double>
        mass_eV_;

    std::vector<double>
        log_mass_;

    std::vector<double>
        radius_km_;

    std::vector<double>
        V_unit_eV_;


private:

    LRIPotentialLibrary(
        std::string filename,
        double km_to_eV_inverse,
        std::vector<double> mass_eV,
        std::vector<double> radius_km,
        std::vector<double> V_unit_eV)
        :
        filename_(
            std::move(filename)),
        km_to_eV_inverse_(
            km_to_eV_inverse),
        mass_eV_(
            std::move(mass_eV)),
        radius_km_(
            std::move(radius_km)),
        V_unit_eV_(
            std::move(V_unit_eV))
    {
        Validate();

        log_mass_.resize(
            mass_eV_.size());

        for (std::size_t i = 0;
             i < mass_eV_.size();
             ++i)
        {
            log_mass_[i] =
                std::log(
                    mass_eV_[i]);
        }
    }


    void Validate() const
    {
        lri_detail::Require(
            mass_eV_.size() >= 2,
            "LRIPotentialLibrary: "
            "se necesitan al menos dos masas."
        );

        lri_detail::Require(
            radius_km_.size() >= 2,
            "LRIPotentialLibrary: "
            "se necesitan al menos dos radios."
        );

        lri_detail::Require(
            lri_detail::
                StrictlyIncreasing(
                    mass_eV_),
            "LRIPotentialLibrary: "
            "eje de masa no creciente."
        );

        lri_detail::Require(
            lri_detail::
                StrictlyIncreasing(
                    radius_km_),
            "LRIPotentialLibrary: "
            "eje radial no creciente."
        );

        lri_detail::Require(
            V_unit_eV_.size()
                ==
            mass_eV_.size()
                *
            radius_km_.size(),
            "LRIPotentialLibrary: "
            "shape inconsistente para V_unit."
        );

        lri_detail::Require(
            std::isfinite(
                km_to_eV_inverse_)
            &&
            km_to_eV_inverse_ > 0.0,
            "LRIPotentialLibrary: "
            "conversion km invalida."
        );


        for (const double mass :
             mass_eV_)
        {
            lri_detail::Require(
                std::isfinite(mass)
                &&
                mass > 0.0,
                "LRIPotentialLibrary: "
                "masa invalida."
            );
        }


        for (const double value :
             V_unit_eV_)
        {
            lri_detail::Require(
                std::isfinite(value)
                &&
                value > 0.0,
                "LRIPotentialLibrary: "
                "V_unit contiene valor invalido."
            );
        }
    }


    double ValueAt(
        std::size_t mass_index,
        std::size_t radial_index) const
    {
        return
            V_unit_eV_[
                mass_index
                * radius_km_.size()
                +
                radial_index
            ];
    }


public:

    // ========================================================================
    // Load
    // ========================================================================

    static std::shared_ptr<const LRIPotentialLibrary>
    LoadFromHDF5(
        const std::string& filename)
    {
        // Disable HDF5 automatic giant error dumps.
        H5Eset_auto2(
            H5E_DEFAULT,
            nullptr,
            nullptr);

        lri_detail::H5Handle file(
            H5Fopen(
                filename.c_str(),
                H5F_ACC_RDONLY,
                H5P_DEFAULT),
            H5Fclose
        );

        lri_detail::Require(
            static_cast<bool>(file),
            "No pude abrir LUT1: "
            + filename
        );


        std::vector<double> mass_eV =
            lri_detail::ReadVector(
                file.Get(),
                "/axes/mass_eV");


        std::vector<double> radius_km =
            lri_detail::ReadVector(
                file.Get(),
                "/axes/radius_km");


        std::vector<double> V_unit_eV =
            lri_detail::ReadMatrix(
                file.Get(),
                "/potential/V_unit_eV",
                mass_eV.size(),
                radius_km.size());


        squids::Const units;

        const double km_to_eV_inverse =
            lri_detail::
                ReadOptionalDoubleAttribute(
                    file.Get(),
                    "km_to_eV_inverse",
                    units.km);


        return
            std::shared_ptr<
                const LRIPotentialLibrary>(
                new LRIPotentialLibrary(
                    filename,
                    km_to_eV_inverse,
                    std::move(
                        mass_eV),
                    std::move(
                        radius_km),
                    std::move(
                        V_unit_eV)));
    }


    // ========================================================================
    // Build profile at arbitrary mediator mass
    // ========================================================================
    //
    // Interpolation:
    //
    //     x = log(m)
    //     y = log(V_unit)
    //
    // using local PCHIP.
    //
    // This occurs ONCE when the radial profile is prepared.
    //
    // During EvolveState(), there is no interpolation in m anymore.
    // ========================================================================

    std::shared_ptr<const VLRIRadialProfile>
    BuildRadialProfile(
        double mediator_mass_eV) const
    {
        lri_detail::Require(
            std::isfinite(
                mediator_mass_eV)
            &&
            mediator_mass_eV > 0.0,
            "BuildRadialProfile: "
            "masa invalida."
        );


        const double minimum_mass =
            mass_eV_.front();

        const double maximum_mass =
            mass_eV_.back();


        const double tolerance =
            128.0
            *
            std::numeric_limits<double>::
                epsilon();


        if (mediator_mass_eV <
                minimum_mass
                * (1.0 - tolerance)
            ||
            mediator_mass_eV >
                maximum_mass
                * (1.0 + tolerance))
        {
            throw std::runtime_error(
                "BuildRadialProfile: masa "
                + std::to_string(
                    mediator_mass_eV)
                + " eV fuera del rango ["
                + std::to_string(
                    minimum_mass)
                + ", "
                + std::to_string(
                    maximum_mass)
                + "] eV."
            );
        }


        if (lri_detail::
            NearlyEqualRelative(
                mediator_mass_eV,
                minimum_mass))
        {
            mediator_mass_eV =
                minimum_mass;
        }


        if (lri_detail::
            NearlyEqualRelative(
                mediator_mass_eV,
                maximum_mass))
        {
            mediator_mass_eV =
                maximum_mass;
        }


        // --------------------------------------------------------------------
        // Exact mass node?
        // --------------------------------------------------------------------

        const auto lower_bound_iterator =
            std::lower_bound(
                mass_eV_.begin(),
                mass_eV_.end(),
                mediator_mass_eV);


        if (lower_bound_iterator !=
            mass_eV_.end())
        {
            const std::size_t index =
                static_cast<std::size_t>(
                    std::distance(
                        mass_eV_.begin(),
                        lower_bound_iterator));


            if (lri_detail::
                NearlyEqualRelative(
                    mediator_mass_eV,
                    mass_eV_[index]))
            {
                std::vector<double> profile(
                    radius_km_.size());

                for (std::size_t r = 0;
                     r < radius_km_.size();
                     ++r)
                {
                    profile[r] =
                        ValueAt(
                            index,
                            r);
                }


                return
                    std::shared_ptr<
                        const VLRIRadialProfile>(
                        new VLRIRadialProfile(
                            filename_,
                            mass_eV_[index],
                            km_to_eV_inverse_,
                            radius_km_,
                            std::move(
                                profile)));
            }
        }


        // --------------------------------------------------------------------
        // Off-grid mass
        // --------------------------------------------------------------------

        lri_detail::Require(
            lower_bound_iterator !=
                mass_eV_.begin()
            &&
            lower_bound_iterator !=
                mass_eV_.end(),
            "BuildRadialProfile: "
            "no pude encontrar intervalo de masa."
        );


        const std::size_t upper_index =
            static_cast<std::size_t>(
                std::distance(
                    mass_eV_.begin(),
                    lower_bound_iterator));


        const std::size_t lower_index =
            upper_index - 1;


        // Local PCHIP needs one neighbor on each side whenever available.
        const std::size_t first_index =
            (
                lower_index > 0
            )
            ?
            lower_index - 1
            :
            0;


        const std::size_t last_index =
            std::min(
                mass_eV_.size() - 1,
                upper_index + 1);


        const std::size_t local_size =
            last_index
            - first_index
            + 1;


        std::vector<double> local_x(
            local_size);


        for (std::size_t i = 0;
             i < local_size;
             ++i)
        {
            local_x[i] =
                log_mass_[
                    first_index
                    + i];
        }


        const std::size_t local_lower =
            lower_index
            - first_index;


        const double requested_log_mass =
            std::log(
                mediator_mass_eV);


        std::vector<double> profile(
            radius_km_.size());


        std::vector<double> local_y(
            local_size);


        for (std::size_t radial_index = 0;
             radial_index < radius_km_.size();
             ++radial_index)
        {
            for (std::size_t local_index = 0;
                 local_index < local_size;
                 ++local_index)
            {
                const double value =
                    ValueAt(
                        first_index
                            + local_index,
                        radial_index);

                local_y[local_index] =
                    std::log(
                        value);
            }


            const std::vector<double>
                derivatives =
                    lri_detail::
                        ComputePchipDerivatives(
                            local_x,
                            local_y);


            const double interpolated_log_V =
                lri_detail::
                    EvaluateHermite(
                        local_x[
                            local_lower],
                        local_x[
                            local_lower + 1],
                        local_y[
                            local_lower],
                        local_y[
                            local_lower + 1],
                        derivatives[
                            local_lower],
                        derivatives[
                            local_lower + 1],
                        requested_log_mass);


            const double interpolated_V =
                std::exp(
                    interpolated_log_V);


            lri_detail::Require(
                std::isfinite(
                    interpolated_V)
                &&
                interpolated_V > 0.0,
                "BuildRadialProfile: "
                "interpolacion en masa invalida."
            );


            profile[radial_index] =
                interpolated_V;
        }


        return
            std::shared_ptr<
                const VLRIRadialProfile>(
                new VLRIRadialProfile(
                    filename_,
                    mediator_mass_eV,
                    km_to_eV_inverse_,
                    radius_km_,
                    std::move(
                        profile)));
    }


    // ========================================================================
    // Getters
    // ========================================================================

    const std::vector<double>&
    GetMassGridEV() const
    {
        return mass_eV_;
    }


    const std::vector<double>&
    GetRadiusGridKm() const
    {
        return radius_km_;
    }


    double GetMinimumMassEV() const
    {
        return mass_eV_.front();
    }


    double GetMaximumMassEV() const
    {
        return mass_eV_.back();
    }


    double GetEarthRadiusKm() const
    {
        return radius_km_.back();
    }


    std::size_t GetNumberMassPoints() const
    {
        return mass_eV_.size();
    }


    std::size_t GetNumberRadiusPoints() const
    {
        return radius_km_.size();
    }


    const std::string&
    GetFilename() const
    {
        return filename_;
    }
};



// ============================================================================
// nuSQUIDSLRI
// ============================================================================
//
// Extension of nuSQUIDS:
//
//     H_total = H_standard + H_LRI
//
// with:
//
//     H_LRI
//       = g'^2 V_unit(R,m)
//         diag(+1,-1,0)
//
// for L_e - L_mu.
//
// nuSQuIDS Hamiltonian quantities are expressed in eV.
//
// ============================================================================

class nuSQUIDSLRI :
    public nuSQUIDS
{
private:

    // Coupling
    double gprime_;
    double gprime_squared_;


    // Fixed radial profile for this mediator mass
    std::shared_ptr<
        const VLRIRadialProfile>
        profile_;


    // Same Track used by nuSQuIDS
    std::shared_ptr<
        EarthDepthBody::Track>
        earth_track_;


    // L_e - L_mu flavor charge operator
    squids::SU_vector
        LRI_operator_;


    // Interaction-picture operator for every energy
    std::vector<
        squids::SU_vector>
        LRI_evolved_;


    // Buffer required by SU_vector constructor
    std::unique_ptr<double[]>
        hamiltonian_buffer_;


    // Diagnostics
    double current_radius_km_;
    double current_V_unit_eV_;
    double current_V_LRI_eV_;


private:

    // ========================================================================
    // Validation
    // ========================================================================

    void ValidateState() const
    {
        lri_detail::Require(
            std::isfinite(
                gprime_)
            &&
            gprime_ >= 0.0,
            "nuSQUIDSLRI: gprime invalido."
        );


        lri_detail::Require(
            static_cast<bool>(
                profile_),
            "nuSQUIDSLRI: perfil LRI nulo."
        );


        lri_detail::Require(
            nsun == 3,
            "nuSQUIDSLRI requiere exactamente 3 sabores."
        );
    }


    // ========================================================================
    // Build Q = diag(+1,-1,0)
    // ========================================================================
    //
    // IMPORTANT:
    //
    // Mixing angles must ALREADY be configured before RotateToB1().
    //
    // ========================================================================

    void BuildLRIOperator()
    {
        assert(
            nsun == 3);


        gsl_matrix_complex* matrix =
            gsl_matrix_complex_calloc(
                3,
                3);


        if (matrix == nullptr)
        {
            throw std::runtime_error(
                "nuSQUIDSLRI: "
                "no pude reservar matriz LRI."
            );
        }


        gsl_matrix_complex_set(
            matrix,
            0,
            0,
            gsl_complex_rect(
                +1.0,
                0.0));


        gsl_matrix_complex_set(
            matrix,
            1,
            1,
            gsl_complex_rect(
                -1.0,
                0.0));


        gsl_matrix_complex_set(
            matrix,
            2,
            2,
            gsl_complex_rect(
                0.0,
                0.0));


        LRI_operator_ =
            squids::SU_vector(
                matrix);


        gsl_matrix_complex_free(
            matrix);


        // Convert flavor-basis charge operator to B1/mass representation
        // used internally by nuSQuIDS.
        LRI_operator_.
            RotateToB1(
                params);
    }


    void RebuildLRIOperator()
    {
        BuildLRIOperator();
    }


    // ========================================================================
    // Current LRI potential
    // ========================================================================

    void UpdateCurrentLRI(
        double x)
    {
        lri_detail::Require(
            static_cast<bool>(
                earth_track_),
            "nuSQUIDSLRI: "
            "Set_Track debe llamarse antes de EvolveState."
        );


        // Make absolutely sure our shared Track has the same x used by
        // nuSQuIDS at this derivative evaluation.
        earth_track_->SetX(
            x);


        const double radius_km =
            earth_track_->
                RadiusKm();


        const double V_unit_eV =
            profile_->
                EvaluateAtRadiusKm(
                    radius_km);


        const double V_LRI_eV =
            gprime_squared_
            *
            V_unit_eV;


        lri_detail::Require(
            std::isfinite(
                radius_km)
            &&
            radius_km >= 0.0,
            "nuSQUIDSLRI: radio invalido."
        );


        lri_detail::Require(
            std::isfinite(
                V_unit_eV)
            &&
            V_unit_eV >= 0.0,
            "nuSQUIDSLRI: V_unit invalido."
        );


        lri_detail::Require(
            std::isfinite(
                V_LRI_eV)
            &&
            V_LRI_eV >= 0.0,
            "nuSQUIDSLRI: V_LRI invalido."
        );


        current_radius_km_ =
            radius_km;

        current_V_unit_eV_ =
            V_unit_eV;

        current_V_LRI_eV_ =
            V_LRI_eV;
    }


protected:

    // ========================================================================
    // AddToPreDerive
    // ========================================================================
    //
    // nuSQuIDS calls this hook before HI().
    //
    // The LRI flavor operator is transformed to the interaction picture:
    //
    //   Q_I(x,E)
    //      = exp(+i H0 x) Q exp(-i H0 x)
    //
    // via SU_vector::Evolve().
    //
    // ========================================================================

    void AddToPreDerive(
        double x) override
    {
        UpdateCurrentLRI(
            x);


        for (unsigned int energy_index = 0;
             energy_index < ne;
             ++energy_index)
        {
            LRI_evolved_[
                energy_index]
                =
                LRI_operator_.
                    Evolve(
                        H0_array[
                            energy_index],
                        x
                        -
                        Get_t_initial());
        }
    }


    // ========================================================================
    // HI
    // ========================================================================
    //
    // Neutrino:
    //
    //     H = H_standard + H_LRI
    //
    // Antineutrino:
    //
    //     H = H_standard - H_LRI
    //
    // ========================================================================

    squids::SU_vector HI(
        unsigned int energy_index,
        unsigned int index_rho) const override
    {
        squids::SU_vector potential(
            nsun,
            hamiltonian_buffer_.get());


        potential =
            current_V_LRI_eV_
            *
            LRI_evolved_[
                energy_index];


        // --------------------------------------------------------------------
        // Neutrino
        // --------------------------------------------------------------------

        if (
            NT == neutrino
            ||
            (
                NT == both
                &&
                index_rho == 0
            )
        )
        {
            return
                nuSQUIDS::HI(
                    energy_index,
                    index_rho)
                +
                potential;
        }


        // --------------------------------------------------------------------
        // Antineutrino
        // --------------------------------------------------------------------

        if (
            NT == antineutrino
            ||
            (
                NT == both
                &&
                index_rho == 1
            )
        )
        {
            return
                nuSQUIDS::HI(
                    energy_index,
                    index_rho)
                -
                potential;
        }


        throw std::runtime_error(
            "nuSQUIDSLRI::HI: "
            "tipo neutrino/index_rho inconsistente."
        );
    }


public:

    // Preserve generic nuSQUIDS Set_Track overloads as well.
    using nuSQUIDS::Set_Track;


    // ========================================================================
    // Constructor
    // ========================================================================
    //
    // NOTE:
    //
    // The profile already corresponds to ONE mediator mass.
    //
    // The expensive mass interpolation has therefore already happened.
    //
    // ========================================================================

    nuSQUIDSLRI(
        double gprime,
        std::shared_ptr<
            const VLRIRadialProfile>
            profile,
        marray<double, 1>
            energy_range,
        int number_neutrinos,
        NeutrinoType neutrino_type,
        bool interactions,
        double theta01 = 0.563942,
        double theta02 = 0.154085,
        double theta12 = 0.785398)
        :
        nuSQUIDS(
            energy_range,
            number_neutrinos,
            neutrino_type,
            interactions),
        gprime_(
            gprime),
        gprime_squared_(
            gprime * gprime),
        profile_(
            std::move(
                profile)),
        earth_track_(
            nullptr),
        LRI_operator_(
            number_neutrinos),
        hamiltonian_buffer_(
            new double[
                static_cast<std::size_t>(
                    number_neutrinos)
                *
                static_cast<std::size_t>(
                    number_neutrinos)]),
        current_radius_km_(
            std::numeric_limits<double>::
                quiet_NaN()),
        current_V_unit_eV_(
            0.0),
        current_V_LRI_eV_(
            0.0)
    {
        if (number_neutrinos != 3)
        {
            throw std::runtime_error(
                "nuSQUIDSLRI requiere exactamente "
                "3 sabores."
            );
        }


        ValidateState();


        // --------------------------------------------------------------------
        // IMPORTANT:
        //
        // Set mixing BEFORE rotating diag(+1,-1,0) to B1.
        //
        // Call the BASE functions explicitly so the operator is only built
        // once after all three angles have been configured.
        // --------------------------------------------------------------------

        nuSQUIDS::
            Set_MixingAngle(
                0,
                1,
                theta01);


        nuSQUIDS::
            Set_MixingAngle(
                0,
                2,
                theta02);


        nuSQUIDS::
            Set_MixingAngle(
                1,
                2,
                theta12);


        BuildLRIOperator();


        LRI_evolved_.resize(
            ne);


        for (unsigned int energy_index = 0;
             energy_index < ne;
             ++energy_index)
        {
            LRI_evolved_[
                energy_index]
                =
                squids::SU_vector(
                    nsun);
        }
    }


    // ========================================================================
    // Set Track
    // ========================================================================
    //
    // This is the SAME shared Track passed to nuSQuIDS.
    //
    // Therefore standard matter and LRI see exactly the same trajectory.
    //
    // ========================================================================

    void Set_Track(
        const std::shared_ptr<
            EarthDepthBody::Track>& track)
    {
        lri_detail::Require(
            static_cast<bool>(
                track),
            "nuSQUIDSLRI::Set_Track: "
            "track nulo."
        );


        earth_track_ =
            track;


        nuSQUIDS::
            Set_Track(
                track);
    }


    // ========================================================================
    // Coupling
    // ========================================================================

    void Set_gprime(
        double gprime)
    {
        lri_detail::Require(
            std::isfinite(
                gprime)
            &&
            gprime >= 0.0,
            "Set_gprime: "
            "gprime debe ser finito y >= 0."
        );


        gprime_ =
            gprime;


        gprime_squared_ =
            gprime
            *
            gprime;
    }


    double Get_gprime() const
    {
        return
            gprime_;
    }


    // ========================================================================
    // Mixing parameters
    // ========================================================================
    //
    // If mixing angles/phases change after construction, Q must be re-rotated.
    //
    // ========================================================================

    void Set_MixingAngle(
        unsigned int i,
        unsigned int j,
        double theta)
    {
        nuSQUIDS::
            Set_MixingAngle(
                i,
                j,
                theta);


        RebuildLRIOperator();
    }


    void Set_CPPhase(
        unsigned int i,
        unsigned int j,
        double delta)
    {
        nuSQUIDS::
            Set_CPPhase(
                i,
                j,
                delta);


        RebuildLRIOperator();
    }


    // ========================================================================
    // Diagnostics / testing
    // ========================================================================

    double GetMediatorMassEV() const
    {
        return
            profile_->
                GetMediatorMassEV();
    }


    double GetCurrentRadiusKm() const
    {
        return
            current_radius_km_;
    }


    double GetCurrentVUnitEV() const
    {
        return
            current_V_unit_eV_;
    }


    double GetCurrentVLRIEV() const
    {
        return
            current_V_LRI_eV_;
    }


    double EvalVUnitAtRadiusKm(
        double radius_km) const
    {
        return
            profile_->
                EvaluateAtRadiusKm(
                    radius_km);
    }


    double EvalVLRIAtRadiusKm(
        double radius_km) const
    {
        return
            gprime_squared_
            *
            EvalVUnitAtRadiusKm(
                radius_km);
    }


    double EvalRadiusFromCurrentTrack() const
    {
        lri_detail::Require(
            static_cast<bool>(
                earth_track_),
            "EvalRadiusFromCurrentTrack: "
            "no hay Track."
        );


        return
            earth_track_->
                RadiusKm();
    }


    std::shared_ptr<
        const VLRIRadialProfile>
    GetRadialProfile() const
    {
        return
            profile_;
    }
};
