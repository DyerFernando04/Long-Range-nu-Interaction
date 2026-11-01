#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>
#include <nuSQuIDS/nuSQuIDS.h>


// ============================================================================
// EarthDepthBody
// ============================================================================
//
// Extensión mínima de nusquids::EarthAtm para permitir que la trayectoria
// termine a una profundidad finita bajo la superficie.
//
// IMPORTANTE:
//
//   - NO se redefine PREM.
//   - NO se redefine Y_e(r).
//   - NO se escriben coeficientes de densidad a mano.
//
// La clase hereda de EarthAtm y utiliza directamente:
//
//   inter_density
//   inter_ye
//
// construidos por nuSQuIDS a partir de su modelo terrestre.
//
// La única física/geometría añadida aquí es que el detector se encuentra en:
//
//   R_det = R_earth - detector_depth
//
// en lugar de estar exactamente en la superficie.
//
// ============================================================================

class EarthDepthBody : public nusquids::EarthAtm
{
public:

    // ========================================================================
    // Track
    // ========================================================================

    class Track : public nusquids::Body::Track
    {
    private:

        double zenith_rad_;
        double cos_zenith_;

        double earth_radius_km_;
        double atmosphere_height_km_;
        double detector_depth_km_;
        double detector_radius_km_;

        double baseline_km_;
        double km_unit_;

    private:

        // --------------------------------------------------------------------
        // Longitud desde la parte superior de la atmósfera hasta el detector.
        //
        // Sea:
        //
        //   R_a   = R_earth + h_atm
        //   R_d   = R_earth - depth
        //
        // y theta_z el ángulo cenital local en el detector.
        //
        // Si s es la distancia hacia atrás desde el detector:
        //
        //   r(s)^2 =
        //       R_d^2
        //       + s^2
        //       + 2 R_d s cos(theta_z)
        //
        // En el inicio de la trayectoria:
        //
        //   r = R_a
        //
        // y por tanto:
        //
        //   L =
        //       -R_d cos(theta_z)
        //       + sqrt(
        //           R_a^2
        //           - R_d^2 sin^2(theta_z)
        //         )
        //
        // --------------------------------------------------------------------

        static double ComputeBaselineKm(
            double zenith_rad,
            double earth_radius_km,
            double atmosphere_height_km,
            double detector_depth_km)
        {
            if (!std::isfinite(zenith_rad))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: zenith no finito.");
            }

            if (!(earth_radius_km > 0.0) ||
                !std::isfinite(earth_radius_km))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: radio terrestre invalido.");
            }

            if (atmosphere_height_km < 0.0 ||
                !std::isfinite(atmosphere_height_km))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: altura atmosferica invalida.");
            }

            if (detector_depth_km < 0.0 ||
                detector_depth_km >= earth_radius_km ||
                !std::isfinite(detector_depth_km))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: profundidad del detector invalida.");
            }

            const double detector_radius_km =
                earth_radius_km
                - detector_depth_km;

            const double atmosphere_radius_km =
                earth_radius_km
                + atmosphere_height_km;

            const double cos_zenith =
                std::cos(zenith_rad);

            const double sin_zenith =
                std::sin(zenith_rad);

            double radicand =
                atmosphere_radius_km
                    * atmosphere_radius_km
                -
                detector_radius_km
                    * detector_radius_km
                    * sin_zenith
                    * sin_zenith;

            const double scale =
                atmosphere_radius_km
                * atmosphere_radius_km;

            if (radicand < 0.0 &&
                std::abs(radicand) <=
                    128.0
                    * std::numeric_limits<double>::epsilon()
                    * scale)
            {
                radicand = 0.0;
            }

            if (radicand < 0.0)
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: geometria imposible.");
            }

            const double baseline_km =
                -detector_radius_km
                    * cos_zenith
                + std::sqrt(radicand);

            if (!(baseline_km > 0.0) ||
                !std::isfinite(baseline_km))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: baseline invalido.");
            }

            return baseline_km;
        }


    public:

        Track(
            double zenith_rad,
            double earth_radius_km,
            double atmosphere_height_km,
            double detector_depth_km,
            double km_unit)
            :
            nusquids::Body::Track(
                0.0,
                ComputeBaselineKm(
                    zenith_rad,
                    earth_radius_km,
                    atmosphere_height_km,
                    detector_depth_km)
                    * km_unit
            ),
            zenith_rad_(zenith_rad),
            cos_zenith_(std::cos(zenith_rad)),
            earth_radius_km_(earth_radius_km),
            atmosphere_height_km_(
                atmosphere_height_km),
            detector_depth_km_(
                detector_depth_km),
            detector_radius_km_(
                earth_radius_km
                - detector_depth_km),
            baseline_km_(
                ComputeBaselineKm(
                    zenith_rad,
                    earth_radius_km,
                    atmosphere_height_km,
                    detector_depth_km)),
            km_unit_(km_unit)
        {
            if (!(km_unit_ > 0.0) ||
                !std::isfinite(km_unit_))
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: "
                    "conversion km -> eV^-1 invalida.");
            }

            if (cos_zenith_ < -1.0 ||
                cos_zenith_ > 1.0)
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track: "
                    "cos(theta_z) fuera de [-1,1].");
            }
        }


        // --------------------------------------------------------------------
        // Radio actual respecto al centro terrestre.
        // --------------------------------------------------------------------

        double RadiusKm() const
        {
            const double x =
                GetX();

            const double x_initial =
                GetInitialX();

            const double x_final =
                GetFinalX();

            const double scale =
                std::max(
                    1.0,
                    std::abs(x_final));

            const double tolerance =
                128.0
                * std::numeric_limits<double>::epsilon()
                * scale;

            if (x < x_initial - tolerance ||
                x > x_final + tolerance)
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track::RadiusKm: "
                    "x esta fuera de la trayectoria.");
            }

            // Distancia que todavía falta para llegar al detector.
            double s_eV_inv =
                x_final - x;

            if (s_eV_inv < 0.0 &&
                std::abs(s_eV_inv) <= tolerance)
            {
                s_eV_inv = 0.0;
            }

            if (s_eV_inv < 0.0)
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track::RadiusKm: "
                    "distancia negativa al detector.");
            }

            const double s_km =
                s_eV_inv
                / km_unit_;

            double radius_squared =
                detector_radius_km_
                    * detector_radius_km_
                +
                s_km
                    * s_km
                +
                2.0
                    * detector_radius_km_
                    * s_km
                    * cos_zenith_;

            const double geometric_scale =
                detector_radius_km_
                    * detector_radius_km_
                +
                s_km
                    * s_km;

            if (radius_squared < 0.0 &&
                std::abs(radius_squared) <=
                    128.0
                    * std::numeric_limits<double>::epsilon()
                    * std::max(
                        1.0,
                        geometric_scale))
            {
                radius_squared = 0.0;
            }

            if (radius_squared < 0.0)
            {
                throw std::runtime_error(
                    "EarthDepthBody::Track::RadiusKm: "
                    "r^2 negativo.");
            }

            return std::sqrt(
                radius_squared);
        }


        double GetZenithRad() const
        {
            return zenith_rad_;
        }


        double GetCosZenith() const
        {
            return cos_zenith_;
        }


        double GetEarthRadiusKm() const
        {
            return earth_radius_km_;
        }


        double GetAtmosphereHeightKm() const
        {
            return atmosphere_height_km_;
        }


        double GetDetectorDepthKm() const
        {
            return detector_depth_km_;
        }


        double GetDetectorRadiusKm() const
        {
            return detector_radius_km_;
        }


        double GetBaselineKm() const
        {
            return baseline_km_;
        }


        // --------------------------------------------------------------------
        // Parámetros adicionales del Track.
        //
        // Body::Track::GetTrackParams() añadirá los parámetros básicos
        // y después estos.
        // --------------------------------------------------------------------

        void FillDerivedParams(
            std::vector<double>& parameters) const override
        {
            parameters.push_back(
                zenith_rad_);

            parameters.push_back(
                cos_zenith_);

            parameters.push_back(
                earth_radius_km_);

            parameters.push_back(
                atmosphere_height_km_);

            parameters.push_back(
                detector_depth_km_);

            parameters.push_back(
                detector_radius_km_);

            parameters.push_back(
                baseline_km_);
        }


        // No necesitaremos serializar Track individualmente para LUT_FLUX.
        void Serialize(
            hid_t group) const override
        {
            (void)group;
        }


        static std::string GetName()
        {
            return "EarthDepthTrack";
        }
    };


private:

    double detector_depth_km_;


private:

    // ========================================================================
    // Evaluación radial del modelo terrestre HEREDADO de EarthAtm
    // ========================================================================

    double DensityFromRadiusKm(
        double radius_km) const
    {
        if (!std::isfinite(radius_km) ||
            radius_km < 0.0)
        {
            throw std::runtime_error(
                "EarthDepthBody::DensityFromRadiusKm: "
                "radio invalido.");
        }

        // --------------------------------------------------------------------
        // Atmósfera.
        //
        // Usamos exactamente el mismo modelo exponencial que EarthAtm:
        //
        // rho(h) = 0.0012 exp(-h / 7.594 km) g/cm^3
        // --------------------------------------------------------------------

        if (radius_km > radius)
        {
            const double height_km =
                radius_km - radius;

            constexpr double atmosphere_scale_height_km =
                7.594;

            constexpr double surface_atmosphere_density =
                0.0012;

            return
                surface_atmosphere_density
                * std::exp(
                    -height_km
                    / atmosphere_scale_height_km);
        }

        // --------------------------------------------------------------------
        // Interior terrestre.
        //
        // La variable del archivo PREM de nuSQuIDS es:
        //
        //   u = r / R_earth
        //
        // y usamos DIRECTAMENTE el AkimaSpline inter_density heredado.
        // --------------------------------------------------------------------

        const double relative_radius =
            radius_km
            / radius;

        if (relative_radius < x_radius_min)
        {
            return x_rho_min;
        }

        if (relative_radius > x_radius_max)
        {
            return x_rho_max;
        }

        return inter_density(
            relative_radius);
    }


    double YeFromRadiusKm(
        double radius_km) const
    {
        if (!std::isfinite(radius_km) ||
            radius_km < 0.0)
        {
            throw std::runtime_error(
                "EarthDepthBody::YeFromRadiusKm: "
                "radio invalido.");
        }

        // Mismo valor atmosférico que EarthAtm.
        if (radius_km > radius)
        {
            return 0.494;
        }

        const double relative_radius =
            radius_km
            / radius;

        if (relative_radius < x_radius_min)
        {
            return x_ye_min;
        }

        if (relative_radius > x_radius_max)
        {
            return x_ye_max;
        }

        return inter_ye(
            relative_radius);
    }


public:

    // ========================================================================
    // Constructores
    // ========================================================================

    explicit EarthDepthBody(
        double detector_depth_km = 1.5)
        :
        nusquids::EarthAtm(),
        detector_depth_km_(
            detector_depth_km)
    {
        Validate();
    }


    EarthDepthBody(
        const std::string& earth_model_file,
        double detector_depth_km = 1.5)
        :
        nusquids::EarthAtm(
            earth_model_file),
        detector_depth_km_(
            detector_depth_km)
    {
        Validate();
    }


private:

    void Validate() const
    {
        if (!(radius > 0.0) ||
            !std::isfinite(radius))
        {
            throw std::runtime_error(
                "EarthDepthBody: "
                "nuSQuIDS devolvio un radio terrestre invalido.");
        }

        if (detector_depth_km_ < 0.0 ||
            detector_depth_km_ >= radius ||
            !std::isfinite(detector_depth_km_))
        {
            throw std::runtime_error(
                "EarthDepthBody: "
                "profundidad del detector invalida.");
        }

        if (atm_height < 0.0 ||
            !std::isfinite(atm_height))
        {
            throw std::runtime_error(
                "EarthDepthBody: "
                "altura atmosferica invalida.");
        }
    }


public:

    // ========================================================================
    // Body interface usada por nuSQuIDS
    // ========================================================================

    double density(
        const nusquids::GenericTrack& generic_track) const override
    {
        const Track& track =
            static_cast<const Track&>(
                generic_track);

        return DensityFromRadiusKm(
            track.RadiusKm());
    }


    double ye(
        const nusquids::GenericTrack& generic_track) const override
    {
        const Track& track =
            static_cast<const Track&>(
                generic_track);

        return YeFromRadiusKm(
            track.RadiusKm());
    }


    // ========================================================================
    // Creación de Track
    // ========================================================================

    std::shared_ptr<Track>
    MakeTrackToDetector(
        double zenith_rad) const
    {
        squids::Const units;

        return std::make_shared<Track>(
            zenith_rad,
            radius,
            atm_height,
            detector_depth_km_,
            units.km);
    }


    std::shared_ptr<Track>
    MakeTrackWithCosineToDetector(
        double cos_zenith) const
    {
        if (!std::isfinite(cos_zenith))
        {
            throw std::runtime_error(
                "EarthDepthBody::MakeTrackWithCosineToDetector: "
                "cos(theta_z) no finito.");
        }

        const double tolerance =
            64.0
            * std::numeric_limits<double>::epsilon();

        if (cos_zenith < -1.0 - tolerance ||
            cos_zenith > 1.0 + tolerance)
        {
            throw std::runtime_error(
                "EarthDepthBody::MakeTrackWithCosineToDetector: "
                "cos(theta_z) fuera de [-1,1].");
        }

        cos_zenith =
            std::max(
                -1.0,
                std::min(
                    1.0,
                    cos_zenith));

        return MakeTrackToDetector(
            std::acos(
                cos_zenith));
    }


    // ========================================================================
    // Getters
    // ========================================================================

    double GetDetectorDepthKm() const
    {
        return detector_depth_km_;
    }


    double GetDetectorRadiusKm() const
    {
        return
            radius
            - detector_depth_km_;
    }


    double GetEarthRadiusKm() const
    {
        return radius;
    }


    double GetAtmosphereHeightKm() const
    {
        return atm_height;
    }


    // Útiles para tests independientes.
    double EvaluateDensityAtRadiusKm(
        double radius_km) const
    {
        return DensityFromRadiusKm(
            radius_km);
    }


    double EvaluateYeAtRadiusKm(
        double radius_km) const
    {
        return YeFromRadiusKm(
            radius_km);
    }


    // No usaremos WriteStateHDF5 en LUT_FLUX.
    // Evitamos serializar este Body custom como si fuera un EarthAtm normal.
    void Serialize(
        hid_t group) const override
    {
        (void)group;
    }


    static std::string GetName()
    {
        return "EarthDepthBody";
    }


    static unsigned int GetId()
    {
        return 1007;
    }
};
