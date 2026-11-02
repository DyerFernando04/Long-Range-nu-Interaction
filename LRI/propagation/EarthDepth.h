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
// Geometria esferica para neutrinos atmosfericos que llegan a un detector
// situado a una profundidad finita bajo la superficie de la Tierra.
//
// Convencion del angulo cenital en el DETECTOR:
//   cos(theta_z) = -1 : neutrinos ascendentes (atraviesan la Tierra)
//   cos(theta_z) =  0 : horizontales
//   cos(theta_z) = +1 : neutrinos descendentes
//
// La trayectoria empieza en R_earth + h_atm y termina en
// R_detector = R_earth - detector_depth.
//
// MODELO FISICO acordado:
//   r > R_earth  : propagacion en vacio; rho = 0 (sin MSW ni colisiones).
//   r <= R_earth : PREM de nuSQuIDS; MSW e interacciones habilitables.
//
// ATENCION: Esta clase NO modifica H_LRI. El corte V_LRI = 0 en la
// atmosfera debe implementarse por separado en LRI.h.
//
// h_atm representa el tope del perfil de MCEq, NO una altura fija de
// produccion. Se configura ANTES de construir el Track mediante
// SetAtmosphereHeightKm(height_edges_km.front()).
// ============================================================================

class EarthDepthBody : public nusquids::EarthAtm
{
public:
    class Track : public nusquids::Body::Track
    {
    private:
        double zenith_rad_;
        double coszen_;
        double earth_radius_km_;
        double atmosphere_height_km_;
        double detector_depth_km_;
        double detector_radius_km_;
        double baseline_km_;
        double km_unit_;

        // s es la distancia recorrida HACIA ATRAS desde el detector.
        // El punto a distancia s del detector tiene radio
        //   r(s)^2 = R_d^2 + s^2 + 2 R_d s cos(theta_z).
        // El cruce entrante con la esfera de radio R >= R_earth es
        //   s(R) = -R_d cos(theta_z)
        //          + sqrt(R^2 - R_d^2 (1-cos^2(theta_z))).
        static double BackwardDistanceKm(
            double target_radius_km,
            double detector_radius_km,
            double coszen)
        {
            const double sine_squared =
                std::max(0.0, 1.0 - coszen * coszen);
            double radicand =
                target_radius_km * target_radius_km
                - detector_radius_km * detector_radius_km * sine_squared;

            const double tol = 128.0
                * std::numeric_limits<double>::epsilon()
                * target_radius_km * target_radius_km;
            if (radicand < 0.0 && radicand >= -tol)
                radicand = 0.0;
            if (radicand < 0.0)
                throw std::runtime_error("EarthDepthBody: interseccion geometrica imposible.");

            return -detector_radius_km * coszen + std::sqrt(radicand);
        }

        static double BaselineKm(
            double zenith_rad,
            double earth_radius_km,
            double atmosphere_height_km,
            double detector_depth_km)
        {
            if (!std::isfinite(zenith_rad)
                || zenith_rad < 0.0 || zenith_rad > std::acos(-1.0))
                throw std::invalid_argument("EarthDepthBody: zenith fuera de [0,pi].");
            if (!std::isfinite(earth_radius_km) || earth_radius_km <= 0.0)
                throw std::invalid_argument("EarthDepthBody: radio terrestre invalido.");
            if (!std::isfinite(atmosphere_height_km) || atmosphere_height_km <= 0.0)
                throw std::invalid_argument("EarthDepthBody: altura atmosferica invalida.");
            if (!std::isfinite(detector_depth_km) || detector_depth_km < 0.0
                || detector_depth_km >= earth_radius_km)
                throw std::invalid_argument("EarthDepthBody: profundidad del detector invalida.");

            const double Rd = earth_radius_km - detector_depth_km;
            const double Ra = earth_radius_km + atmosphere_height_km;
            const double L = BackwardDistanceKm(Ra, Rd, std::cos(zenith_rad));
            if (!std::isfinite(L) || L <= 0.0)
                throw std::runtime_error("EarthDepthBody: baseline invalido.");
            return L;
        }

    public:
        Track(double zenith_rad,
              double earth_radius_km,
              double atmosphere_height_km,
              double detector_depth_km,
              double km_unit)
            : nusquids::Body::Track(
                  0.0,
                  BaselineKm(zenith_rad, earth_radius_km,
                             atmosphere_height_km, detector_depth_km) * km_unit),
              zenith_rad_(zenith_rad),
              coszen_(std::cos(zenith_rad)),
              earth_radius_km_(earth_radius_km),
              atmosphere_height_km_(atmosphere_height_km),
              detector_depth_km_(detector_depth_km),
              detector_radius_km_(earth_radius_km - detector_depth_km),
              baseline_km_(BaselineKm(zenith_rad, earth_radius_km,
                                     atmosphere_height_km, detector_depth_km)),
              km_unit_(km_unit)
        {
            if (!std::isfinite(km_unit_) || km_unit_ <= 0.0)
                throw std::invalid_argument("EarthDepthBody: unidad km invalida.");
        }

        // Radio a cualquier posicion x del Track (x en eV^-1).
        double RadiusKm() const
        {
            const double x = GetX();
            const double start = GetInitialX();
            // The physical detector is ALWAYS at the full baseline,
            // even when the numerical interval ends at the Earth surface.
            const double end = GetFinalX();
            const double physical_end = baseline_km_ * km_unit_;
            const double tolerance = 128.0
                * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(end));

            if (!std::isfinite(x) || x < start - tolerance || x > end + tolerance)
                throw std::runtime_error("EarthDepthBody::Track: x fuera de trayectoria.");

            const double s_km = std::max(0.0, physical_end - x) / km_unit_;
            double r2 = detector_radius_km_ * detector_radius_km_
                        + s_km * s_km
                        + 2.0 * detector_radius_km_ * s_km * coszen_;

            const double geom_scale =
                detector_radius_km_ * detector_radius_km_ + s_km * s_km;
            if (r2 < 0.0 && r2 >= -128.0
                    * std::numeric_limits<double>::epsilon() * geom_scale)
                r2 = 0.0;
            if (r2 < 0.0)
                throw std::runtime_error("EarthDepthBody::Track: radio al cuadrado negativo.");

            return std::sqrt(r2);
        }

        // Coordenada x [eV^-1] del punto de ALTURA h en el tramo
        // ATMOSFERICO ENTRANTE, en orden desde el tope hasta la superficie.
        // Para neutrinos ascendentes corresponde al hemisferio opuesto.
        double XAtIncomingHeightKm(double height_km) const
        {
            const double tolerance_km = 128.0
                * std::numeric_limits<double>::epsilon()
                * std::max(1.0, atmosphere_height_km_);
            if (!std::isfinite(height_km) || height_km < -tolerance_km
                || height_km > atmosphere_height_km_ + tolerance_km)
                throw std::invalid_argument("EarthDepthBody::Track: altura fuera de atmosfera.");

            const double h = std::clamp(height_km, 0.0, atmosphere_height_km_);
            const double s_km = BackwardDistanceKm(
                earth_radius_km_ + h, detector_radius_km_, coszen_);
            const double x_km = baseline_km_ - s_km;
            if (x_km < -tolerance_km || x_km > baseline_km_ + tolerance_km)
                throw std::runtime_error("EarthDepthBody::Track: posicion de altura invalida.");

            return std::clamp(x_km, 0.0, baseline_km_) * km_unit_;
        }

        double XAtEarthEntry() const
        {
            return XAtIncomingHeightKm(0.0);
        }

        double GetAtmospherePathLengthKm() const
        {
            return XAtEarthEntry() / km_unit_;
        }

        double GetEarthPathLengthKm() const
        {
            return baseline_km_ - GetAtmospherePathLengthKm();
        }

        // ---------------------------------------------------------------
        // Split the NUMERICAL propagation without changing the physical
        // geometry. The first interval is production atmosphere; the second
        // traverses PREM Earth down to the detector. The same nuSQUIDS
        // object must be kept so off-diagonal coherences are not lost.
        //
        // nuSQUIDS::EvolveState() advances by (xend-xini), hence updating
        // both protected endpoints makes successive calls incremental.
        // ---------------------------------------------------------------
        void SetIntegrationInterval(double beginning, double finish)
        {
            const double total = baseline_km_ * km_unit_;
            const double tol = 128.0 *
                std::numeric_limits<double>::epsilon() * std::max(1.0, total);

            if (!std::isfinite(beginning) || !std::isfinite(finish) ||
                beginning < -tol || finish > total + tol ||
                finish <= beginning)
                throw std::invalid_argument(
                    "EarthDepthBody::Track: invalid propagation segment.");

            // NuSQuIDS advances its internal time to the precise endpoint,
            // but Track::GetX() may still be at the final RHS evaluation
            // (not necessarily the exact endpoint). Compare against the
            // PREVIOUS interval endpoint instead of GetX().
            const bool starting_from_top =
                beginning == 0.0 && GetInitialX() == 0.0;
            if (!starting_from_top && std::abs(GetFinalX()-beginning) > tol)
                throw std::logic_error(
                    "EarthDepthBody::Track: segments are not contiguous.");

            xini = std::clamp(beginning, 0.0, total);
            xend = std::clamp(finish, 0.0, total);
            x = xini;
        }

        void SelectAtmosphericSegment()
        {
            SetIntegrationInterval(0.0, XAtEarthEntry());
        }

        void SelectEarthSegment()
        {
            SetIntegrationInterval(XAtEarthEntry(), baseline_km_ * km_unit_);
        }

        double GetZenithRad() const { return zenith_rad_; }
        double GetCosZenith() const { return coszen_; }
        double GetEarthRadiusKm() const { return earth_radius_km_; }
        double GetAtmosphereHeightKm() const { return atmosphere_height_km_; }
        double GetDetectorDepthKm() const { return detector_depth_km_; }
        double GetDetectorRadiusKm() const { return detector_radius_km_; }
        double GetBaselineKm() const { return baseline_km_; }

        void FillDerivedParams(std::vector<double>& params) const override
        {
            params.push_back(zenith_rad_);
            params.push_back(coszen_);
            params.push_back(earth_radius_km_);
            params.push_back(atmosphere_height_km_);
            params.push_back(detector_depth_km_);
            params.push_back(detector_radius_km_);
            params.push_back(baseline_km_);
        }

        // Este Track no se serializa de forma independiente.
        void Serialize(hid_t group) const override { (void)group; }
        static std::string GetName() { return "EarthDepthTrack"; }
    };

private:
    double detector_depth_km_;

    double DensityFromRadiusKm(double r_km) const
    {
        if (!std::isfinite(r_km) || r_km < 0.0)
            throw std::invalid_argument("EarthDepthBody: radio invalido.");

        // Solo vacio durante todo el tramo atmosferico. La densidad
        // fisica del aire ya esta incorporada en la PRODUCCION MCEq.
        if (r_km > radius)
            return 0.0;

        // PREM se lee de EarthAtm; no modificamos la interpolacion radial.
        const double relative_radius = r_km / radius;
        if (relative_radius < x_radius_min)
            return x_rho_min;
        if (relative_radius > x_radius_max)
            return x_rho_max;
        return inter_density(relative_radius);
    }

    double YeFromRadiusKm(double r_km) const
    {
        if (!std::isfinite(r_km) || r_km < 0.0)
            throw std::invalid_argument("EarthDepthBody: radio invalido.");
        if (r_km > radius)
            return 0.494; // Inocuo: rho = 0 en la atmosfera.

        const double relative_radius = r_km / radius;
        if (relative_radius < x_radius_min)
            return x_ye_min;
        if (relative_radius > x_radius_max)
            return x_ye_max;
        return inter_ye(relative_radius);
    }

    void Validate() const
    {
        if (!std::isfinite(radius) || radius <= 0.0)
            throw std::runtime_error("EarthDepthBody: radio PREM invalido.");
        if (!std::isfinite(detector_depth_km_) || detector_depth_km_ < 0.0
            || detector_depth_km_ >= radius)
            throw std::invalid_argument("EarthDepthBody: profundidad invalida.");
        if (!std::isfinite(atm_height) || atm_height <= 0.0)
            throw std::runtime_error("EarthDepthBody: altura atmosferica invalida.");
    }

public:
    explicit EarthDepthBody(double detector_depth_km = 1.5)
        : nusquids::EarthAtm(), detector_depth_km_(detector_depth_km)
    {
        Validate();
    }

    EarthDepthBody(const std::string& earth_model_file,
                   double detector_depth_km = 1.5)
        : nusquids::EarthAtm(earth_model_file),
          detector_depth_km_(detector_depth_km)
    {
        Validate();
    }

    // Configurar con height_edges_km.front() leido del HDF5 de MCEq.
    // Se DEBE llamar antes de crear los Track.
    void SetAtmosphereHeightKm(double height_km)
    {
        if (!std::isfinite(height_km) || height_km <= 0.0)
            throw std::invalid_argument("EarthDepthBody: altura atmosferica invalida.");
        // Los dos miembros son heredados de EarthAtm. Esta actualizacion
        // funciona tambien con versiones sin SetAtmosphereHeight().
        atm_height = height_km;
        earth_with_atm_radius = radius + height_km;
    }

    double density(const nusquids::GenericTrack& track) const override
    {
        return DensityFromRadiusKm(static_cast<const Track&>(track).RadiusKm());
    }

    double ye(const nusquids::GenericTrack& track) const override
    {
        return YeFromRadiusKm(static_cast<const Track&>(track).RadiusKm());
    }

    std::shared_ptr<Track> MakeTrackToDetector(double zenith_rad) const
    {
        squids::Const units;
        return std::make_shared<Track>(
            zenith_rad, radius, atm_height, detector_depth_km_, units.km);
    }

    std::shared_ptr<Track> MakeTrackWithCosineToDetector(double coszen) const
    {
        if (!std::isfinite(coszen))
            throw std::invalid_argument("EarthDepthBody: coszen no finito.");
        const double tol = 64.0 * std::numeric_limits<double>::epsilon();
        if (coszen < -1.0 - tol || coszen > 1.0 + tol)
            throw std::invalid_argument("EarthDepthBody: coszen fuera de [-1,1].");
        return MakeTrackToDetector(std::acos(std::clamp(coszen, -1.0, 1.0)));
    }

    double GetDetectorDepthKm() const { return detector_depth_km_; }
    double GetDetectorRadiusKm() const { return radius - detector_depth_km_; }
    double GetEarthRadiusKm() const { return radius; }
    double GetAtmosphereHeightKm() const { return atm_height; }

    // Util para pruebas de la transicion vacio -> PREM.
    double EvaluateDensityAtRadiusKm(double r_km) const
    {
        return DensityFromRadiusKm(r_km);
    }

    double EvaluateYeAtRadiusKm(double r_km) const
    {
        return YeFromRadiusKm(r_km);
    }

    // Igual que el codigo anterior: no serializamos el Body personalizado.
    void Serialize(hid_t group) const override { (void)group; }
    static std::string GetName() { return "EarthDepthBody"; }
    static unsigned int GetId() { return 1007; }
};
