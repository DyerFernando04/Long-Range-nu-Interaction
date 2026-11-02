#pragma once

// PotentialLibrary.h
//
// Lector e interpolador de la LUT terrestre de build_vlri_lut.cpp.
// Este archivo NO calcula integrales de Yukawa ni resuelve oscilaciones.
//
//  - Le-Lmu y Le-Ltau usan la misma integral electronica I_e(R,m).
//  - Lmu-Ltau usa la integral neutronica I_n(R,m).
//  - Se interpola UNA VEZ en masa para construir un perfil radial.
//  - Durante la propagacion solo se interpola en radio (PCHIP).
//  - R > R_Tierra devuelve V=0: aproximacion de vacio atmosferico
//    acordada para el propagador, no el campo de Yukawa exacto exterior.
//
// HDF5 requerido:
//   /axes/mass_eV                         [mass]
//   /axes/radius_km                       [radius]
//   /potential/V_unit_electron_eV         [mass,radius]
//   /potential/V_unit_neutron_eV          [mass,radius]
//   atributo raiz: earth_radius_km
//
// Convencion:
//   Le-Lmu, Le-Ltau: V = gprime^2 * I_e.
//   Lmu-Ltau:        V = e*gprime*(xi-sin(theta_W)*chi)
//                           / (sin(theta_W)*cos(theta_W)) * I_n.
// Las cargas de sabor se aplican en el Hamiltoniano, NO en la LUT.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>

namespace lri {

enum class Model {
    LeLmu,
    LeLtau,
    LmuLtau
};

inline std::array<double, 3> FlavorCharges(Model model) {
    switch (model) {
        case Model::LeLmu:   return { 1.0, -1.0,  0.0};
        case Model::LeLtau:  return { 1.0,  0.0, -1.0};
        case Model::LmuLtau: return { 0.0,  1.0, -1.0};
    }
    throw std::invalid_argument("Modelo LRI desconocido");
}

struct Couplings {
    double gprime = 0.0;

    // Obligatorios para Lmu-Ltau si gprime != 0.
    // Se dejan sin valor deliberadamente: no asumimos xi ni chi.
    double xi = std::numeric_limits<double>::quiet_NaN();
    double chi = std::numeric_limits<double>::quiet_NaN();

    // Valores de referencia de baja energia; pueden sustituirse
    // por los valores/convenios adoptados en el analisis.
    double sin_theta_w = std::sqrt(0.23122);
    double electric_charge = std::sqrt(4.0 * 3.14159265358979323846
                                       / 137.035999084);
};

inline double EffectivePrefactor(Model model, const Couplings& p) {
    if (!std::isfinite(p.gprime) || p.gprime < 0.0)
        throw std::invalid_argument("gprime debe ser finito y no negativo");

    if (model == Model::LeLmu || model == Model::LeLtau)
        return p.gprime * p.gprime;

    if (model != Model::LmuLtau)
        throw std::invalid_argument("Modelo LRI desconocido");

    if (p.gprime == 0.0)
        return 0.0;

    if (!std::isfinite(p.xi) || !std::isfinite(p.chi))
        throw std::invalid_argument(
            "Lmu-Ltau requiere especificar xi y chi; no se infieren de gprime");

    const double s = p.sin_theta_w;
    if (!std::isfinite(s) || !(s > 0.0 && s < 1.0) ||
        !std::isfinite(p.electric_charge) || p.electric_charge <= 0.0)
        throw std::invalid_argument("Constantes electrodébiles invalidas");

    const double c = std::sqrt(1.0 - s * s);
    const double factor = p.electric_charge * p.gprime *
                          (p.xi - s * p.chi) / (s * c);
    if (!std::isfinite(factor))
        throw std::runtime_error("Prefactor Lmu-Ltau no finito");
    return factor;  // Puede ser negativo: no hay que aplicar abs().
}

namespace detail {

inline void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// HDF5: RAII para evitar handles abiertos cuando se producen excepciones.
class H5Object {
    hid_t id_ = -1;
    herr_t (*close_)(hid_t) = nullptr;

public:
    H5Object(hid_t id, herr_t (*closer)(hid_t)) : id_(id), close_(closer) {}
    ~H5Object() { if (id_ >= 0 && close_) close_(id_); }
    H5Object(const H5Object&) = delete;
    H5Object& operator=(const H5Object&) = delete;
    H5Object(H5Object&& other) noexcept : id_(other.id_), close_(other.close_) {
        other.id_ = -1;
    }
    hid_t Get() const { return id_; }
    bool Valid() const { return id_ >= 0; }
};

inline std::vector<double> ReadArray(hid_t file, const char* path,
                                     const std::vector<hsize_t>& expected_shape = {}) {
    H5Object dataset(H5Dopen2(file, path, H5P_DEFAULT), H5Dclose);
    Require(dataset.Valid(), std::string("Dataset HDF5 no encontrado: ") + path);

    H5Object space(H5Dget_space(dataset.Get()), H5Sclose);
    Require(space.Valid(), std::string("Dataspace invalido: ") + path);
    const int ndims = H5Sget_simple_extent_ndims(space.Get());
    Require(ndims > 0 && ndims <= 2, std::string("Rango HDF5 invalido: ") + path);

    std::vector<hsize_t> dims(static_cast<std::size_t>(ndims));
    Require(H5Sget_simple_extent_dims(space.Get(), dims.data(), nullptr) >= 0,
            std::string("No se pudieron leer las dimensiones: ") + path);
    if (!expected_shape.empty())
        Require(dims == expected_shape, std::string("Shape inesperado: ") + path);

    std::size_t size = 1;
    for (auto d : dims) {
        Require(d > 0, std::string("Dimension vacia: ") + path);
        size *= static_cast<std::size_t>(d);
    }
    std::vector<double> values(size);
    Require(H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL,
                    H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
            std::string("No se pudo leer: ") + path);
    return values;
}

inline double ReadDoubleAttribute(hid_t file, const char* name) {
    H5Object attribute(H5Aopen(file, name, H5P_DEFAULT), H5Aclose);
    Require(attribute.Valid(), std::string("Atributo HDF5 ausente: ") + name);
    double value = 0.0;
    Require(H5Aread(attribute.Get(), H5T_NATIVE_DOUBLE, &value) >= 0,
            std::string("Error leyendo atributo: ") + name);
    return value;
}

// Derivadas de PCHIP (Fritsch-Butland), preservando monotonia.
// Se usa sobre log(V) al interpolar en masa, y sobre V al interpolar en radio.
inline std::vector<double> PchipSlopes(const std::vector<double>& x,
                                       const std::vector<double>& y) {
    const std::size_t n = x.size();
    Require(n >= 2 && n == y.size(), "PCHIP requiere >=2 nodos");

    std::vector<double> h(n - 1), delta(n - 1), d(n);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        h[i] = x[i + 1] - x[i];
        Require(h[i] > 0.0 && std::isfinite(h[i]), "Eje PCHIP no creciente");
        delta[i] = (y[i + 1] - y[i]) / h[i];
        Require(std::isfinite(delta[i]), "Pendiente PCHIP no finita");
    }

    if (n == 2) {
        d[0] = delta[0];
        d[1] = delta[0];
        return d;
    }

    // Pendientes interiores: media armonica ponderada.
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double a = delta[i - 1];
        const double b = delta[i];
        if (a == 0.0 || b == 0.0 || std::signbit(a) != std::signbit(b)) {
            d[i] = 0.0;
        } else {
            const double w1 = 2.0 * h[i] + h[i - 1];
            const double w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / a + w2 / b);
        }
    }

    const auto endpoint = [](double h0, double h1, double a, double b) {
        double v = ((2.0 * h0 + h1) * a - h0 * b) / (h0 + h1);
        if (std::signbit(v) != std::signbit(a) || a == 0.0) return 0.0;
        if (std::signbit(a) != std::signbit(b) && std::abs(v) > 3.0 * std::abs(a))
            return 3.0 * a;
        return v;
    };

    d.front() = endpoint(h[0], h[1], delta[0], delta[1]);
    d.back() = endpoint(h[n-2], h[n-3], delta[n-2], delta[n-3]);
    return d;
}

inline double Hermite(double x0, double x1, double y0, double y1,
                      double d0, double d1, double x) {
    const double h = x1 - x0;
    const double t = (x - x0) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    return (2.0*t3 - 3.0*t2 + 1.0)*y0
         + (t3 - 2.0*t2 + t)*h*d0
         + (-2.0*t3 + 3.0*t2)*y1
         + (t3 - t2)*h*d1;
}

inline bool SameMass(double a, double b) {
    return std::abs(a - b) <= 1.0e-12 * std::max(std::abs(a), std::abs(b));
}

}  // namespace detail

// Un perfil radial para una masa fija. Se construye una vez por masa/modelo
// y se puede compartir entre hilos OpenMP (evaluaciones const).
class RadialProfile {
    Model model_;
    double mass_eV_;
    double earth_radius_km_;
    std::vector<double> radius_km_;
    std::vector<double> unit_eV_;
    std::vector<double> radial_slopes_;

public:
    RadialProfile(Model model, double mass_eV, double earth_radius_km,
                  std::vector<double> radius_km, std::vector<double> unit_eV)
        : model_(model), mass_eV_(mass_eV), earth_radius_km_(earth_radius_km),
          radius_km_(std::move(radius_km)), unit_eV_(std::move(unit_eV)) {
        detail::Require(radius_km_.size() == unit_eV_.size(),
                        "Perfil radial con dimensiones incompatibles");
        for (double v : unit_eV_)
            detail::Require(v > 0.0 && std::isfinite(v),
                            "Perfil radial contiene potencial invalido");
        radial_slopes_ = detail::PchipSlopes(radius_km_, unit_eV_);
    }

    Model GetModel() const { return model_; }
    double GetMassEV() const { return mass_eV_; }
    double GetEarthRadiusKm() const { return earth_radius_km_; }
    const std::vector<double>& RadiusGridKm() const { return radius_km_; }
    const std::vector<double>& UnitValuesEV() const { return unit_eV_; }

    // Regla del proyecto: V_LRI se anula en la atmosfera (r > R_earth).
    // NO es la extrapolacion fisica Yukawa exterior de una fuente terrestre.
    double UnitAtRadiusKm(double r_km) const {
        if (!std::isfinite(r_km) || r_km < 0.0)
            throw std::invalid_argument("Radio de evaluacion invalido");
        if (r_km > earth_radius_km_) return 0.0;
        if (r_km <= radius_km_.front()) return unit_eV_.front();
        if (r_km >= radius_km_.back()) return unit_eV_.back();

        const auto it = std::upper_bound(radius_km_.begin(), radius_km_.end(), r_km);
        const std::size_t j = static_cast<std::size_t>(it - radius_km_.begin()) - 1;
        double v = detail::Hermite(radius_km_[j], radius_km_[j+1],
                                   unit_eV_[j], unit_eV_[j+1],
                                   radial_slopes_[j], radial_slopes_[j+1], r_km);
        // Guard contra errores de redondeo en la interpolacion.
        if (v < 0.0 && v > -1.0e-12 * std::max(unit_eV_[j], unit_eV_[j+1]))
            v = 0.0;
        detail::Require(std::isfinite(v) && v >= 0.0,
                        "Potencial radial interpolado invalido");
        return v;
    }

    double PotentialAtRadiusKm(double r_km, const Couplings& p) const {
        return EffectivePrefactor(model_, p) * UnitAtRadiusKm(r_km);
    }
};

// Lee ambas fuentes una sola vez. La interpolacion log-log en masa
// construye un RadialProfile que se reutiliza durante EvolveState().
class PotentialLibrary {
    std::string filename_;
    double earth_radius_km_;
    double detector_depth_km_;
    std::vector<double> masses_eV_;
    std::vector<double> log_masses_;
    std::vector<double> radii_km_;
    std::vector<double> electron_eV_;
    std::vector<double> neutron_eV_;

public:
    explicit PotentialLibrary(const std::string& filename) : filename_(filename) {
        detail::H5Object file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                              H5Fclose);
        detail::Require(file.Valid(), "No se pudo abrir la LUT: " + filename);

        masses_eV_ = detail::ReadArray(file.Get(), "/axes/mass_eV");
        radii_km_ = detail::ReadArray(file.Get(), "/axes/radius_km");
        detail::Require(masses_eV_.size() >= 2 && radii_km_.size() >= 2,
                        "LUT requiere al menos dos masas y dos radios");
        const std::vector<hsize_t> dims = {
            static_cast<hsize_t>(masses_eV_.size()),
            static_cast<hsize_t>(radii_km_.size())
        };
        electron_eV_ = detail::ReadArray(
            file.Get(), "/potential/V_unit_electron_eV", dims);
        neutron_eV_ = detail::ReadArray(
            file.Get(), "/potential/V_unit_neutron_eV", dims);
        earth_radius_km_ = detail::ReadDoubleAttribute(file.Get(), "earth_radius_km");
        detector_depth_km_ = detail::ReadDoubleAttribute(file.Get(), "detector_depth_km");

        detail::Require(std::isfinite(earth_radius_km_) && earth_radius_km_ > 0.0,
                        "Radio terrestre invalido en LUT");
        detail::Require(std::isfinite(detector_depth_km_) && detector_depth_km_ >= 0.0,
                        "Profundidad del detector invalida en LUT");
        detail::Require(std::abs(radii_km_.front()) < 1.0e-7 &&
                        std::abs(radii_km_.back() - earth_radius_km_) < 1.0e-7,
                        "La malla radial debe ir del centro a la superficie");

        for (std::size_t i = 0; i < masses_eV_.size(); ++i) {
            const double m = masses_eV_[i];
            detail::Require(std::isfinite(m) && m > 0.0, "Masa invalida en LUT");
            if (i > 0) detail::Require(m > masses_eV_[i-1], "Masas no crecientes");
            log_masses_.push_back(std::log(m));
        }
        for (std::size_t i = 0; i < radii_km_.size(); ++i) {
            detail::Require(std::isfinite(radii_km_[i]), "Radio no finito en LUT");
            if (i > 0) detail::Require(radii_km_[i] > radii_km_[i-1],
                                       "Radios no crecientes");
        }
        for (const auto* table : {&electron_eV_, &neutron_eV_}) {
            for (double v : *table)
                detail::Require(std::isfinite(v) && v > 0.0,
                                "LUT de Yukawa tiene valores no positivos");
        }
    }

    const std::string& Filename() const { return filename_; }
    double EarthRadiusKm() const { return earth_radius_km_; }
    double DetectorDepthKm() const { return detector_depth_km_; }
    const std::vector<double>& MassGridEV() const { return masses_eV_; }
    const std::vector<double>& RadiusGridKm() const { return radii_km_; }

    RadialProfile BuildRadialProfile(Model model, double mass_eV) const {
        if (!std::isfinite(mass_eV) || mass_eV <= 0.0)
            throw std::invalid_argument("Masa del mediador invalida");
        const std::vector<double>* table = nullptr;
        switch (model) {
            case Model::LeLmu:
            case Model::LeLtau: table = &electron_eV_; break;
            case Model::LmuLtau: table = &neutron_eV_; break;
        }
        detail::Require(table != nullptr, "Modelo LRI desconocido");

        const std::size_t nr = radii_km_.size();
        const std::size_t nm = masses_eV_.size();
        const double low = masses_eV_.front();
        const double high = masses_eV_.back();
        if (mass_eV < low && detail::SameMass(mass_eV, low)) mass_eV = low;
        if (mass_eV > high && detail::SameMass(mass_eV, high)) mass_eV = high;
        detail::Require(mass_eV >= low && mass_eV <= high,
                        "Masa fuera de la LUT: [" + std::to_string(low) + ", " +
                        std::to_string(high) + "] eV");

        const auto it = std::lower_bound(masses_eV_.begin(), masses_eV_.end(), mass_eV);
        const std::size_t upper = static_cast<std::size_t>(it - masses_eV_.begin());
        std::vector<double> profile(nr);

        // Nodo de masa exacto: copiar sin interpolar.
        std::size_t exact = nm;
        if (upper < nm && detail::SameMass(mass_eV, masses_eV_[upper]))
            exact = upper;
        else if (upper > 0 && detail::SameMass(mass_eV, masses_eV_[upper - 1]))
            exact = upper - 1;
        if (exact < nm) {
            std::copy_n(table->begin() + exact * nr, nr, profile.begin());
            return RadialProfile(model, mass_eV, earth_radius_km_,
                                 radii_km_, std::move(profile));
        }

        // Masa fuera de nodos: PCHIP local en (log(m), log(I_f)).
        detail::Require(upper > 0 && upper < nm, "Intervalo de masas invalido");
        const std::size_t lower = upper - 1;
        const std::size_t first = (lower > 0) ? lower - 1 : 0;
        const std::size_t last = std::min(nm - 1, upper + 1);
        const std::size_t nlocal = last - first + 1;
        const std::size_t interval = lower - first;
        std::vector<double> x(nlocal), y(nlocal);
        for (std::size_t j = 0; j < nlocal; ++j)
            x[j] = log_masses_[first + j];

        const double lm = std::log(mass_eV);
        for (std::size_t ir = 0; ir < nr; ++ir) {
            for (std::size_t j = 0; j < nlocal; ++j)
                y[j] = std::log((*table)[(first + j) * nr + ir]);
            const auto slopes = detail::PchipSlopes(x, y);
            profile[ir] = std::exp(detail::Hermite(
                x[interval], x[interval+1], y[interval], y[interval+1],
                slopes[interval], slopes[interval+1], lm));
            detail::Require(std::isfinite(profile[ir]) && profile[ir] > 0.0,
                            "Interpolacion del potencial en masa invalida");
        }
        return RadialProfile(model, mass_eV, earth_radius_km_,
                             radii_km_, std::move(profile));
    }
};

}  // namespace lri
