// build_vlri_lut.cpp
//
// Tablas de integrales unitarias de Yukawa para dos fuentes terrestres:
// electrones (Le-Lmu y Le-Ltau) y neutrones (Lmu-Ltau).
//
// I_f(R,m) = 1/(4 pi) integral_Tierra d^3r n_f(r) exp(-m|R-r|)/|R-r|
//
//   Le-Lmu, Le-Ltau : V = g'^2 * I_e(R,m)
//   Lmu-Ltau       : V = [e/(sin(theta_W)cos(theta_W))]
//                          * g' * (xi - sin(theta_W)*chi) * I_n(R,m)
//
// La LUT guarda SOLO I_e e I_n (eV), sin prefactores de acoplamiento.
// Los operadores de carga de sabor y los acoplamientos van en propagacion.
// La atmosfera NO se incluye como fuente. Su Hamiltoniano se define luego.
//
// Ejecucion desde la raiz del repositorio:
//   ./LRI/potential/build_vlri_lut
//   ./LRI/potential/build_vlri_lut LRI/output/potential/vlri_unit.h5
//   ./LRI/potential/build_vlri_lut output.h5 EARTH_MODEL.dat
//
// Se conserva el formato HDF5 del generador previo:
//   /axes/mass_eV, /axes/log10_mass_eV, /axes/radius_km
//   /earth/rho_g_cm3, /earth/Ye, /earth/ne_cm3, /earth/nn_cm3, ...
//   /potential/V_unit_electron_eV [mass,radius]
//   /potential/V_unit_neutron_eV  [mass,radius]
//   /potential/V_unit_eV -> hard link a la tabla electronica (compatibilidad)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <nuSQuIDS/nuSQuIDS.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double LOG10_MASS_MIN = -16.0;
constexpr double LOG10_MASS_MAX = -11.0;
constexpr std::size_t N_MASS = 201;
constexpr double RADIAL_STEP_KM = 0.25;
constexpr int GL_ORDER = 16;
constexpr double DETECTOR_DEPTH_KM = 1.5;
constexpr long double N_AVOGADRO = 6.02214076e23L;
constexpr double VALIDATION_TOLERANCE = 1.0e-6;
constexpr double GSL_VALIDATION_TOLERANCE = 1.0e-9;
constexpr std::size_t GSL_WORKSPACE_SIZE = 5000;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

enum class Source { Electron, Neutron };

struct EarthSample {
    double rho_g_cm3;
    double ye;
    double ne_cm3;
    double ne_eV3;
    double nn_cm3;
    double nn_eV3;
};

// Usamos el PREM y la interpolacion propios de la clase Earth de nuSQuIDS.
class EarthSampler {
    squids::Const units_;
    std::shared_ptr<nusquids::Earth> earth_;
    std::unique_ptr<nusquids::Earth::Track> track_;
    double radius_km_;

public:
    explicit EarthSampler(const std::string& earth_model) {
        if (earth_model.empty())
            earth_ = std::make_shared<nusquids::Earth>();
        else
            earth_ = std::make_shared<nusquids::Earth>(earth_model);

        radius_km_ = earth_->GetRadius();
        Check(std::isfinite(radius_km_) && radius_km_ > 6000.0 &&
              radius_km_ < 7000.0, "Radio terrestre inesperado");

        // Track diametral: en la coordenada x = R_earth - r,
        // Earth::density(track) y Earth::ye(track) evaluan el radio r.
        track_ = std::make_unique<nusquids::Earth::Track>(
            2.0 * radius_km_ * units_.km);
    }

    double RadiusKm() const { return radius_km_; }
    double KmToEVInverse() const { return units_.km; }
    double CmToEVInverse() const { return units_.cm; }

    EarthSample Sample(double radius_km) {
        Check(radius_km >= 0.0 && radius_km <= radius_km_ &&
              std::isfinite(radius_km), "Radio fuera de la Tierra");
        track_->SetX((radius_km_ - radius_km) * units_.km);
        const double rho = earth_->density(*track_);
        const double ye = earth_->ye(*track_);
        Check(std::isfinite(rho) && std::isfinite(ye) &&
              rho >= 0.0 && ye >= 0.0 && ye <= 1.0,
              "PREM ha devuelto rho/Ye invalidos");
        // Neutralidad electrica: protones por nucleon = Ye.
        // Neutrones por nucleon = 1 - Ye (aproximacion PREM).
        const long double ne = static_cast<long double>(rho) * ye * N_AVOGADRO;
        const long double nn = static_cast<long double>(rho) * (1.0L - ye) * N_AVOGADRO;
        const long double c = units_.cm;
        return {rho, ye, static_cast<double>(ne),
                static_cast<double>(ne / (c * c * c)),
                static_cast<double>(nn),
                static_cast<double>(nn / (c * c * c))};
    }
};

std::vector<double> MassGrid() {
    std::vector<double> masses(N_MASS);
    for (std::size_t i = 0; i < N_MASS; ++i) {
        const double logm = LOG10_MASS_MIN +
            (LOG10_MASS_MAX - LOG10_MASS_MIN) *
            static_cast<double>(i) / static_cast<double>(N_MASS - 1);
        masses[i] = std::pow(10.0, logm);
    }
    return masses;
}

std::vector<double> RadiusGrid(double earth_radius_km) {
    // Incluimos exactamente el radio del detector y el limite nucleo-manto.
    std::vector<double> r;
    for (std::size_t i = 0;
         i * RADIAL_STEP_KM < earth_radius_km; ++i)
        r.push_back(i * RADIAL_STEP_KM);
    r.push_back(earth_radius_km);
    r.push_back(earth_radius_km - DETECTOR_DEPTH_KM);
    r.push_back(3480.0);
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end(), [](double a, double b) {
        return std::abs(a - b) < 1.0e-10;
    }), r.end());
    return r;
}

std::size_t ExactRadiusIndex(const std::vector<double>& r, double target) {
    auto it = std::lower_bound(r.begin(), r.end(), target);
    if (it != r.end() && std::abs(*it - target) < 1.0e-8)
        return static_cast<std::size_t>(it - r.begin());
    throw std::runtime_error("El radio solicitado no esta en la malla");
}

struct EarthProfile {
    std::vector<double> rho, ye, ne_cm3, ne_eV3, nn_cm3, nn_eV3;
};

EarthProfile SampleEarthProfile(EarthSampler& earth,
                               const std::vector<double>& r) {
    EarthProfile p;
    for (const double radius : r) {
        const EarthSample s = earth.Sample(radius);
        p.rho.push_back(s.rho_g_cm3);
        p.ye.push_back(s.ye);
        p.ne_cm3.push_back(s.ne_cm3);
        p.ne_eV3.push_back(s.ne_eV3);
        p.nn_cm3.push_back(s.nn_cm3);
        p.nn_eV3.push_back(s.nn_eV3);
    }
    return p;
}

struct QuadratureCache {
    std::vector<long double> radius_eV_inv;
    std::vector<long double> weighted_electron_eV;
    std::vector<long double> weighted_neutron_eV;
};

QuadratureCache BuildSourceCache(EarthSampler& earth,
                                const std::vector<double>& radii) {
    QuadratureCache cache;
    const std::size_t total = (radii.size() - 1) * GL_ORDER;
    cache.radius_eV_inv.reserve(total);
    cache.weighted_electron_eV.reserve(total);
    cache.weighted_neutron_eV.reserve(total);

    gsl_integration_glfixed_table* table =
        gsl_integration_glfixed_table_alloc(GL_ORDER);
    Check(table != nullptr, "No se pudo crear la cuadratura Gauss-Legendre");

    const long double km = earth.KmToEVInverse();
    for (std::size_t i = 0; i + 1 < radii.size(); ++i) {
        for (int k = 0; k < GL_ORDER; ++k) {
            double rq_km = 0.0, weight_km = 0.0;
            const int status = gsl_integration_glfixed_point(
                radii[i], radii[i + 1], k, &rq_km, &weight_km, table);
            Check(status == GSL_SUCCESS, "Fallo nodo Gauss-Legendre");
            const EarthSample s = earth.Sample(rq_km);
            const long double rq = static_cast<long double>(rq_km) * km;
            const long double dr = static_cast<long double>(weight_km) * km;
            cache.radius_eV_inv.push_back(rq);
            cache.weighted_electron_eV.push_back(
                dr * rq * static_cast<long double>(s.ne_eV3));
            cache.weighted_neutron_eV.push_back(
                dr * rq * static_cast<long double>(s.nn_eV3));
        }
    }
    gsl_integration_glfixed_table_free(table);
    return cache;
}

// Para cada fuente f=e,n, simetria esferica:
//
//   I_f(R,m) = exp(-mR)/(mR) * int_0^R dr r n_f(r) sinh(mr)
//            + sinh(mR)/(mR) * int_R^Re dr r n_f(r) exp(-mr)
//
// En R=0: I_f(0,m) = int_0^Re dr r n_f(r) exp(-mr).
// Las integrales llevan r y dr en eV^-1 y n_f en eV^3.
// Se calculan AMBAS fuentes en la misma cuadratura y bucle de masas.
struct PotentialTables {
    std::vector<double> electron;
    std::vector<double> neutron;
};

PotentialTables BuildPotentials(const std::vector<double>& masses,
                                const std::vector<double>& radii,
                                const QuadratureCache& cache,
                                double km_to_eV_inv) {
    const std::size_t nr = radii.size();
    const std::size_t ns = nr - 1;
    PotentialTables table;
    table.electron.resize(masses.size() * nr);
    table.neutron.resize(masses.size() * nr);

#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
#else
    std::cout << "OpenMP no habilitado; calculo secuencial\n";
#endif

#pragma omp parallel for schedule(dynamic)
    for (long long im = 0; im < static_cast<long long>(masses.size()); ++im) {
        const long double mass = masses[static_cast<std::size_t>(im)];
        std::vector<long double> Ae(nr, 0.0L), Be(nr, 0.0L);
        std::vector<long double> An(nr, 0.0L), Bn(nr, 0.0L);
        std::vector<long double> segmentBe(ns), segmentBn(ns);

        for (std::size_t is = 0; is < ns; ++is) {
            long double ca_e = 0.0L, cb_e = 0.0L;
            long double ca_n = 0.0L, cb_n = 0.0L;
            for (int k = 0; k < GL_ORDER; ++k) {
                const std::size_t q = is * GL_ORDER + k;
                const long double r = cache.radius_eV_inv[q];
                const long double sh = std::sinh(mass * r);
                const long double ex = std::exp(-mass * r);
                const long double we = cache.weighted_electron_eV[q];
                const long double wn = cache.weighted_neutron_eV[q];
                ca_e += we * sh;
                cb_e += we * ex;
                ca_n += wn * sh;
                cb_n += wn * ex;
            }
            Ae[is + 1] = Ae[is] + ca_e;
            An[is + 1] = An[is] + ca_n;
            segmentBe[is] = cb_e;
            segmentBn[is] = cb_n;
        }
        for (std::size_t is = ns; is > 0; --is) {
            Be[is - 1] = Be[is] + segmentBe[is - 1];
            Bn[is - 1] = Bn[is] + segmentBn[is - 1];
        }

        for (std::size_t ir = 0; ir < nr; ++ir) {
            long double ve = 0.0L, vn = 0.0L;
            if (ir == 0) {
                ve = Be[0];
                vn = Bn[0];
            } else {
                const long double R =
                    static_cast<long double>(radii[ir]) * km_to_eV_inv;
                const long double z = mass * R;
                const long double coefA = std::exp(-z) / z;
                const long double coefB = std::sinh(z) / z;
                ve = coefA * Ae[ir] + coefB * Be[ir];
                vn = coefA * An[ir] + coefB * Bn[ir];
            }
            Check(std::isfinite(ve) && ve > 0.0L &&
                  std::isfinite(vn) && vn > 0.0L,
                  "Potencial unitario invalido para electrones o neutrones");
            const std::size_t idx = static_cast<std::size_t>(im) * nr + ir;
            table.electron[idx] = static_cast<double>(ve);
            table.neutron[idx] = static_cast<double>(vn);
        }
    }
    return table;
}

// Comprobacion independiente por integracion GSL adaptativa.
// Solo se usa en varios puntos representativos y NO durante la propagacion.
struct DirectContext {
    EarthSampler* earth;
    Source source;
    long double m, R;
};

double DirectIntegrand(double r_km, void* context) {
    auto& c = *static_cast<DirectContext*>(context);
    const long double km = c.earth->KmToEVInverse();
    const EarthSample s = c.earth->Sample(r_km);
    const long double r = static_cast<long double>(r_km) * km;
    const long double m = c.m;
    const long double R = c.R;
    long double kernel;
    if (R == 0.0L)
        kernel = std::exp(-m * r);
    else if (r <= R)
        kernel = std::exp(-m * R) * std::sinh(m * r) / (m * R);
    else
        kernel = std::exp(-m * r) * std::sinh(m * R) / (m * R);
    const double density = c.source == Source::Electron
        ? s.ne_eV3 : s.nn_eV3;
    return static_cast<double>(r * density * kernel * km);
}

double DirectPotential(EarthSampler& earth, Source source,
                       double mass, double evaluation_km) {
    auto workspace = std::unique_ptr<gsl_integration_workspace,
        decltype(&gsl_integration_workspace_free)>(
            gsl_integration_workspace_alloc(GSL_WORKSPACE_SIZE),
            &gsl_integration_workspace_free);
    Check(static_cast<bool>(workspace), "No se pudo crear workspace GSL");
    DirectContext context{&earth, source,
                          mass, evaluation_km * earth.KmToEVInverse()};
    gsl_function integrand;
    integrand.function = &DirectIntegrand;
    integrand.params = &context;

    double total = 0.0;
    auto integrate = [&](double a, double b) {
        if (b <= a) return;
        double val = 0.0, error = 0.0;
        const int status = gsl_integration_qag(
            &integrand, a, b, 0.0, GSL_VALIDATION_TOLERANCE,
            GSL_WORKSPACE_SIZE, GSL_INTEG_GAUSS61,
            workspace.get(), &val, &error);
        Check(status == GSL_SUCCESS,
              "La validacion GSL fallo: " + std::string(gsl_strerror(status)));
        total += val;
    };
    if (evaluation_km > 0.0 && evaluation_km < earth.RadiusKm()) {
        integrate(0.0, evaluation_km);
        integrate(evaluation_km, earth.RadiusKm());
    } else {
        integrate(0.0, earth.RadiusKm());
    }
    return total;
}

double ValidatePotential(EarthSampler& earth, Source source,
                         const std::vector<double>& masses,
                         const std::vector<double>& radii,
                         const std::vector<double>& table) {
    const std::vector<std::size_t> m_indices = {
        0, masses.size() / 2, masses.size() - 1};
    const std::vector<double> test_radii = {
        0.0, 3480.0, earth.RadiusKm() - DETECTOR_DEPTH_KM,
        earth.RadiusKm()};
    double max_error = 0.0;
    std::cout << "Validacion GSL de fuente "
              << (source == Source::Electron ? "electronica" : "neutronica")
              << ":\n";
    for (std::size_t im : m_indices) {
        for (double R : test_radii) {
            const std::size_t ir = ExactRadiusIndex(radii, R);
            const double approx = table[im * radii.size() + ir];
            const double direct = DirectPotential(earth, source, masses[im], R);
            const double rel = std::abs(approx - direct) /
                std::max(std::abs(direct), std::numeric_limits<double>::min());
            max_error = std::max(max_error, rel);
            std::cout << "  m=" << std::scientific << masses[im]
                      << " eV, r=" << R << " km, rel=" << rel << '\n';
        }
    }
    Check(max_error <= VALIDATION_TOLERANCE,
          "La LUT no supero la validacion numerica (error > 1e-6)");
    return max_error;
}

void CheckH5(herr_t status, const std::string& what) {
    Check(status >= 0, "HDF5: " + what);
}

void AttributeString(hid_t object, const char* key, const std::string& value) {
    hid_t datatype = H5Tcopy(H5T_C_S1);
    Check(datatype >= 0, "HDF5 string datatype");
    CheckH5(H5Tset_size(datatype, value.size() + 1), "string size");
    CheckH5(H5Tset_strpad(datatype, H5T_STR_NULLTERM), "string padding");
    hid_t space = H5Screate(H5S_SCALAR);
    Check(space >= 0, "HDF5 scalar space");
    hid_t attr = H5Acreate2(object, key, datatype, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    Check(attr >= 0, "HDF5 create attribute");
    CheckH5(H5Awrite(attr, datatype, value.c_str()), "write string attribute");
    H5Aclose(attr); H5Sclose(space); H5Tclose(datatype);
}

void AttributeDouble(hid_t object, const char* key, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    Check(space >= 0, "HDF5 scalar space");
    hid_t attr = H5Acreate2(object, key, H5T_IEEE_F64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    Check(attr >= 0, "HDF5 create numeric attribute");
    CheckH5(H5Awrite(attr, H5T_NATIVE_DOUBLE, &value), "write numeric attribute");
    H5Aclose(attr); H5Sclose(space);
}

void AttributeUInt64(hid_t object, const char* key, std::uint64_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    Check(space >= 0, "HDF5 scalar space");
    hid_t attr = H5Acreate2(object, key, H5T_STD_U64LE, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    Check(attr >= 0, "HDF5 create integer attribute");
    CheckH5(H5Awrite(attr, H5T_NATIVE_ULLONG, &value), "write integer attribute");
    H5Aclose(attr); H5Sclose(space);
}

void WriteVector(hid_t group, const char* name,
                 const std::vector<double>& values, const char* unit) {
    const hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    CheckH5(H5LTmake_dataset_double(group, name, 1, dims, values.data()),
            std::string("write ") + name);
    hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
    Check(dataset >= 0, "HDF5 open dataset");
    AttributeString(dataset, "unit", unit);
    H5Dclose(dataset);
}

void WriteMatrix(hid_t group, const char* name,
                 const std::vector<double>& values,
                 std::size_t rows, std::size_t columns, const char* unit) {
    Check(values.size() == rows * columns, "HDF5 matrix shape incompatible");
    const hsize_t dims[2] = {static_cast<hsize_t>(rows),
                             static_cast<hsize_t>(columns)};
    CheckH5(H5LTmake_dataset_double(group, name, 2, dims, values.data()),
            std::string("write ") + name);
    hid_t dataset = H5Dopen2(group, name, H5P_DEFAULT);
    Check(dataset >= 0, "HDF5 open dataset");
    AttributeString(dataset, "unit", unit);
    H5Dclose(dataset);
}

void WriteLUT(const std::string& filename, const std::string& earth_model,
              const EarthSampler& earth, const std::vector<double>& masses,
              const std::vector<double>& radii,
              const EarthProfile& profile, const PotentialTables& V,
              double validation_error_e, double validation_error_n) {
    const auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    Check(!std::filesystem::exists(filename),
          "El HDF5 ya existe y no se sobrescribira: " + filename);

    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_EXCL,
                           H5P_DEFAULT, H5P_DEFAULT);
    Check(file >= 0, "No se pudo crear el HDF5 de salida");
    try {
        AttributeString(file, "format_name",
                        "nuSQuIDS-based terrestrial LRI unit-potential library");
        AttributeString(file, "format_version", "2.0");
        AttributeString(file, "electron_model_potential",
                        "V_ebeta = g_ebeta^2 * I_e(R,m), beta=mu,tau");
        AttributeString(file, "neutron_model_potential",
                        "V_mutau = [e/(sin(theta_W)*cos(theta_W))] * g_mutau * "
                        "(xi - sin(theta_W)*chi) * I_n(R,m)");
        AttributeString(file, "unit_potential_definition",
                        "I_f = (1/4pi) integral_Tierra d3r n_f(r) exp(-m|R-r|)/|R-r|");
        AttributeString(file, "earth_source", earth_model.empty()
            ? "nuSQuIDS default Earth()" : earth_model);
        AttributeString(file, "earth_sampling",
                        "rho and Ye queried using nuSQuIDS Earth::density/ye");
        AttributeString(file, "earth_interpolation_semantics",
                        "Uses the native nuSQuIDS Earth PREM interpolation");
        AttributeString(file, "integration_method",
                        "16-point Gauss-Legendre per 0.25 km segment, spherical Yukawa reduction");
        AttributeString(file, "source_region", "terrestrial electrons and neutrons");
        AttributeString(file, "coupling_convention",
                        "No couplings applied to I_e/I_n; model dependent prefactor");
        AttributeString(file, "source_electron_density", "rho * Ye * N_A");
        AttributeString(file, "source_neutron_density", "rho * (1 - Ye) * N_A");
        AttributeString(file, "potential_dimension_order", "mass, radius");
        AttributeDouble(file, "earth_radius_km", earth.RadiusKm());
        AttributeDouble(file, "detector_depth_km", DETECTOR_DEPTH_KM);
        AttributeDouble(file, "detector_radius_km",
                        earth.RadiusKm() - DETECTOR_DEPTH_KM);
        AttributeDouble(file, "radial_step_km", RADIAL_STEP_KM);
        AttributeDouble(file, "km_to_eV_inverse", earth.KmToEVInverse());
        AttributeDouble(file, "cm_to_eV_inverse", earth.CmToEVInverse());
        AttributeDouble(file, "avogadro_constant", static_cast<double>(N_AVOGADRO));
        AttributeDouble(file, "maximum_validation_relative_error",
                        std::max(validation_error_e, validation_error_n));
        AttributeDouble(file, "validation_relative_error_electron", validation_error_e);
        AttributeDouble(file, "validation_relative_error_neutron", validation_error_n);
        AttributeUInt64(file, "number_mass_points", masses.size());
        AttributeUInt64(file, "number_radius_points", radii.size());
        AttributeUInt64(file, "gauss_legendre_order", GL_ORDER);
        AttributeUInt64(file, "detector_radius_index",
                        ExactRadiusIndex(radii, earth.RadiusKm() - DETECTOR_DEPTH_KM));

        hid_t axes = H5Gcreate2(file, "/axes", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        Check(axes >= 0, "No se pudo crear /axes");
        std::vector<double> logs(masses.size());
        for (std::size_t i = 0; i < masses.size(); ++i)
            logs[i] = std::log10(masses[i]);
        WriteVector(axes, "mass_eV", masses, "eV");
        WriteVector(axes, "log10_mass_eV", logs, "log10(eV)");
        WriteVector(axes, "radius_km", radii, "km");
        H5Gclose(axes);

        hid_t earth_group = H5Gcreate2(file, "/earth", H5P_DEFAULT,
                                       H5P_DEFAULT, H5P_DEFAULT);
        Check(earth_group >= 0, "No se pudo crear /earth");
        WriteVector(earth_group, "rho_g_cm3", profile.rho, "g/cm^3");
        WriteVector(earth_group, "Ye", profile.ye, "dimensionless");
        WriteVector(earth_group, "ne_cm3", profile.ne_cm3, "cm^-3");
        WriteVector(earth_group, "ne_eV3", profile.ne_eV3, "eV^3");
        WriteVector(earth_group, "nn_cm3", profile.nn_cm3, "cm^-3");
        WriteVector(earth_group, "nn_eV3", profile.nn_eV3, "eV^3");
        H5Gclose(earth_group);

        hid_t pot = H5Gcreate2(file, "/potential", H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
        Check(pot >= 0, "No se pudo crear /potential");
        WriteMatrix(pot, "V_unit_electron_eV", V.electron,
                    masses.size(), radii.size(), "eV");
        WriteMatrix(pot, "V_unit_neutron_eV", V.neutron,
                    masses.size(), radii.size(), "eV");

        // Alias HDF5 fisico (sin duplicar datos) para lectores anteriores.
        CheckH5(H5Lcreate_hard(pot, "V_unit_electron_eV",
                               pot, "V_unit_eV", H5P_DEFAULT, H5P_DEFAULT),
                "alias V_unit_eV");

        const auto ir = ExactRadiusIndex(
            radii, earth.RadiusKm() - DETECTOR_DEPTH_KM);
        std::vector<double> detector_e(masses.size()), detector_n(masses.size());
        for (std::size_t im = 0; im < masses.size(); ++im) {
            detector_e[im] = V.electron[im * radii.size() + ir];
            detector_n[im] = V.neutron[im * radii.size() + ir];
        }
        WriteVector(pot, "V_unit_electron_at_detector_eV", detector_e, "eV");
        WriteVector(pot, "V_unit_neutron_at_detector_eV", detector_n, "eV");
        CheckH5(H5Lcreate_hard(pot, "V_unit_electron_at_detector_eV",
                               pot, "V_unit_at_detector_eV", H5P_DEFAULT, H5P_DEFAULT),
                "alias V_unit_at_detector_eV");
        for (const char* name : {"V_unit_electron_at_detector_eV",
                                 "V_unit_neutron_at_detector_eV"}) {
            hid_t ds = H5Dopen2(pot, name, H5P_DEFAULT);
            Check(ds >= 0, "No se pudo abrir vector de potencial en detector");
            AttributeDouble(ds, "radius_km", earth.RadiusKm() - DETECTOR_DEPTH_KM);
            AttributeString(ds, "mass_axis", "/axes/mass_eV");
            H5Dclose(ds);
        }
        H5Gclose(pot);

        CheckH5(H5Fflush(file, H5F_SCOPE_GLOBAL), "flush");
        H5Fclose(file);
    } catch (...) {
        H5Fclose(file);
        std::filesystem::remove(filename); // No conservar una LUT parcial.
        throw;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        gsl_set_error_handler_off();
        if (argc > 3) {
            std::cerr << "Uso: " << argv[0] << " [output.h5] [earth_model.dat]\n";
            return 1;
        }

        const std::string output = argc >= 2 ? argv[1]
            : "LRI/output/potential/vlri_unit.h5";
        const std::string earth_model = argc >= 3 ? argv[2] : "";

        std::cout << "LRI unit potential | nuSQuIDS Earth/PREM\n";
        std::cout << "Earth model: " << (earth_model.empty()
            ? "nuSQuIDS default" : earth_model) << '\n';
        EarthSampler earth(earth_model);
        const auto masses = MassGrid();
        const auto radii = RadiusGrid(earth.RadiusKm());
        std::cout << "Masses: " << masses.size() << " ["
                  << masses.front() << ", " << masses.back() << "] eV\n";
        std::cout << "Radii: " << radii.size() << " | R_E = "
                  << earth.RadiusKm() << " km\n";
        std::cout << "Sampling PREM...\n";
        const EarthProfile profile = SampleEarthProfile(earth, radii);
        const QuadratureCache cache = BuildSourceCache(earth, radii);
        std::cout << "Calculating electron and neutron potential profiles...\n";
        const auto values = BuildPotentials(masses, radii, cache,
                                            earth.KmToEVInverse());
        const double error_e = ValidatePotential(
            earth, Source::Electron, masses, radii, values.electron);
        const double error_n = ValidatePotential(
            earth, Source::Neutron, masses, radii, values.neutron);
        WriteLUT(output, earth_model, earth, masses, radii, profile,
                 values, error_e, error_n);
        std::cout << "LUT: " << std::filesystem::absolute(output)
                  << "\nElectron validation relative error: " << std::scientific
                  << error_e << "\nNeutron validation relative error: "
                  << error_n << "\nBoth source integrals have NO coupling prefactor.\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << '\n';
        return 1;
    }
}
