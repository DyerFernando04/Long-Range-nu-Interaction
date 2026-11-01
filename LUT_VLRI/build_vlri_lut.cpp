#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include <nuSQuIDS/nuSQuIDS.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace nusquids;

namespace {

// ============================================================================
// Configuración general
// ============================================================================

constexpr double RADIAL_STEP_KM = 0.25;

// m' = 10^-15 ... 10^-11 eV
// constexpr double LOG10_MASS_MIN = -15.0;
constexpr double LOG10_MASS_MIN = -16.0;
constexpr double LOG10_MASS_MAX = -11.0;
constexpr std::size_t N_MASS = 201;

// Cuadratura Gauss-Legendre usada dentro de cada intervalo radial.
// No estamos aproximando PREM mediante polinomios propios.
// rho(r) y Ye(r) se preguntan directamente a nuSQuIDS.
constexpr std::size_t GL_ORDER = 16;

// Profundidad de IceCube que veníamos usando.
constexpr double DETECTOR_DEPTH_KM = 1.5;

// Número de Avogadro.
// Si rho está en g/cm^3 y Ye = Ne/Nnucleons:
//
// n_e [cm^-3] = rho * Ye * N_A
//
constexpr long double N_AVOGADRO =
    6.02214076e23L;

// Validación independiente.
constexpr double VALIDATION_REL_TOL = 1.0e-6;
constexpr double GSL_REL_TOL = 1.0e-9;
constexpr std::size_t GSL_LIMIT = 5000;


// ============================================================================
// Utilidades
// ============================================================================

void CheckHDF5Status(
    herr_t status,
    const std::string& message)
{
    if (status < 0) {
        throw std::runtime_error(message);
    }
}


void CheckHDF5Id(
    hid_t id,
    const std::string& message)
{
    if (id < 0) {
        throw std::runtime_error(message);
    }
}


bool AlmostEqual(
    double a,
    double b,
    double rtol = 1.0e-12,
    double atol = 1.0e-12)
{
    return std::abs(a - b)
        <= atol
        + rtol * std::max(std::abs(a), std::abs(b));
}


// ============================================================================
// Atributos HDF5
// ============================================================================

void WriteStringAttribute(
    hid_t object_id,
    const std::string& name,
    const std::string& value)
{
    hid_t datatype_id =
        H5Tcopy(H5T_C_S1);

    CheckHDF5Id(
        datatype_id,
        "No pude crear datatype para atributo string.");

    CheckHDF5Status(
        H5Tset_size(
            datatype_id,
            value.size() + 1),
        "No pude fijar tamaño del atributo string.");

    CheckHDF5Status(
        H5Tset_strpad(
            datatype_id,
            H5T_STR_NULLTERM),
        "No pude fijar padding del atributo string.");

    hid_t dataspace_id =
        H5Screate(H5S_SCALAR);

    CheckHDF5Id(
        dataspace_id,
        "No pude crear dataspace de atributo string.");

    hid_t attribute_id =
        H5Acreate2(
            object_id,
            name.c_str(),
            datatype_id,
            dataspace_id,
            H5P_DEFAULT,
            H5P_DEFAULT);

    CheckHDF5Id(
        attribute_id,
        "No pude crear atributo: " + name);

    CheckHDF5Status(
        H5Awrite(
            attribute_id,
            datatype_id,
            value.c_str()),
        "No pude escribir atributo: " + name);

    H5Aclose(attribute_id);
    H5Sclose(dataspace_id);
    H5Tclose(datatype_id);
}


void WriteDoubleAttribute(
    hid_t object_id,
    const std::string& name,
    double value)
{
    hid_t dataspace_id =
        H5Screate(H5S_SCALAR);

    CheckHDF5Id(
        dataspace_id,
        "No pude crear dataspace de atributo double.");

    hid_t attribute_id =
        H5Acreate2(
            object_id,
            name.c_str(),
            H5T_NATIVE_DOUBLE,
            dataspace_id,
            H5P_DEFAULT,
            H5P_DEFAULT);

    CheckHDF5Id(
        attribute_id,
        "No pude crear atributo: " + name);

    CheckHDF5Status(
        H5Awrite(
            attribute_id,
            H5T_NATIVE_DOUBLE,
            &value),
        "No pude escribir atributo: " + name);

    H5Aclose(attribute_id);
    H5Sclose(dataspace_id);
}


void WriteUInt64Attribute(
    hid_t object_id,
    const std::string& name,
    std::uint64_t value)
{
    hid_t dataspace_id =
        H5Screate(H5S_SCALAR);

    CheckHDF5Id(
        dataspace_id,
        "No pude crear dataspace de atributo uint64.");

    hid_t attribute_id =
        H5Acreate2(
            object_id,
            name.c_str(),
            H5T_NATIVE_ULLONG,
            dataspace_id,
            H5P_DEFAULT,
            H5P_DEFAULT);

    CheckHDF5Id(
        attribute_id,
        "No pude crear atributo: " + name);

    unsigned long long stored =
        static_cast<unsigned long long>(value);

    CheckHDF5Status(
        H5Awrite(
            attribute_id,
            H5T_NATIVE_ULLONG,
            &stored),
        "No pude escribir atributo: " + name);

    H5Aclose(attribute_id);
    H5Sclose(dataspace_id);
}


// ============================================================================
// Datasets HDF5
// ============================================================================

void Write1DDataset(
    hid_t group_id,
    const std::string& name,
    const std::vector<double>& data,
    const std::string& unit = "")
{
    const hsize_t dimensions[1] = {
        static_cast<hsize_t>(data.size())
    };

    CheckHDF5Status(
        H5LTmake_dataset_double(
            group_id,
            name.c_str(),
            1,
            dimensions,
            data.data()),
        "No pude escribir dataset: " + name);

    if (!unit.empty()) {
        hid_t dataset_id =
            H5Dopen2(
                group_id,
                name.c_str(),
                H5P_DEFAULT);

        CheckHDF5Id(
            dataset_id,
            "No pude reabrir dataset: " + name);

        WriteStringAttribute(
            dataset_id,
            "unit",
            unit);

        H5Dclose(dataset_id);
    }
}


void Write2DDataset(
    hid_t group_id,
    const std::string& name,
    const std::vector<double>& data,
    std::size_t n0,
    std::size_t n1,
    const std::string& unit = "")
{
    if (data.size() != n0 * n1) {
        throw std::runtime_error(
            "Write2DDataset: tamaño inconsistente.");
    }

    const hsize_t dimensions[2] = {
        static_cast<hsize_t>(n0),
        static_cast<hsize_t>(n1)
    };

    CheckHDF5Status(
        H5LTmake_dataset_double(
            group_id,
            name.c_str(),
            2,
            dimensions,
            data.data()),
        "No pude escribir dataset 2D: " + name);

    hid_t dataset_id =
        H5Dopen2(
            group_id,
            name.c_str(),
            H5P_DEFAULT);

    CheckHDF5Id(
        dataset_id,
        "No pude reabrir dataset 2D: " + name);

    if (!unit.empty()) {
        WriteStringAttribute(
            dataset_id,
            "unit",
            unit);
    }

    WriteStringAttribute(
        dataset_id,
        "dimension_order",
        "mass, radius");

    H5Dclose(dataset_id);
}


// ============================================================================
// Fuente terrestre tomada DIRECTAMENTE de nuSQuIDS
// ============================================================================

struct EarthSample {
    double rho_g_cm3;
    double ye;
    double ne_cm3;
    double ne_eV3;
};


class NuSQuIDSEarthSampler {
private:
    squids::Const units_;

    std::shared_ptr<Earth> earth_;

    std::unique_ptr<Earth::Track>
        radial_track_;

    double earth_radius_km_;

public:
    explicit NuSQuIDSEarthSampler(
        const std::string& earth_model_filename = "")
    {
        if (earth_model_filename.empty()) {
            // Usa directamente el modelo terrestre por defecto
            // distribuido con nuSQuIDS.
            earth_ =
                std::make_shared<Earth>();
        } else {
            earth_ =
                std::make_shared<Earth>(
                    earth_model_filename);
        }

        earth_radius_km_ =
            earth_->GetRadius();

        if (
            !std::isfinite(earth_radius_km_)
            || earth_radius_km_ < 6000.0
            || earth_radius_km_ > 7000.0
        ) {
            throw std::runtime_error(
                "Earth::GetRadius() devolvió un valor inesperado.");
        }

        // ------------------------------------------------------------
        // Truco geométrico:
        //
        // Earth::density(track) calcula internamente:
        //
        // r^2 = R_E^2 + x^2 - L x.
        //
        // Elegimos una trayectoria diametral:
        //
        // L = 2 R_E.
        //
        // Entonces:
        //
        // r = |R_E - x|.
        //
        // Para consultar un radio r entre centro y superficie:
        //
        // x = R_E - r.
        //
        // Así NO implementamos el PREM nosotros.
        // La transformación geométrica solo sirve para preguntarle
        // a Earth qué rho y Ye tiene a ese radio.
        // ------------------------------------------------------------

        const double baseline =
            2.0
            * earth_radius_km_
            * units_.km;

        radial_track_ =
            std::make_unique<Earth::Track>(
                baseline);
    }

    double GetEarthRadiusKm() const
    {
        return earth_radius_km_;
    }

    double GetKmToEVInv() const
    {
        return units_.km;
    }

    double GetCmToEVInv() const
    {
        return units_.cm;
    }

    EarthSample Sample(double radius_km)
    {
        if (
            !std::isfinite(radius_km)
            || radius_km < 0.0
            || radius_km > earth_radius_km_
        ) {
            throw std::runtime_error(
                "Se intentó evaluar la Tierra en un radio inválido.");
        }

        const double x =
            (earth_radius_km_ - radius_km)
            * units_.km;

        radial_track_->SetX(x);

        const double rho =
            earth_->density(
                *radial_track_);

        const double ye =
            earth_->ye(
                *radial_track_);

        if (
            !std::isfinite(rho)
            || !std::isfinite(ye)
            || rho < 0.0
            || ye < 0.0
            || ye > 1.0
        ) {
            throw std::runtime_error(
                "nuSQuIDS devolvió rho o Ye inválido.");
        }

        // ------------------------------------------------------------
        // Ye = número de electrones por nucleón.
        //
        // n_e [cm^-3]
        //     = rho [g/cm^3] * N_A * Ye
        //
        // ------------------------------------------------------------

        const long double ne_cm3 =
            static_cast<long double>(rho)
            * static_cast<long double>(ye)
            * N_AVOGADRO;

        // 1 cm = units_.cm eV^-1.
        //
        // Por tanto:
        //
        // 1 cm^-3 = 1 / units_.cm^3  eV^3.
        //
        const long double cm_to_evinv =
            static_cast<long double>(
                units_.cm);

        const long double ne_eV3 =
            ne_cm3
            / (
                cm_to_evinv
                * cm_to_evinv
                * cm_to_evinv
            );

        EarthSample result;

        result.rho_g_cm3 = rho;
        result.ye = ye;
        result.ne_cm3 =
            static_cast<double>(ne_cm3);
        result.ne_eV3 =
            static_cast<double>(ne_eV3);

        return result;
    }
};


// ============================================================================
// Mallas
// ============================================================================

std::vector<double> BuildMassGrid()
{
    std::vector<double> masses(
        N_MASS);

    const double step =
        (LOG10_MASS_MAX - LOG10_MASS_MIN)
        / static_cast<double>(N_MASS - 1);

    for (std::size_t i = 0;
         i < N_MASS;
         ++i) {

        const double log10_mass =
            LOG10_MASS_MIN
            + step * static_cast<double>(i);

        masses[i] =
            std::pow(
                10.0,
                log10_mass);
    }

    return masses;
}


std::vector<double> BuildLogMassGrid(
    const std::vector<double>& mass_eV)
{
    std::vector<double> result(
        mass_eV.size());

    for (std::size_t i = 0;
         i < mass_eV.size();
         ++i) {

        result[i] =
            std::log10(
                mass_eV[i]);
    }

    return result;
}


std::vector<double> BuildRadiusGrid(
    double earth_radius_km)
{
    const long long number_intervals =
        std::llround(
            earth_radius_km
            / RADIAL_STEP_KM);

    if (
        !AlmostEqual(
            number_intervals
                * RADIAL_STEP_KM,
            earth_radius_km,
            0.0,
            1.0e-10)
    ) {
        throw std::runtime_error(
            "El radio terrestre no es divisible exactamente "
            "por RADIAL_STEP_KM.");
    }

    std::vector<double> radius(
        static_cast<std::size_t>(
            number_intervals + 1));

    for (long long i = 0;
         i <= number_intervals;
         ++i) {

        radius[
            static_cast<std::size_t>(i)
        ] =
            static_cast<double>(i)
            * RADIAL_STEP_KM;
    }

    radius.back() =
        earth_radius_km;

    return radius;
}


std::size_t FindExactRadiusIndex(
    const std::vector<double>& radius_km,
    double target_km)
{
    const auto it =
        std::lower_bound(
            radius_km.begin(),
            radius_km.end(),
            target_km);

    if (
        it != radius_km.end()
        && AlmostEqual(
            *it,
            target_km,
            0.0,
            1.0e-10)
    ) {
        return static_cast<std::size_t>(
            std::distance(
                radius_km.begin(),
                it));
    }

    throw std::runtime_error(
        "El radio solicitado no aparece exactamente en la malla: "
        + std::to_string(target_km)
        + " km");
}


// ============================================================================
// Valores de rho, Ye y ne en la malla que se guardará
// ============================================================================

struct EarthProfile {
    std::vector<double> rho_g_cm3;
    std::vector<double> ye;
    std::vector<double> ne_cm3;
    std::vector<double> ne_eV3;
};


EarthProfile BuildStoredEarthProfile(
    NuSQuIDSEarthSampler& sampler,
    const std::vector<double>& radius_km)
{
    EarthProfile profile;

    const std::size_t n =
        radius_km.size();

    profile.rho_g_cm3.resize(n);
    profile.ye.resize(n);
    profile.ne_cm3.resize(n);
    profile.ne_eV3.resize(n);

    for (std::size_t i = 0;
         i < n;
         ++i) {

        const EarthSample sample =
            sampler.Sample(
                radius_km[i]);

        profile.rho_g_cm3[i] =
            sample.rho_g_cm3;

        profile.ye[i] =
            sample.ye;

        profile.ne_cm3[i] =
            sample.ne_cm3;

        profile.ne_eV3[i] =
            sample.ne_eV3;
    }

    return profile;
}


// ============================================================================
// Cuadratura de la FUENTE n_e(r)
// ============================================================================
//
// En lugar de aproximar PREM con nuestros propios polinomios,
// preguntamos a nuSQuIDS rho(r), Ye(r) en nodos de Gauss-Legendre.
//
// Para cada nodo guardamos:
//
//    r_q
//
// y:
//
//    W_q = dr_q * r_q * n_e(r_q)
//
// cuyas unidades son eV.
//
// Después, para una masa concreta, basta multiplicar W_q
// por las funciones Yukawa.
// ============================================================================

struct QuadratureCache {
    std::size_t n_segments;
    std::size_t order;

    std::vector<long double>
        radius_eV_inv;

    std::vector<long double>
        weighted_source_eV;
};


QuadratureCache BuildQuadratureCache(
    NuSQuIDSEarthSampler& sampler,
    const std::vector<double>& radius_km)
{
    if (radius_km.size() < 2) {
        throw std::runtime_error(
            "La malla radial necesita al menos dos puntos.");
    }

    const std::size_t n_segments =
        radius_km.size() - 1;

    const std::size_t total_points =
        n_segments * GL_ORDER;

    QuadratureCache cache;

    cache.n_segments =
        n_segments;

    cache.order =
        GL_ORDER;

    cache.radius_eV_inv.resize(
        total_points);

    cache.weighted_source_eV.resize(
        total_points);

    gsl_integration_glfixed_table* table =
        gsl_integration_glfixed_table_alloc(
            GL_ORDER);

    if (table == nullptr) {
        throw std::runtime_error(
            "No pude crear tabla Gauss-Legendre.");
    }

    const long double km_to_evinv =
        static_cast<long double>(
            sampler.GetKmToEVInv());

    std::cout
        << "Construyendo fuente terrestre desde nuSQuIDS...\n";

    for (std::size_t segment = 0;
         segment < n_segments;
         ++segment) {

        const double a_km =
            radius_km[segment];

        const double b_km =
            radius_km[segment + 1];

        for (std::size_t k = 0;
             k < GL_ORDER;
             ++k) {

            double rq_km = 0.0;
            double weight_km = 0.0;

            const int status =
                gsl_integration_glfixed_point(
                    a_km,
                    b_km,
                    k,
                    &rq_km,
                    &weight_km,
                    table);

            if (status != GSL_SUCCESS) {
                gsl_integration_glfixed_table_free(
                    table);

                throw std::runtime_error(
                    "Falló gsl_integration_glfixed_point.");
            }

            const EarthSample sample =
                sampler.Sample(
                    rq_km);

            const long double rq_evinv =
                static_cast<long double>(
                    rq_km)
                * km_to_evinv;

            const long double dr_weight_evinv =
                static_cast<long double>(
                    weight_km)
                * km_to_evinv;

            const long double ne_eV3 =
                static_cast<long double>(
                    sample.ne_eV3);

            const std::size_t index =
                segment * GL_ORDER + k;

            cache.radius_eV_inv[index] =
                rq_evinv;

            // dr * r * ne
            cache.weighted_source_eV[index] =
                dr_weight_evinv
                * rq_evinv
                * ne_eV3;
        }

        if (
            (segment + 1) % 5000 == 0
            || segment + 1 == n_segments
        ) {
            std::cout
                << "  segmentos: "
                << (segment + 1)
                << "/"
                << n_segments
                << '\n';
        }
    }

    gsl_integration_glfixed_table_free(
        table);

    return cache;
}


// ============================================================================
// Construcción del potencial
// ============================================================================
//
// Partimos de:
//
// V_unit(R,m)
//
// = 1/(2 m R)
//   int_0^Re dr r n_e(r)
//   [
//       exp(-m |R-r|)
//       - exp(-m(R+r))
//   ].
//
// Esta expresión ya incluye el 1/(4 pi) del potencial original:
//
// V_LRI = g'^2/(4 pi)
//         int d^3r n_e(r)
//         exp(-m s)/s.
//
// Para evitar cancelaciones numéricas, separamos:
//
// para r < R:
//
// e^{-m|R-r|} - e^{-m(R+r)}
// = 2 e^{-mR} sinh(mr)
//
// y para r > R:
//
// = 2 e^{-mr} sinh(mR).
//
// Definimos:
//
// A(R) = int_0^R dr r ne(r) sinh(mr)
//
// B(R) = int_R^Re dr r ne(r) exp(-mr)
//
// Entonces:
//
// V_unit(R,m)
//
// = exp(-mR)/(mR) A(R)
//   + sinh(mR)/(mR) B(R).
//
// En R=0:
//
// V_unit(0,m)
//
// = int_0^Re dr r ne(r) exp(-mr)
// = B(0).
//
// ============================================================================

std::vector<double> BuildPotentialLibrary(
    const std::vector<double>& mass_eV,
    const std::vector<double>& radius_km,
    const QuadratureCache& cache,
    double km_to_evinv)
{
    const std::size_t n_mass =
        mass_eV.size();

    const std::size_t n_radius =
        radius_km.size();

    const std::size_t n_segments =
        n_radius - 1;

    std::vector<long double>
        radius_eV_inv(
            n_radius);

    for (std::size_t i = 0;
         i < n_radius;
         ++i) {

        radius_eV_inv[i] =
            static_cast<long double>(
                radius_km[i])
            * static_cast<long double>(
                km_to_evinv);
    }

    std::vector<double> V_unit_eV(
        n_mass * n_radius,
        0.0);

    std::cout << '\n';
    std::cout
        << "Calculando V_unit(R,m)...\n";

#ifdef _OPENMP
    std::cout
        << "OpenMP threads: "
        << omp_get_max_threads()
        << '\n';
#else
    std::cout
        << "OpenMP no habilitado.\n";
#endif

#pragma omp parallel for schedule(dynamic)
    for (long long mass_index_signed = 0;
         mass_index_signed
            < static_cast<long long>(n_mass);
         ++mass_index_signed) {

        const std::size_t mass_index =
            static_cast<std::size_t>(
                mass_index_signed);

        const long double mass =
            static_cast<long double>(
                mass_eV[mass_index]);

        std::vector<long double> A(
            n_radius,
            0.0L);

        std::vector<long double> B(
            n_radius,
            0.0L);

        std::vector<long double>
            segment_B(
                n_segments,
                0.0L);

        // ------------------------------------------------------------
        // Forward:
        //
        // A(R) = int_0^R r ne(r) sinh(mr) dr
        //
        // Al mismo tiempo calculamos la contribución de cada
        // segmento a B.
        // ------------------------------------------------------------

        for (std::size_t segment = 0;
             segment < n_segments;
             ++segment) {

            long double contribution_A =
                0.0L;

            long double contribution_B =
                0.0L;

            const std::size_t offset =
                segment * cache.order;

            for (std::size_t k = 0;
                 k < cache.order;
                 ++k) {

                const std::size_t q =
                    offset + k;

                const long double r =
                    cache.radius_eV_inv[q];

                const long double weighted_source =
                    cache.weighted_source_eV[q];

                const long double z =
                    mass * r;

                contribution_A +=
                    weighted_source
                    * std::sinh(z);

                contribution_B +=
                    weighted_source
                    * std::exp(-z);
            }

            A[segment + 1] =
                A[segment]
                + contribution_A;

            segment_B[segment] =
                contribution_B;
        }

        // ------------------------------------------------------------
        // Backward:
        //
        // B(R) = int_R^Re r ne(r) exp(-mr) dr
        // ------------------------------------------------------------

        B[n_radius - 1] =
            0.0L;

        for (std::size_t reverse = n_segments;
             reverse > 0;
             --reverse) {

            const std::size_t segment =
                reverse - 1;

            B[segment] =
                B[segment + 1]
                + segment_B[segment];
        }

        // ------------------------------------------------------------
        // Construir potencial.
        // ------------------------------------------------------------

        for (std::size_t radial_index = 0;
             radial_index < n_radius;
             ++radial_index) {

            long double value = 0.0L;

            if (radial_index == 0) {

                // Límite R -> 0.
                value =
                    B[0];

            } else {

                const long double R =
                    radius_eV_inv[
                        radial_index];

                const long double z =
                    mass * R;

                if (!(z > 0.0L)) {
                    throw std::runtime_error(
                        "mR no positivo inesperadamente.");
                }

                const long double term_inner =
                    std::exp(-z)
                    * A[radial_index]
                    / z;

                const long double term_outer =
                    std::sinh(z)
                    * B[radial_index]
                    / z;

                value =
                    term_inner
                    + term_outer;
            }

            if (
                !std::isfinite(value)
                || value <= 0.0L
            ) {
                throw std::runtime_error(
                    "Se produjo V_unit no finito o no positivo.");
            }

            V_unit_eV[
                mass_index * n_radius
                + radial_index
            ] =
                static_cast<double>(
                    value);
        }

#pragma omp critical
        {
            std::cout
                << "  masa "
                << (mass_index + 1)
                << "/"
                << n_mass
                << " : "
                << std::scientific
                << std::setprecision(4)
                << mass_eV[mass_index]
                << " eV\n";
        }
    }

    return V_unit_eV;
}


// ============================================================================
// Validación independiente mediante GSL adaptativo
// ============================================================================

struct DirectIntegrationContext {
    NuSQuIDSEarthSampler* sampler;
    long double mass_eV;
    long double evaluation_radius_eV_inv;
};


double DirectPotentialIntegrandKm(
    double source_radius_km,
    void* parameters)
{
    auto* context =
        static_cast<
            DirectIntegrationContext*
        >(parameters);

    const EarthSample sample =
        context->sampler->Sample(
            source_radius_km);

    const long double km_to_evinv =
        static_cast<long double>(
            context->sampler
                ->GetKmToEVInv());

    const long double r =
        static_cast<long double>(
            source_radius_km)
        * km_to_evinv;

    const long double ne =
        static_cast<long double>(
            sample.ne_eV3);

    const long double m =
        context->mass_eV;

    const long double R =
        context->evaluation_radius_eV_inv;

    long double angular_factor = 0.0L;

    if (R == 0.0L) {

        angular_factor =
            std::exp(-m * r);

    } else {

        const long double zR =
            m * R;

        if (r <= R) {

            angular_factor =
                std::exp(-zR)
                * std::sinh(m * r)
                / zR;

        } else {

            angular_factor =
                std::exp(-m * r)
                * std::sinh(zR)
                / zR;
        }
    }

    // Integrando natural:
    //
    // r * ne * angular_factor
    //
    // integraremos en km, así que necesitamos:
    //
    // dr_eV^-1 = dr_km * km_to_eVinv.
    //
    const long double result =
        r
        * ne
        * angular_factor
        * km_to_evinv;

    return static_cast<double>(
        result);
}


double DirectAdaptivePotential(
    NuSQuIDSEarthSampler& sampler,
    double mass_eV,
    double evaluation_radius_km)
{
    gsl_integration_workspace* workspace =
        gsl_integration_workspace_alloc(
            GSL_LIMIT);

    if (workspace == nullptr) {
        throw std::runtime_error(
            "No pude crear workspace GSL.");
    }

    DirectIntegrationContext context;

    context.sampler =
        &sampler;

    context.mass_eV =
        static_cast<long double>(
            mass_eV);

    context.evaluation_radius_eV_inv =
        static_cast<long double>(
            evaluation_radius_km)
        * static_cast<long double>(
            sampler.GetKmToEVInv());

    gsl_function function;

    function.function =
        &DirectPotentialIntegrandKm;

    function.params =
        &context;

    const double earth_radius_km =
        sampler.GetEarthRadiusKm();

    double total_result = 0.0;

    auto integrate_piece =
        [&](double a, double b) {

            if (!(b > a)) {
                return;
            }

            double result = 0.0;
            double error = 0.0;

            const int status =
                gsl_integration_qag(
                    &function,
                    a,
                    b,
                    0.0,
                    GSL_REL_TOL,
                    GSL_LIMIT,
                    GSL_INTEG_GAUSS61,
                    workspace,
                    &result,
                    &error);

            if (status != GSL_SUCCESS) {
                gsl_integration_workspace_free(
                    workspace);

                throw std::runtime_error(
                    "Falló validación adaptativa GSL: "
                    + std::string(
                        gsl_strerror(status)));
            }

            total_result += result;
        };

    // Dividimos explícitamente en R porque allí cambia la
    // expresión analítica del kernel.
    if (
        evaluation_radius_km > 0.0
        && evaluation_radius_km < earth_radius_km
    ) {

        integrate_piece(
            0.0,
            evaluation_radius_km);

        integrate_piece(
            evaluation_radius_km,
            earth_radius_km);

    } else {

        integrate_piece(
            0.0,
            earth_radius_km);
    }

    gsl_integration_workspace_free(
        workspace);

    return total_result;
}


double ValidatePotentialLibrary(
    NuSQuIDSEarthSampler& sampler,
    const std::vector<double>& mass_eV,
    const std::vector<double>& radius_km,
    const std::vector<double>& V_unit_eV)
{
    const std::vector<std::size_t>
        mass_indices = {
            0,
            N_MASS / 2,
            N_MASS - 1
        };

    const double earth_radius_km =
        sampler.GetEarthRadiusKm();

    const double detector_radius_km =
        earth_radius_km
        - DETECTOR_DEPTH_KM;

    const std::vector<double>
        validation_radii_km = {
            0.0,
            3480.0,
            detector_radius_km,
            earth_radius_km
        };

    double maximum_relative_error =
        0.0;

    std::cout << '\n';
    std::cout
        << "Validación independiente GSL:\n";

    for (const std::size_t mass_index
         : mass_indices) {

        for (const double R_km
             : validation_radii_km) {

            const std::size_t radial_index =
                FindExactRadiusIndex(
                    radius_km,
                    R_km);

            const double table_value =
                V_unit_eV[
                    mass_index
                    * radius_km.size()
                    + radial_index
                ];

            const double direct_value =
                DirectAdaptivePotential(
                    sampler,
                    mass_eV[mass_index],
                    R_km);

            const double relative_error =
                std::abs(
                    table_value
                    - direct_value)
                /
                std::max(
                    std::abs(
                        direct_value),
                    std::numeric_limits<double>
                        ::min());

            maximum_relative_error =
                std::max(
                    maximum_relative_error,
                    relative_error);

            std::cout
                << "  m = "
                << std::scientific
                << std::setprecision(4)
                << mass_eV[mass_index]
                << " eV"
                << "   R = "
                << std::fixed
                << std::setprecision(1)
                << R_km
                << " km"
                << "   rel.err = "
                << std::scientific
                << std::setprecision(3)
                << relative_error
                << '\n';
        }
    }

    std::cout
        << "Maximum relative error = "
        << std::scientific
        << std::setprecision(6)
        << maximum_relative_error
        << '\n';

    if (
        maximum_relative_error
        > VALIDATION_REL_TOL
    ) {
        throw std::runtime_error(
            "La LUT no superó la validación. "
            "Error relativo máximo = "
            + std::to_string(
                maximum_relative_error));
    }

    return maximum_relative_error;
}


// ============================================================================
// Guardar HDF5
// ============================================================================

void WriteLibrary(
    const std::string& output_filename,
    const std::string& earth_model_filename,
    NuSQuIDSEarthSampler& sampler,
    const std::vector<double>& mass_eV,
    const std::vector<double>& log10_mass,
    const std::vector<double>& radius_km,
    const EarthProfile& earth_profile,
    const std::vector<double>& V_unit_eV,
    double maximum_validation_relative_error)
{
    const std::size_t n_mass =
        mass_eV.size();

    const std::size_t n_radius =
        radius_km.size();

    const double earth_radius_km =
        sampler.GetEarthRadiusKm();

    const double detector_radius_km =
        earth_radius_km
        - DETECTOR_DEPTH_KM;

    const std::size_t detector_index =
        FindExactRadiusIndex(
            radius_km,
            detector_radius_km);

    std::vector<double>
        V_unit_at_detector_eV(
            n_mass);

    for (std::size_t i = 0;
         i < n_mass;
         ++i) {

        V_unit_at_detector_eV[i] =
            V_unit_eV[
                i * n_radius
                + detector_index
            ];
    }

    hid_t file_id =
        H5Fcreate(
            output_filename.c_str(),
            H5F_ACC_TRUNC,
            H5P_DEFAULT,
            H5P_DEFAULT);

    CheckHDF5Id(
        file_id,
        "No pude crear HDF5: "
        + output_filename);

    try {

        // ------------------------------------------------------------
        // Metadatos generales
        // ------------------------------------------------------------

        WriteStringAttribute(
            file_id,
            "format_name",
            "nuSQuIDS-based terrestrial LRI unit-potential library");

        WriteStringAttribute(
            file_id,
            "format_version",
            "1.0");

        WriteStringAttribute(
            file_id,
            "potential_definition",
            "V_LRI(R,m,gprime) = gprime^2 * V_unit(R,m)");

        WriteStringAttribute(
            file_id,
            "unit_potential_definition",
            "V_unit = (1/4pi) integral d^3r ne(r) exp(-m|R-r|)/|R-r|");

        WriteStringAttribute(
            file_id,
            "earth_source",
            earth_model_filename.empty()
                ? "nuSQuIDS default Earth()"
                : earth_model_filename);

        WriteStringAttribute(
            file_id,
            "earth_sampling",
            "rho(r) and Ye(r) queried directly through "
            "nusquids::Earth::density() and nusquids::Earth::ye()");

        WriteStringAttribute(
            file_id,
            "earth_interpolation_semantics",
            "Uses exactly the radial interpolation implemented "
            "inside the selected nuSQuIDS Earth model");

        WriteStringAttribute(
            file_id,
            "integration_method",
            "16-point Gauss-Legendre per 0.25 km radial segment "
            "plus stable cumulative spherical Yukawa reduction");

        WriteStringAttribute(
            file_id,
            "coupling_convention",
            "Multiply V_unit by gprime^2");

        WriteDoubleAttribute(
            file_id,
            "earth_radius_km",
            earth_radius_km);

        WriteDoubleAttribute(
            file_id,
            "detector_depth_km",
            DETECTOR_DEPTH_KM);

        WriteDoubleAttribute(
            file_id,
            "detector_radius_km",
            detector_radius_km);

        WriteDoubleAttribute(
            file_id,
            "radial_step_km",
            RADIAL_STEP_KM);

        WriteDoubleAttribute(
            file_id,
            "km_to_eV_inverse",
            sampler.GetKmToEVInv());

        WriteDoubleAttribute(
            file_id,
            "cm_to_eV_inverse",
            sampler.GetCmToEVInv());

        WriteDoubleAttribute(
            file_id,
            "avogadro_constant",
            static_cast<double>(
                N_AVOGADRO));

        WriteDoubleAttribute(
            file_id,
            "maximum_validation_relative_error",
            maximum_validation_relative_error);

        WriteUInt64Attribute(
            file_id,
            "number_mass_points",
            static_cast<std::uint64_t>(
                n_mass));

        WriteUInt64Attribute(
            file_id,
            "number_radius_points",
            static_cast<std::uint64_t>(
                n_radius));

        WriteUInt64Attribute(
            file_id,
            "gauss_legendre_order",
            static_cast<std::uint64_t>(
                GL_ORDER));

        WriteUInt64Attribute(
            file_id,
            "detector_radius_index",
            static_cast<std::uint64_t>(
                detector_index));

        // ------------------------------------------------------------
        // /axes
        // ------------------------------------------------------------

        hid_t axes_group =
            H5Gcreate2(
                file_id,
                "/axes",
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT);

        CheckHDF5Id(
            axes_group,
            "No pude crear /axes.");

        Write1DDataset(
            axes_group,
            "mass_eV",
            mass_eV,
            "eV");

        Write1DDataset(
            axes_group,
            "log10_mass_eV",
            log10_mass,
            "log10(eV)");

        Write1DDataset(
            axes_group,
            "radius_km",
            radius_km,
            "km");

        H5Gclose(
            axes_group);

        // ------------------------------------------------------------
        // /earth
        // ------------------------------------------------------------

        hid_t earth_group =
            H5Gcreate2(
                file_id,
                "/earth",
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT);

        CheckHDF5Id(
            earth_group,
            "No pude crear /earth.");

        Write1DDataset(
            earth_group,
            "rho_g_cm3",
            earth_profile.rho_g_cm3,
            "g/cm^3");

        Write1DDataset(
            earth_group,
            "Ye",
            earth_profile.ye,
            "dimensionless");

        Write1DDataset(
            earth_group,
            "ne_cm3",
            earth_profile.ne_cm3,
            "cm^-3");

        Write1DDataset(
            earth_group,
            "ne_eV3",
            earth_profile.ne_eV3,
            "eV^3");

        H5Gclose(
            earth_group);

        // ------------------------------------------------------------
        // /potential
        // ------------------------------------------------------------

        hid_t potential_group =
            H5Gcreate2(
                file_id,
                "/potential",
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT);

        CheckHDF5Id(
            potential_group,
            "No pude crear /potential.");

        Write2DDataset(
            potential_group,
            "V_unit_eV",
            V_unit_eV,
            n_mass,
            n_radius,
            "eV");

        Write1DDataset(
            potential_group,
            "V_unit_at_detector_eV",
            V_unit_at_detector_eV,
            "eV");

        hid_t detector_dataset =
            H5Dopen2(
                potential_group,
                "V_unit_at_detector_eV",
                H5P_DEFAULT);

        CheckHDF5Id(
            detector_dataset,
            "No pude abrir V_unit_at_detector_eV.");

        WriteDoubleAttribute(
            detector_dataset,
            "radius_km",
            detector_radius_km);

        WriteStringAttribute(
            detector_dataset,
            "mass_axis",
            "/axes/mass_eV");

        H5Dclose(
            detector_dataset);

        H5Gclose(
            potential_group);

        CheckHDF5Status(
            H5Fflush(
                file_id,
                H5F_SCOPE_GLOBAL),
            "No pude hacer flush del HDF5.");

        H5Fclose(
            file_id);

    } catch (...) {

        H5Fclose(
            file_id);

        throw;
    }
}


// ============================================================================
// MAIN
// ============================================================================

}  // namespace


int main(
    int argc,
    char* argv[])
{
    try {

        gsl_set_error_handler_off();

        // ------------------------------------------------------------
        // Uso:
        //
        // ./build_vlri_lut
        //
        // -> vlri_unit.h5
        //
        // o:
        //
        // ./build_vlri_lut output.h5
        //
        // o:
        //
        // ./build_vlri_lut output.h5 EARTH_MODEL.dat
        //
        // Si no se proporciona EARTH_MODEL.dat, se usa
        // nusquids::Earth() directamente.
        // ------------------------------------------------------------

        if (argc > 3) {
            std::cerr
                << "Uso:\n"
                << "  "
                << argv[0]
                << " [output.h5] [earth_model.dat]\n";

            return 1;
        }

        const std::string output_filename =
            (argc >= 2)
                ? argv[1]
                : "vlri_unit.h5";

        const std::string earth_model_filename =
            (argc >= 3)
                ? argv[2]
                : "";

        std::cout
            << "============================================================\n"
            << "   nuSQuIDS-based terrestrial LRI LUT generator\n"
            << "============================================================\n\n";

        if (earth_model_filename.empty()) {

            std::cout
                << "Earth model : nuSQuIDS default Earth()\n";

        } else {

            std::cout
                << "Earth model : "
                << earth_model_filename
                << '\n';
        }

        // ------------------------------------------------------------
        // 1. Crear Tierra de nuSQuIDS.
        // ------------------------------------------------------------

        NuSQuIDSEarthSampler sampler(
            earth_model_filename);

        const double earth_radius_km =
            sampler.GetEarthRadiusKm();

        const double detector_radius_km =
            earth_radius_km
            - DETECTOR_DEPTH_KM;

        std::cout
            << std::fixed
            << std::setprecision(6)
            << "Earth radius : "
            << earth_radius_km
            << " km\n"
            << "Detector R   : "
            << detector_radius_km
            << " km\n"
            << "Radial step  : "
            << RADIAL_STEP_KM
            << " km\n\n";

        // ------------------------------------------------------------
        // 2. Mallas.
        // ------------------------------------------------------------

        const std::vector<double> mass_eV =
            BuildMassGrid();

        const std::vector<double> log10_mass =
            BuildLogMassGrid(
                mass_eV);

        const std::vector<double> radius_km =
            BuildRadiusGrid(
                earth_radius_km);

        std::cout
            << "Mass points   : "
            << mass_eV.size()
            << '\n'
            << "Radius points : "
            << radius_km.size()
            << '\n'
            << "m_min         : "
            << std::scientific
            << mass_eV.front()
            << " eV\n"
            << "m_max         : "
            << mass_eV.back()
            << " eV\n\n";

        // Asegurarnos de que el detector está exactamente en la malla.
        const std::size_t detector_index =
            FindExactRadiusIndex(
                radius_km,
                detector_radius_km);

        std::cout
            << "Detector radial index = "
            << detector_index
            << "\n\n";

        // ------------------------------------------------------------
        // 3. Guardar también rho(r), Ye(r), ne(r).
        // ------------------------------------------------------------

        const EarthProfile earth_profile =
            BuildStoredEarthProfile(
                sampler,
                radius_km);

        // ------------------------------------------------------------
        // Sanity check especialmente importante para este proyecto.
        // ------------------------------------------------------------

        std::cout
            << "Earth-profile sanity check:\n";

        const std::vector<double>
            check_radii = {
                0.0,
                3400.0,
                3480.0,
                3500.0,
                detector_radius_km
            };

        for (const double R
             : check_radii) {

            const EarthSample sample =
                sampler.Sample(R);

            std::cout
                << "  R = "
                << std::fixed
                << std::setprecision(1)
                << R
                << " km"
                << "   rho = "
                << std::setprecision(6)
                << sample.rho_g_cm3
                << " g/cm^3"
                << "   Ye = "
                << std::setprecision(8)
                << sample.ye
                << "   ne = "
                << std::scientific
                << std::setprecision(6)
                << sample.ne_cm3
                << " cm^-3\n";
        }

        std::cout << '\n';

        // ------------------------------------------------------------
        // 4. Precalcular la fuente en nodos Gauss-Legendre.
        // ------------------------------------------------------------

        const QuadratureCache cache =
            BuildQuadratureCache(
                sampler,
                radius_km);

        // ------------------------------------------------------------
        // 5. Calcular V_unit(R,m).
        // ------------------------------------------------------------

        const std::vector<double> V_unit_eV =
            BuildPotentialLibrary(
                mass_eV,
                radius_km,
                cache,
                sampler.GetKmToEVInv());

        // ------------------------------------------------------------
        // 6. Validación completamente independiente.
        // ------------------------------------------------------------

        const double maximum_validation_relative_error =
            ValidatePotentialLibrary(
                sampler,
                mass_eV,
                radius_km,
                V_unit_eV);

        // ------------------------------------------------------------
        // 7. Guardar HDF5.
        // ------------------------------------------------------------

        WriteLibrary(
            output_filename,
            earth_model_filename,
            sampler,
            mass_eV,
            log10_mass,
            radius_km,
            earth_profile,
            V_unit_eV,
            maximum_validation_relative_error);

        const double size_mb =
            static_cast<double>(
                std::filesystem::file_size(
                    output_filename))
            / (1024.0 * 1024.0);

        std::cout
            << '\n'
            << "============================================================\n"
            << "LUT generada correctamente\n"
            << "============================================================\n"
            << "Archivo        : "
            << std::filesystem::absolute(
                output_filename)
            << '\n'
            << "Tamaño         : "
            << std::fixed
            << std::setprecision(3)
            << size_mb
            << " MB\n"
            << "Shape V_unit   : ["
            << mass_eV.size()
            << ", "
            << radius_km.size()
            << "]\n"
            << "Max val. error : "
            << std::scientific
            << std::setprecision(6)
            << maximum_validation_relative_error
            << '\n'
            << "Formula        : "
            << "V_LRI = gprime^2 * V_unit\n"
            << "============================================================\n";

        return 0;

    } catch (const std::exception& error) {

        std::cerr
            << "\n[ERROR] "
            << error.what()
            << '\n';

        return 1;
    }
}
