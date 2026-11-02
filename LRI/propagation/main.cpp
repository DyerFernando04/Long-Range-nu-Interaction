// ============================================================================
// LRI/propagation/main.cpp
// ============================================================================
// Short CLI examples (from the repository root):
//   ./LRI/propagation/generate_flux_lut atmosphere 16 smoke
//   ./LRI/propagation/generate_flux_lut standard 16 smoke
//   ./LRI/propagation/generate_flux_lut Le-Lmu 32
//   ./LRI/propagation/generate_flux_lut Lmu-Ltau 16 smoke
// All existing --mode / --model / --threads / --smoke options still work.
//
// Input: altitude-resolved MCEq production HDF5, and (only for LRI) the
// terrestrial electron/neutron Yukawa potential HDF5.
//
// Output: chunked HDF5, one file for the standard reference and one file
// per LRI model; a 'done' dataset provides resumable computation.
//
// Atmospheric MCEq production is propagated analytically in vacuum ONCE
// per (coszen, energy), retaining all 9 SU(3) density-matrix components for
// neutrinos and antineutrinos. The cache is separate from detector flux files.
// Subsequent propagations begin at Earth entry and do not integrate the
// atmosphere again. Assumption: LRI potential is zero above Earth's surface.
// The genuine exterior Yukawa field would require a model-dependent cache.
// ============================================================================

#include "LRI.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>
#include <gsl/gsl_odeiv2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::size_t NF = 6;
constexpr std::size_t NNU = 3;
constexpr double PI = 3.141592653589793238462643383279502884;

struct Options {
    std::string mode = "lri";                         // atmosphere, standard, lri, both
    lri::Model model = lri::Model::LeLmu;
    std::string mceq = "data/atmospheric/mceq_gsf2019_ddm_height.h5";
    std::string potential = "results/potential/vlri_unit.h5";
    std::string output_standard;
    std::string output_atmosphere;
    std::string output_lri;

    // These grids are entirely independent of the potential-LUT mass grid.
    double log_m_min = -15.0, log_m_max = -11.0;
    std::size_t n_mass = 41;
    double log_c_min = -30.0, log_c_max = -24.0;
    std::size_t n_coupling = 61;

    // For Lmu-Ltau: "bigg" is the DIRECT effective prefactor G_mu_tau.
    // Alternatively "gprime" requires explicit --xi and --chi.
    std::string coupling_kind = "auto";
    bool custom_coupling_range = false;
    double bigg_sign = 1.0;
    double xi = std::numeric_limits<double>::quiet_NaN();
    double chi = std::numeric_limits<double>::quiet_NaN();

    // Standard oscillation parameters (same nominal set as old LUT_FLUX).
    double theta12 = 0.563942, theta13 = 0.154085, theta23 = 0.785398;
    double delta_cp = 0.0;
    double dm21 = 7.65e-5;
    double dm31 = 2.47e-3;  // convention: m3^2 - m1^2

    bool interactions = true;
    bool tau_regen = true;
    bool glashow = false;
    double detector_depth_km = 1.5;
    double h_max_km = 100.0;
    double source_steps_per_layer = 3.0; // upper-bound step <= min(layer width)/3
    double relative_tolerance = 1e-8;
    double absolute_tolerance = 1e-8;
    bool compress = false;
    bool smoke = false;
    bool overwrite = false;
    std::size_t angle_begin = 0;
    std::size_t angle_end = std::numeric_limits<std::size_t>::max();
    std::size_t mass_begin = 0;
    std::size_t mass_end = std::numeric_limits<std::size_t>::max();
    std::size_t coupling_begin = 0;
    std::size_t coupling_end = std::numeric_limits<std::size_t>::max();
    int threads = 0;
    std::size_t batch_size = 16;
    std::size_t flush_every = 4;   // number of completed batches
};

std::string ModelName(lri::Model m) {
    if (m == lri::Model::LeLmu) return "Le-Lmu";
    if (m == lri::Model::LeLtau) return "Le-Ltau";
    return "Lmu-Ltau";
}

[[noreturn]] void Fail(const std::string& what) {
    throw std::runtime_error(what);
}

bool FinitePositive(double x) { return std::isfinite(x) && x > 0.0; }

std::size_t ParseSize(const std::string& s, const std::string& key) {
    std::size_t pos = 0;
    unsigned long long n = 0;
    try { n = std::stoull(s, &pos); }
    catch (...) { Fail("Parametro entero invalido: " + key + "=" + s); }
    if (pos != s.size() || n > std::numeric_limits<std::size_t>::max())
        Fail("Parametro entero invalido: " + key + "=" + s);
    return static_cast<std::size_t>(n);
}

double ParseDouble(const std::string& s, const std::string& key) {
    std::size_t pos = 0;
    double v = 0.0;
    try { v = std::stod(s, &pos); }
    catch (...) { Fail("Parametro real invalido: " + key + "=" + s); }
    if (pos != s.size() || !std::isfinite(v))
        Fail("Parametro real invalido: " + key + "=" + s);
    return v;
}

bool ParseBool(const std::string& s) {
    if (s == "on" || s == "true" || s == "1") return true;
    if (s == "off" || s == "false" || s == "0") return false;
    Fail("Valor booleano invalido: " + s + " (use on/off)");
}

void Help(const char* exe) {
    std::cout
      << "Usage (short): " << exe << " standard|atmosphere [THREADS] [smoke] [options]\n"
      << "               " << exe << " Le-Lmu|Le-Ltau|Lmu-Ltau [THREADS] [smoke] [options]\n"
      << "               " << exe << " lri|both [MODEL] [THREADS] [smoke] [options]\n"
      << "Usage (long):  " << exe << " --mode atmosphere|standard|lri|both [options]\n\n"
      << "  Example: " << exe << " standard 16 smoke\n"
      << "  Example: " << exe << " Le-Lmu 32\n"
      << "  The MCEq and potential HDF5 paths have built-in defaults.\n\n"
      << "  --model Le-Lmu|Le-Ltau|Lmu-Ltau    (LRI only)\n"
      << "  --mceq PATH                        MCEq height-resolved HDF5\n"
      << "  --potential PATH                   V_unit electron/neutron LUT\n"
      << "  --output-standard PATH             standard reference HDF5\n"
      << "  --output-atmosphere PATH           coherent atmosphere cache HDF5\n"
      << "  --output-lri PATH                  selected-model LRI HDF5\n"
      << "  --logm-min X  --logm-max X --nmass N\n"
      << "  --logc-min X  --logc-max X --ncoupling N\n"
      << "  --coupling-kind gprime|bigg        BigG only for Lmu-Ltau\n"
      << "  --bigg-sign +1|-1                  BigG scan sign\n"
      << "  --xi X --chi X                     required for Lmu-Ltau gprime scan\n"
      << "  --theta12 X --theta13 X --theta23 X --delta-cp X [radians]\n"
      << "  --dm21 X --dm31 X                  mass squared differences [eV^2]\n"
      << "  --interactions on|off --tau-regen on|off --glashow on|off\n"
      << "  --detector-depth-km X\n"
      << "  --h-max-km X                       Earth integration maximum step\n"
      << "  --source-steps-per-layer X         legacy option (not used analytically)\n"
      << "  --rtol X --atol X --threads N --batch-size N\n"
      << "  --angle-begin I --angle-end I       half-open [I,J), MCEq grid indices\n"
      << "  --mass-begin I --mass-end I         half-open parameter index range\n"
      << "  --coupling-begin I --coupling-end I half-open parameter index range\n"
      << "  --smoke                             2 masses x 2 couplings x 3 angles\n"
      << "  --compress                          gzip level 1 (slower; less disk)\n"
      << "  --overwrite                         replace compatible output path\n"
      << "  --help\n\n"
      << "Atmosphere is calculated analytically and cached; never repeated per LRI point.\n";
}

bool IsModelName(const std::string& s) {
    return s == "Le-Lmu" || s == "Le-Ltau" || s == "Lmu-Ltau";
}

void SetModelName(Options& o, const std::string& s) {
    if (s == "Le-Lmu") o.model = lri::Model::LeLmu;
    else if (s == "Le-Ltau") o.model = lri::Model::LeLtau;
    else if (s == "Lmu-Ltau") o.model = lri::Model::LmuLtau;
    else Fail("Modelo desconocido: " + s);
}

bool IsUnsignedNumber(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
    });
}

void SetThreadCount(Options& o, const std::string& value) {
    const std::size_t count = ParseSize(value, "threads");
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        Fail("Numero de hilos demasiado grande: " + value);
    o.threads = static_cast<int>(count);
}

Options ParseOptions(int argc, char** argv) {
    Options o;

    // Optional positional shorthand, always at the beginning. Long-form flags
    // can follow it; the previous CLI therefore remains fully compatible.
    // Examples:  standard 16 smoke; Le-Lmu 32; lri Lmu-Ltau 8 --smoke.
    int first_option = 1;
    if (first_option < argc && argv[first_option][0] != '-') {
        const std::string first = argv[first_option++];
        if (first == "standard" || first == "atmosphere" ||
            first == "lri" || first == "both") {
            o.mode = first;
            if ((first == "lri" || first == "both") &&
                first_option < argc && IsModelName(argv[first_option])) {
                SetModelName(o, argv[first_option++]);
            }
        } else if (IsModelName(first)) {
            o.mode = "lri";
            SetModelName(o, first);
        } else {
            Fail("Modo/modelo desconocido: " + first + ". Usa --help.");
        }

        if (first_option < argc && IsUnsignedNumber(argv[first_option]))
            SetThreadCount(o, argv[first_option++]);
        if (first_option < argc && std::string(argv[first_option]) == "smoke") {
            o.smoke = true;
            ++first_option;
        }
        if (first_option < argc && argv[first_option][0] != '-')
            Fail("Argumento posicional inesperado: " +
                 std::string(argv[first_option]) + ". Usa --help.");
    }

    for (int i = first_option; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") { Help(argv[0]); std::exit(0); }
        if (key == "--smoke") { o.smoke = true; continue; }
        if (key == "--overwrite") { o.overwrite = true; continue; }
        if (key == "--compress") { o.compress = true; continue; }
        if (i + 1 >= argc) Fail("Falta valor para " + key);
        const std::string v = argv[++i];
        if (key == "--mode") o.mode = v;
        else if (key == "--model") SetModelName(o, v);
        else if (key == "--mceq") o.mceq = v;
        else if (key == "--potential") o.potential = v;
        else if (key == "--output-standard") o.output_standard = v;
        else if (key == "--output-atmosphere") o.output_atmosphere = v;
        else if (key == "--output-lri") o.output_lri = v;
        else if (key == "--logm-min") o.log_m_min = ParseDouble(v,key);
        else if (key == "--logm-max") o.log_m_max = ParseDouble(v,key);
        else if (key == "--nmass") o.n_mass = ParseSize(v,key);
        else if (key == "--logc-min") { o.log_c_min = ParseDouble(v,key); o.custom_coupling_range=true; }
        else if (key == "--logc-max") { o.log_c_max = ParseDouble(v,key); o.custom_coupling_range=true; }
        else if (key == "--ncoupling") o.n_coupling = ParseSize(v,key);
        else if (key == "--coupling-kind") o.coupling_kind = v;
        else if (key == "--bigg-sign") o.bigg_sign = ParseDouble(v,key);
        else if (key == "--xi") o.xi = ParseDouble(v,key);
        else if (key == "--chi") o.chi = ParseDouble(v,key);
        else if (key == "--theta12") o.theta12 = ParseDouble(v,key);
        else if (key == "--theta13") o.theta13 = ParseDouble(v,key);
        else if (key == "--theta23") o.theta23 = ParseDouble(v,key);
        else if (key == "--delta-cp") o.delta_cp = ParseDouble(v,key);
        else if (key == "--dm21") o.dm21 = ParseDouble(v,key);
        else if (key == "--dm31") o.dm31 = ParseDouble(v,key);
        else if (key == "--interactions") o.interactions = ParseBool(v);
        else if (key == "--tau-regen") o.tau_regen = ParseBool(v);
        else if (key == "--glashow") o.glashow = ParseBool(v);
        else if (key == "--detector-depth-km") o.detector_depth_km = ParseDouble(v,key);
        else if (key == "--h-max-km") o.h_max_km = ParseDouble(v,key);
        else if (key == "--source-steps-per-layer") o.source_steps_per_layer = ParseDouble(v,key);
        else if (key == "--rtol") o.relative_tolerance = ParseDouble(v,key);
        else if (key == "--atol") o.absolute_tolerance = ParseDouble(v,key);
        else if (key == "--angle-begin") o.angle_begin = ParseSize(v,key);
        else if (key == "--angle-end") o.angle_end = ParseSize(v,key);
        else if (key == "--mass-begin") o.mass_begin = ParseSize(v,key);
        else if (key == "--mass-end") o.mass_end = ParseSize(v,key);
        else if (key == "--coupling-begin") o.coupling_begin = ParseSize(v,key);
        else if (key == "--coupling-end") o.coupling_end = ParseSize(v,key);
        else if (key == "--threads") SetThreadCount(o, v);
        else if (key == "--batch-size") o.batch_size = ParseSize(v,key);
        else Fail("Opcion desconocida: " + key);
    }
    if (o.mode != "lri" && o.mode != "standard" && o.mode != "both" && o.mode != "atmosphere")
        Fail("--mode debe ser atmosphere, lri, standard o both");
    if (o.coupling_kind == "auto")
        o.coupling_kind = o.model == lri::Model::LmuLtau ? "bigg" : "gprime";
    if (o.coupling_kind == "bigg" && !o.custom_coupling_range) {
        o.log_c_min=-50.0; o.log_c_max=-48.0;
    }
    if (o.coupling_kind != "gprime" && o.coupling_kind != "bigg")
        Fail("--coupling-kind debe ser gprime o bigg");
    if (o.model != lri::Model::LmuLtau && o.coupling_kind != "gprime")
        Fail("BigG solo corresponde a Lmu-Ltau");
    if (o.bigg_sign != 1.0 && o.bigg_sign != -1.0)
        Fail("--bigg-sign debe ser +1 o -1");
    if (o.model == lri::Model::LmuLtau && o.coupling_kind == "gprime" &&
        o.mode != "standard" && o.mode != "atmosphere" && (!std::isfinite(o.xi) || !std::isfinite(o.chi)))
        Fail("Para Lmu-Ltau con gprime debes indicar --xi y --chi");
    if (!FinitePositive(o.detector_depth_km) || !FinitePositive(o.h_max_km) ||
        !FinitePositive(o.source_steps_per_layer) ||
        !FinitePositive(o.relative_tolerance) || !FinitePositive(o.absolute_tolerance) ||
        o.batch_size == 0 || o.n_mass == 0 || o.n_coupling == 0 ||
        (o.n_mass > 1 && !(o.log_m_min < o.log_m_max)) ||
        (o.n_coupling > 1 && !(o.log_c_min < o.log_c_max)))
        Fail("Resolucion, tolerancias o malla de parametros invalidas");
    if (o.n_mass == 1 && !std::isfinite(o.log_m_min)) Fail("Masa invalida");
    if (o.n_coupling == 1 && !std::isfinite(o.log_c_min)) Fail("Acoplamiento invalido");
    if (o.threads < 0) Fail("Numero de hilos invalido");
    if (o.smoke) {
        // Independent small grid; do not allocate the full 41x61 output.
        o.n_mass = 2;
        o.log_m_min = -14.0;
        o.log_m_max = -13.0;
        o.n_coupling = 2;
        o.log_c_min = (o.coupling_kind == "bigg") ? -50.0 : -26.0;
        o.log_c_max = (o.coupling_kind == "bigg") ? -48.0 : -25.0;
    }
    return o;
}

std::vector<double> LogGrid(double a, double b, std::size_t n) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::pow(10.0, a + (n == 1 ? 0.0 :
            (b-a) * static_cast<double>(i) / static_cast<double>(n-1)));
    for (double x : out) if (!FinitePositive(x)) Fail("Malla logaritmica invalida");
    return out;
}

struct H5 {
    hid_t id = -1;
    herr_t (*close)(hid_t) = nullptr;
    H5(hid_t i, herr_t (*f)(hid_t)) : id(i), close(f) {
        if (i < 0) Fail("No se pudo abrir/crear objeto HDF5");
    }
    // An unopened handle can be initialized as a placeholder.
    explicit H5(herr_t (*f)(hid_t)) : close(f) {}
    ~H5() { if (id >= 0 && close) close(id); }
    H5(const H5&) = delete;
    H5& operator=(const H5&) = delete;
};

void Check(herr_t status, const std::string& what) {
    if (status < 0) Fail("Error HDF5: " + what);
}

void WriteSignature(hid_t file, const std::string& signature) {
    H5 type(H5Tcopy(H5T_C_S1), H5Tclose);
    Check(H5Tset_size(type.id, signature.size()+1), "signature type size");
    Check(H5Tset_strpad(type.id, H5T_STR_NULLTERM), "signature string type");
    H5 space(H5Screate(H5S_SCALAR), H5Sclose);
    H5 attr(H5Acreate2(file, "config_signature", type.id, space.id,
                        H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    Check(H5Awrite(attr.id, type.id, signature.c_str()), "signature write");
}

std::string ReadSignature(hid_t file) {
    H5 attr(H5Aopen(file, "config_signature", H5P_DEFAULT), H5Aclose);
    H5 type(H5Aget_type(attr.id), H5Tclose);
    std::vector<char> s(H5Tget_size(type.id)+1, '\0');
    Check(H5Aread(attr.id, type.id, s.data()), "signature read");
    return std::string(s.data());
}

void WriteAxis(hid_t file, const char* name, const std::vector<double>& data) {
    if (data.empty()) Fail("Intento de escribir eje HDF5 vacio");
    const hsize_t dims[] = {static_cast<hsize_t>(data.size())};
    H5 space(H5Screate_simple(1,dims,nullptr), H5Sclose);
    H5 ds(H5Dcreate2(file,name,H5T_IEEE_F64LE,space.id,H5P_DEFAULT,
                     H5P_DEFAULT,H5P_DEFAULT), H5Dclose);
    Check(H5Dwrite(ds.id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                   H5P_DEFAULT,data.data()), "axis write");
}

void WriteAttributeString(hid_t object, const char* key, const std::string& value) {
    H5 type(H5Tcopy(H5T_C_S1), H5Tclose);
    Check(H5Tset_size(type.id,value.size()+1),"attribute string length");
    H5 scalar(H5Screate(H5S_SCALAR),H5Sclose);
    H5 a(H5Acreate2(object,key,type.id,scalar.id,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    Check(H5Awrite(a.id,type.id,value.c_str()),"attribute string write");
}

void WriteAttributeDouble(hid_t object, const char* key, double val) {
    H5 scalar(H5Screate(H5S_SCALAR),H5Sclose);
    H5 a(H5Acreate2(object,key,H5T_IEEE_F64LE,scalar.id,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    Check(H5Awrite(a.id,H5T_NATIVE_DOUBLE,&val),"attribute double write");
}

std::string Signature(const Options& o, const lri::AtmosphericSource& src,
                      const std::vector<double>& masses,
                      const std::vector<double>& couplings,
                      const std::string& run_kind) {
    // Verify resumed files represent exactly the SAME physical setup.
    std::ostringstream s;
    s << std::setprecision(17) << "v5;kind=" << run_kind;
    // Standard reference depends on standard oscillation physics only;
    // switching the requested LRI model does NOT invalidate standard.h5.
    if (run_kind == "lri")
      s << ";model=" << ModelName(o.model)
        << ";kind_c=" << o.coupling_kind
        << ";bigg_sign=" << o.bigg_sign
        << ";xi=" << o.xi << ";chi=" << o.chi;
    s
      << ";mceq=" << fs::absolute(o.mceq).string()
      << ";mceq_size=" << fs::file_size(o.mceq)
      << ";src_top=" << src.TopHeightKm()
      << ";pot=" << (run_kind == "lri" ? fs::absolute(o.potential).string() : "none")
      << ";pot_size=" << (run_kind == "lri" ? fs::file_size(o.potential) : 0)
      << ";theta=" << o.theta12 << ',' << o.theta13 << ',' << o.theta23 << ',' << o.delta_cp
      << ";dm=" << o.dm21 << ',' << o.dm31
      << ";int=" << (run_kind=="atmosphere"?false:o.interactions)
      << ";tau=" << (run_kind=="atmosphere"?false:o.tau_regen)
      << ";glashow=" << (run_kind=="atmosphere"?false:o.glashow)
      << ";depth=" << o.detector_depth_km
      << ";integrator=analytic-atmo-gauss3+earth-v1"
      << ";h_max=" << (run_kind=="atmosphere"?0:o.h_max_km)
      << ";rtol=" << o.relative_tolerance << ";atol=" << o.absolute_tolerance
      << ";compress=" << o.compress
      << ";mass=[";
    for (double v : masses) s << v << ',';
    s << "];coupling=[";
    for (double v : couplings) s << v << ',';
    s << ']';
    return s.str();
}

class OutputFile {
    H5 file_;
    H5 data_;
    H5 done_;
    std::size_t nE_;
    std::size_t nAngles_;
    std::size_t nM_;
    std::size_t nC_;
    bool lri_;

    static hid_t CreateFile(const std::string& filename, bool overwrite) {
        const fs::path path(filename);
        if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
        return fs::exists(path) && !overwrite
            ? H5Fopen(filename.c_str(), H5F_ACC_RDWR,H5P_DEFAULT)
            : H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    }

    static hid_t OpenOrMakeDataset(hid_t file, bool create,
                                   const char* name, hid_t disk_type,
                                   const std::vector<hsize_t>& dims,
                                   const std::vector<hsize_t>& chunks,
                                   bool compress) {
        if (!create) return H5Dopen2(file,name,H5P_DEFAULT);
        H5 space(H5Screate_simple(static_cast<int>(dims.size()),dims.data(),nullptr),H5Sclose);
        H5 dcpl(H5Pcreate(H5P_DATASET_CREATE),H5Pclose);
        Check(H5Pset_chunk(dcpl.id,static_cast<int>(chunks.size()),chunks.data()),"chunking");
        if (compress) Check(H5Pset_deflate(dcpl.id,1),"compression filter");
        return H5Dcreate2(file,name,disk_type,space.id,H5P_DEFAULT,dcpl.id,H5P_DEFAULT);
    }

    static void WriteSlab(hid_t dataset, hid_t native_type,
                          const std::vector<hsize_t>& start,
                          const std::vector<hsize_t>& count,
                          const void* ptr) {
        H5 filespace(H5Dget_space(dataset),H5Sclose);
        Check(H5Sselect_hyperslab(filespace.id,H5S_SELECT_SET,start.data(),
                                  nullptr,count.data(),nullptr),"select slab");
        H5 memspace(H5Screate_simple(static_cast<int>(count.size()),count.data(),nullptr),H5Sclose);
        Check(H5Dwrite(dataset,native_type,memspace.id,filespace.id,H5P_DEFAULT,ptr),"write slab");
    }

    std::size_t FlagIndex(std::size_t m, std::size_t c, std::size_t angle) const {
        return (m*nC_+c)*nAngles_+angle;
    }

public:
    OutputFile(const std::string& path, const std::string& signature,
               const std::vector<double>& energies,
               const std::vector<double>& angles,
               const std::vector<double>& masses,
               const std::vector<double>& couplings,
               const Options& o, bool lri_file)
        : file_(CreateFile(path,o.overwrite),H5Fclose),
          data_(H5Dclose),done_(H5Dclose),
          nE_(energies.size()),nAngles_(angles.size()),
          nM_(lri_file?masses.size():1),nC_(lri_file?couplings.size():1),lri_(lri_file)
    {
        const bool is_new = H5Aexists(file_.id,"config_signature") <= 0;
        if (!is_new) {
            if (ReadSignature(file_.id) != signature)
                Fail("Checkpoint existente incompatible con la configuracion: " + path +
                     " (use otra ruta o --overwrite)");
        } else {
            WriteSignature(file_.id, signature);
            WriteAxis(file_.id,"energy_GeV",energies);
            WriteAxis(file_.id,"coszen",angles);
            WriteAttributeString(file_.id,"species_order",
                "nue,numu,nutau,antinue,antinumu,antinutau");
            WriteAttributeString(file_.id,"mode",lri_?"lri":"standard");
            WriteAttributeString(file_.id,"model",lri_?ModelName(o.model):"standard");
            WriteAttributeString(file_.id,"coupling_kind",lri_?o.coupling_kind:"none");
            WriteAttributeString(file_.id,"source","MCEq altitude-integrated production, injected in nuSQuIDS");
            WriteAttributeString(file_.id,"atmosphere","vacuum; Earth=PREM+MSW+optional LRI");
            WriteAttributeDouble(file_.id,"detector_depth_km",o.detector_depth_km);
            if (lri_) {
                WriteAxis(file_.id,"mass_eV",masses);
                WriteAxis(file_.id,"coupling_values",couplings);
            }
        }

        std::vector<hsize_t> ddims, dchunks, mdims, mchunks;
        if (lri_) {
            ddims={hsize_t(nM_),hsize_t(nC_),hsize_t(nAngles_),hsize_t(nE_),hsize_t(NF)};
            dchunks={1,1,1,hsize_t(nE_),hsize_t(NF)};
            mdims={hsize_t(nM_),hsize_t(nC_),hsize_t(nAngles_)};
            mchunks={1,1,hsize_t(std::min<std::size_t>(nAngles_,64))};
        } else {
            ddims={hsize_t(nAngles_),hsize_t(nE_),hsize_t(NF)};
            dchunks={1,hsize_t(nE_),hsize_t(NF)};
            mdims={hsize_t(nAngles_)};
            mchunks={hsize_t(std::min<std::size_t>(nAngles_,64))};
        }
        data_.id=OpenOrMakeDataset(file_.id,is_new,"flux",H5T_IEEE_F32LE,
                                   ddims,dchunks,o.compress);
        done_.id=OpenOrMakeDataset(file_.id,is_new,"done",H5T_STD_U8LE,
                                   mdims,mchunks,false);
    }

    std::vector<std::uint8_t> ReadDone() const {
        std::vector<std::uint8_t> result(nM_*nC_*nAngles_,0);
        Check(H5Dread(done_.id,H5T_NATIVE_UINT8,H5S_ALL,H5S_ALL,
                      H5P_DEFAULT,result.data()),"read done flags");
        return result;
    }

    void WritePoint(std::size_t m, std::size_t c, std::size_t angle,
                    const std::vector<float>& flux) {
        if (flux.size()!=nE_*NF) Fail("Output HDF5: flux length mismatch");
        if (lri_) WriteSlab(data_.id,H5T_NATIVE_FLOAT,
                            {hsize_t(m),hsize_t(c),hsize_t(angle),0,0},
                            {1,1,1,hsize_t(nE_),hsize_t(NF)},flux.data());
        else WriteSlab(data_.id,H5T_NATIVE_FLOAT,
                       {hsize_t(angle),0,0},{1,hsize_t(nE_),hsize_t(NF)},flux.data());
        const std::uint8_t one=1;
        if (lri_) WriteSlab(done_.id,H5T_NATIVE_UINT8,
                            {hsize_t(m),hsize_t(c),hsize_t(angle)},
                            {1,1,1},&one);
        else WriteSlab(done_.id,H5T_NATIVE_UINT8,{hsize_t(angle)},{1},&one);
    }

    void Flush() { Check(H5Fflush(file_.id,H5F_SCOPE_GLOBAL),"flush checkpoint"); }
};

// Absolute MCEq production flux magnitude can be tiny.  Normalize ALL six
// species by the SAME factor, for the whole trajectory. Output multiply back.
// This is only a numerical scaling, not a physics nuisance parameter.
double Normalization(const lri::AtmosphericSource::AngleSlice& slice) {
    double maxflux=0.0;
    for (std::size_t e=0;e<slice.NumberEnergies();++e) {
        const auto v=slice.IntegratedUnoscillatedFluxAtNode(e);
        for (double s:v) maxflux=std::max(maxflux,s);
    }
    if (!(maxflux>0.0) || !std::isfinite(maxflux))
        Fail("MCEq returned no positive atmospheric flux in this direction");
    return maxflux;
}

void ConfigureOscillations(nusquids::nuSQUIDS& solver, const Options& o,
                           double max_step_km) {
    solver.Set_MixingAngle(0,1,o.theta12);
    solver.Set_MixingAngle(0,2,o.theta13);
    solver.Set_MixingAngle(1,2,o.theta23);
    solver.Set_SquareMassDifference(1,o.dm21);
    solver.Set_SquareMassDifference(2,o.dm31);
    solver.Set_CPPhase(0,2,o.delta_cp);
    solver.Set_TauRegeneration(o.interactions && o.tau_regen);
    solver.Set_GlashowResonance(o.interactions && o.glashow);
    solver.Set_rel_error(o.relative_tolerance);
    solver.Set_abs_error(o.absolute_tolerance);
    solver.Set_GSL_step(gsl_odeiv2_step_rkf45);
    squids::Const units;
    solver.Set_h_max(max_step_km*units.km);
}

nusquids::marray<double,1> MakeEnergyGrid(const std::vector<double>& energy_GeV) {
    squids::Const units;
    nusquids::marray<double,1> e({energy_GeV.size()});
    for (std::size_t i=0;i<energy_GeV.size();++i) e[i]=energy_GeV[i]*units.GeV;
    return e;
}

// ============================================================================
// Coherent atmosphere state, mass basis; nine real SU(3) coefficients per rho.
// ============================================================================
struct SurfaceState {
    double normalization = 0.0;
    std::vector<double> components; // [energy][neutrino/antineutrino][9]
};

class VacuumStateCalculator final : public nusquids::nuSQUIDS {
public:
    explicit VacuumStateCalculator(nusquids::marray<double,1> energies)
        : nusquids::nuSQUIDS(std::move(energies),3,nusquids::both,false) {}

    SurfaceState Calculate(const Options& /*o*/, const lri::AtmosphericSource& source,
                           const EarthDepthBody::Track& geometry)
    {
        // Initialize nuSQuIDS projectors and vacuum mass Hamiltonian once.
        // Avoid PMNS-parameterization differences: use its OWN mass-basis
        // flavor projectors b1_proj[nu/antinu][flavor] and H0_array[E].
        nusquids::marray<double,3> zero({source.NumberEnergies(),2u,3u});
        std::fill(zero.begin(),zero.end(),0.0);
        Set_initial_state(zero,nusquids::mass);

        const auto raw=source.PrepareAngle(geometry,1.0);
        SurfaceState surface;
        surface.normalization=Normalization(raw);
        const auto slice=source.PrepareAngle(geometry,surface.normalization);
        const auto& edges=slice.LayerBoundariesX();
        const double xsurface=geometry.XAtEarthEntry();
        const std::size_t nE=source.NumberEnergies();
        surface.components.resize(nE*2u*9u);

        // Three-point Gauss-Legendre integration of the UNIFORM source within
        // each altitude bin. Exact vacuum evolution at each quadrature node.
        // No GSL ODE; no variable step that could skip thin source layers.
        constexpr double gauss_z[3] = {
            -0.7745966692414833770359, 0.0, +0.7745966692414833770359};
        constexpr double gauss_w[3] = {5.0/18.0, 8.0/18.0, 5.0/18.0};

        for(std::size_t ie=0;ie<nE;++ie) {
            for(std::size_t rho=0;rho<2u;++rho) {
                squids::SU_vector total(3);
                total.SetAllComponents(0.0);
                for(std::size_t layer=0;layer<slice.NumberLayers();++layer) {
                    const double lo=edges[layer], hi=edges[layer+1];
                    const double mid=0.5*(lo+hi), half=0.5*(hi-lo);
                    const auto source_density=slice.SourceAtNode(ie,mid);
                    for(std::size_t flavor=0;flavor<3u;++flavor) {
                        const double total_weight=source_density[rho*3u+flavor]*(hi-lo);
                        if(total_weight==0.0) continue;
                        for(std::size_t k=0;k<3u;++k) {
                            const double xpos=mid+gauss_z[k]*half;
                            const double flight=xsurface-xpos;
                            // Evolve(P,H0,-L) produces Schrödinger-evolved
                            // density, whereas nuSQuIDS measures using P(+L).
                            const squids::SU_vector contribution =
                                b1_proj[rho][flavor].Evolve(H0_array[ie],-flight);
                            total += (gauss_w[k]*total_weight)*contribution;
                        }
                    }
                }
                const auto coeffs=total.GetComponents();
                if(coeffs.size()!=9) Fail("Unexpected SU(3) component count");
                const std::size_t offset=(ie*2u+rho)*9u;
                for(std::size_t k=0;k<9u;++k) {
                    if(!std::isfinite(coeffs[k])) Fail("Nonfinite atmospheric state");
                    surface.components[offset+k]=coeffs[k];
                }
            }
        }
        return surface;
    }
};

// Older nuSQuIDS exposes only diagonal Set_initial_state(). A subclass may
// safely populate protected SQuIDS::state after the base initializer set up
// the mass-basis projectors and Hamiltonian. We do NOT reconstruct diagonal
// flavor fluxes (that would discard off-diagonal coherences).
class StandardSurfaceSolver final : public nusquids::nuSQUIDS {
public:
    StandardSurfaceSolver(nusquids::marray<double,1> energies,bool interactions)
        : nusquids::nuSQUIDS(std::move(energies),3,nusquids::both,interactions) {}

    void SetCoherentInitialState(const std::vector<double>& values) {
        const std::size_t nE=GetNumE();
        if(values.size()!=nE*2u*9u) Fail("Bad cached atmospheric density shape");
        nusquids::marray<double,3> zero({nE,2u,3u});
        std::fill(zero.begin(),zero.end(),0.0);
        Set_initial_state(zero,nusquids::mass);
        for(std::size_t ie=0;ie<nE;++ie)
            for(std::size_t rho=0;rho<2u;++rho) {
                for(std::size_t k=0;k<9u;++k)
                    state[ie].rho[rho][static_cast<unsigned>(k)]=
                        values[(ie*2u+rho)*9u+k];
            }
    }
};

class AtmosphereCache {
    H5 file_;
    H5 states_;
    H5 norms_;
    H5 done_;
    std::size_t ne_, na_;

    static hid_t OpenFile(const std::string& filename,bool overwrite) {
        const fs::path p(filename);
        if(!p.parent_path().empty()) fs::create_directories(p.parent_path());
        return fs::exists(p) && !overwrite
            ? H5Fopen(filename.c_str(),H5F_ACC_RDWR,H5P_DEFAULT)
            : H5Fcreate(filename.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    }
    static hid_t MakeData(hid_t file,const char* name,hid_t dtype,
                          const std::vector<hsize_t>& dims,
                          const std::vector<hsize_t>& chunks,bool fresh) {
        if(!fresh)return H5Dopen2(file,name,H5P_DEFAULT);
        H5 space(H5Screate_simple(static_cast<int>(dims.size()),dims.data(),nullptr),H5Sclose);
        H5 creation(H5Pcreate(H5P_DATASET_CREATE),H5Pclose);
        Check(H5Pset_chunk(creation.id,static_cast<int>(chunks.size()),chunks.data()),"cache chunk");
        return H5Dcreate2(file,name,dtype,space.id,H5P_DEFAULT,creation.id,H5P_DEFAULT);
    }
    static void Slab(hid_t ds,bool write,hid_t memtype,
                     const std::vector<hsize_t>& start,
                     const std::vector<hsize_t>& count,void* data) {
        H5 filespace(H5Dget_space(ds),H5Sclose);
        Check(H5Sselect_hyperslab(filespace.id,H5S_SELECT_SET,start.data(),nullptr,
                                  count.data(),nullptr),"cache hyperslab");
        H5 memspace(H5Screate_simple(static_cast<int>(count.size()),count.data(),nullptr),H5Sclose);
        if(write)Check(H5Dwrite(ds,memtype,memspace.id,filespace.id,H5P_DEFAULT,data),"cache write");
        else Check(H5Dread(ds,memtype,memspace.id,filespace.id,H5P_DEFAULT,data),"cache read");
    }
public:
    AtmosphereCache(const std::string& path,const std::string& signature,
                    const std::vector<double>& energies,
                    const std::vector<double>& angles,bool overwrite)
        : file_(OpenFile(path,overwrite),H5Fclose),
          states_(H5Dclose),norms_(H5Dclose),done_(H5Dclose),
          ne_(energies.size()),na_(angles.size())
    {
        const bool fresh=H5Aexists(file_.id,"config_signature")<=0;
        if(fresh) {
            WriteSignature(file_.id,signature);
            WriteAxis(file_.id,"energy_GeV",energies);
            WriteAxis(file_.id,"coszen",angles);
            WriteAttributeString(file_.id,"representation","SU(3) mass basis, 9 real components per density");
            WriteAttributeString(file_.id,"state_layout","coszen,energy,nu_or_antinu,SU3_components");
            WriteAttributeString(file_.id,"physics","uniform MCEq production; Gauss3 exact-vacuum evolution; zero LRI in atmosphere");
        }else if(ReadSignature(file_.id)!=signature)
            Fail("Incompatible atmosphere cache: "+path+" (choose new path or --overwrite)");
        states_.id=MakeData(file_.id,"state",H5T_IEEE_F64LE,
                       {hsize_t(na_),hsize_t(ne_),2,9},
                       {1,hsize_t(ne_),2,9},fresh);
        norms_.id=MakeData(file_.id,"normalization",H5T_IEEE_F64LE,
                       {hsize_t(na_)},{hsize_t(std::min<std::size_t>(na_,64))},fresh);
        done_.id=MakeData(file_.id,"done",H5T_STD_U8LE,
                       {hsize_t(na_)},{hsize_t(std::min<std::size_t>(na_,64))},fresh);
    }
    std::vector<std::uint8_t> ReadDone()const {
        std::vector<std::uint8_t> d(na_,0);
        Check(H5Dread(done_.id,H5T_NATIVE_UINT8,H5S_ALL,H5S_ALL,H5P_DEFAULT,d.data()),"cache flags");
        return d;
    }
    void Write(std::size_t angle,SurfaceState& s) {
        if(s.components.size()!=ne_*18u || !FinitePositive(s.normalization))
            Fail("Invalid atmosphere density state");
        Slab(states_.id,true,H5T_NATIVE_DOUBLE,{hsize_t(angle),0,0,0},
             {1,hsize_t(ne_),2,9},s.components.data());
        Slab(norms_.id,true,H5T_NATIVE_DOUBLE,{hsize_t(angle)},{1},&s.normalization);
        H5Fflush(file_.id,H5F_SCOPE_GLOBAL);
        std::uint8_t done=1;
        Slab(done_.id,true,H5T_NATIVE_UINT8,{hsize_t(angle)},{1},&done);
        H5Fflush(file_.id,H5F_SCOPE_GLOBAL);
    }
    SurfaceState Read(std::size_t angle) {
        SurfaceState s;
        s.components.resize(ne_*18u);
        Slab(states_.id,false,H5T_NATIVE_DOUBLE,{hsize_t(angle),0,0,0},
             {1,hsize_t(ne_),2,9},s.components.data());
        Slab(norms_.id,false,H5T_NATIVE_DOUBLE,{hsize_t(angle)},{1},&s.normalization);
        if(!FinitePositive(s.normalization)) Fail("Invalid cached atmosphere normalization");
        return s;
    }
};

std::vector<float> RunAngle(
    const Options& o,
    const lri::AtmosphericSource& source,
    const SurfaceState& surface,
    std::shared_ptr<const lri::RadialProfile> profile,
    double coupling,
    double coszen,
    bool is_lri)
{
    // EARTH ONLY: the atmosphere is not integrated by nuSQuIDS here.
    auto earth=std::make_shared<EarthDepthBody>(o.detector_depth_km);
    earth->SetAtmosphereHeightKm(source.TopHeightKm());
    auto track=earth->MakeTrackWithCosineToDetector(coszen);
    // EarthDepth.h requires segment endpoints to be contiguous.
    // Select the atmosphere geometrically (without evolving it), then
    // immediately select Earth as the actual integration interval.
    track->SelectAtmosphericSegment();
    track->SelectEarthSegment();

    auto energies=MakeEnergyGrid(source.EnergyGeV());
    std::unique_ptr<nusquids::nuSQUIDS> solver;
    if(is_lri) {
        lri::Couplings p;
        p.gprime=(o.coupling_kind=="gprime")?coupling:0.0;
        p.xi=o.xi; p.chi=o.chi;
        auto ptr=std::make_unique<nuSQUIDSLRI>(
            std::move(profile),p,std::move(energies),nusquids::both,o.interactions);
        if(o.coupling_kind=="bigg")ptr->SetBigG(coupling);
        solver=std::move(ptr);
    }else {
        solver=std::make_unique<StandardSurfaceSolver>(
            std::move(energies),o.interactions);
    }
    ConfigureOscillations(*solver,o,o.h_max_km);
    if(is_lri){
        auto* ptr=static_cast<nuSQUIDSLRI*>(solver.get());
        ptr->Set_Body(earth);
        ptr->Set_Track(track);
        ptr->SetCoherentInitialState(surface.components);
    }else{
        auto* ptr=static_cast<StandardSurfaceSolver*>(solver.get());
        ptr->Set_Body(earth);
        ptr->Set_Track(track);
        ptr->SetCoherentInitialState(surface.components);
    }
    solver->EvolveState();

    std::vector<float> flux(source.NumberEnergies()*NF);
    for(std::size_t ie=0;ie<source.NumberEnergies();++ie)
        for(std::size_t fl=0;fl<NNU;++fl)
            for(std::size_t rho=0;rho<2u;++rho) {
                const double v=solver->EvalFlavorAtNode(static_cast<unsigned>(fl),
                       static_cast<unsigned>(ie),static_cast<unsigned>(rho))
                       *surface.normalization;
                if(!std::isfinite(v)||v<-1e-7*surface.normalization)
                    Fail("Nonfinite/negative propagated flux");
                if(v>std::numeric_limits<float>::max())Fail("Float32 flux overflow");
                flux[ie*NF+rho*3u+fl]=static_cast<float>(std::max(0.0,v));
            }
    return flux;
}

// Reconstruct the signed MCEq zenith grid from its validated absolute grid.
// AtmosphericSource checks this symmetry against /axes/coszen on load.
std::vector<double> SignedCosZenithGrid(const lri::AtmosphericSource& source) {
    const auto& abs = source.AbsCosZenithGrid();
    std::vector<double> signed_angles;
    signed_angles.reserve(2*abs.size()-1);
    for (std::size_t i=abs.size();i-->1;) signed_angles.push_back(-abs[i]);
    for (double a:abs) signed_angles.push_back(a);
    return signed_angles;
}

std::vector<std::size_t> AngleIndices(const Options& o, std::size_t count) {
    if (o.smoke) return {0, count/2, count-1};
    const auto last=std::min(count,o.angle_end);
    if(o.angle_begin>=last) Fail("Invalid angle range");
    std::vector<std::size_t> v;
    v.reserve(last-o.angle_begin);
    for(std::size_t i=o.angle_begin;i<last;++i) v.push_back(i);
    return v;
}

std::size_t StartIndex(std::size_t lower,std::size_t count) {
    if(lower>=count) Fail("Parameter start index out of range");
    return lower;
}

std::size_t StopIndex(std::size_t upper,std::size_t count) {
    return std::min(upper,count);
}

// Calculate requested coherent atmosphere states ONCE, then hold them in RAM
// across the entire mass/coupling scan. Can resume from a dedicated HDF5.
std::vector<SurfaceState> GetAtmosphereStates(
    const Options& o,const lri::AtmosphericSource& source,
    const std::vector<double>& angles)
{
    const std::string sig=Signature(o,source,{}, {},"atmosphere");
    AtmosphereCache cache(o.output_atmosphere,sig,source.EnergyGeV(),
                          angles,o.overwrite);
    auto done=cache.ReadDone();
    const auto selected=AngleIndices(o,angles.size());
    std::vector<std::size_t> pending;
    for(auto angle:selected)if(!done[angle])pending.push_back(angle);
    std::cout<<"\nAtmosphere cache="<<o.output_atmosphere
             <<"\nPending atmosphere angles="<<pending.size()
             <<" energies="<<source.NumberEnergies()<<'\n';

    std::vector<std::string> errors(pending.size());
    std::vector<SurfaceState> computed(pending.size());
    // One (energy x 2 x 9) state per zenith angle, no mass/coupling axis.
    #pragma omp parallel for schedule(dynamic,1)
    for(std::int64_t k=0;k<static_cast<std::int64_t>(pending.size());++k){
        try{
            const auto ia=pending[static_cast<std::size_t>(k)];
            auto earth=std::make_shared<EarthDepthBody>(o.detector_depth_km);
            earth->SetAtmosphereHeightKm(source.TopHeightKm());
            auto track=earth->MakeTrackWithCosineToDetector(angles[ia]);
            track->SelectAtmosphericSegment();
            track->SelectEarthSegment();
            VacuumStateCalculator solver(MakeEnergyGrid(source.EnergyGeV()));
            solver.Set_MixingAngle(0,1,o.theta12);
            solver.Set_MixingAngle(0,2,o.theta13);
            solver.Set_MixingAngle(1,2,o.theta23);
            solver.Set_CPPhase(0,2,o.delta_cp);
            solver.Set_SquareMassDifference(1,o.dm21);
            solver.Set_SquareMassDifference(2,o.dm31);
            solver.Set_Body(earth);
            solver.Set_Track(track);
            computed[static_cast<std::size_t>(k)]=solver.Calculate(o,source,*track);
        }catch(const std::exception& e){errors[static_cast<std::size_t>(k)]=e.what();}
    }
    for(std::size_t k=0;k<pending.size();++k){
        if(!errors[k].empty())Fail("Atmosphere angle="+std::to_string(pending[k])+": "+errors[k]);
        cache.Write(pending[k],computed[k]);
        if(k%4==0 || k+1==pending.size())
            std::cout<<"  atmosphere completed "<<k+1<<'/'<<pending.size()<<'\n';
    }

    std::vector<SurfaceState> states(angles.size());
    for(auto a:selected)states[a]=cache.Read(a);
    return states;
}

void DoMode(const Options& o, const lri::AtmosphericSource& source,
            const std::vector<double>& masses,
            const std::vector<double>& couplings,
            const lri::PotentialLibrary* potential,
            const std::vector<SurfaceState>& surface_states,
            bool is_lri, const std::string& path)
{
    const auto& energies=source.EnergyGeV();
    // Source's signed grid and output's signed grid are identical.
    const std::vector<double> angles=SignedCosZenithGrid(source);
    const auto sig=Signature(o,source,is_lri?masses:std::vector<double>{},
                            is_lri?couplings:std::vector<double>{},
                            is_lri?"lri":"standard");
    OutputFile output(path,sig,energies,angles,masses,couplings,o,is_lri);
    auto done=output.ReadDone();
    const auto selected_angles=AngleIndices(o,angles.size());
    const std::size_t mstart=is_lri?StartIndex(o.mass_begin,masses.size()):0;
    const std::size_t mend=is_lri?StopIndex(o.mass_end,masses.size()):1;
    const std::size_t cstart=is_lri?StartIndex(o.coupling_begin,couplings.size()):0;
    const std::size_t cend=is_lri?StopIndex(o.coupling_end,couplings.size()):1;
    if(mstart>=mend || cstart>=cend) Fail("Empty mass/coupling selection");

    std::uint64_t pending=0;
    for(std::size_t m=mstart;m<mend;++m)
      for(std::size_t c=cstart;c<cend;++c)
        for(auto a:selected_angles)
          if(!done[(m*(is_lri?couplings.size():1)+c)*angles.size()+a]) ++pending;

    std::cout<<"\nMode="<<(is_lri?"lri":"standard")
             <<" model="<<(is_lri?ModelName(o.model):"standard")
             <<" output="<<path<<"\nPending="<<pending
             <<"  angles="<<selected_angles.size()
             <<"  energies="<<energies.size()<<'\n';
    if(pending==0) return;

    std::uint64_t completed=0;
    std::size_t batches=0;
    for(std::size_t im=mstart;im<mend;++im) {
        std::shared_ptr<const lri::RadialProfile> radial;
        if(is_lri) radial=std::make_shared<lri::RadialProfile>(
            potential->BuildRadialProfile(o.model,masses[im]));
        for(std::size_t ic=cstart;ic<cend;++ic) {
            for(std::size_t a0=0;a0<selected_angles.size();a0+=o.batch_size) {
                const std::size_t end=std::min(selected_angles.size(),a0+o.batch_size);
                const std::size_t n=end-a0;
                std::vector<std::vector<float>> computed(n);
                std::vector<std::string> errors(n);
                // HDF5 is accessed only outside this parallel block.
                // Every worker owns its Body, Track, nuSQUIDS and AngleSlice.
                #pragma omp parallel for schedule(dynamic,1)
                for(std::int64_t bi=0;bi<static_cast<std::int64_t>(n);++bi) {
                    const auto i=static_cast<std::size_t>(bi);
                    const auto ia=selected_angles[a0+i];
                    const auto maskindex=(im*(is_lri?couplings.size():1)+ic)*angles.size()+ia;
                    if(done[maskindex]) continue;
                    try {
                        computed[i]=RunAngle(o,source,surface_states[ia],radial,
                              is_lri?couplings[ic]:0.0,angles[ia],is_lri);
                    } catch(const std::exception& error) {
                        errors[i]=error.what();
                    } catch(...) { errors[i]="unknown propagation failure"; }
                }
                for(std::size_t i=0;i<n;++i) {
                    if(!errors[i].empty()) {
                        output.Flush();
                        Fail("Propagation failed at mass="+std::to_string(im)+
                             " coupling="+std::to_string(ic)+
                             " angle="+std::to_string(selected_angles[a0+i])+": "+errors[i]);
                    }
                    if(!computed[i].empty()) {
                        const auto ia=selected_angles[a0+i];
                        output.WritePoint(im,ic,ia,computed[i]);
                        done[(im*(is_lri?couplings.size():1)+ic)*angles.size()+ia]=1;
                        ++completed;
                    }
                }
                ++batches;
                if(batches%o.flush_every==0) output.Flush();
                if(completed && (completed%std::max<std::uint64_t>(1,pending/20))<n)
                    std::cout<<"  completed "<<completed<<'/'<<pending<<"\n";
            }
        }
        output.Flush();
    }
    output.Flush();
    std::cout<<"Saved "<<completed<<" new rows. Checkpoint complete for requested selection.\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        H5Eset_auto2(H5E_DEFAULT,nullptr,nullptr);
        Options opt=ParseOptions(argc,argv);

        // Accept both launch directories:
        //   from the repository root: ./LRI/propagation/generate_flux_lut ...
        //   from LRI/propagation:     ./generate_flux_lut ...
        // Keep all relative default paths rooted at the repository root.
        if (!fs::exists(opt.mceq)) {
            const fs::path exe=fs::absolute(fs::path(argv[0])).lexically_normal();
            const fs::path parent=exe.parent_path();
            if (parent.filename()=="propagation" &&
                parent.parent_path().filename()=="LRI") {
                const fs::path repo=parent.parent_path().parent_path();
                if (fs::exists(repo / opt.mceq)) fs::current_path(repo);
            }
        }

        #ifdef _OPENMP
        if(opt.threads>0) omp_set_num_threads(opt.threads);
        #else
        if(opt.threads>1) Fail("Program compiled without OpenMP");
        #endif

        if(!fs::exists(opt.mceq)) Fail("Missing MCEq HDF5: "+opt.mceq);
        lri::AtmosphericSource source(opt.mceq);
        const auto angles=SignedCosZenithGrid(source);
        if(angles.size()<3) Fail("At least 3 angles needed for smoke/grid");
        const auto masses=LogGrid(opt.log_m_min,opt.log_m_max,opt.n_mass);
        auto couplings=LogGrid(opt.log_c_min,opt.log_c_max,opt.n_coupling);
        if(opt.coupling_kind=="bigg")
            for(auto& g:couplings)g*=opt.bigg_sign;

        const std::string suffix=opt.smoke?"_smoke":"";
        if(opt.output_atmosphere.empty())
            opt.output_atmosphere="results/oscillated_flux/atmosphere_state"+suffix+".h5";
        if(opt.output_standard.empty())
            opt.output_standard="results/oscillated_flux/standard"+suffix+".h5";
        if(opt.output_lri.empty())
            opt.output_lri="results/oscillated_flux/"+ModelName(opt.model)+suffix+".h5";
        if(opt.output_atmosphere==opt.output_standard || opt.output_atmosphere==opt.output_lri)
            Fail("Atmosphere cache filename must differ from final flux filename");
        if(opt.output_lri==opt.output_standard && opt.mode=="both")
            Fail("Standard and LRI output must have different filenames");

        std::unique_ptr<lri::PotentialLibrary> potential;
        if(opt.mode=="lri"||opt.mode=="both") {
            if(!fs::exists(opt.potential)) Fail("Missing potential LUT: "+opt.potential);
            potential=std::make_unique<lri::PotentialLibrary>(opt.potential);
            if(std::abs(potential->EarthRadiusKm()-source.EarthRadiusKm())>1e-6)
                Fail("Earth radius mismatch between MCEq and potential LUT");
            if(std::abs(potential->DetectorDepthKm()-opt.detector_depth_km)>1e-6)
                Fail("Detector depth mismatch between LUT and propagation");
            for(double m:masses)
              if(m<potential->MassGridEV().front()||m>potential->MassGridEV().back())
                  Fail("Requested mass outside potential LUT");
        }
        const auto atmosphere=GetAtmosphereStates(opt,source,angles);
        if(opt.mode=="standard"||opt.mode=="both")
            DoMode(opt,source,masses,couplings,nullptr,atmosphere,false,opt.output_standard);
        if(opt.mode=="lri"||opt.mode=="both")
            DoMode(opt,source,masses,couplings,potential.get(),atmosphere,true,opt.output_lri);
        std::cout<<"\nDone. Standard is computed only when --mode standard|both.\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"\n[ERROR] "<<e.what()<<'\n';
        return 1;
    }
}
