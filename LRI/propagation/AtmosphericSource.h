#pragma once

// ============================================================================
// AtmosphericSource.h
// ============================================================================
// MCEq atmospheric neutrino production by altitude, for nuSQuIDS propagation.
//
// Input HDF5 (from flux_mceq_height_parallel.py / future MCEq/flux.py):
//   /axes/energy_GeV             [energy]
//   /axes/coszen                 [signed_coszen]
//   /axes/abs_coszen_mceq        [abs_coszen]
//   /production/height_edges_km  [layer + 1], DESCENDING
//   /production/{species}        [abs_coszen, layer, energy]
//   /flux/{species}              [signed_coszen, energy]
//
// species = nue, numu, nutau, antinue, antinumu, antinutau.
//
// IMPORTANT:
//   - /production contains integrated DELTA FLUX per layer, not dPhi/dx.
//   - AngleSlice turns each delta flux into an injection rate per unit
//     nuSQuIDS track coordinate x (eV^-1).
//   - The source contributes only on the incoming atmospheric segment.
//   - There is NO LRI or MSW potential in this class; use EarthDepthBody
//     (vacuum atmosphere) and the correct Hamiltonian separately.
//   - To avoid missing narrow source layers, the propagator MUST resolve
//     the layer boundaries returned by LayerBoundariesX().
//   - Production is symmetric in up/down angle in the input MCEq file;
//     angular interpolation uses the absolute LOCAL cosine at the surface,
//     not blindly the detector cosine (important near the horizon).
//   - Only atmospheric neutrino SOURCES are modeled here, not atmospheric
//     muon backgrounds.
//
// Threading: AtmosphericSource is immutable after construction. It can be
// shared by OpenMP workers. Each worker/track gets its own AngleSlice.
//
// Does not read YAML and does not depend on a new configuration framework.
// ============================================================================

#include "EarthDepth.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>

namespace lri {

class AtmosphericSource final
{
public:
    static constexpr std::size_t NSPECIES = 6;
    using Spectrum6 = std::array<double, NSPECIES>;

    // Fixed species ordering throughout propagation:
    // [nue, numu, nutau, anti-nue, anti-numu, anti-nutau]
    enum Species : std::size_t {
        NuE = 0, NuMu = 1, NuTau = 2,
        AntiNuE = 3, AntiNuMu = 4, AntiNuTau = 5
    };

private:
    struct H5Handle
    {
        hid_t id = -1;
        herr_t (*close)(hid_t) = nullptr;

        H5Handle(hid_t id_, herr_t (*close_)(hid_t)) : id(id_), close(close_) {
            if (id < 0) throw std::runtime_error("AtmosphericSource: HDF5 open failed");
        }
        ~H5Handle() { if (id >= 0 && close) close(id); }
        H5Handle(const H5Handle&) = delete;
        H5Handle& operator=(const H5Handle&) = delete;
        H5Handle(H5Handle&& o) noexcept : id(o.id), close(o.close) { o.id = -1; }
        H5Handle& operator=(H5Handle&&) = delete;
    };

    static std::vector<hsize_t> DatasetShape(hid_t file, const std::string& path)
    {
        H5Handle dataset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose);
        H5Handle space(H5Dget_space(dataset.id), H5Sclose);
        const int rank = H5Sget_simple_extent_ndims(space.id);
        if (rank < 1 || rank > 3)
            throw std::runtime_error("AtmosphericSource: invalid rank in " + path);
        std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
        if (H5Sget_simple_extent_dims(space.id, dims.data(), nullptr) < 0)
            throw std::runtime_error("AtmosphericSource: cannot get shape of " + path);
        for (hsize_t d : dims) {
            if (d == 0 || d > static_cast<hsize_t>(std::numeric_limits<std::size_t>::max()))
                throw std::runtime_error("AtmosphericSource: invalid extent in " + path);
        }
        return dims;
    }

    static void ExpectShape(const std::vector<hsize_t>& found,
                            const std::vector<hsize_t>& expected,
                            const std::string& path)
    {
        if (found != expected)
            throw std::runtime_error("AtmosphericSource: unexpected dataset shape: " + path);
    }

    template <typename T>
    static std::vector<T> ReadArray(hid_t file, const std::string& path,
                                    const std::vector<hsize_t>& expected, hid_t type)
    {
        ExpectShape(DatasetShape(file, path), expected, path);
        H5Handle dataset(H5Dopen2(file, path.c_str(), H5P_DEFAULT), H5Dclose);
        std::size_t n = 1;
        for (hsize_t d : expected) {
            const auto size_d = static_cast<std::size_t>(d);
            if (n > std::numeric_limits<std::size_t>::max() / size_d)
                throw std::runtime_error("AtmosphericSource: array too large: " + path);
            n *= size_d;
        }
        std::vector<T> out(n);
        if (H5Dread(dataset.id, type, H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, out.data()) < 0)
            throw std::runtime_error("AtmosphericSource: cannot read: " + path);
        return out;
    }

    static std::vector<double> ReadAxis(hid_t file, const std::string& path)
    {
        const auto dims = DatasetShape(file, path);
        if (dims.size() != 1)
            throw std::runtime_error("AtmosphericSource: axis not 1D: " + path);
        return ReadArray<double>(file, path, dims, H5T_NATIVE_DOUBLE);
    }

    static double ReadEarthRadius(hid_t file)
    {
        if (H5Aexists(file, "earth_radius_km") <= 0)
            throw std::runtime_error("AtmosphericSource: missing earth_radius_km metadata");
        H5Handle attr(H5Aopen(file, "earth_radius_km", H5P_DEFAULT), H5Aclose);
        double value = 0.0;
        if (H5Aread(attr.id, H5T_NATIVE_DOUBLE, &value) < 0 ||
            !std::isfinite(value) || value <= 0.0)
            throw std::runtime_error("AtmosphericSource: invalid earth_radius_km");
        return value;
    }

    static void CheckIncreasing(const std::vector<double>& v,
                                 const std::string& name, bool nonnegative)
    {
        if (v.size() < 2) throw std::runtime_error("AtmosphericSource: short axis " + name);
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (!std::isfinite(v[i]) || (nonnegative && v[i] < 0.0) ||
                (i > 0 && !(v[i] > v[i-1])))
                throw std::runtime_error("AtmosphericSource: non-increasing or invalid " + name);
        }
    }

    static bool Close(double a, double b, double rel = 1e-8, double abs = 1e-10)
    {
        return std::abs(a-b) <= abs + rel*std::max(std::abs(a), std::abs(b));
    }

    static std::size_t LowerIndex(const std::vector<double>& axis, double x)
    {
        if (x <= axis.front()) return 0;
        if (x >= axis.back()) return axis.size()-2;
        auto it = std::upper_bound(axis.begin(), axis.end(), x);
        return static_cast<std::size_t>(it - axis.begin())-1;
    }

    std::string filename_;
    std::vector<double> energy_GeV_;
    std::vector<double> abs_coszen_;
    std::vector<double> coszen_;
    std::vector<double> heights_km_;
    double earth_radius_km_ = 0.0;
    std::array<std::vector<float>, NSPECIES> production_;

    static constexpr std::array<const char*, NSPECIES> SpeciesNames() {
        return {{"nue", "numu", "nutau", "antinue", "antinumu", "antinutau"}};
    }

public:
    // Per-trajectory source, independent of the LRI mass and coupling.
    class AngleSlice final
    {
        friend class AtmosphericSource;
        std::vector<double> energies_;
        std::vector<double> layer_boundaries_x_; // [0, x_1, ..., x_surface]
        std::vector<double> source_per_x_;       // [layer, energy, species]
        double normalization_ = 1.0;
        double surface_coszen_abs_ = 0.0;

        std::size_t Offset(std::size_t layer, std::size_t energy,
                           std::size_t species) const noexcept
        {
            return (layer*energies_.size()+energy)*NSPECIES+species;
        }

    public:
        std::size_t NumberEnergies() const noexcept { return energies_.size(); }
        std::size_t NumberLayers() const noexcept { return layer_boundaries_x_.size()-1; }
        const std::vector<double>& EnergyGeV() const noexcept { return energies_; }
        const std::vector<double>& LayerBoundariesX() const noexcept { return layer_boundaries_x_; }
        double SurfaceAbsCosZenith() const noexcept { return surface_coszen_abs_; }
        double FluxNormalization() const noexcept { return normalization_; }

        // Layer containing x, or -1 outside the incoming atmosphere.
        // At the surface x == x_entry there is no injection.
        int LayerAtX(double x) const
        {
            if (!std::isfinite(x)) throw std::invalid_argument("AtmosphericSource: non-finite x");
            if (x < layer_boundaries_x_.front() || x >= layer_boundaries_x_.back())
                return -1;
            const auto it = std::upper_bound(layer_boundaries_x_.begin(),
                                             layer_boundaries_x_.end(), x);
            return static_cast<int>(it-layer_boundaries_x_.begin()-1);
        }

        // dPhi/dx, with x in eV^-1 as used by nuSQuIDS.
        // Output has differential flux units (from MCEq) per eV^-1.
        // The caller must add these as flavor-diagonal density-matrix sources
        // in the nuSQuIDS evolution equations, with consistent normalization.
        Spectrum6 SourceAtNode(std::size_t energy_index, double x) const
        {
            if (energy_index >= energies_.size())
                throw std::out_of_range("AtmosphericSource: energy index out of range");
            const int layer = LayerAtX(x);
            Spectrum6 out{};
            if (layer < 0) return out;
            const std::size_t base = Offset(static_cast<std::size_t>(layer), energy_index, 0);
            for (std::size_t s = 0; s < NSPECIES; ++s)
                out[s] = source_per_x_[base+s];
            return out;
        }

        // Optional interpolation on log-energy axis; linear in the flux itself.
        Spectrum6 SourceAtEnergy(double energy_GeV, double x) const
        {
            if (!std::isfinite(energy_GeV) || energy_GeV <= 0.0 ||
                energy_GeV < energies_.front() || energy_GeV > energies_.back())
                throw std::out_of_range("AtmosphericSource: energy outside MCEq grid");
            const int layer = LayerAtX(x);
            Spectrum6 out{};
            if (layer < 0) return out;
            const auto i = LowerIndex(energies_, energy_GeV);
            const double denom = std::log(energies_[i+1] / energies_[i]);
            const double f = std::clamp(std::log(energy_GeV / energies_[i]) / denom, 0.0, 1.0);
            for (std::size_t s=0; s<NSPECIES; ++s) {
                const double y0 = source_per_x_[Offset(static_cast<std::size_t>(layer), i, s)];
                const double y1 = source_per_x_[Offset(static_cast<std::size_t>(layer), i+1, s)];
                out[s] = (1.0-f)*y0+f*y1;
            }
            return out;
        }

        // Invariant: integrate sources over all atmospheric layers (without
        // oscillations/interactions) to recover the MCEq surface spectrum.
        Spectrum6 IntegratedUnoscillatedFluxAtNode(std::size_t energy_index) const
        {
            if (energy_index >= energies_.size())
                throw std::out_of_range("AtmosphericSource: energy index out of range");
            Spectrum6 total{};
            for (std::size_t l=0; l<NumberLayers(); ++l) {
                const double dx = layer_boundaries_x_[l+1]-layer_boundaries_x_[l];
                for (std::size_t s=0; s<NSPECIES; ++s)
                    total[s] += source_per_x_[Offset(l,energy_index,s)]*dx;
            }
            return total;
        }
    };

    explicit AtmosphericSource(const std::string& filename) : filename_(filename)
    {
        if (H5Fis_hdf5(filename.c_str()) <= 0)
            throw std::runtime_error("AtmosphericSource: missing/invalid HDF5: " + filename);
        H5Handle file(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);

        energy_GeV_ = ReadAxis(file.id, "/axes/energy_GeV");
        abs_coszen_ = ReadAxis(file.id, "/axes/abs_coszen_mceq");
        coszen_ = ReadAxis(file.id, "/axes/coszen");
        heights_km_ = ReadAxis(file.id, "/production/height_edges_km");
        earth_radius_km_ = ReadEarthRadius(file.id);

        CheckIncreasing(energy_GeV_, "energy_GeV", true);
        if (!(energy_GeV_.front() > 0.0))
            throw std::runtime_error("AtmosphericSource: energy must be positive");
        CheckIncreasing(abs_coszen_, "abs_coszen_mceq", true);
        CheckIncreasing(coszen_, "coszen", false);
        if (!Close(abs_coszen_.front(),0.0) || !Close(abs_coszen_.back(),1.0))
            throw std::runtime_error("AtmosphericSource: expected |coszen| from 0 to 1");
        if (coszen_.size() != 2*abs_coszen_.size()-1)
            throw std::runtime_error("AtmosphericSource: incompatible signed/absolute coszen axes");
        for (std::size_t i=0; i<abs_coszen_.size(); ++i) {
            if (!Close(coszen_[abs_coszen_.size()-1+i],abs_coszen_[i]) ||
                (i>0 && !Close(coszen_[abs_coszen_.size()-1-i],-abs_coszen_[i])))
                throw std::runtime_error("AtmosphericSource: inconsistent signed coszen grid");
        }
        if (heights_km_.size() < 2)
            throw std::runtime_error("AtmosphericSource: no atmospheric layers");
        for (std::size_t i=0; i<heights_km_.size(); ++i) {
            if (!std::isfinite(heights_km_[i]) || heights_km_[i] < 0.0 ||
                (i>0 && !(heights_km_[i] < heights_km_[i-1])))
                throw std::runtime_error("AtmosphericSource: height edges must descend");
        }
        if (!Close(heights_km_.back(),0.0,0.0,1e-8))
            throw std::runtime_error("AtmosphericSource: last height must be ground level");

        const std::vector<hsize_t> prod_shape = {
            static_cast<hsize_t>(abs_coszen_.size()),
            static_cast<hsize_t>(NumberLayers()),
            static_cast<hsize_t>(energy_GeV_.size())
        };
        const std::vector<hsize_t> flux_shape = {
            static_cast<hsize_t>(coszen_.size()),
            static_cast<hsize_t>(energy_GeV_.size())
        };

        for (std::size_t sp=0; sp<NSPECIES; ++sp) {
            const std::string prod_path = "/production/"+std::string(SpeciesNames()[sp]);
            production_[sp] = ReadArray<float>(file.id, prod_path, prod_shape, H5T_NATIVE_FLOAT);
            for (float v: production_[sp]) {
                if (!std::isfinite(v) || v < 0.0f)
                    throw std::runtime_error("AtmosphericSource: negative or non-finite production in "+prod_path);
            }

            // Cross-check layer sums against independently stored surface flux.
            const std::string surface_path = "/flux/"+std::string(SpeciesNames()[sp]);
            const auto flux = ReadArray<double>(file.id, surface_path, flux_shape, H5T_NATIVE_DOUBLE);
            const std::size_t nE = energy_GeV_.size();
            for (std::size_t a=0; a<abs_coszen_.size(); ++a) {
                for (std::size_t e=0; e<nE; ++e) {
                    double sum=0.0;
                    for (std::size_t layer=0; layer<NumberLayers(); ++layer)
                        sum += production_[sp][(a*NumberLayers()+layer)*nE+e];
                    const double expected=flux[(abs_coszen_.size()-1+a)*nE+e];
                    if (!std::isfinite(expected) || expected < 0.0 ||
                        std::abs(sum-expected) > 1e-4*std::max(expected,1e-30))
                        throw std::runtime_error("AtmosphericSource: production does not reconstruct surface flux for "+surface_path);
                }
            }
        }
    }

    const std::string& Filename() const noexcept { return filename_; }
    std::size_t NumberEnergies() const noexcept { return energy_GeV_.size(); }
    std::size_t NumberAngles() const noexcept { return abs_coszen_.size(); }
    std::size_t NumberLayers() const noexcept { return heights_km_.size()-1; }
    double EarthRadiusKm() const noexcept { return earth_radius_km_; }
    double TopHeightKm() const noexcept { return heights_km_.front(); }
    const std::vector<double>& EnergyGeV() const noexcept { return energy_GeV_; }
    const std::vector<double>& AbsCosZenithGrid() const noexcept { return abs_coszen_; }
    const std::vector<double>& HeightEdgesKm() const noexcept { return heights_km_; }

    // One angle-dependent preparation per track; reused for ALL mass/gprime
    // values sharing the same standard-oscillation/source parameters.
    // normalization is the SAME flux normalization used in the propagator.
    AngleSlice PrepareAngle(const EarthDepthBody::Track& track,
                            double normalization = 1.0) const
    {
        if (!(normalization > 0.0) || !std::isfinite(normalization))
            throw std::invalid_argument("AtmosphericSource: invalid normalization");
        if (!Close(track.GetEarthRadiusKm(), earth_radius_km_,1e-9,1e-6))
            throw std::invalid_argument("AtmosphericSource: MCEq and PREM Earth radii differ");
        if (!Close(track.GetAtmosphereHeightKm(), TopHeightKm(),1e-9,1e-6))
            throw std::invalid_argument("AtmosphericSource: set atmosphere height from MCEq before creating track");

        // The source atmosphere is on the incoming side of the Earth. The
        // detector zenith is defined at R_detector, while MCEq's zenith is
        // evaluated at the atmospheric observation surface R_earth.
        // Conserved ray impact parameter: b = R_detector*sin(theta_det).
        const double c = std::clamp(track.GetCosZenith(),-1.0,1.0);
        const double sin_detector=std::sqrt(std::max(0.0,1.0-c*c));
        const double b = track.GetDetectorRadiusKm()*sin_detector;
        const double sin_surface=std::clamp(b/earth_radius_km_,0.0,1.0);
        const double abs_surface_cos=std::sqrt(std::max(0.0,1.0-sin_surface*sin_surface));

        const auto k = LowerIndex(abs_coszen_,abs_surface_cos);
        const double a=(abs_surface_cos-abs_coszen_[k])/(abs_coszen_[k+1]-abs_coszen_[k]);
        const double w=std::clamp(a,0.0,1.0);

        AngleSlice result;
        result.energies_ = energy_GeV_;
        result.normalization_ = normalization;
        result.surface_coszen_abs_ = abs_surface_cos;
        result.layer_boundaries_x_.resize(heights_km_.size());
        for (std::size_t l=0; l<heights_km_.size(); ++l)
            result.layer_boundaries_x_[l]=track.XAtIncomingHeightKm(heights_km_[l]);

        result.layer_boundaries_x_.front()=0.0; // exact track start
        const std::size_t nE = NumberEnergies();
        result.source_per_x_.resize(NumberLayers()*nE*NSPECIES);
        for (std::size_t l=0; l<NumberLayers(); ++l) {
            const double dx=result.layer_boundaries_x_[l+1]-result.layer_boundaries_x_[l];
            if (!std::isfinite(dx) || !(dx>0.0))
                throw std::runtime_error("AtmosphericSource: invalid path length for atmospheric layer");
            for (std::size_t e=0; e<nE; ++e) {
                for (std::size_t s=0; s<NSPECIES; ++s) {
                    const auto index0=(k*NumberLayers()+l)*nE+e;
                    const auto index1=((k+1)*NumberLayers()+l)*nE+e;
                    const double delta=(1.0-w)*production_[s][index0] + w*production_[s][index1];
                    result.source_per_x_[result.Offset(l,e,s)] = delta/(dx*normalization);
                }
            }
        }
        return result;
    }
};

} // namespace lri
