#pragma once

// ============================================================================
// LRI/propagation/LRI.h
// ============================================================================
// Three-flavor atmospheric neutrino propagation using nuSQuIDS.
//
// Depends on:
//   ../potential/PotentialLibrary.h   (electron/neutron Yukawa integrals)
//   EarthDepth.h                      (DeepCore geometry and vacuum atmosphere)
//   AtmosphericSource.h               (MCEq production by height)
//
// PHYSICS:
//
//   H_nu     = H_standard_nu     + V_LRI(R) Q_model
//   H_antinu = H_standard_antinu - V_LRI(R) Q_model
//
//   Q(Le-Lmu)  = diag(+1,-1, 0)     V_LRI = (g')^2 I_e(R,m)
//   Q(Le-Ltau) = diag(+1, 0,-1)     V_LRI = (g')^2 I_e(R,m)
//   Q(Lmu-Ltau)= diag( 0,+1,-1)     V_LRI = G_mu_tau I_n(R,m)
//
//   G_mu_tau = e g' (xi - sin(theta_W) chi) / (sin(theta_W) cos(theta_W))
//
//   This is the prefactor convention of PotentialLibrary.h: I_e and I_n
//   ALREADY include the angular 1/(4*pi) factor of the Yukawa kernel.
//   Do not multiply by 1/(4*pi) again. G_mu_tau can have either sign.
//
//   Standard MSW and collisions within Earth are handled by nuSQuIDS.
//   In the atmosphere: rho=0 and V_LRI=0 (model approximation, NOT the
//   exact exterior Yukawa field). Vacuum oscillations remain active.
//
// ATMOSPHERIC PRODUCTION:
//   AtmosphericEarthBody implements the *native nuSQuIDS injection hook*.
//   SourceAtNode() returns dPhi/dx on the INCOMING atmospheric segment.
//   The normalizations of source and nuSQuIDS initial state must match.
//   Start the propagation with ZERO initial flux when enabling sources,
//   to avoid double-counting the MCEq-produced flux.
//
//   IMPORTANT: the injection profile is piecewise constant in x.
//   In main.cpp, control the numerical integration at atmospheric layer
//   boundaries (or validate a sufficiently small step size). Do NOT allow
//   an adaptive integrator to skip entire thin production layers.
//
// No YAML; no likelihood/Feldman-Cousins; this file does not run fits.
// ============================================================================

#include "EarthDepth.h"
#include "AtmosphericSource.h"
#include "../potential/PotentialLibrary.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nuSQuIDS/nuSQuIDS.h>

// ============================================================================
// AtmosphericEarthBody
// ============================================================================
// This is an EarthDepthBody with the nuSQuIDS injection callback enabled.
// One body and one AngleSlice belong to ONE propagation trajectory.
// Never mutate the AngleSlice while nuSQuIDS is evolving.
// ============================================================================

class AtmosphericEarthBody final : public EarthDepthBody
{
private:
    std::shared_ptr<const lri::AtmosphericSource::AngleSlice> source_;
    nusquids::NeutrinoType neutrino_type_ = nusquids::both;

public:
    explicit AtmosphericEarthBody(double detector_depth_km = 1.5)
        : EarthDepthBody(detector_depth_km)
    {
    }

    AtmosphericEarthBody(const std::string& earth_model_file,
                         double detector_depth_km = 1.5)
        : EarthDepthBody(earth_model_file, detector_depth_km)
    {
    }

    void SetAtmosphericSource(
        std::shared_ptr<const lri::AtmosphericSource::AngleSlice> source)
    {
        if (!source)
            throw std::invalid_argument(
                "AtmosphericEarthBody: la fuente atmosferica es nula.");
        source_ = std::move(source);
    }

    bool HasAtmosphericSource() const noexcept
    {
        return static_cast<bool>(source_);
    }

    const lri::AtmosphericSource::AngleSlice& GetAtmosphericSource() const
    {
        if (!source_)
            throw std::logic_error(
                "AtmosphericEarthBody: no se ha configurado una fuente.");
        return *source_;
    }

    // COMPATIBILITY:
    // The installed nuSQuIDS build can use an EarthAtm with only the
    // density and electron-fraction (Ye) PREM splines, without an isotope
    // composition() interface. Do not reference newer nuSQuIDS members
    // such as inter_composition here: they may not exist in this build.
    // The physical matter profile for the current pipeline is supplied
    // by EarthDepthBody::density() and EarthDepthBody::ye().

    // Older nuSQuIDS releases have no public GetNeutrinoType() getter.
    // The body stores this value separately for its injection callback.
    // Defaults to both (as required by propagation/main.cpp).
    void SetNeutrinoType(nusquids::NeutrinoType type) noexcept
    {
        neutrino_type_ = type;
    }

    // Shape: [energy, rho, flavor]. rho=0 for neutrinos, rho=1 for
    // antineutrinos when NeutrinoType=both. For a single component the
    // rho index is 0, irrespective of neutrino vs antineutrino.
    void injected_neutrino_flux(
        nusquids::marray<double, 3>& flux,
        const nusquids::GenericTrack& generic_track,
        const nusquids::nuSQUIDS& engine) override
    {
        // Mandatory: clear previous values at EVERY derivative evaluation.
        std::fill(flux.begin(), flux.end(), 0.0);

        if (!source_)
            return;

        const double x = generic_track.GetX();
        if (source_->LayerAtX(x) < 0)
            return;

        // The engine API does not expose GetNeutrinoType() in every
        // nuSQuIDS version. Its type is configured with SetNeutrinoType().
        (void)engine;
        const auto type = neutrino_type_;
        const std::size_t nE = source_->NumberEnergies();

        if (flux.extent(0) != nE || flux.extent(2) != 3 ||
            flux.extent(1) != (type == nusquids::both ? 2u : 1u))
            throw std::runtime_error(
                "AtmosphericEarthBody: shape de inyeccion incompatible.");

        for (std::size_t ie = 0; ie < nE; ++ie)
        {
            const auto s = source_->SourceAtNode(ie, x);

            if (type == nusquids::neutrino || type == nusquids::both)
            {
                for (std::size_t flavor = 0; flavor < 3; ++flavor)
                    flux[ie][0][flavor] = s[flavor];
            }

            if (type == nusquids::antineutrino || type == nusquids::both)
            {
                const std::size_t rho =
                    (type == nusquids::both) ? 1u : 0u;
                for (std::size_t flavor = 0; flavor < 3; ++flavor)
                    flux[ie][rho][flavor] = s[flavor + 3];
            }
        }
    }
};

// ============================================================================
// nuSQUIDSLRI
// ============================================================================
// One object per propagation track and parameter point. The immutable radial
// profile may be shared across tracks/threads. Mass interpolation occurs when
// PotentialLibrary::BuildRadialProfile is called, not inside EvolveState().
//
// Rather than manually rotating Q to the mass basis, the class constructs
// Q_I from the already-evolved flavor projectors in nuSQuIDS. This matters
// when CP phases differ between neutrinos and antineutrinos.
// ============================================================================

class nuSQUIDSLRI final : public nusquids::nuSQUIDS
{
private:
    std::shared_ptr<const lri::RadialProfile> profile_;
    lri::Model model_;
    std::array<double, 3> charges_;
    lri::Couplings couplings_;

    // Optional direct G_mu_tau, instead of computing it from xi/chi/gprime.
    // No additional factor of gprime is applied in this mode.
    bool direct_big_g_ = false;
    double big_g_override_ = 0.0;
    double effective_prefactor_ = 0.0;

    std::shared_ptr<EarthDepthBody> earth_body_;
    std::shared_ptr<EarthDepthBody::Track> earth_track_;

    double current_radius_km_ = std::numeric_limits<double>::quiet_NaN();
    double current_V_unit_eV_ = 0.0;
    double current_V_LRI_eV_ = 0.0;

    static bool SameLength(double a, double b)
    {
        return std::abs(a - b) <=
            1.0e-6 + 1.0e-9 * std::max(std::abs(a), std::abs(b));
    }

    void ValidateProfile() const
    {
        if (!profile_)
            throw std::invalid_argument("nuSQUIDSLRI: perfil radial nulo.");
        if (profile_->GetModel() != model_)
            throw std::invalid_argument("nuSQUIDSLRI: modelo y perfil diferentes.");
        if (GetNumNeu() != 3)
            throw std::invalid_argument("nuSQUIDSLRI requiere 3 sabores.");
    }

    void RefreshPrefactor()
    {
        if (direct_big_g_)
        {
            if (model_ != lri::Model::LmuLtau)
                throw std::logic_error("G directo solo es valido para Lmu-Ltau.");
            effective_prefactor_ = big_g_override_;
        }
        else
        {
            effective_prefactor_ = lri::EffectivePrefactor(model_, couplings_);
        }

        if (!std::isfinite(effective_prefactor_))
            throw std::runtime_error("nuSQUIDSLRI: prefactor LRI no finito.");
    }

    void ValidateSourceEnergyGrid(const AtmosphericEarthBody& body) const
    {
        if (!body.HasAtmosphericSource())
            return;

        const auto& energies = body.GetAtmosphericSource().EnergyGeV();
        if (energies.size() != GetNumE())
            throw std::invalid_argument(
                "nuSQUIDSLRI: la malla energetica de MCEq no coincide con nuSQuIDS.");

        const auto E_nusquids = GetERange();
        squids::Const units;
        for (std::size_t i = 0; i < energies.size(); ++i)
        {
            const double expected_eV = energies[i] * units.GeV;
            const double actual_eV = E_nusquids[i];
            const double tolerance = 1.0e-9 * expected_eV;
            if (!std::isfinite(actual_eV) ||
                std::abs(actual_eV - expected_eV) > tolerance)
                throw std::invalid_argument(
                    "nuSQUIDSLRI: nodos E de nuSQuIDS y MCEq distintos.");
        }
    }

    void UpdateCurrentLRI()
    {
        if (!earth_track_ || !earth_body_)
            throw std::logic_error(
                "nuSQUIDSLRI: usar Set_Body y Set_Track antes de evolucionar.");

        // nuSQuIDS::PreDerive already updated the common Track with its
        // internal time offset; do NOT call SetX(x) again here.
        current_radius_km_ = earth_track_->RadiusKm();
        current_V_unit_eV_ = profile_->UnitAtRadiusKm(current_radius_km_);
        current_V_LRI_eV_ = effective_prefactor_ * current_V_unit_eV_;

        if (!std::isfinite(current_V_LRI_eV_))
            throw std::runtime_error("nuSQUIDSLRI: potencial LRI no finito.");
        // V_LRI can be negative in Lmu-Ltau when G_mu_tau < 0.
    }

protected:
    void AddToPreDerive(double x) override
    {
        (void)x;
        UpdateCurrentLRI();
    }

    squids::SU_vector HI(unsigned int ie, unsigned int irho) const override
    {
        // Base HI already contains vacuum evolution and MSW matter effects.
        // The MSW term changes sign internally for antineutrinos.
        auto hamiltonian = nusquids::nuSQUIDS::HI(ie, irho);

        if (current_V_LRI_eV_ == 0.0)
            return hamiltonian;

        // Flavor-charge operator in the nuSQuIDS interaction picture:
        // Q_I = q_e P_e,I + q_mu P_mu,I + q_tau P_tau,I
        // Use the rho-specific projectors to preserve the correct CP phase.
        // Explicit SU_vector: 'auto' would deduce MultiplicationProxy,
        // which does not support operator+= (SQuIDS expression templates).
        squids::SU_vector charge_operator =
            charges_[0] * evol_b1_proj[irho][0][ie];
        charge_operator += charges_[1] * evol_b1_proj[irho][1][ie];
        charge_operator += charges_[2] * evol_b1_proj[irho][2][ie];

        const bool is_antineutrino =
            (NT == nusquids::antineutrino) ||
            (NT == nusquids::both && irho == 1);

        const double sign = is_antineutrino ? -1.0 : +1.0;
        hamiltonian += (sign * current_V_LRI_eV_) * charge_operator;
        return hamiltonian;
    }

public:
    // No need to pre-rotate the charge operator. nuSQuIDS handles mixing
    // matrices, including delta_CP, through its flavor projectors.
    nuSQUIDSLRI(
        std::shared_ptr<const lri::RadialProfile> profile,
        const lri::Couplings& couplings,
        nusquids::marray<double, 1> energy_range,
        nusquids::NeutrinoType neutrino_type = nusquids::both,
        bool interactions = true)
        : nusquids::nuSQUIDS(
              std::move(energy_range), 3, neutrino_type, interactions),
          profile_(std::move(profile)),
          model_(profile_ ? profile_->GetModel() : lri::Model::LeLmu),
          charges_(lri::FlavorCharges(model_)),
          couplings_(couplings)
    {
        ValidateProfile();
        RefreshPrefactor();
    }

    // Set_Body: enforce exactly the geometry used by the potential LUT.
    // An AtmosphericEarthBody automatically enables MCEq injection when it
    // has an attached AngleSlice. A plain EarthDepthBody does not inject.
    void Set_Body(const std::shared_ptr<EarthDepthBody>& body)
    {
        if (!body)
            throw std::invalid_argument("nuSQUIDSLRI::Set_Body: body nulo.");
        if (!SameLength(body->GetEarthRadiusKm(), profile_->GetEarthRadiusKm()))
            throw std::invalid_argument(
                "nuSQUIDSLRI::Set_Body: R_earth no coincide con LUT.");

        auto emitting = std::dynamic_pointer_cast<AtmosphericEarthBody>(body);
        if (emitting)
            emitting->SetNeutrinoType(NT);
        if (emitting && emitting->HasAtmosphericSource())
        {
            ValidateSourceEnergyGrid(*emitting);
            const auto& boundaries =
                emitting->GetAtmosphericSource().LayerBoundariesX();
            if (boundaries.size() < 2 || !std::isfinite(boundaries.back()) ||
                boundaries.back() <= 0.0)
                throw std::invalid_argument(
                    "nuSQUIDSLRI: bordes de fuente atmosferica invalidos.");
        }

        earth_body_ = body;
        nusquids::nuSQUIDS::Set_Body(body);
        nusquids::nuSQUIDS::Set_NeutrinoSources(
            emitting && emitting->HasAtmosphericSource());
    }

    void Set_Track(const std::shared_ptr<EarthDepthBody::Track>& track)
    {
        if (!track)
            throw std::invalid_argument("nuSQUIDSLRI::Set_Track: track nulo.");
        if (!earth_body_)
            throw std::logic_error("nuSQUIDSLRI::Set_Track: primero Set_Body.");
        if (!SameLength(track->GetEarthRadiusKm(), earth_body_->GetEarthRadiusKm()) ||
            !SameLength(track->GetDetectorDepthKm(), earth_body_->GetDetectorDepthKm()) ||
            !SameLength(track->GetAtmosphereHeightKm(), earth_body_->GetAtmosphereHeightKm()))
            throw std::invalid_argument("nuSQUIDSLRI::Set_Track: geometria no coincide con body.");

        // The injected source must have been prepared with THIS trajectory.
        auto emitting = std::dynamic_pointer_cast<AtmosphericEarthBody>(earth_body_);
        if (emitting && emitting->HasAtmosphericSource())
        {
            const double x_surface = track->XAtEarthEntry();
            const double x_source =
                emitting->GetAtmosphericSource().LayerBoundariesX().back();
            const double tol = 1.0e-8 * std::max(1.0, std::abs(x_surface));
            if (!std::isfinite(x_source) || std::abs(x_source - x_surface) > tol)
                throw std::invalid_argument(
                    "nuSQUIDSLRI::Set_Track: fuente MCEq preparada para otra trayectoria.");
        }

        earth_track_ = track;
        nusquids::nuSQUIDS::Set_Track(track);
    }

    // Load a full, coherent 3x3 density matrix in the mass basis at the
    // START OF THE EARTH SEGMENT. Layout [energy, rho, 9 SU(3) components].
    // These are normalized dimensionless densities; multiply output by
    // the corresponding MCEq normalization only AFTER propagation.
    // A flavor-diagonal Set_initial_state would erase off-diagonal phases.
    void SetCoherentInitialState(const std::vector<double>& components)
    {
        const std::size_t nE = GetNumE();
        if (components.size() != nE * 2u * 9u)
            throw std::invalid_argument(
                "nuSQUIDSLRI: invalid full density-state shape (expected [E,2,9]).");

        nusquids::marray<double,3> zeros({nE, 2u, 3u});
        std::fill(zeros.begin(), zeros.end(), 0.0);
        nusquids::nuSQUIDS::Set_initial_state(zeros, nusquids::mass);

        for (std::size_t ie=0; ie<nE; ++ie)
            for (std::size_t rho=0; rho<2; ++rho) {
                const auto first = components.begin() + (ie*2u+rho)*9u;
                const std::vector<double> values(first, first+9u);
                for (double v : values)
                    if (!std::isfinite(v))
                        throw std::invalid_argument("nuSQUIDSLRI: nonfinite surface state.");
                for (std::size_t k=0;k<9u;++k)
                    state[ie].rho[rho][static_cast<unsigned>(k)] = values[k];
            }
    }

    // Change coupling without rebuilding the interpolated mass profile.
    void SetCouplings(const lri::Couplings& couplings)
    {
        couplings_ = couplings;
        direct_big_g_ = false;
        RefreshPrefactor();
    }

    void Set_gprime(double gprime)
    {
        lri::Couplings updated = couplings_;
        updated.gprime = gprime;
        SetCouplings(updated);
    }

    // Explicit G_mu_tau = e*g'*(xi-sin(thetaW)*chi)/(sin(thetaW)*cos(thetaW)).
    // This setter is useful when G_mu_tau is already computed outside.
    // It is NOT gprime, and is NOT squared or multiplied again.
    void SetBigG(double big_g)
    {
        if (model_ != lri::Model::LmuLtau)
            throw std::logic_error("SetBigG solo corresponde a Lmu-Ltau.");
        if (!std::isfinite(big_g))
            throw std::invalid_argument("SetBigG: G_mu_tau no finito.");
        direct_big_g_ = true;
        big_g_override_ = big_g;
        RefreshPrefactor();
    }

    // Reuse the same engine for a different mediator mass, after the caller
    // has built/interpolated the new radial profile outside EvolveState().
    void SetRadialProfile(std::shared_ptr<const lri::RadialProfile> profile)
    {
        if (!profile)
            throw std::invalid_argument("SetRadialProfile: perfil nulo.");
        if (profile->GetModel() != model_)
            throw std::invalid_argument(
                "SetRadialProfile: no cambiar de modelo en el mismo objeto.");
        if (earth_body_ &&
            !SameLength(profile->GetEarthRadiusKm(), earth_body_->GetEarthRadiusKm()))
            throw std::invalid_argument("SetRadialProfile: radio terrestre incompatible.");
        profile_ = std::move(profile);
    }

    lri::Model GetModel() const noexcept { return model_; }
    double GetMediatorMassEV() const { return profile_->GetMassEV(); }
    double Get_gprime() const noexcept { return couplings_.gprime; }
    double GetEffectivePrefactor() const noexcept { return effective_prefactor_; }
    double GetBigG() const
    {
        if (model_ != lri::Model::LmuLtau)
            throw std::logic_error("GetBigG solo corresponde a Lmu-Ltau.");
        return effective_prefactor_;
    }

    const lri::Couplings& GetCouplings() const noexcept { return couplings_; }
    double GetCurrentRadiusKm() const noexcept { return current_radius_km_; }
    double GetCurrentVUnitEV() const noexcept { return current_V_unit_eV_; }
    double GetCurrentVLRIEV() const noexcept { return current_V_LRI_eV_; }

    double EvalVUnitAtRadiusKm(double r_km) const
    {
        return profile_->UnitAtRadiusKm(r_km);
    }

    double EvalVLRIAtRadiusKm(double r_km) const
    {
        return effective_prefactor_ * EvalVUnitAtRadiusKm(r_km);
    }
};
