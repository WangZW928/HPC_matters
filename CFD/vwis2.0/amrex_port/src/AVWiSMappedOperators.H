#ifndef AVWIS_MAPPED_OPERATORS_H_
#define AVWIS_MAPPED_OPERATORS_H_

#include "AVWiSMetricData.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <string>
#include <vector>

enum class CoordinateSystemMode { Cartesian, Mapped };
enum class ProjectionOperatorMode { CartesianMLMG, OrthogonalMLMG, DeferredNonorthogonal };

/** Explicit mapping/operator selection. The default preserves the Cartesian path. */
struct MappingOperatorConfig {
    CoordinateSystemMode coordinates = CoordinateSystemMode::Cartesian;
    std::string mapping_type = "identity";
    ProjectionOperatorMode projection = ProjectionOperatorMode::CartesianMLMG;
    AnalyticOrthogonalMappingParameters analytic_parameters{};
    amrex::Real shear_epsilon = 0.0;
    int deferred_max_iterations = 40;
    // Only the opt-in incremental Cartesian/orthogonal solve uses this budget.
    int diagonal_max_iterations = 1600;
    // The default preserves the experimental transported-cell/face-mass path.
    std::string state_contract = "experimental_dual_state";
    std::string pressure_contract = "accumulated_correction";
    std::string failure_contract = "fatal";
};

enum class MetricFluxPart { Full, Diagonal, Cross };

/** Shared stencil for pressure and component diffusion. Caller supplies fresh
 * corner halos. Normal two-cell derivative, tangential centered average.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real metric_face_gradient_value(
    amrex::Array4<amrex::Real const> const& phi,
    amrex::Array4<amrex::Real const> const& q,
    int i, int j, int k, int direction, int component,
    amrex::GpuArray<amrex::Real, 3> const& inverse_h,
    MetricFluxPart part = MetricFluxPart::Full) noexcept
{
    int const il=i-(direction==0), jl=j-(direction==1), kl=k-(direction==2);
    amrex::Real value = part == MetricFluxPart::Cross ? 0.0 :
        q(i,j,k,direction)*(phi(i,j,k,component)-phi(il,jl,kl,component))*inverse_h[direction];
    if (part != MetricFluxPart::Diagonal) {
        for (int t=0;t<3;++t) {
            if (t==direction) continue;
            int const di=t==0, dj=t==1, dk=t==2;
            value += q(i,j,k,t)*0.25*inverse_h[t]*
                (phi(i+di,j+dj,k+dk,component)-phi(i-di,j-dj,k-dk,component)+
                 phi(il+di,jl+dj,kl+dk,component)-phi(il-di,jl-dj,kl-dk,component));
        }
    }
    return value;
}

struct DeferredProjectionResult {
    int iterations = 0;
    amrex::Real full_relative_residual = 0.0;
    amrex::Real full_max_residual = 0.0;
    int inner_iterations = 0;
};

/** Opt-in observation only; no additional reductions or changed stopping rule. */
struct DeferredProjectionIteration {
    int outer, inner;
    amrex::Real inner_initial, inner_final, removed_mean;
    amrex::Real full_max, full_relative;
};
using DeferredProjectionTrace = std::vector<DeferredProjectionIteration>;

/** Solver-owned scratch storage bound to one immutable metric object/epoch/layout.
 * No flow state or checkpoint payload. Values are overwritten for every solve,
 * including after failure; only allocations are reused, not MLMG/hierarchy state.
 * Also supplies candidate/RHS storage for periodic diagonal incremental solves.
 * A workspace may not be shared concurrently or reused after metric rebuild.
 */
struct DeferredProjectionWorkspace {
    explicit DeferredProjectionWorkspace(MetricData const& metric);
private:
    friend class AVWiSSolver;
    friend class AVWiSContractTestAccess;
    friend DeferredProjectionResult solve_deferred_metric_poisson(
        amrex::MultiFab&,amrex::MultiFab const&,MetricData const&,std::uint64_t,
        MappingOperatorConfig const&,amrex::Geometry const&,DeferredProjectionWorkspace&,
        DeferredProjectionTrace*);
    MetricData const* const metric_owner;
    std::uint64_t const metric_epoch;
    amrex::Array<amrex::MultiFab,3> b, flux;
    amrex::MultiFab bc, work_rhs, action, residual, physical_rhs, weighted;
    amrex::MultiFab cell_gradient;
    // Face-authority predictor and pre-commit projection candidates. Never
    // checkpointed; partial numerical solves do not publish these candidates.
    amrex::MultiFab predictor_rate, reconstructed_cell;
    amrex::Array<amrex::MultiFab,3> face_candidate;
};

CoordinateSystemMode parse_coordinate_system_mode(std::string const& name);
ProjectionOperatorMode parse_projection_operator_mode(std::string const& name);
char const* coordinate_system_mode_name(CoordinateSystemMode mode) noexcept;
char const* projection_operator_mode_name(ProjectionOperatorMode mode) noexcept;

/** Reject unsupported combinations and stale or differently mapped MetricData. */
void validate_mapping_operator_config(MappingOperatorConfig const& config,
                                      MetricData const& metric,
                                      std::uint64_t expected_metric_epoch);

/** Finite-volume divergence of already integrated face flux: net flux / V exactly once. */
void compute_metric_divergence(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& integrated_flux,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& divergence);

/** Signed cell net integrated flux, without a divide/multiply volume round trip.
 * Face-authoritative pressure RHS and deferred cross action share this path.
 * Direction differences are added in x,y,z order; no compatibility mean repair.
 */
void compute_metric_net_flux(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& integrated_flux,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& net_flux);

/** Cell physical gradient using centered logical differences and grad(xi) at the cell. */
void compute_metric_cell_gradient(amrex::MultiFab const& scalar,
                                  MetricData const& metric,
                                  std::uint64_t expected_metric_epoch,
                                  MappingOperatorConfig const& config,
                                  amrex::MultiFab& gradient);

/** Full/diagonal/cross integrated pressure-gradient flux; owner synchronized. */
void compute_metric_face_gradient_flux(amrex::MultiFab const& scalar, int direction,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::MultiFab& face_flux, MetricFluxPart part = MetricFluxPart::Full);

/** Periodic -V*L(phi)=integrated_rhs, using diagonal MLMG preconditioning.
 * Reject incompatible RHS and exhaustion; return only after full physical
 * residual passes 1e-11 relative. No silent solvability repair.
 */
DeferredProjectionResult solve_deferred_metric_poisson(
    amrex::MultiFab& phi, amrex::MultiFab const& integrated_rhs,
    MetricData const& metric, std::uint64_t epoch,
    MappingOperatorConfig const& config, amrex::Geometry const& geometry);

/** Storage-reusing variant. Same numerical/failure contract; workspace values
 * and Phi may change on failure, while the caller owns flow-state commit.
 */
DeferredProjectionResult solve_deferred_metric_poisson(
    amrex::MultiFab& phi, amrex::MultiFab const& integrated_rhs,
    MetricData const& metric, std::uint64_t epoch,
    MappingOperatorConfig const& config, amrex::Geometry const& geometry,
    DeferredProjectionWorkspace& workspace, DeferredProjectionTrace* trace = nullptr);

/** Orthogonal pressure face flux G^m=beta Q^{mm}(phi_R-phi_L)/h_m. */
void compute_orthogonal_face_gradient_flux(amrex::MultiFab const& scalar,
                                           int direction,
                                           MetricData const& metric,
                                           std::uint64_t expected_metric_epoch,
                                           MappingOperatorConfig const& config,
                                           amrex::MultiFab& face_flux);

/** Supported orthogonal/fixed-shear velocity transforms. Reconstruction is the
 * normalized six-face LS specialization for constant shear normals. */
void sync_metric_ucont_from_ucat(amrex::MultiFab const& ucat,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::Periodicity const& periodicity);
void sync_metric_ucat_from_ucont(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::MultiFab& ucat);

/** Compatibility entry for orthogonal/separable velocity transforms. */
void sync_orthogonal_ucont_from_ucat(
    amrex::MultiFab const& ucat,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::Periodicity const& periodicity);

/** Orthogonal/separable integrated face flux to cell Cartesian velocity. */
void sync_orthogonal_ucat_from_ucont(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& ucat);

#endif
