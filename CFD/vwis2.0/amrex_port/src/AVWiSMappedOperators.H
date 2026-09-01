#ifndef AVWIS_MAPPED_OPERATORS_H_
#define AVWIS_MAPPED_OPERATORS_H_

#include "AVWiSMetricData.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <string>

enum class CoordinateSystemMode { Cartesian, Mapped };
enum class ProjectionOperatorMode { CartesianMLMG, OrthogonalMLMG };

/** Explicit mapping/operator selection. The default preserves the Cartesian path. */
struct MappingOperatorConfig {
    CoordinateSystemMode coordinates = CoordinateSystemMode::Cartesian;
    std::string mapping_type = "identity";
    ProjectionOperatorMode projection = ProjectionOperatorMode::CartesianMLMG;
    AnalyticOrthogonalMappingParameters analytic_parameters{};
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

/** Cell physical gradient using centered logical differences and grad(xi) at the cell. */
void compute_metric_cell_gradient(amrex::MultiFab const& scalar,
                                  MetricData const& metric,
                                  std::uint64_t expected_metric_epoch,
                                  MappingOperatorConfig const& config,
                                  amrex::MultiFab& gradient);

/** Orthogonal pressure face flux G^m=beta Q^{mm}(phi_R-phi_L)/h_m. */
void compute_orthogonal_face_gradient_flux(amrex::MultiFab const& scalar,
                                           int direction,
                                           MetricData const& metric,
                                           std::uint64_t expected_metric_epoch,
                                           MappingOperatorConfig const& config,
                                           amrex::MultiFab& face_flux);

/** Orthogonal/separable cell Cartesian velocity to integrated mapped face flux. */
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
