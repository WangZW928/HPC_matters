#ifndef AVWIS_SOLVER_H_
#define AVWIS_SOLVER_H_

#include "CartesianBoundaryConfig.H"
#include "AVWiSMetricData.H"
#include "AVWiSMappedOperators.H"

#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <string>
#include <array>
#include <cstdint>
#include <vector>

/** Physical staggering of a registered field; fields are never implicitly converted. */
enum class FieldLocation { Cell, XFace, YFace, ZFace };

struct P3Diagnostics {
    std::uint64_t valid_epoch = 0;
    std::uint64_t halo_epoch = 0;
    std::uint64_t physical_epoch = 0;
    amrex::Real outward_boundary_flux = 0.0;
    amrex::Real integrated_divergence = 0.0;
    amrex::Real max_abs_divergence = 0.0;
};

struct ProjectionDiagnostics {
    amrex::Real max_divergence_before = 0.0;
    amrex::Real max_divergence_after = 0.0;
    amrex::Real rhs_mean = 0.0;
    amrex::Real final_residual = 0.0;
    bool singular = false;
};

struct TimeStepDiagnostics {
    amrex::Real advective_cfl = 0.0;
    amrex::Real diffusive_number = 0.0;
    amrex::Real projection_time_coefficient = 1.0;
};

/** P8-003 cell-containing point sample on the uniform Cartesian level. */
struct UniformPointSample {
    amrex::Array<int, AMREX_SPACEDIM> cell{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> cell_center{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> velocity{};
    amrex::Real pressure = 0.0;
    amrex::Real divergence = 0.0;
};

/** P8-003 area-weighted cell section summary plus collocated normal flow. */
struct UniformPlaneStatistics {
    int direction = 0;
    int cell_index = 0;
    amrex::Real coordinate = 0.0;
    amrex::Long cell_count = 0;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> mean_velocity{};
    amrex::Real mean_pressure = 0.0;
    amrex::Real normal_flow = 0.0;
};

/** Globally reduced instantaneous integral/statistical diagnostics. */
struct UniformFlowDiagnostics {
    amrex::Real integrated_divergence = 0.0;
    amrex::Real max_abs_divergence = 0.0;
    amrex::Real net_mass_flux = 0.0;
    amrex::Real outlet_flow = 0.0;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> momentum{};
    amrex::Real kinetic_energy = 0.0;
    amrex::Real pressure_mean = 0.0;
    amrex::Real pressure_min = 0.0;
    amrex::Real pressure_max = 0.0;
};

/**
 * P0--P2 field contract. `ghost_cells` is allocated same-level halo width;
 * ghost reads require fill_ghost_cells() after a valid-region write.  The
 * `time_layer` is descriptive metadata, not evidence of an integrator.
 */
struct FieldSpec {
    std::string name;
    FieldLocation location;
    int components;
    int ghost_cells;
    std::string units;
    std::string time_layer;
    std::string component_names;
    std::string ownership;
};

/** Cartesian metric names and values; no curvilinear metric is implied. */
struct MetricSpec {
    std::string name;
    std::string location;
    std::string value;
    std::string meaning;
};

/**
 * Single-level Cartesian P0--P8 data owner. MultiFabs remain owned here for the
 * complete solver lifetime, while tile Array4 views never escape an MFIter.
 * Thus DistributionMapping is the sole MPI ownership authority; valid cells
 * may be written independently and only fill_ghost_cells() exchanges halos.
 * P4 adds a single-level Cartesian pressure projection of integrated face
 * volume flux. P5 adds conservative Cartesian advective and constant-
 * coefficient viscous RHS operators and a minimal explicit fractional step; no variable viscosity,
 * curvilinear operator, LES, IBM, or FSI exists.
 */
class AVWiSSolver {
public:
    /**
     * Define the sole Cartesian level. n_cell is cell count, max_grid_size controls
     * BoxArray splitting, nghost allocates halos for every registered field,
     * and is_periodic controls only geometry/halo wrapping. physical_domain is
     * Cartesian geometry, never a curvilinear metric input.
     */
    AVWiSSolver(amrex::Vector<int> const& n_cell, int max_grid_size,
                    int nghost, amrex::RealBox const& physical_domain,
                    amrex::Vector<int> const& is_periodic,
                    CartesianBoundaryConfig const& boundary = {},
                    MappingOperatorConfig const& mapping_operator = {});

    /** Initialize valid state to zero and, when configured, execute the P3 boundary pipeline. */
    void initialize();
    /** Sync shared face owners, then exchange same-level/MPI/periodic ghosts only. */
    void fill_ghost_cells();
    /** Fill non-periodic physical ghosts and impose boundary-normal Ucont faces. */
    void fill_physical_ghost_cells();
    /** Required order: same-level/MPI/periodic halo, then physical ghosts/faces. */
    void apply_boundary_pipeline(char const* stage);
    /** Provisional explicit Euler predictor plus Cartesian projection; not legacy SNES. */
    void advance_one_step(amrex::Real dt, amrex::Real viscosity = 0.1);
    /** Conservative explicit-step diagnostics; limits are guards, not a nonlinear stability proof. */
    TimeStepDiagnostics time_step_diagnostics(amrex::Real dt, amrex::Real viscosity) const;
    /** Recover Cartesian velocity by averaging adjacent face volume-flux densities. */
    void sync_ucat_from_ucont();
    /** Build Cartesian face volume flux from interpolated normal velocity times area. */
    void sync_ucont_from_ucat();
    /** Solve the P4 Cartesian pressure correction and project integrated face fluxes. */
    ProjectionDiagnostics project_cartesian(amrex::Real dt, amrex::Real time_coefficient = 1.0);
    /** C2.2 periodic diagonal/separable mapped pressure projection. */
    ProjectionDiagnostics project_orthogonal(amrex::Real dt, amrex::Real time_coefficient = 1.0);
    /** Conservative Cartesian -div(Ucont[d] * face_average(Ucat)) / cell volume. */
    void compute_cartesian_advection_rhs(amrex::MultiFab& rhs);
    /** Uniform Cartesian constant-coefficient nu*Laplacian(Ucat), component-wise. */
    void compute_cartesian_viscous_rhs(amrex::MultiFab& rhs, amrex::Real viscosity);
    /** Globally reduced P3 boundary/divergence/freshness report; diagnostic only, never a RHS. */
    P3Diagnostics p3_diagnostics(char const* stage, bool require_fresh = true) const;
    /** IO-rank layout/schema diagnostics, safe after initialization or a completed step. */
    void diagnostics() const;
    /** Write schema-only metadata; this never writes a plotfile/restart payload. */
    void write_metadata_manifest(std::string const& path) const;
    /** Write an AMReX VisMF checkpoint plus a strict, versioned Header manifest. */
    void write_checkpoint(std::string const& path) const;
    /** Restore every persistent Cartesian field and time layer from a strict checkpoint. */
    void read_checkpoint(std::string const& path);
    /** Sample the containing cell; the current P8-003 implementation is CPU/single-rank. */
    UniformPointSample sample_uniform_point(
        amrex::Array<amrex::Real, AMREX_SPACEDIM> const& position) const;
    /** Reduce a cell plane and its collocated integrated normal face flux. */
    UniformPlaneStatistics uniform_plane_statistics(int direction, int cell_index) const;
    /** Write every cell on one section as deterministic CSV (not an AMReX plotfile). */
    void write_uniform_plane_csv(std::string const& path, int direction, int cell_index) const;
    /** Compute conservation, energy, momentum, and pressure statistics. */
    UniformFlowDiagnostics uniform_flow_diagnostics() const;
    /** Read-only identity metric owned for the complete solver lifetime. */
    MetricData const& metric_data() const noexcept { return m_metric_data; }
    /** Epoch frozen when the solver's identity metric was successfully built. */
    std::uint64_t metric_epoch() const noexcept { return m_metric_epoch; }
    MappingOperatorConfig const& mapping_operator_config() const noexcept {
        return m_mapping_operator;
    }

private:
    // Case setup/export must coordinate solver-owned fields. One internal
    // bridge avoids exposing raw MultiFabs or case-specific declarations here.
    friend class AVWiSCaseRunnerAccess;
    friend class AVWiSContractTestAccess;

    void register_fields();
    void register_metric_metadata();
    void define_boundary_metadata();
    void validate_boundary_config() const;
    void validate_projection_boundary_policy() const;
    void mark_valid_modified();
    void require_ghosts_fresh(char const* consumer) const;
    amrex::Real boundary_flux(int dir, bool high) const;
    amrex::Real inlet_profile_integral(int dir, bool high) const;
    void fill_physical_ghost_cells_impl(bool impose_boundary_flux);
    void sync_ucat_from_ucont_impl(bool reapply_boundary_flux);
    void compute_cartesian_divergence(amrex::MultiFab& divergence) const;
    amrex::Real validate_singular_rhs_compatibility(amrex::MultiFab const& rhs) const;

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;
    MetricData m_metric_data;
    std::uint64_t m_metric_epoch = 0;
    MappingOperatorConfig m_mapping_operator;

    // Cell IndexType. Nvert remains a legacy classifier, never an EB volume fraction.
    amrex::MultiFab m_p;
    amrex::MultiFab m_phi;
    amrex::MultiFab m_nvert;
    amrex::MultiFab m_ucat;
    amrex::MultiFab m_ucat_old;
    amrex::MultiFab m_ucat_older;
    amrex::MultiFab m_projection_rhs;
    amrex::MultiFab m_projection_bc;

    // MAC face IndexType: each field has one normal volume-flux component.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_old;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_older;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_phi_operator_flux;
    std::vector<FieldSpec> m_fields;
    std::vector<MetricSpec> m_metrics;
    std::vector<amrex::BCRec> m_cell_bcs;
    CartesianBoundaryConfig m_boundary;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_dx{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_face_area{};
    amrex::Real m_cell_volume = 0.0;
    int m_nghost;
    amrex::Real m_initialize_seconds = 0.0;
    amrex::Real m_last_advance_seconds = 0.0;
    amrex::Real m_time = 0.0;
    std::uint64_t m_step = 0;
    int m_history_depth = 1;
    std::uint64_t m_valid_epoch = 0;
    std::uint64_t m_halo_epoch = 0;
    std::uint64_t m_physical_epoch = 0;
};

#endif
