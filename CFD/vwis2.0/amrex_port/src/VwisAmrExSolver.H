#ifndef VWIS_AMREX_SOLVER_H_
#define VWIS_AMREX_SOLVER_H_

#include "CartesianBoundaryConfig.H"

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
class VwisAmrExSolver {
public:
    /**
     * Define the sole Cartesian level. n_cell is cell count, max_grid_size controls
     * BoxArray splitting, nghost allocates halos for every registered field,
     * and is_periodic controls only geometry/halo wrapping. physical_domain is
     * Cartesian geometry, never a curvilinear metric input.
     */
    VwisAmrExSolver(amrex::Vector<int> const& n_cell, int max_grid_size,
                    int nghost, amrex::RealBox const& physical_domain,
                    amrex::Vector<int> const& is_periodic,
                    CartesianBoundaryConfig const& boundary = {});

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
    /** Exercise base cell layout, initialization, periodic halo, and physical-BC limitation contracts. */
    void run_runtime_contract_checks();
    /** Recover Cartesian velocity by averaging adjacent face volume-flux densities. */
    void sync_ucat_from_ucont();
    /** Build Cartesian face volume flux from interpolated normal velocity times area. */
    void sync_ucont_from_ucat();
    /** Verify P2-003/004/005 manufactured transforms, metrics, and multi-box layout. */
    void run_p2_transform_layout_checks();
    /** Verify P3 classification, cell/face physical ghosts, flux, and stale-ghost diagnostics. */
    void run_p3_boundary_contract_checks();
    /** Solve the P4 Cartesian pressure correction and project integrated face fluxes. */
    ProjectionDiagnostics project_cartesian(amrex::Real dt, amrex::Real time_coefficient = 1.0);
    /** Manufactured checks for the configured periodic, closed, or inflow/outflow pressure policy. */
    void run_p4_projection_contract_checks(amrex::Real dt, amrex::Real time_coefficient = 1.0);
    /** Conservative Cartesian -div(Ucont[d] * face_average(Ucat)) / cell volume. */
    void compute_cartesian_advection_rhs(amrex::MultiFab& rhs);
    /** Constant and smooth manufactured P5-001 checks, including configured physical boundaries. */
    void run_p5_advection_contract_checks();
    /** Uniform Cartesian constant-coefficient nu*Laplacian(Ucat), component-wise. */
    void compute_cartesian_viscous_rhs(amrex::MultiFab& rhs, amrex::Real viscosity);
    /** P5-002 periodic manufactured/conservation or physical-boundary multi-Box checks. */
    void run_p5_viscous_contract_checks(amrex::Real viscosity);
    /** P5-004 periodic temporal-order, conservation, divergence, CFL, and history checks. */
    void run_p5_time_contract_checks(amrex::Real dt, amrex::Real final_time,
                                     amrex::Real viscosity);
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
    /** Real disk round-trip: uninterrupted N steps versus restart after K steps. */
    void run_p8_restart_contract_checks(std::string const& path, amrex::Real dt,
                                        int total_steps, int checkpoint_step,
                                        amrex::Real viscosity);

private:
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
