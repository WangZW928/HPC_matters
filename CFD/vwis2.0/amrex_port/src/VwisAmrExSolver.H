#ifndef VWIS_AMREX_SOLVER_H_
#define VWIS_AMREX_SOLVER_H_

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

/** P3 single-level Cartesian physical-boundary semantics. */
enum class CartesianBC { Periodic, NoSlipWall, SlipWall, Symmetry, Inflow, Outflow };

struct BoundarySideSpec {
    CartesianBC velocity = CartesianBC::Periodic;
    int legacy_code = -999; // -999 means a new named Cartesian BC, not a legacy code.
    amrex::Real pressure = 0.0; // fixed only for Outflow; otherwise homogeneous Neumann.
};

struct CartesianBoundaryConfig {
    amrex::Array<BoundarySideSpec, 2 * AMREX_SPACEDIM> sides{}; // lo,hi per direction
    std::string inlet_profile = "uniform"; // uniform or linear_plane
    amrex::Real inlet_target_flux = 1.0;    // positive volume entering the domain
    amrex::Real profile_offset = 1.0;
    amrex::Real profile_slope_0 = 0.0;
    amrex::Real profile_slope_1 = 0.0;
    bool constrain_outlet_flux = true;
    bool enabled = false;
};

struct P3Diagnostics {
    std::uint64_t valid_epoch = 0;
    std::uint64_t halo_epoch = 0;
    std::uint64_t physical_epoch = 0;
    amrex::Real outward_boundary_flux = 0.0;
    amrex::Real integrated_divergence = 0.0;
    amrex::Real max_abs_divergence = 0.0;
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
 * Single-level Cartesian P0--P3 data owner. MultiFabs remain owned here for the
 * complete solver lifetime, while tile Array4 views never escape an MFIter.
 * Thus DistributionMapping is the sole MPI ownership authority; valid cells
 * may be written independently and only fill_ghost_cells() exchanges halos.
 * P3 adds only physical ghost/boundary flux plumbing.  No momentum, pressure
 * projection/RHS, time integration, LES, IBM, or FSI algorithm exists.
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
    /** P1 placeholder with an explicit positive-dt contract; it does not rotate time layers. */
    void advance_one_step(amrex::Real dt);
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
    /** Globally reduced P3 boundary/divergence/freshness report; diagnostic only, never a RHS. */
    P3Diagnostics p3_diagnostics(char const* stage, bool require_fresh = true) const;
    /** IO-rank layout/schema diagnostics, safe after initialization or a no-op step. */
    void diagnostics() const;
    /** Write schema-only metadata; this never writes a plotfile/restart payload. */
    void write_metadata_manifest(std::string const& path) const;

private:
    void register_fields();
    void register_metric_metadata();
    void define_boundary_metadata();
    void validate_boundary_config() const;
    void mark_valid_modified();
    void require_ghosts_fresh(char const* consumer) const;
    amrex::Real boundary_flux(int dir, bool high) const;
    amrex::Real inlet_profile_integral(int dir, bool high) const;

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // Cell IndexType. Nvert remains a legacy classifier, never an EB volume fraction.
    amrex::MultiFab m_p;
    amrex::MultiFab m_phi;
    amrex::MultiFab m_nvert;
    amrex::MultiFab m_ucat;
    amrex::MultiFab m_ucat_old;

    // MAC face IndexType: each field has one normal volume-flux component.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_old;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_older;
    std::vector<FieldSpec> m_fields;
    std::vector<MetricSpec> m_metrics;
    std::vector<amrex::BCRec> m_cell_bcs;
    CartesianBoundaryConfig m_boundary;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_dx{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_face_area{};
    amrex::Real m_cell_volume = 0.0;
    int m_nghost;
    amrex::Real m_initialize_seconds = 0.0;
    amrex::Real m_last_noop_seconds = 0.0;
    std::uint64_t m_valid_epoch = 0;
    std::uint64_t m_halo_epoch = 0;
    std::uint64_t m_physical_epoch = 0;
};

#endif
