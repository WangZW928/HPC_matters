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
#include <vector>

/** Physical staggering of a registered field; fields are never implicitly converted. */
enum class FieldLocation { Cell, XFace, YFace, ZFace };

/**
 * P0/P1 field contract. `ghost_cells` is allocated same-level halo width;
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
};

/**
 * Single-level Cartesian P1 data owner. MultiFabs remain owned here for the
 * complete solver lifetime, while tile Array4 views never escape an MFIter.
 * Thus DistributionMapping is the sole MPI ownership authority; valid cells
 * may be written independently and only fill_ghost_cells() exchanges halos.
 * No momentum, projection, LES, IBM, FSI, or physical-boundary algorithm exists.
 */
class VwisAmrExSolver {
public:
    /**
     * Define the sole P1 level. n_cell is cell count, max_grid_size controls
     * BoxArray splitting, nghost allocates halos for every registered field,
     * and is_periodic controls only geometry/halo wrapping. physical_domain is
     * Cartesian geometry, never a curvilinear metric input.
     */
    VwisAmrExSolver(amrex::Vector<int> const& n_cell, int max_grid_size,
                    int nghost, amrex::RealBox const& physical_domain,
                    amrex::Vector<int> const& is_periodic);

    /** Initialize valid state to zero; physical ghost values remain undefined. */
    void initialize();
    /** Same-level/MPI/periodic exchange only; no physical BC is evaluated. */
    void fill_ghost_cells();
    /** P1 placeholder with an explicit positive-dt contract; it does not rotate time layers. */
    void advance_one_step(amrex::Real dt);
    /** IO-rank layout/schema diagnostics, safe after initialization or a no-op step. */
    void diagnostics() const;
    /** Write schema-only metadata; this never writes a plotfile/restart payload. */
    void write_metadata_manifest(std::string const& path) const;

private:
    void register_fields();
    void define_boundary_metadata();

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // Cell IndexType. Nvert remains a legacy classifier, never an EB volume fraction.
    amrex::MultiFab m_p;
    amrex::MultiFab m_phi;
    amrex::MultiFab m_nvert;
    amrex::MultiFab m_ucat;
    amrex::MultiFab m_ucat_old;

    // MAC face IndexType: each field has one normal flux/velocity component.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_old;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont_older;
    std::vector<FieldSpec> m_fields;
    std::vector<amrex::BCRec> m_cell_bcs;
    int m_nghost;
    amrex::Real m_initialize_seconds = 0.0;
    amrex::Real m_last_noop_seconds = 0.0;
};

#endif
