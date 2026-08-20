#ifndef VWIS_AMREX_SOLVER_H_
#define VWIS_AMREX_SOLVER_H_

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

// Deliberately a data-layout and execution skeleton.  It does not implement
// the VWiS momentum, Poisson, LES, immersed-boundary, or FSI algorithms.
class VwisAmrExSolver {
public:
    VwisAmrExSolver(amrex::Vector<int> const& n_cell, int max_grid_size,
                    int nghost, amrex::RealBox const& physical_domain,
                    amrex::Vector<int> const& is_periodic);

    void initialize();
    void fill_ghost_cells();
    void advance_one_step(amrex::Real dt);
    void diagnostics() const;

private:
    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    // Cell centered: pressure, pressure-correction placeholder, solid marker.
    amrex::MultiFab m_p;
    amrex::MultiFab m_phi;
    amrex::MultiFab m_nvert;

    // MAC layout: one scalar normal flux/velocity field on each face direction.
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_ucont;
    int m_nghost;
};

#endif
