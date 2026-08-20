#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelFor.H>
#include <AMReX_Print.H>

VwisAmrExSolver::VwisAmrExSolver(
    amrex::Vector<int> const& n_cell, int max_grid_size, int nghost,
    amrex::RealBox const& physical_domain,
    amrex::Vector<int> const& is_periodic)
    : m_nghost(nghost)
{
    amrex::IntVect small_end(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect big_end(AMREX_D_DECL(n_cell[0] - 1, n_cell[1] - 1,
                                        n_cell[2] - 1));
    amrex::Box domain(small_end, big_end);
    amrex::Array<int, AMREX_SPACEDIM> periodicity{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        periodicity[dir] = is_periodic[dir];
    }

    m_geom.define(domain, &physical_domain, 0, periodicity.data());
    m_ba.define(domain);
    m_ba.maxSize(max_grid_size);
    m_dm = amrex::DistributionMapping(m_ba);

    m_p.define(m_ba, m_dm, 1, m_nghost);
    m_phi.define(m_ba, m_dm, 1, m_nghost);
    m_nvert.define(m_ba, m_dm, 1, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = amrex::convert(
            m_ba, amrex::IntVect::TheDimensionVector(dir));
        m_ucont[dir].define(face_ba, m_dm, 1, m_nghost);
    }
}

void VwisAmrExSolver::initialize()
{
    for (amrex::MFIter mfi(m_p); mfi.isValid(); ++mfi) {
        auto const& p = m_p.array(mfi);
        auto const& phi = m_phi.array(mfi);
        auto const& nvert = m_nvert.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE
                           (int i, int j, int k) noexcept {
            p(i, j, k) = 0.0;
            phi(i, j, k) = 0.0;
            nvert(i, j, k) = 0.0; // Fluid everywhere in this skeleton.
        });
    }
    for (auto& velocity : m_ucont) {
        velocity.setVal(0.0);
    }
    fill_ghost_cells();
}

void VwisAmrExSolver::fill_ghost_cells()
{
    // Periodic/inter-box exchange only. Physical BC filling belongs to a later BC module.
    m_p.FillBoundary(m_geom.periodicity());
    m_phi.FillBoundary(m_geom.periodicity());
    m_nvert.FillBoundary(m_geom.periodicity());
    for (auto& velocity : m_ucont) {
        velocity.FillBoundary(m_geom.periodicity());
    }
}

void VwisAmrExSolver::advance_one_step(amrex::Real dt)
{
    (void)dt;
    fill_ghost_cells();
    // Placeholder only: no RHS/LES, time integration, Poisson solve,
    // projection, IBM, FSI, or physical-boundary update is performed.
}

void VwisAmrExSolver::diagnostics() const
{
    amrex::Print() << "skeleton diagnostics: max(|P|)="
                   << m_p.norm0(0, 0, true)
                   << ", max(|Nvert|)=" << m_nvert.norm0(0, 0, true)
                   << "\n";
}
