#include "AVWiSSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <stdexcept>

void AVWiSSolver::compute_cartesian_advection_rhs(amrex::MultiFab& rhs)
{
    if (rhs.boxArray() != m_ba || rhs.nComp() != AMREX_SPACEDIM) {
        throw std::runtime_error("P5 Cartesian advection RHS must be a three-component cell MultiFab");
    }
    if (m_nghost < 1) {
        throw std::runtime_error("P5 Cartesian advection requires avwis.nghost>=1");
    }
    if (m_boundary.enabled) {
        apply_boundary_pipeline("pre-p5-advection");
    } else {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P5 Cartesian advection requires explicit vwisbcs on non-periodic directions");
            }
        }
        fill_ghost_cells();
    }
    require_ghosts_fresh("P5 Cartesian advection");

    const amrex::Real inverse_volume = 1.0 / m_cell_volume;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = rhs.array(mfi);
        auto const u = m_ucat.const_array(mfi);
        auto const fx = m_ucont[0].const_array(mfi);
        auto const fy = m_ucont[1].const_array(mfi);
        auto const fz = m_ucont[2].const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real flux_x_hi = fx(i+1,j,k) * 0.5 * (u(i,j,k,comp) + u(i+1,j,k,comp));
                const amrex::Real flux_x_lo = fx(i,j,k)   * 0.5 * (u(i-1,j,k,comp) + u(i,j,k,comp));
                const amrex::Real flux_y_hi = fy(i,j+1,k) * 0.5 * (u(i,j,k,comp) + u(i,j+1,k,comp));
                const amrex::Real flux_y_lo = fy(i,j,k)   * 0.5 * (u(i,j-1,k,comp) + u(i,j,k,comp));
                const amrex::Real flux_z_hi = fz(i,j,k+1) * 0.5 * (u(i,j,k,comp) + u(i,j,k+1,comp));
                const amrex::Real flux_z_lo = fz(i,j,k)   * 0.5 * (u(i,j,k-1,comp) + u(i,j,k,comp));
                adv(i,j,k,comp) = -((flux_x_hi-flux_x_lo) +
                                     (flux_y_hi-flux_y_lo) +
                                     (flux_z_hi-flux_z_lo)) * inverse_volume;
            });
    }
    amrex::Gpu::streamSynchronize();
}
