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

void AVWiSSolver::compute_cartesian_viscous_rhs(
    amrex::MultiFab& rhs, amrex::Real viscosity)
{
    if (rhs.boxArray() != m_ba || rhs.nComp() != AMREX_SPACEDIM) {
        throw std::runtime_error("P5 Cartesian viscous RHS must be a three-component cell MultiFab");
    }
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
        throw std::runtime_error("P5 Cartesian viscosity must be finite and non-negative");
    }
    if (m_nghost < 1) {
        throw std::runtime_error("P5 Cartesian viscosity requires avwis.nghost>=1");
    }
    if (m_boundary.enabled) {
        apply_boundary_pipeline("pre-p5-viscosity");
    } else {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P5 Cartesian viscosity requires explicit vwisbcs on non-periodic directions");
            }
        }
        fill_ghost_cells();
    }
    require_ghosts_fresh("P5 Cartesian viscosity");

    const amrex::Real idx2 = 1.0 / (m_dx[0] * m_dx[0]);
    const amrex::Real idy2 = 1.0 / (m_dx[1] * m_dx[1]);
    const amrex::Real idz2 = 1.0 / (m_dx[2] * m_dx[2]);
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const visc = rhs.array(mfi);
        auto const u = m_ucat.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                visc(i,j,k,comp) = viscosity * (
                    (u(i+1,j,k,comp) - 2.0*u(i,j,k,comp) + u(i-1,j,k,comp)) * idx2 +
                    (u(i,j+1,k,comp) - 2.0*u(i,j,k,comp) + u(i,j-1,k,comp)) * idy2 +
                    (u(i,j,k+1,comp) - 2.0*u(i,j,k,comp) + u(i,j,k-1,comp)) * idz2);
            });
    }
    amrex::Gpu::streamSynchronize();
}
