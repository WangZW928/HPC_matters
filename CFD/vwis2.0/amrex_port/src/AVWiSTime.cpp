#include "AVWiSSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

TimeStepDiagnostics AVWiSSolver::time_step_diagnostics(
    amrex::Real dt, amrex::Real viscosity) const
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        throw std::runtime_error(
            "C2.2 mapped mode has no metric-aware advection/viscosity time advance");
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error("P5-004 explicit dt must be finite and positive");
    }
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
        throw std::runtime_error("P5-004 viscosity must be finite and non-negative");
    }

    TimeStepDiagnostics result;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Real max_velocity = m_ucat.norm0(dir, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(max_velocity);
        result.advective_cfl += dt * max_velocity / m_dx[dir];
        result.diffusive_number += 2.0 * viscosity * dt / (m_dx[dir] * m_dx[dir]);
    }
    return result;
}

void AVWiSSolver::advance_one_step(amrex::Real dt, amrex::Real viscosity)
{
    const TimeStepDiagnostics stability = time_step_diagnostics(dt, viscosity);
    if (stability.advective_cfl > 1.0 + 64.0 * std::numeric_limits<amrex::Real>::epsilon() ||
        stability.diffusive_number > 1.0 + 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
        throw std::runtime_error(
            "P5-004 explicit step rejected: advective CFL and diffusive number must each be <= 1");
    }

    const amrex::Real start = amrex::second();
    amrex::MultiFab advective_rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab viscous_rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    compute_cartesian_advection_rhs(advective_rhs);
    compute_cartesian_viscous_rhs(viscous_rhs, viscosity);

    // After the step these layers are (n+1,n,n-1).  The explicit scheme reads
    // only layer n; retaining both older layers fixes restart/BDF2 data shape
    // without claiming that either an implicit or BDF2 solve exists.
    amrex::MultiFab::Copy(m_ucat_older, m_ucat_old, 0, 0, AMREX_SPACEDIM, m_nghost);
    amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont_old[dir], 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
    }

    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        auto const adv = advective_rhs.const_array(mfi);
        auto const visc = viscous_rhs.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                u(i,j,k,comp) += dt * (adv(i,j,k,comp) + visc(i,j,k,comp));
            });
    }
    amrex::Gpu::streamSynchronize();
    mark_valid_modified();
    sync_ucont_from_ucat();
    (void)project_cartesian(dt, stability.projection_time_coefficient);

    m_time += dt;
    ++m_step;
    m_history_depth = std::min(3, m_history_depth + 1);
    m_last_advance_seconds = amrex::second() - start;
}
