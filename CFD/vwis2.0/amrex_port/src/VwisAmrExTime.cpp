#include "VwisAmrExSolver.H"

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

TimeStepDiagnostics VwisAmrExSolver::time_step_diagnostics(
    amrex::Real dt, amrex::Real viscosity) const
{
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

void VwisAmrExSolver::advance_one_step(amrex::Real dt, amrex::Real viscosity)
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

void VwisAmrExSolver::run_p5_time_contract_checks(
    amrex::Real dt, amrex::Real final_time, amrex::Real viscosity)
{
    if (!std::isfinite(final_time) || final_time <= 0.0 ||
        !std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::runtime_error("P5-004 time contract requires positive final_time and viscosity");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5-004 manufactured time contract requires a fully periodic domain");
        }
    }
    const amrex::Real coarse_steps_real = final_time / dt;
    const int coarse_steps = static_cast<int>(std::llround(coarse_steps_real));
    if (coarse_steps < 2 ||
        std::abs(coarse_steps_real - coarse_steps) > 128.0 * std::numeric_limits<amrex::Real>::epsilon() * coarse_steps) {
        throw std::runtime_error("P5-004 final_time must be an integer multiple of dt with at least two steps");
    }

    struct SequenceResult {
        amrex::Real error = 0.0;
        amrex::Real momentum_drift = 0.0;
        amrex::Real max_divergence = 0.0;
        amrex::Real history_error = 0.0;
        TimeStepDiagnostics stability;
    };

    auto max_difference = [&](amrex::MultiFab const& lhs, amrex::MultiFab const& rhs,
                              int components) {
        amrex::MultiFab difference(lhs.boxArray(), lhs.DistributionMap(), components, 0);
        amrex::MultiFab::Copy(difference, lhs, 0, 0, components, 0);
        amrex::MultiFab::Subtract(difference, rhs, 0, 0, components, 0);
        amrex::Real result = 0.0;
        for (int comp = 0; comp < components; ++comp) {
            result = amrex::max(result, difference.norm0(comp, 0, true));
        }
        amrex::ParallelDescriptor::ReduceRealMax(result);
        return result;
    };

    auto run_sequence = [&](amrex::Real step_dt, int steps) {
        SequenceResult result;
        const amrex::Real xlo = m_geom.ProbLo(0);
        const amrex::Real length = m_geom.ProbLength(0);
        const amrex::Real dx = m_dx[0];
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                u(i,j,k,0) = 0.0;
                u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
                u(i,j,k,2) = 0.0;
            });
        }
        mark_valid_modified();
        sync_ucont_from_ucat();
        amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab::Copy(m_ucat_older, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab initial_cells(m_ba, m_dm, AMREX_SPACEDIM, 0);
        amrex::MultiFab::Copy(initial_cells, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> initial_faces;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            initial_faces[dir].define(m_ucont[dir].boxArray(), m_dm, 1, 0);
            amrex::MultiFab::Copy(initial_faces[dir], m_ucont[dir], 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
            amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont[dir], 0, 0, 1, m_nghost);
        }
        m_time = 0.0;
        m_step = 0;
        m_history_depth = 1;
        result.stability = time_step_diagnostics(step_dt, viscosity);

        amrex::Real initial_momentum = m_ucat.sum(1, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealSum(initial_momentum);
        amrex::MultiFab previous(m_ba, m_dm, AMREX_SPACEDIM, 0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> previous_faces;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            previous_faces[dir].define(m_ucont[dir].boxArray(), m_dm, 1, 0);
        }

        for (int step = 0; step < steps; ++step) {
            amrex::MultiFab::Copy(previous, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                amrex::MultiFab::Copy(previous_faces[dir], m_ucont[dir], 0, 0, 1, 0);
            }
            advance_one_step(step_dt, viscosity);
            result.history_error = amrex::max(
                result.history_error, max_difference(m_ucat_old, previous, AMREX_SPACEDIM));
            if (step == 1) {
                result.history_error = amrex::max(
                    result.history_error, max_difference(m_ucat_older, initial_cells, AMREX_SPACEDIM));
            }
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                result.history_error = amrex::max(
                    result.history_error, max_difference(m_ucont_old[dir], previous_faces[dir], 1));
                if (step == 1) {
                    result.history_error = amrex::max(
                        result.history_error, max_difference(m_ucont_older[dir], initial_faces[dir], 1));
                }
            }
        }

        const amrex::Real pi = 3.141592653589793238462643383279502884;
        const amrex::Real eigenvalue = -4.0 * viscosity *
            std::pow(std::sin(pi * dx / length) / dx, 2);
        const amrex::Real exact_factor = std::exp(eigenvalue * final_time);
        amrex::MultiFab error(m_ba, m_dm, 1, 0);
        for (amrex::MFIter mfi(error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi);
            auto const err = error.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                const amrex::Real exact = exact_factor * amrex::Math::sinpi(2.0 * (x-xlo) / length);
                err(i,j,k) = u(i,j,k,1) - exact;
            });
        }
        result.error = error.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(result.error);
        amrex::Real final_momentum = m_ucat.sum(1, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealSum(final_momentum);
        result.momentum_drift = std::abs(final_momentum - initial_momentum);
        amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
        compute_cartesian_divergence(divergence);
        result.max_divergence = divergence.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(result.max_divergence);
        return result;
    };

    const SequenceResult coarse = run_sequence(dt, coarse_steps);
    const SequenceResult fine = run_sequence(0.5 * dt, 2 * coarse_steps);
    const amrex::Real order_ratio = coarse.error / fine.error;
    const amrex::Real roundoff = 32768.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (!(order_ratio > 1.8 && order_ratio < 2.2) ||
        fine.momentum_drift > roundoff || fine.max_divergence > roundoff ||
        fine.history_error > roundoff || m_step != static_cast<std::uint64_t>(2 * coarse_steps) ||
        m_history_depth != 3 || std::abs(m_time - final_time) > roundoff ||
        coarse.stability.advective_cfl > 1.0 || coarse.stability.diffusive_number > 1.0) {
        throw std::runtime_error("P5-004 explicit temporal order/conservation/history contract failed");
    }

    amrex::Print() << "VWiS AMReX P5-004 explicit Euler time contract: PASS"
                   << " coarse_error=" << coarse.error
                   << " fine_error=" << fine.error
                   << " ratio=" << order_ratio
                   << " advective_CFL=" << coarse.stability.advective_cfl
                   << " diffusive_number=" << coarse.stability.diffusive_number
                   << " momentum_drift=" << fine.momentum_drift
                   << " max_divergence=" << fine.max_divergence
                   << " history_error=" << fine.history_error
                   << " projection_time_coefficient="
                   << fine.stability.projection_time_coefficient << "\n";
}
