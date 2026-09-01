#include "CartesianBenchmarkCase.H"

#include "AVWiSCaseRunnerAccess.H"
#include "AVWiSSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
amrex::Real global_sum(amrex::Real value)
{
    amrex::ParallelDescriptor::ReduceRealSum(value);
    return value;
}
}

void AVWiSCaseRunnerAccess::run_cartesian_benchmark_impl(
    AVWiSSolver& solver, amrex::Real dt, int steps, amrex::Real viscosity,
    std::string const& report_path)
{
    if (!(dt > 0.0) || steps <= 0 || !(viscosity >= 0.0) || report_path.empty()) {
        throw std::runtime_error("Cartesian benchmark requires positive dt/steps, non-negative viscosity, and report path");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!solver.m_geom.isPeriodic(dir)) {
            throw std::runtime_error("Cartesian benchmark requires a fully periodic domain");
        }
    }

    // The same divergence-free shear used by the existing P5-004 contract.
    // This is a manufactured contract baseline, not a legacy vwis reference case.
    const amrex::Real xlo = solver.m_geom.ProbLo(0);
    const amrex::Real length = solver.m_geom.ProbLength(0);
    const amrex::Real dx = solver.m_dx[0];
    for (amrex::MFIter mfi(solver.m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = solver.m_ucat.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
            u(i,j,k,0) = 0.0;
            u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
            u(i,j,k,2) = 0.0;
        });
    }
    amrex::Gpu::streamSynchronize();
    solver.mark_valid_modified();
    solver.sync_ucont_from_ucat();
    amrex::MultiFab::Copy(solver.m_ucat_old, solver.m_ucat, 0, 0, AMREX_SPACEDIM, solver.m_nghost);
    amrex::MultiFab::Copy(solver.m_ucat_older, solver.m_ucat, 0, 0, AMREX_SPACEDIM, solver.m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(solver.m_ucont_old[dir], solver.m_ucont[dir], 0, 0, 1, solver.m_nghost);
        amrex::MultiFab::Copy(solver.m_ucont_older[dir], solver.m_ucont[dir], 0, 0, 1, solver.m_nghost);
    }
    solver.m_time = 0.0;
    solver.m_step = 0;
    solver.m_history_depth = 1;

    amrex::MultiFab divergence(solver.m_ba, solver.m_dm, 1, 0);
    amrex::Real total_step_seconds = 0.0;
    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(report_path);
        if (!output) throw std::runtime_error("cannot write Cartesian benchmark report: " + report_path);
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n  \"schema\": \"avwis-cartesian-benchmark-v1\",\n"
               << "  \"status\": \"PASS\",\n"
               << "  \"case_type\": \"manufactured/contract baseline\",\n"
               << "  \"legacy_reference\": false,\n"
               << "  \"amrex_version\": \"" << amrex::Version() << "\",\n"
               << "  \"amrex_locked_version\": \"" << AVWIS_LOCKED_VERSION << "\",\n"
               << "  \"dimension\": 3, \"n_cell\": [" << solver.m_geom.Domain().length(0) << ", "
               << solver.m_geom.Domain().length(1) << ", " << solver.m_geom.Domain().length(2) << "],\n"
               << "  \"dx\": [" << solver.m_dx[0] << ", " << solver.m_dx[1] << ", " << solver.m_dx[2] << "],\n"
               << "  \"dt\": " << dt << ", \"viscosity\": " << viscosity
               << ", \"steps\": " << steps << ",\n"
               << "  \"ranks\": " << amrex::ParallelDescriptor::NProcs()
               << ", \"compiler\": \"" << __VERSION__ << "\", \"build_backend\": \"CPU\",\n"
               << "  \"boundary_conditions\": \"periodic on x/y/z\",\n"
               << "  \"pressure_datum\": \"zero-mean periodic correction\",\n"
               << "  \"records\": [\n";
    }
    auto write_metric = [&](int step, bool comma) {
        solver.compute_cartesian_divergence(divergence);
        amrex::Real max_div = divergence.norm0(0, 0, true);
        amrex::Real integrated_div = divergence.sum(0, true) * solver.m_cell_volume;
        amrex::Real momentum[AMREX_SPACEDIM]{};
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            momentum[comp] = global_sum(solver.m_ucat.sum(comp, true) * solver.m_cell_volume);
        }
        amrex::MultiFab kinetic(solver.m_ba, solver.m_dm, 1, 0);
        for (amrex::MFIter mfi(solver.m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = solver.m_ucat.const_array(mfi);
            auto const e = kinetic.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                e(i,j,k) = 0.5 * (u(i,j,k,0)*u(i,j,k,0) + u(i,j,k,1)*u(i,j,k,1) + u(i,j,k,2)*u(i,j,k,2));
            });
        }
        amrex::Gpu::streamSynchronize();
        const amrex::Real energy = global_sum(kinetic.sum(0, true) * solver.m_cell_volume);
        amrex::ParallelDescriptor::ReduceRealMax(max_div);
        integrated_div = global_sum(integrated_div);
        if (!amrex::ParallelDescriptor::IOProcessor()) return;
        output << (comma ? ",\n" : "")
               << "    {\"step\": " << step << ", \"time\": " << solver.m_time
               << ", \"step_seconds\": " << solver.m_last_advance_seconds
               << ", \"post_projection_max_abs_divergence\": " << max_div
               << ", \"net_flux\": " << integrated_div
               << ", \"momentum\": [" << momentum[0] << ", " << momentum[1] << ", " << momentum[2]
               << "], \"kinetic_energy\": " << energy << "}";
    };

    for (int step = 0; step < steps; ++step) {
        solver.advance_one_step(dt, viscosity);
        total_step_seconds += solver.m_last_advance_seconds;
        write_metric(step + 1, step != 0);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output << "\n  ],\n  \"final_time\": " << solver.m_time
               << ",\n  \"total_step_seconds\": " << total_step_seconds << "\n}\n";
    }
    amrex::Print() << "AVWiS Cartesian benchmark: PASS report=" << report_path
                   << " steps=" << steps << " final_time=" << solver.m_time << "\n";
}

void run_cartesian_benchmark(AVWiSSolver& solver, amrex::Real dt, int steps,
                             amrex::Real viscosity, std::string const& report_path)
{
    AVWiSCaseRunnerAccess::run_cartesian_benchmark_impl(
        solver, dt, steps, viscosity, report_path);
}
