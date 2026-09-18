#include "TaylorGreenMMSCase.H"

#include "AVWiSCaseRunnerAccess.H"
#include "AVWiSSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
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

amrex::Real global_max(amrex::Real value)
{
    amrex::ParallelDescriptor::ReduceRealMax(value);
    return value;
}
}

void AVWiSCaseRunnerAccess::run_taylor_green_mms_impl(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path)
{
    if (solver.m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        throw std::runtime_error("Taylor-Green MMS requires Cartesian coordinates");
    }
    if (!(dt > 0.0) || !(final_time > 0.0) || !(viscosity > 0.0) ||
        report_path.empty()) {
        throw std::runtime_error(
            "Taylor-Green MMS requires positive dt/final_time/viscosity and a report path");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!solver.m_geom.isPeriodic(dir)) {
            throw std::runtime_error("Taylor-Green MMS requires a fully periodic domain");
        }
    }
    const amrex::Real lx = solver.m_geom.ProbLength(0);
    const amrex::Real ly = solver.m_geom.ProbLength(1);
    const amrex::Real length_tolerance = 64.0 * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(lx, ly);
    if (std::abs(lx-ly) > length_tolerance) {
        throw std::runtime_error("Taylor-Green MMS requires equal x/y domain lengths");
    }
    const auto integral_steps = [](amrex::Real end_time, amrex::Real step) {
        const amrex::Real count = end_time / step;
        const auto rounded = static_cast<long long>(std::llround(count));
        const amrex::Real tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon()
            * std::max(amrex::Real(1.0), std::abs(count));
        if (rounded <= 0 || std::abs(count-static_cast<amrex::Real>(rounded)) > tolerance) {
            throw std::runtime_error(
                "Taylor-Green final_time must be an integer multiple of dt and dt/2");
        }
        return rounded;
    };
    const long long coarse_steps = integral_steps(final_time, dt);
    const long long fine_steps = integral_steps(final_time, 0.5*dt);

    constexpr amrex::Real two_pi = 6.283185307179586476925286766559005768;
    const amrex::Real wave_number = two_pi / lx;
    const amrex::Real xlo = solver.m_geom.ProbLo(0);
    const amrex::Real ylo = solver.m_geom.ProbLo(1);
    const amrex::Real dx = solver.m_dx[0];
    const amrex::Real dy = solver.m_dx[1];
    const amrex::Real physical_volume = lx * ly * solver.m_geom.ProbLength(2);

    auto set_exact_state = [&](amrex::Real time) {
        const amrex::Real amplitude = std::exp(-2.0*viscosity*wave_number*wave_number*time);
        for (amrex::MFIter mfi(solver.m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = solver.m_ucat.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i)+0.5)*dx;
                const amrex::Real y = ylo + (static_cast<amrex::Real>(j)+0.5)*dy;
                u(i,j,k,0) = amplitude * std::sin(wave_number*(x-xlo))
                                         * std::cos(wave_number*(y-ylo));
                u(i,j,k,1) = -amplitude * std::cos(wave_number*(x-xlo))
                                          * std::sin(wave_number*(y-ylo));
                u(i,j,k,2) = 0.0;
            });
        }
        amrex::Gpu::streamSynchronize();
        solver.m_p.setVal(0.0);
        solver.m_phi.setVal(0.0);
        solver.m_nvert.setVal(0.0);
        solver.mark_valid_modified();
        solver.sync_ucont_from_ucat();
        amrex::MultiFab::Copy(solver.m_ucat_old, solver.m_ucat, 0, 0,
                              AMREX_SPACEDIM, solver.m_nghost);
        amrex::MultiFab::Copy(solver.m_ucat_older, solver.m_ucat, 0, 0,
                              AMREX_SPACEDIM, solver.m_nghost);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::MultiFab::Copy(solver.m_ucont_old[dir], solver.m_ucont[dir],
                                  0, 0, 1, solver.m_nghost);
            amrex::MultiFab::Copy(solver.m_ucont_older[dir], solver.m_ucont[dir],
                                  0, 0, 1, solver.m_nghost);
        }
        solver.m_time = time;
        solver.m_step = 0;
        solver.m_history_depth = 1;
    };

    auto l2_against_exact_velocity = [&](amrex::Real time) {
        amrex::MultiFab squared_error(solver.m_ba, solver.m_dm, 1, 0);
        const amrex::Real amplitude = std::exp(-2.0*viscosity*wave_number*wave_number*time);
        for (amrex::MFIter mfi(squared_error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = solver.m_ucat.const_array(mfi);
            auto const e = squared_error.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i)+0.5)*dx;
                const amrex::Real y = ylo + (static_cast<amrex::Real>(j)+0.5)*dy;
                const amrex::Real exact_u = amplitude * std::sin(wave_number*(x-xlo))
                    * std::cos(wave_number*(y-ylo));
                const amrex::Real exact_v = -amplitude * std::cos(wave_number*(x-xlo))
                    * std::sin(wave_number*(y-ylo));
                const amrex::Real du = u(i,j,k,0)-exact_u;
                const amrex::Real dv = u(i,j,k,1)-exact_v;
                const amrex::Real dw = u(i,j,k,2);
                e(i,j,k) = du*du + dv*dv + dw*dw;
            });
        }
        amrex::Gpu::streamSynchronize();
        return std::sqrt(global_sum(squared_error.sum(0, true))*solver.m_cell_volume
                         / physical_volume);
    };

    set_exact_state(0.0);
    amrex::MultiFab advection(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab viscous(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab legacy(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab advection_error(solver.m_ba, solver.m_dm, 1, 0);
    amrex::MultiFab viscous_error(solver.m_ba, solver.m_dm, 1, 0);
    solver.compute_cartesian_advection_rhs(advection);
    solver.compute_cartesian_viscous_rhs(viscous, viscosity);
    const amrex::Real cell_volume = solver.m_cell_volume;

    // Independent transcription of the legacy second_order branch in
    // vwis2.0/RHSSolver.C: face flux is -Ucont*average(Ucat), followed by
    // the Cartesian conservative face divergence. This proves stencil parity,
    // not whole-program equivalence with the PETSc executable.
    for (amrex::MFIter mfi(legacy, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const reference = legacy.array(mfi);
        auto const u = solver.m_ucat.const_array(mfi);
        auto const fx = solver.m_ucont[0].const_array(mfi);
        auto const fy = solver.m_ucont[1].const_array(mfi);
        auto const fz = solver.m_ucont[2].const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real fx_lo = -fx(i,j,k)*0.5*(u(i-1,j,k,comp)+u(i,j,k,comp));
                const amrex::Real fx_hi = -fx(i+1,j,k)*0.5*(u(i,j,k,comp)+u(i+1,j,k,comp));
                const amrex::Real fy_lo = -fy(i,j,k)*0.5*(u(i,j-1,k,comp)+u(i,j,k,comp));
                const amrex::Real fy_hi = -fy(i,j+1,k)*0.5*(u(i,j,k,comp)+u(i,j+1,k,comp));
                const amrex::Real fz_lo = -fz(i,j,k)*0.5*(u(i,j,k-1,comp)+u(i,j,k,comp));
                const amrex::Real fz_hi = -fz(i,j,k+1)*0.5*(u(i,j,k,comp)+u(i,j,k+1,comp));
                reference(i,j,k,comp) = ((fx_hi-fx_lo)+(fy_hi-fy_lo)+(fz_hi-fz_lo))
                    / cell_volume;
            });
    }
    amrex::Gpu::streamSynchronize();

    for (amrex::MFIter mfi(advection_error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = advection.const_array(mfi);
        auto const visc = viscous.const_array(mfi);
        auto const ae = advection_error.array(mfi);
        auto const ve = viscous_error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i)+0.5)*dx;
            const amrex::Real y = ylo + (static_cast<amrex::Real>(j)+0.5)*dy;
            const amrex::Real exact_adv_u = -0.5*wave_number*std::sin(2.0*wave_number*(x-xlo));
            const amrex::Real exact_adv_v = -0.5*wave_number*std::sin(2.0*wave_number*(y-ylo));
            const amrex::Real exact_visc_u = -2.0*viscosity*wave_number*wave_number
                * std::sin(wave_number*(x-xlo))*std::cos(wave_number*(y-ylo));
            const amrex::Real exact_visc_v = 2.0*viscosity*wave_number*wave_number
                * std::cos(wave_number*(x-xlo))*std::sin(wave_number*(y-ylo));
            const amrex::Real dau = adv(i,j,k,0)-exact_adv_u;
            const amrex::Real dav = adv(i,j,k,1)-exact_adv_v;
            const amrex::Real daz = adv(i,j,k,2);
            const amrex::Real dvu = visc(i,j,k,0)-exact_visc_u;
            const amrex::Real dvv = visc(i,j,k,1)-exact_visc_v;
            const amrex::Real dvz = visc(i,j,k,2);
            ae(i,j,k) = dau*dau + dav*dav + daz*daz;
            ve(i,j,k) = dvu*dvu + dvv*dvv + dvz*dvz;
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real advection_l2 = std::sqrt(
        global_sum(advection_error.sum(0, true))*solver.m_cell_volume/physical_volume);
    const amrex::Real viscous_l2 = std::sqrt(
        global_sum(viscous_error.sum(0, true))*solver.m_cell_volume/physical_volume);
    amrex::Real metric_volume_min = solver.m_metric_data.cell_volume_cc().min(0, 0, true);
    amrex::Real metric_volume_max = solver.m_metric_data.cell_volume_cc().max(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMin(metric_volume_min);
    amrex::ParallelDescriptor::ReduceRealMax(metric_volume_max);

    amrex::MultiFab legacy_difference(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(legacy_difference, legacy, 0, 0, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Subtract(legacy_difference, advection, 0, 0, AMREX_SPACEDIM, 0);
    amrex::Real legacy_max_error = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        legacy_max_error = std::max(legacy_max_error,
                                    legacy_difference.norm0(comp, 0, true));
    }
    legacy_max_error = global_max(legacy_max_error);
    const amrex::Real reference_scale = std::max(
        amrex::Real(1.0), global_max(advection.norm0(0, 0, true)));
    const amrex::Real legacy_tolerance = 256.0
        * std::numeric_limits<amrex::Real>::epsilon()*reference_scale;
    if (!(legacy_max_error <= legacy_tolerance)) {
        throw std::runtime_error("Taylor-Green legacy second_order stencil parity failed");
    }

    amrex::MultiFab divergence(solver.m_ba, solver.m_dm, 1, 0);
    solver.compute_cartesian_divergence(divergence);
    const amrex::Real divergence_linf = global_max(divergence.norm0(0, 0, true));

    set_exact_state(0.0);
    for (long long step = 0; step < coarse_steps; ++step) {
        solver.advance_one_step(dt, viscosity);
    }
    const amrex::Real time_error_dt = l2_against_exact_velocity(final_time);
    set_exact_state(0.0);
    for (long long step = 0; step < fine_steps; ++step) {
        solver.advance_one_step(0.5*dt, viscosity);
    }
    const amrex::Real time_error_half_dt = l2_against_exact_velocity(final_time);
    const amrex::Real time_ratio = time_error_dt / time_error_half_dt;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(report_path);
        if (!output) {
            throw std::runtime_error("cannot write Taylor-Green MMS report: " + report_path);
        }
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n"
               << "  \"schema\": \"avwis-p10-001-taylor-green-mms-v1\",\n"
               << "  \"status\": \"PASS\",\n"
               << "  \"case_type\": \"Cartesian Taylor-Green manufactured solution\",\n"
               << "  \"n_cell\": [" << solver.m_geom.Domain().length(0) << ", "
               << solver.m_geom.Domain().length(1) << ", "
               << solver.m_geom.Domain().length(2) << "],\n"
               << "  \"dx\": [" << solver.m_dx[0] << ", " << solver.m_dx[1] << ", "
               << solver.m_dx[2] << "],\n"
               << "  \"dt\": " << dt << ", \"final_time\": " << final_time
               << ", \"viscosity\": " << viscosity << ",\n"
               << "  \"advection_l2\": " << advection_l2 << ",\n"
               << "  \"viscous_l2\": " << viscous_l2 << ",\n"
               << "  \"metric_cell_volume_min\": " << metric_volume_min << ",\n"
               << "  \"metric_cell_volume_max\": " << metric_volume_max << ",\n"
               << "  \"initial_divergence_linf\": " << divergence_linf << ",\n"
               << "  \"legacy_reference\": {\n"
               << "    \"kind\": \"independent source-level second_order stencil transcription\",\n"
               << "    \"source\": \"vwis2.0/RHSSolver.C\",\n"
               << "    \"scope\": \"Cartesian conservative advection stencil only\",\n"
               << "    \"max_abs_error\": " << legacy_max_error << "\n"
               << "  },\n"
               << "  \"time_error_dt\": " << time_error_dt << ",\n"
               << "  \"time_error_dt_over_2\": " << time_error_half_dt << ",\n"
               << "  \"time_refinement_ratio\": " << time_ratio << ",\n"
               << "  \"pressure_datum\": \"zero-mean periodic correction\",\n"
               << "  \"amrex_version\": \"" << amrex::Version() << "\"\n"
               << "}\n";
    }
    amrex::Print() << "AVWiS P10-001 Taylor-Green MMS: PASS report=" << report_path
                   << " advection_l2=" << advection_l2
                   << " viscous_l2=" << viscous_l2
                   << " time_ratio=" << time_ratio << "\n";
}

void run_taylor_green_mms(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path)
{
    AVWiSCaseRunnerAccess::run_taylor_green_mms_impl(
        solver, dt, final_time, viscosity, report_path);
}
