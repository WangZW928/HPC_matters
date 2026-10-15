#include "MappedPressureWaveCase.H"
#include "AVWiSCaseRunnerAccess.H"
#include "AVWiSSolver.H"

#include <AMReX_Arena.H>
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
constexpr amrex::Real two_pi = 6.283185307179586476925286766559;
constexpr amrex::Real velocity_amplitude = 0.25;
constexpr amrex::Real pressure_amplitude = 0.2;

amrex::Real global_sum(amrex::Real x) {
    amrex::ParallelDescriptor::ReduceRealSum(x);
    return x;
}

void check_io(bool local_ok, char const* message) {
    int ok = local_ok;
    amrex::ParallelDescriptor::ReduceIntMin(ok);
    if (!ok) throw std::runtime_error(message);
}
}

void AVWiSCaseRunnerAccess::run_mapped_pressure_wave_case_impl(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path,
    std::string const& field_path)
{
    auto const& cfg = solver.m_mapping_operator;
    if (cfg.analytic_parameters.stretch != amrex::Array<amrex::Real,3>{0,0,0} ||
        cfg.state_contract != "face_authoritative" ||
        cfg.pressure_contract != "incremental_pressure") {
        throw std::runtime_error(
            "pressure wave requires an affine periodic mapping, face_authoritative and incremental_pressure");
    }
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(final_time) ||
        final_time <= 0 || !std::isfinite(viscosity) || viscosity <= 0 || report_path.empty()) {
        throw std::runtime_error("mapped pressure wave requires positive finite dt/final_time/viscosity and a report path");
    }
    auto const count = final_time / dt;
    if (!std::isfinite(count) || count < 1 || count > std::numeric_limits<int>::max())
        throw std::runtime_error("mapped pressure wave step count is out of range");
    int const steps = static_cast<int>(std::llround(count));
    if (std::abs(count - steps) > 128 * std::numeric_limits<amrex::Real>::epsilon() * std::max(amrex::Real(1), count))
        throw std::runtime_error("mapped pressure wave final_time must be an integer multiple of dt");

    auto const& metric = solver.m_metric_data;
    auto const epoch = solver.m_metric_epoch;
    auto const domain = solver.m_geom.Domain();
    auto const h = metric.logical_grid().spacing;
    auto const wx = two_pi / solver.m_geom.ProbLength(0);
    auto const wy = two_pi / solver.m_geom.ProbLength(1);
    // x=sx*xi+epsilon*sy*eta, y=sy*eta. For theta=wx*xi+wy*eta,
    // grad(theta)=K and B perpendicular to K give div(u)=u.grad(u)=0.
    bool const identity = cfg.mapping_type == "identity";
    auto const kx = wx / (identity ? 1.0 : cfg.analytic_parameters.scale[0]);
    auto const ky = wy / (identity ? 1.0 : cfg.analytic_parameters.scale[1]) - cfg.shear_epsilon * kx;
    auto const k2 = kx*kx + ky*ky;
    auto const bx = ky / std::sqrt(k2);
    auto const by = -kx / std::sqrt(k2);
    auto const ilo = domain.smallEnd(0), jlo = domain.smallEnd(1);
    auto const periodicity = solver.m_geom.periodicity();
    // Exact fields: u=.25 exp(-nu |K|^2 t) B sin(theta),
    // p=.2 exp(-t) cos(theta), f=grad(p). Thus u_t=nu Laplacian(u)
    // and the pressure/body-force pair cancels in the continuum momentum PDE.
    // The forcing below is analytic, never manufactured with a discrete G/D.
    auto const sinc_x = std::sin(0.5*wx*h[0]) / (0.5*wx*h[0]);
    auto const sinc_y = std::sin(0.5*wy*h[1]) / (0.5*wy*h[1]);
    amrex::Array<amrex::MultiFab const*, 3> faces{};
    for (int d=0; d<3; ++d) {
        auto const& area = metric.face_area_vector_fc(d);
        for (amrex::MFIter mfi(solver.m_ucont[d], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const f = solver.m_ucont[d].array(mfi);
            auto const s = area.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                auto const theta = wx*(i-ilo+(d==0 ? 0.0 : 0.5))*h[0]
                                 + wy*(j-jlo+(d==1 ? 0.0 : 0.5))*h[1];
                auto const average = (d==0 ? sinc_y : d==1 ? sinc_x : sinc_x*sinc_y);
                f(i,j,k) = velocity_amplitude*(bx*s(i,j,k,0)+by*s(i,j,k,1))
                    *std::sin(theta)*average;
            });
        }
        solver.m_ucont[d].OverrideSync(periodicity);
        solver.m_ucont[d].FillBoundary(periodicity);
        faces[d] = &solver.m_ucont[d];
    }
    // Start from exact integrated face averages, rather than projecting an
    // interpolated cell field and importing that projection's pressure noise.
    sync_metric_ucat_from_ucont(faces, metric, epoch, cfg, solver.m_ucat);
    for (amrex::MFIter mfi(solver.m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = solver.m_p.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            auto const theta = wx*(i-ilo+0.5)*h[0] + wy*(j-jlo+0.5)*h[1];
            p(i,j,k) = pressure_amplitude*std::cos(theta);
        });
    }
    solver.m_phi.setVal(0);
    solver.mark_valid_modified();
    solver.fill_ghost_cells();
    for (auto* older : {&solver.m_ucat_old, &solver.m_ucat_older})
        amrex::MultiFab::Copy(*older, solver.m_ucat, 0, 0, 3, solver.m_nghost);
    for (int d=0; d<3; ++d)
        for (auto* older : {&solver.m_ucont_old[d], &solver.m_ucont_older[d]})
            amrex::MultiFab::Copy(*older, solver.m_ucont[d], 0, 0, 1, solver.m_nghost);
    solver.m_time = 0;
    solver.m_step = 0;
    solver.m_history_depth = 1;

    amrex::MultiFab forcing(solver.m_ba, solver.m_dm, 3, 0);
    amrex::MultiFab previous(solver.m_ba, solver.m_dm, 3, 0);
    amrex::MultiFab gradient(solver.m_ba, solver.m_dm, 3, 0);
    amrex::MultiFab errors(solver.m_ba, solver.m_dm, 6, 0);
    amrex::MultiFab divergence(solver.m_ba, solver.m_dm, 1, 0);
    compute_metric_divergence(faces, metric, epoch, cfg, divergence);
    auto initial_divergence = divergence.norm0(0,0,true);
    amrex::ParallelDescriptor::ReduceRealMax(initial_divergence);
    if (!std::isfinite(initial_divergence) || initial_divergence > 1e-10)
        throw std::runtime_error("mapped pressure wave exact face initialization failed divergence gate");
    auto const volume = global_sum(metric.cell_volume_cc().sum(0, true));

    std::ofstream report;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        report.open(report_path);
        report.precision(17);
    }
    check_io(!amrex::ParallelDescriptor::IOProcessor() || bool(report), "cannot open pressure wave report");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        report << "{\"schema\":\"avwis-mapped-pressure-wave-v1\",\"status\":\"COMPLETED\","
               << "\"pressure_time_layer\":\"explicit n; velocity stored at n+1\","
               << "\"forcing\":\"analytic continuum grad(p) at n\","
               << "\"coordinates\":\"" << coordinate_system_mode_name(cfg.coordinates) << "\","
               << "\"mapping_type\":\"" << cfg.mapping_type << "\","
               << "\"pressure_solver\":\"" << projection_operator_mode_name(cfg.projection) << "\","
               << "\"wave_vector\":[" << kx << ',' << ky << ",0],"
               << "\"ranks\":" << amrex::ParallelDescriptor::NProcs()
               << ",\"n_cell\":[" << domain.length(0) << ',' << domain.length(1) << ',' << domain.length(2)
               << "],\"epsilon\":" << cfg.shear_epsilon << ",\"dt\":" << dt
               << ",\"viscosity\":" << viscosity << ",\"steps\":" << steps
               << ",\"initial_divergence\":" << initial_divergence << ",\"series\":[";
    }
    amrex::Real max_divergence = initial_divergence;
    amrex::Real max_residual = 0;
    for (int n=0; n<steps; ++n) {
        auto const time_n = n*dt;
        auto const time_new = (n+1)*dt;
        auto const pa = pressure_amplitude*std::exp(-time_n);
        auto const pa_new = pressure_amplitude*std::exp(-time_new);
        auto const ua = velocity_amplitude*std::exp(-viscosity*k2*time_n);
        auto const ua_new = velocity_amplitude*std::exp(-viscosity*k2*time_new);
        for (amrex::MFIter mfi(forcing, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const f = forcing.array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                auto const theta = wx*(i-ilo+0.5)*h[0] + wy*(j-jlo+0.5)*h[1];
                auto const value = -pa*std::sin(theta);
                f(i,j,k,0) = kx*value;
                f(i,j,k,1) = ky*value;
                f(i,j,k,2) = 0;
            });
        }
        amrex::MultiFab::Copy(previous, solver.m_ucat, 0, 0, 3, 0);
        solver.advance_one_step(dt, viscosity, {}, &forcing);
        compute_metric_cell_gradient(solver.m_p, metric, epoch, cfg, gradient);
        for (amrex::MFIter mfi(errors, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const e = errors.array(mfi);
            auto const u = solver.m_ucat.const_array(mfi);
            auto const old = previous.const_array(mfi);
            auto const p = solver.m_p.const_array(mfi);
            auto const g = gradient.const_array(mfi);
            auto const f = forcing.const_array(mfi);
            auto const v = metric.cell_volume_cc().const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                auto const theta = wx*(i-ilo+0.5)*h[0] + wy*(j-jlo+0.5)*h[1];
                auto const sine = std::sin(theta), cosine = std::cos(theta);
                amrex::Real ue=0, ge=0, balance=0;
                for (int c=0; c<3; ++c) {
                    auto const b = c==0 ? bx : c==1 ? by : 0.0;
                    auto const du = u(i,j,k,c) - ua_new*b*sine;
                    auto const dg = g(i,j,k,c) - f(i,j,k,c);
                    // Independent continuum momentum residual at time n:
                    // delta(u)/dt + grad(P) - nu Laplacian(u_exact^n) - f^n.
                    auto const r = (u(i,j,k,c)-old(i,j,k,c))/dt + g(i,j,k,c)
                                 + viscosity*k2*ua*b*sine - f(i,j,k,c);
                    ue += du*du; ge += dg*dg; balance += r*r;
                }
                auto const pe = p(i,j,k)-pa*cosine;
                auto const pn = p(i,j,k)-pa_new*cosine;
                e(i,j,k,0) = ue*v(i,j,k);
                e(i,j,k,1) = pe*pe*v(i,j,k);
                e(i,j,k,2) = ge*v(i,j,k);
                e(i,j,k,3) = balance*v(i,j,k);
                e(i,j,k,4) = pn*pn*v(i,j,k);
                e(i,j,k,5) = p(i,j,k)*v(i,j,k);
            });
        }
        amrex::Array<amrex::Real, 6> norms{};
        for (int c=0; c<6; ++c) {
            auto const value = global_sum(errors.sum(c, true))/volume;
            norms[c] = c==5 ? value : std::sqrt(value);
            if (!std::isfinite(norms[c])) throw std::runtime_error("pressure wave produced a nonfinite error");
        }
        auto const& projection = solver.last_step_projection();
        max_divergence = std::max(max_divergence, projection.max_divergence_after);
        max_residual = std::max(max_residual, projection.full_relative_residual);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (n) report << ',';
            report << "{\"step\":" << n+1 << ",\"velocity_time\":" << time_new
                   << ",\"pressure_time\":" << time_n << ",\"velocity_l2\":" << norms[0]
                   << ",\"pressure_l2\":" << norms[1] << ",\"pressure_gradient_l2\":" << norms[2]
                   << ",\"momentum_balance_l2\":" << norms[3]
                   << ",\"pressure_at_velocity_time_l2\":" << norms[4]
                   << ",\"pressure_mean\":" << norms[5] << '}';
        }
    }
    if (amrex::ParallelDescriptor::IOProcessor())
        report << "],\"max_divergence\":" << max_divergence
               << ",\"max_full_relative_residual\":" << max_residual << "}\n";
    check_io(!amrex::ParallelDescriptor::IOProcessor() || bool(report), "cannot write pressure wave report");

    if (!field_path.empty()) {
        amrex::MultiFab fields(solver.m_ba, solver.m_dm, 9, 0);
        for (amrex::MFIter mfi(fields, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const a = fields.array(mfi);
            auto const u = solver.m_ucat.const_array(mfi);
            auto const p = solver.m_p.const_array(mfi);
            auto const phi = solver.m_phi.const_array(mfi);
            auto const xyz = metric.cell_center_coordinates_cc().const_array(mfi);
            auto const pa = pressure_amplitude*std::exp(-(steps-1)*dt);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int c=0; c<3; ++c) { a(i,j,k,c)=xyz(i,j,k,c); a(i,j,k,3+c)=u(i,j,k,c); }
                a(i,j,k,6)=p(i,j,k); a(i,j,k,7)=phi(i,j,k);
                a(i,j,k,8)=pa*std::cos(wx*(i-ilo+0.5)*h[0]+wy*(j-jlo+0.5)*h[1]);
            });
        }
        amrex::DistributionMapping io(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
        amrex::MultiFab gathered(amrex::BoxArray(domain), io, 9, 0);
        gathered.ParallelCopy(fields);
        amrex::Gpu::streamSynchronize();
        bool ok = true;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::FArrayBox host(domain, 9, amrex::The_Pinned_Arena());
            amrex::Gpu::dtoh_memcpy(host.dataPtr(), gathered[0].dataPtr(), host.size()*sizeof(amrex::Real));
            std::ofstream csv(field_path); csv.precision(17);
            csv << "i,j,k,x,y,z,u,v,w,P,Phi,P_exact_at_n\n";
            auto const a = host.const_array();
            for (int k=domain.smallEnd(2); k<=domain.bigEnd(2); ++k)
                for (int j=domain.smallEnd(1); j<=domain.bigEnd(1); ++j)
                    for (int i=domain.smallEnd(0); i<=domain.bigEnd(0); ++i) {
                        csv << i << ',' << j << ',' << k;
                        for (int c=0; c<9; ++c) csv << ',' << a(i,j,k,c);
                        csv << '\n';
                    }
            ok = bool(csv);
        }
        check_io(ok, "cannot write pressure wave field CSV");
    }
    amrex::Print() << "Mapped pressure wave completed: steps=" << steps
                   << " velocity_time=" << steps*dt << " pressure_time=" << (steps-1)*dt
                   << " max_divergence=" << max_divergence << '\n';
}

void run_mapped_pressure_wave_case(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path,
    std::string const& field_path)
{
    AVWiSCaseRunnerAccess::run_mapped_pressure_wave_case_impl(
        solver, dt, final_time, viscosity, report_path, field_path);
}
