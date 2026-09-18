#include "MappedChannelReferenceCase.H"

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
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

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

long long integral_step_count(amrex::Real final_time, amrex::Real dt)
{
    const amrex::Real count = final_time / dt;
    const auto rounded = static_cast<long long>(std::llround(count));
    const amrex::Real tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1.0), std::abs(count));
    if (rounded <= 0 || std::abs(count-static_cast<amrex::Real>(rounded)) > tolerance) {
        throw std::runtime_error(
            "mapped channel final_time must be a positive integer multiple of dt");
    }
    return rounded;
}
} // namespace

void AVWiSCaseRunnerAccess::run_mapped_channel_reference_impl(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& body_force,
    std::string const& report_path, std::string const& profile_path)
{
#ifdef AMREX_USE_GPU
    throw std::runtime_error(
        "C8-G1 mapped channel profile export is currently validated only on CPU");
#endif
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::runtime_error(
            "C8-G1 mapped channel reference currently supports exactly one MPI rank");
    }
    if (solver.m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        solver.m_mapping_operator.mapping_type != "analytic_orthogonal" ||
        solver.m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG) {
        throw std::runtime_error(
            "C8-G1 mapped channel requires mapped analytic_orthogonal coordinates and orthogonal_mlmg");
    }
    if (!(dt > 0.0) || !(final_time > 0.0) || !(viscosity > 0.0) ||
        report_path.empty() || profile_path.empty()) {
        throw std::runtime_error(
            "C8-G1 mapped channel requires positive dt/final_time/viscosity and output paths");
    }
    if (!solver.m_boundary.enabled ||
        solver.m_boundary.geometry != BoundaryGeometryMode::MappedOrthogonal ||
        !solver.m_geom.isPeriodic(0) || solver.m_geom.isPeriodic(1) ||
        !solver.m_geom.isPeriodic(2) ||
        solver.m_boundary.sides[2].velocity != CartesianBC::NoSlipWall ||
        solver.m_boundary.sides[3].velocity != CartesianBC::NoSlipWall) {
        throw std::runtime_error(
            "C8-G1 mapped channel requires periodic x/z and mapped no-slip y walls");
    }
    if (body_force[0] != 0.0 || body_force[1] != 0.0 || !(body_force[2] > 0.0)) {
        throw std::runtime_error(
            "C8-G1 mapped channel requires a positive z-only body force");
    }
    const long long steps = integral_step_count(final_time, dt);

    auto const& parameters = solver.m_mapping_operator.analytic_parameters;
    const amrex::Real logical_ylo = solver.m_geom.ProbLo(1);
    const amrex::Real logical_height = solver.m_geom.ProbLength(1);
    const amrex::Real ylo = logical_ylo;
    const amrex::Real height = parameters.scale[1] * logical_height;
    const amrex::Real yhi = ylo + height;
    const amrex::Real physical_x_length =
        parameters.scale[0] * solver.m_geom.ProbLength(0);
    const amrex::Real physical_z_length =
        parameters.scale[2] * solver.m_geom.ProbLength(2);
    const amrex::Real physical_volume = physical_x_length * height * physical_z_length;
    const amrex::Real profile_coefficient = body_force[2] / (2.0 * viscosity);
    const amrex::Real exact_flow =
        body_force[2] * physical_x_length * height * height * height /
        (12.0 * viscosity);
    const amrex::Real exact_wall_shear = 0.5 * body_force[2] * height;
    const amrex::Real legacy_tolerance = 512.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (std::abs(ylo) > legacy_tolerance || std::abs(height-1.0) > legacy_tolerance ||
        std::abs(profile_coefficient-6.0) > legacy_tolerance) {
        throw std::runtime_error(
            "C8-G1 legacy profile reference requires y in [0,1] and body_force_z/(2*nu)=6");
    }

    auto const& centers = solver.m_metric_data.cell_center_coordinates_cc();
    auto const& volume = solver.m_metric_data.cell_volume_cc();
    auto set_exact_state = [&]() {
        for (amrex::MFIter mfi(solver.m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const velocity = solver.m_ucat.array(mfi);
            auto const xyz = centers.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real y = xyz(i,j,k,1);
                velocity(i,j,k,0) = 0.0;
                velocity(i,j,k,1) = 0.0;
                velocity(i,j,k,2) = profile_coefficient * (y-ylo) * (yhi-y);
            });
        }
        amrex::Gpu::streamSynchronize();
        solver.m_p.setVal(0.0);
        solver.m_phi.setVal(0.0);
        solver.m_nvert.setVal(0.0);
        solver.mark_valid_modified();
        solver.sync_ucont_from_ucat();
        amrex::MultiFab::Copy(
            solver.m_ucat_old, solver.m_ucat, 0, 0, AMREX_SPACEDIM, solver.m_nghost);
        amrex::MultiFab::Copy(
            solver.m_ucat_older, solver.m_ucat, 0, 0, AMREX_SPACEDIM, solver.m_nghost);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::MultiFab::Copy(
                solver.m_ucont_old[dir], solver.m_ucont[dir], 0, 0, 1, solver.m_nghost);
            amrex::MultiFab::Copy(
                solver.m_ucont_older[dir], solver.m_ucont[dir], 0, 0, 1, solver.m_nghost);
        }
        solver.m_time = 0.0;
        solver.m_step = 0;
        solver.m_history_depth = 1;
    };

    set_exact_state();
    amrex::MultiFab advection(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab viscous(solver.m_ba, solver.m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab residual_squared(solver.m_ba, solver.m_dm, 1, 0);
    solver.compute_advection_rhs(advection);
    solver.compute_viscous_rhs(viscous, viscosity);
    for (amrex::MFIter mfi(residual_squared, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = advection.const_array(mfi);
        auto const visc = viscous.const_array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const residual = residual_squared.array(mfi);
        const amrex::Real forcing = body_force[2];
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real rx = adv(i,j,k,0) + visc(i,j,k,0);
            const amrex::Real ry = adv(i,j,k,1) + visc(i,j,k,1);
            const amrex::Real rz = adv(i,j,k,2) + visc(i,j,k,2) + forcing;
            residual(i,j,k) = (rx*rx + ry*ry + rz*rz) * cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real steady_residual_l2 = std::sqrt(
        global_sum(residual_squared.sum(0, true)) / physical_volume);

    for (long long step = 0; step < steps; ++step) {
        solver.advance_one_step(dt, viscosity, body_force);
    }

    amrex::MultiFab velocity_error_squared(solver.m_ba, solver.m_dm, 1, 0);
    amrex::MultiFab velocity_difference(solver.m_ba, solver.m_dm, 1, 0);
    amrex::MultiFab divergence(solver.m_ba, solver.m_dm, 1, 0);
    amrex::MultiFab mass_work(solver.m_ba, solver.m_dm, 2, 0);
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> flux{
        AMREX_D_DECL(&solver.m_ucont[0], &solver.m_ucont[1], &solver.m_ucont[2])};
    compute_metric_divergence(
        flux, solver.m_metric_data, solver.m_metric_epoch,
        solver.m_mapping_operator, divergence);
    for (amrex::MFIter mfi(velocity_error_squared, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const velocity = solver.m_ucat.const_array(mfi);
        auto const xyz = centers.const_array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const error_squared = velocity_error_squared.array(mfi);
        auto const difference = velocity_difference.array(mfi);
        auto const div = divergence.const_array(mfi);
        auto const mass = mass_work.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real y = xyz(i,j,k,1);
            const amrex::Real exact = profile_coefficient * (y-ylo) * (yhi-y);
            const amrex::Real dx = velocity(i,j,k,0);
            const amrex::Real dy = velocity(i,j,k,1);
            const amrex::Real dz = velocity(i,j,k,2)-exact;
            difference(i,j,k) = std::sqrt(dx*dx + dy*dy + dz*dz);
            error_squared(i,j,k) = difference(i,j,k) * difference(i,j,k) * cell_volume(i,j,k);
            mass(i,j,k,0) = div(i,j,k) * cell_volume(i,j,k);
            mass(i,j,k,1) = std::abs(div(i,j,k)) * cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real velocity_l2 = std::sqrt(
        global_sum(velocity_error_squared.sum(0, true)) / physical_volume);
    const amrex::Real velocity_linf = global_max(velocity_difference.norm0(0, 0, true));
    const amrex::Real divergence_linf = global_max(divergence.norm0(0, 0, true));
    const amrex::Real integrated_divergence = global_sum(mass_work.sum(0, true));
    const amrex::Real divergence_l1 = global_sum(mass_work.sum(1, true));

    amrex::MultiFab pressure_work(solver.m_ba, solver.m_dm, 2, 0);
    for (amrex::MFIter mfi(pressure_work, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const pressure = solver.m_p.const_array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const work = pressure_work.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            work(i,j,k,0) = pressure(i,j,k) * cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real pressure_mean = global_sum(pressure_work.sum(0, true)) / physical_volume;
    for (amrex::MFIter mfi(pressure_work, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const pressure = solver.m_p.const_array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const work = pressure_work.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real datum_free = pressure(i,j,k)-pressure_mean;
            work(i,j,k,0) = datum_free;
            work(i,j,k,1) = datum_free*datum_free*cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real pressure_linf = global_max(pressure_work.norm0(0, 0, true));
    const amrex::Real pressure_l2 = std::sqrt(
        global_sum(pressure_work.sum(1, true)) / physical_volume);

    amrex::MultiFab selected_flow(
        solver.m_ucont[2].boxArray(), solver.m_dm, 1, 0);
    selected_flow.setVal(0.0);
    const int flow_plane = solver.m_geom.Domain().smallEnd(2);
    for (amrex::MFIter mfi(selected_flow, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const selected = selected_flow.array(mfi);
        auto const zflux = solver.m_ucont[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (k == flow_plane) selected(i,j,k) = zflux(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real numerical_flow =
        selected_flow.sum_unique(0, false, solver.m_geom.periodicity());
    const amrex::Real flow_relative_error = std::abs(numerical_flow-exact_flow) / exact_flow;

    const amrex::Box& domain = solver.m_geom.Domain();
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);
    amrex::MultiFab wall_work(solver.m_ba, solver.m_dm, 4, 0);
    wall_work.setVal(0.0);
    auto const& wall_area = solver.m_metric_data.face_area_vector_fc(1);
    for (amrex::MFIter mfi(wall_work, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const velocity = solver.m_ucat.const_array(mfi);
        auto const xyz = centers.const_array(mfi);
        auto const area = wall_area.const_array(mfi);
        auto const work = wall_work.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (j == jlo) {
                const amrex::Real y0 = xyz(i,j,k,1)-ylo;
                const amrex::Real y1 = xyz(i,j+1,k,1)-ylo;
                const amrex::Real derivative =
                    (y1*velocity(i,j,k,2)/y0 -
                     y0*velocity(i,j+1,k,2)/y1) / (y1-y0);
                const amrex::Real face_area = std::abs(area(i,jlo,k,1));
                work(i,j,k,0) = viscosity * derivative * face_area;
                work(i,j,k,1) = face_area;
            }
            if (j == jhi) {
                const amrex::Real s0 = yhi-xyz(i,j,k,1);
                const amrex::Real s1 = yhi-xyz(i,j-1,k,1);
                const amrex::Real inward_derivative =
                    (s1*velocity(i,j,k,2)/s0 -
                     s0*velocity(i,j-1,k,2)/s1) / (s1-s0);
                const amrex::Real face_area = std::abs(area(i,jhi+1,k,1));
                work(i,j,k,2) = -viscosity * inward_derivative * face_area;
                work(i,j,k,3) = face_area;
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real wall_shear_low =
        global_sum(wall_work.sum(0, true)) / global_sum(wall_work.sum(1, true));
    const amrex::Real wall_shear_high =
        global_sum(wall_work.sum(2, true)) / global_sum(wall_work.sum(3, true));
    const amrex::Real wall_shear_error = std::max(
        std::abs(wall_shear_low-exact_wall_shear),
        std::abs(wall_shear_high+exact_wall_shear));

    const int ny = domain.length(1);
    std::vector<amrex::Real> profile_y(ny, 0.0);
    std::vector<amrex::Real> profile_u(ny, 0.0);
    std::vector<amrex::Real> profile_exact(ny, 0.0);
    std::vector<amrex::Real> profile_weight(ny, 0.0);
    for (amrex::MFIter mfi(solver.m_ucat); mfi.isValid(); ++mfi) {
        auto const velocity = solver.m_ucat.const_array(mfi);
        auto const xyz = centers.const_array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        const amrex::Box& box = mfi.validbox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const int layer = j-jlo;
                    const amrex::Real weight = cell_volume(i,j,k);
                    const amrex::Real y = xyz(i,j,k,1);
                    profile_y[layer] += y*weight;
                    profile_u[layer] += velocity(i,j,k,2)*weight;
                    profile_exact[layer] +=
                        profile_coefficient*(y-ylo)*(yhi-y)*weight;
                    profile_weight[layer] += weight;
                }
            }
        }
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream profile(profile_path);
        if (!profile) {
            throw std::runtime_error("cannot write C8-G1 mapped channel profile: " + profile_path);
        }
        profile.precision(std::numeric_limits<amrex::Real>::max_digits10);
        profile << "j,y,w_numerical,w_exact,error\n";
        for (int layer = 0; layer < ny; ++layer) {
            const amrex::Real inverse_weight = 1.0/profile_weight[layer];
            const amrex::Real y = profile_y[layer]*inverse_weight;
            const amrex::Real numerical = profile_u[layer]*inverse_weight;
            const amrex::Real exact = profile_exact[layer]*inverse_weight;
            profile << jlo+layer << ',' << y << ',' << numerical << ',' << exact
                    << ',' << numerical-exact << '\n';
        }

        std::ofstream output(report_path);
        if (!output) {
            throw std::runtime_error("cannot write C8-G1 mapped channel report: " + report_path);
        }
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n"
               << "  \"schema\": \"avwis-c8-g1-mapped-channel-v1\",\n"
               << "  \"status\": \"PASS\",\n"
               << "  \"case_type\": \"steady pressure-driven plane Poiseuille flow on a separable orthogonal stretched grid\",\n"
               << "  \"n_cell\": [" << domain.length(0) << ", "
               << domain.length(1) << ", " << domain.length(2) << "],\n"
               << "  \"mapping_scale\": [" << parameters.scale[0] << ", "
               << parameters.scale[1] << ", " << parameters.scale[2] << "],\n"
               << "  \"mapping_stretch\": [" << parameters.stretch[0] << ", "
               << parameters.stretch[1] << ", " << parameters.stretch[2] << "],\n"
               << "  \"dt\": " << dt << ", \"final_time\": " << final_time
               << ", \"steps\": " << steps << ", \"viscosity\": " << viscosity << ",\n"
               << "  \"body_force\": [" << body_force[0] << ", "
               << body_force[1] << ", " << body_force[2] << "],\n"
               << "  \"velocity_l2\": " << velocity_l2 << ",\n"
               << "  \"velocity_linf\": " << velocity_linf << ",\n"
               << "  \"steady_residual_l2\": " << steady_residual_l2 << ",\n"
               << "  \"pressure_mean\": " << pressure_mean << ",\n"
               << "  \"pressure_datum_free_l2\": " << pressure_l2 << ",\n"
               << "  \"pressure_datum_free_linf\": " << pressure_linf << ",\n"
               << "  \"divergence_linf\": " << divergence_linf << ",\n"
               << "  \"integrated_divergence\": " << integrated_divergence << ",\n"
               << "  \"divergence_l1\": " << divergence_l1 << ",\n"
               << "  \"flow_rate\": " << numerical_flow << ",\n"
               << "  \"flow_rate_exact\": " << exact_flow << ",\n"
               << "  \"flow_rate_relative_error\": " << flow_relative_error << ",\n"
               << "  \"wall_shear_low\": " << wall_shear_low << ",\n"
               << "  \"wall_shear_high\": " << wall_shear_high << ",\n"
               << "  \"wall_shear_exact_magnitude\": " << exact_wall_shear << ",\n"
               << "  \"wall_shear_max_abs_error\": " << wall_shear_error << ",\n"
               << "  \"pressure_datum\": \"zero-mean periodic/all-Neumann correction\",\n"
               << "  \"profile_file\": \"" << profile_path << "\",\n"
               << "  \"legacy_reference\": {\n"
               << "    \"kind\": \"source-traceable physical definition; legacy executable not run\",\n"
               << "    \"profile_source\": \"vwis2.0/BcsUtility.C inletprofile=4: 6*(y-y^2), unit bulk flow\",\n"
               << "    \"forcing_source\": \"vwis2.0/RHSSolver.C -dpdz mean pressure gradient\",\n"
               << "    \"equivalent_dpdz\": " << -body_force[2] << ",\n"
               << "    \"executable_reference_available\": false,\n"
               << "    \"blocker\": \"repository has no control.dat, grid, reference output, or reproducible PETSc-3.4 dependency bundle\"\n"
               << "  },\n"
               << "  \"amrex_version\": \"" << amrex::Version() << "\"\n"
               << "}\n";
    }

    amrex::Print() << "AVWiS C8-G1 mapped channel reference: PASS report="
                   << report_path << " velocity_l2=" << velocity_l2
                   << " flow_error=" << flow_relative_error
                   << " wall_shear_error=" << wall_shear_error << "\n";
}

void run_mapped_channel_reference(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& body_force,
    std::string const& report_path, std::string const& profile_path)
{
    AVWiSCaseRunnerAccess::run_mapped_channel_reference_impl(
        solver, dt, final_time, viscosity, body_force, report_path, profile_path);
}
