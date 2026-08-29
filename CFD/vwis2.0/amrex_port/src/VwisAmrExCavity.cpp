#include "VwisAmrExSolver.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VWIS_AMREX_LOCKED_VERSION
#define VWIS_AMREX_LOCKED_VERSION "unknown"
#endif
#ifndef VWIS_AMREX_LOCKED_GIT_SHA
#define VWIS_AMREX_LOCKED_GIT_SHA "unknown"
#endif

namespace {

void require_output(std::ofstream const& output, std::string const& path)
{
    if (!output) throw std::runtime_error("lid cavity cannot write output: " + path);
}

bool finite_flow(UniformFlowDiagnostics const& value)
{
    bool finite = std::isfinite(value.integrated_divergence) &&
        std::isfinite(value.max_abs_divergence) &&
        std::isfinite(value.net_mass_flux) && std::isfinite(value.outlet_flow) &&
        std::isfinite(value.kinetic_energy) && std::isfinite(value.pressure_mean) &&
        std::isfinite(value.pressure_min) && std::isfinite(value.pressure_max);
    for (auto component : value.momentum) finite = finite && std::isfinite(component);
    return finite;
}

} // namespace

void VwisAmrExSolver::run_lid_driven_cavity(
    amrex::Real dt, int steps, amrex::Real viscosity,
    std::string const& report_path, std::string const& field_path,
    std::string const& centerline_path, std::string const& history_path)
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::runtime_error("lid cavity demonstration currently supports exactly one MPI rank");
    }
#ifdef AMREX_USE_GPU
    throw std::runtime_error("lid cavity CSV demonstration is validated only with an AMReX CPU build");
#endif
    const auto& domain = m_geom.Domain();
    if (steps <= 0 || !std::isfinite(dt) || dt <= 0.0 ||
        !std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::runtime_error("lid cavity requires positive finite dt, steps, and viscosity");
    }
    if (domain.length(0) != domain.length(1) || domain.length(2) != 1 ||
        m_geom.ProbLength(0) != m_geom.ProbLength(1) || !m_geom.isPeriodic(2) ||
        m_geom.isPeriodic(0) || m_geom.isPeriodic(1)) {
        throw std::runtime_error(
            "lid cavity requires square Nx x Ny x 1 geometry with only z periodic");
    }
    const CartesianBC expected[4] = {
        CartesianBC::NoSlipWall, CartesianBC::NoSlipWall,
        CartesianBC::NoSlipWall, CartesianBC::MovingWall};
    for (int slot = 0; slot < 4; ++slot) {
        if (m_boundary.sides[slot].velocity != expected[slot]) {
            throw std::runtime_error(
                "lid cavity requires noslip xlo/xhi/ylo and moving_wall yhi");
        }
    }
    const amrex::Real lid_speed = m_boundary.moving_wall_velocity[0];
    if (!(lid_speed > 0.0) || m_boundary.moving_wall_velocity[1] != 0.0 ||
        m_boundary.moving_wall_velocity[2] != 0.0) {
        throw std::runtime_error("lid cavity requires positive x-directed tangential lid velocity");
    }

    std::ofstream history(history_path, std::ios::trunc);
    require_output(history, history_path);
    history << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
            << "step,time,post_projection_max_abs_divergence,integrated_divergence,"
               "net_mass_flux,kinetic_energy,center_u,center_v,advective_cfl,diffusive_number\n";

    amrex::Real max_post_divergence = 0.0;
    amrex::Real max_mass_imbalance = 0.0;
    bool all_finite = true;
    UniformFlowDiagnostics final_flow{};
    UniformPointSample center{};
    TimeStepDiagnostics final_stability{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> center_position{
        AMREX_D_DECL(m_geom.ProbLo(0) + 0.5 * m_geom.ProbLength(0),
                     m_geom.ProbLo(1) + 0.5 * m_geom.ProbLength(1),
                     m_geom.ProbLo(2) + 0.5 * m_geom.ProbLength(2))};

    for (int local_step = 0; local_step < steps; ++local_step) {
        advance_one_step(dt, viscosity);
        final_flow = uniform_flow_diagnostics();
        center = sample_uniform_point(center_position);
        final_stability = time_step_diagnostics(dt, viscosity);
        max_post_divergence = std::max(max_post_divergence, final_flow.max_abs_divergence);
        max_mass_imbalance = std::max(max_mass_imbalance, std::abs(final_flow.net_mass_flux));
        all_finite = all_finite && finite_flow(final_flow) &&
            std::isfinite(center.velocity[0]) && std::isfinite(center.velocity[1]) &&
            std::isfinite(final_stability.advective_cfl) &&
            std::isfinite(final_stability.diffusive_number);
        if (final_flow.max_abs_divergence > 1.0e-8) {
            throw std::runtime_error(
                "lid cavity post-projection divergence exceeds 1e-8; refine BoxArray splitting for the one-cell periodic direction");
        }
        history << m_step << ',' << m_time << ',' << final_flow.max_abs_divergence << ','
                << final_flow.integrated_divergence << ',' << final_flow.net_mass_flux << ','
                << final_flow.kinetic_energy << ',' << center.velocity[0] << ','
                << center.velocity[1] << ',' << final_stability.advective_cfl << ','
                << final_stability.diffusive_number << '\n';
    }
    history.close();
    require_output(history, history_path);
    if (!all_finite) throw std::runtime_error("lid cavity produced NaN or Inf diagnostics");

    fill_ghost_cells();
    fill_physical_ghost_cells();
    amrex::Real wall_velocity_error = 0.0;
    const int yhi = domain.bigEnd(1);
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        if (box.smallEnd(1) > yhi || box.bigEnd(1) < yhi) continue;
        auto const velocity = m_ucat.const_array(mfi);
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                wall_velocity_error = std::max(
                    wall_velocity_error,
                    std::abs(0.5 * (velocity(i,yhi,k,0) + velocity(i,yhi+1,k,0)) - lid_speed));
                wall_velocity_error = std::max(
                    wall_velocity_error,
                    std::abs(0.5 * (velocity(i,yhi,k,1) + velocity(i,yhi+1,k,1))));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(wall_velocity_error);
    const amrex::Real boundary_tolerance =
        128.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (wall_velocity_error > boundary_tolerance ||
        max_mass_imbalance > boundary_tolerance) {
        throw std::runtime_error(
            "lid cavity wall velocity or closed-boundary mass sanity check failed");
    }

    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    compute_cartesian_divergence(divergence);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int ilo = domain.smallEnd(0);
    const int jlo = domain.smallEnd(1);
    const int klo = domain.smallEnd(2);
    std::vector<amrex::Real> u(nx * ny), v(nx * ny), w(nx * ny), p(nx * ny), div(nx * ny);
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        auto const velocity = m_ucat.const_array(mfi);
        auto const pressure = m_p.const_array(mfi);
        auto const divergence_value = divergence.const_array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const std::size_t at = static_cast<std::size_t>(j-jlo) * nx + (i-ilo);
                u[at] = velocity(i,j,klo,0);
                v[at] = velocity(i,j,klo,1);
                w[at] = velocity(i,j,klo,2);
                p[at] = pressure(i,j,klo);
                div[at] = divergence_value(i,j,klo);
                all_finite = all_finite && std::isfinite(u[at]) && std::isfinite(v[at]) &&
                    std::isfinite(w[at]) && std::isfinite(p[at]) && std::isfinite(div[at]);
            }
        }
    }
    if (!all_finite) throw std::runtime_error("lid cavity final field contains NaN or Inf");

    std::ofstream field(field_path, std::ios::trunc);
    require_output(field, field_path);
    field << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
          << "x,y,u,v,w,velocity_magnitude,pressure,divergence\n";
    for (int j = 0; j < ny; ++j) {
        const amrex::Real y = m_geom.ProbLo(1) + (j + 0.5) * m_dx[1];
        for (int i = 0; i < nx; ++i) {
            const amrex::Real x = m_geom.ProbLo(0) + (i + 0.5) * m_dx[0];
            const std::size_t at = static_cast<std::size_t>(j) * nx + i;
            const amrex::Real speed = std::sqrt(u[at]*u[at] + v[at]*v[at] + w[at]*w[at]);
            field << x << ',' << y << ',' << u[at] << ',' << v[at] << ',' << w[at]
                  << ',' << speed << ',' << p[at] << ',' << div[at] << '\n';
        }
    }
    field.close();
    require_output(field, field_path);

    auto bracket = [](int count) {
        const amrex::Real cell_coordinate = 0.5 * count - 0.5;
        const int lower = std::max(0, std::min(count-1, static_cast<int>(std::floor(cell_coordinate))));
        const int upper = std::max(0, std::min(count-1, lower + 1));
        return std::array<amrex::Real,3>{static_cast<amrex::Real>(lower),
                                         static_cast<amrex::Real>(upper),
                                         cell_coordinate - lower};
    };
    const auto xb = bracket(nx);
    const auto yb = bracket(ny);
    std::ofstream centerline(centerline_path, std::ios::trunc);
    require_output(centerline, centerline_path);
    centerline << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
               << "profile,coordinate,velocity\n";
    for (int j = 0; j < ny; ++j) {
        const int il = static_cast<int>(xb[0]);
        const int ir = static_cast<int>(xb[1]);
        const amrex::Real value = (1.0-xb[2]) * u[static_cast<std::size_t>(j)*nx+il] +
                                  xb[2] * u[static_cast<std::size_t>(j)*nx+ir];
        centerline << "u_at_x_0.5," << m_geom.ProbLo(1) + (j+0.5)*m_dx[1] << ',' << value << '\n';
    }
    for (int i = 0; i < nx; ++i) {
        const int jl = static_cast<int>(yb[0]);
        const int jr = static_cast<int>(yb[1]);
        const amrex::Real value = (1.0-yb[2]) * v[static_cast<std::size_t>(jl)*nx+i] +
                                  yb[2] * v[static_cast<std::size_t>(jr)*nx+i];
        centerline << "v_at_y_0.5," << m_geom.ProbLo(0) + (i+0.5)*m_dx[0] << ',' << value << '\n';
    }
    centerline.close();
    require_output(centerline, centerline_path);

    std::ofstream report(report_path, std::ios::trunc);
    require_output(report, report_path);
    report << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
           << "{\n  \"schema\": \"vwis-lid-driven-cavity-v1\",\n"
           << "  \"status\": \"demonstration / engineering result; not CFD validation\",\n"
           << "  \"case_type\": \"2D-equivalent 3D Cartesian lid-driven square cavity\",\n"
           << "  \"amrex_release_locked\": \"" << VWIS_AMREX_LOCKED_VERSION << "\",\n"
           << "  \"amrex_git_sha\": \"" << VWIS_AMREX_LOCKED_GIT_SHA << "\",\n"
           << "  \"amrex_runtime_version\": \"" << amrex::Version() << "\",\n"
           << "  \"compiler\": \"" << __VERSION__ << "\",\n"
           << "  \"grid\": [" << nx << ", " << ny << ", 1],\n"
           << "  \"periodic\": [false, false, true],\n"
           << "  \"dt\": " << dt << ",\n  \"steps\": " << steps
           << ",\n  \"final_time\": " << m_time << ",\n  \"lid_velocity\": ["
           << m_boundary.moving_wall_velocity[0] << ", "
           << m_boundary.moving_wall_velocity[1] << ", "
           << m_boundary.moving_wall_velocity[2] << "],\n"
           << "  \"viscosity\": " << viscosity << ",\n  \"reynolds_number\": "
           << lid_speed * m_geom.ProbLength(0) / viscosity << ",\n"
           << "  \"time_integrator\": \"explicit Euler predictor plus Cartesian pressure projection\",\n"
           << "  \"spatial_scheme\": \"cell-centred second-order central advection and viscosity; MAC projection\",\n"
           << "  \"initial_condition\": \"fluid at rest\",\n"
           << "  \"boundary_conditions\": {\"xlo\":\"noslip\",\"xhi\":\"noslip\","
              "\"ylo\":\"noslip\",\"yhi\":\"moving_wall\",\"z\":\"periodic\"},\n"
           << "  \"centerline_method\": \"linear interpolation between adjacent cell centres to x=0.5 or y=0.5; no smoothing\",\n"
           << "  \"representative_velocity_method\": \"velocity in the cell containing (0.5,0.5,0.5)\",\n"
           << "  \"raw_outputs\": {\"field\":\"" << std::filesystem::path(field_path).filename().string()
           << "\",\"centerlines\":\"" << std::filesystem::path(centerline_path).filename().string()
           << "\",\"history\":\"" << std::filesystem::path(history_path).filename().string() << "\"},\n"
           << "  \"sanity\": {\n    \"all_finite\": true,\n"
           << "    \"no_nan_or_inf\": true,\n"
           << "    \"max_post_projection_abs_divergence\": " << max_post_divergence << ",\n"
           << "    \"final_post_projection_abs_divergence\": " << final_flow.max_abs_divergence << ",\n"
           << "    \"final_integrated_divergence\": " << final_flow.integrated_divergence << ",\n"
           << "    \"max_abs_net_mass_flux\": " << max_mass_imbalance << ",\n"
           << "    \"final_net_mass_flux\": " << final_flow.net_mass_flux << ",\n"
           << "    \"moving_wall_reconstruction_max_error\": " << wall_velocity_error << ",\n"
           << "    \"final_kinetic_energy\": " << final_flow.kinetic_energy << ",\n"
           << "    \"final_center_velocity\": [" << center.velocity[0] << ", " << center.velocity[1] << ", " << center.velocity[2] << "],\n"
           << "    \"final_advective_cfl\": " << final_stability.advective_cfl << ",\n"
           << "    \"diffusive_number\": " << final_stability.diffusive_number << "\n  },\n"
           << "  \"limitations\": \"single-level uniform CPU/single-rank demonstration; no reference comparison, AMR, IBM/EB, LES, or steady-state claim\"\n}\n";
    report.close();
    require_output(report, report_path);
    amrex::Print() << "VWiS lid-driven cavity demonstration complete: report=" << report_path
                   << " steps=" << steps << " final_time=" << m_time << "\n";
}
