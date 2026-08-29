#include "VwisAmrExSolver.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

void VwisAmrExSolver::run_physical_benchmark(
    amrex::Real dt, int steps, amrex::Real viscosity, std::string const& report_path)
{
    if (!(dt > 0.0) || steps <= 0 || !(viscosity >= 0.0) || report_path.empty()) {
        throw std::runtime_error("physical channel benchmark requires positive dt/steps, non-negative viscosity, and report path");
    }
    if (m_geom.isPeriodic(0) || !m_geom.isPeriodic(2) || !m_boundary.enabled ||
        m_boundary.sides[0].velocity != CartesianBC::Inflow ||
        m_boundary.sides[1].velocity != CartesianBC::Outflow ||
        m_boundary.sides[2].velocity != CartesianBC::NoSlipWall ||
        m_boundary.sides[3].velocity != CartesianBC::NoSlipWall) {
        throw std::runtime_error("physical channel requires x inflow/outflow, y no-slip walls, and z periodicity");
    }

    // Rest fluid is the physical initial condition.  The inlet supplies the
    // momentum; no body force or manufactured source is added.
    m_ucat.setVal(0.0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].setVal(0.0);
        m_ucont_old[dir].setVal(0.0);
        m_ucont_older[dir].setVal(0.0);
    }
    mark_valid_modified();
    apply_boundary_pipeline("physical-channel-initialize");
    amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    amrex::MultiFab::Copy(m_ucat_older, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont[dir], 0, 0, 1, m_nghost);
    }
    m_time = 0.0;
    m_step = 0;
    m_history_depth = 1;

    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(report_path);
        if (!output) throw std::runtime_error("cannot write physical channel report: " + report_path);
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n  \"schema\": \"vwis-physical-channel-v1\",\n"
               << "  \"status\": \"physical run / not yet validated\",\n"
               << "  \"reference_available\": false,\n"
               << "  \"case_type\": \"physical Cartesian plane channel\",\n"
               << "  \"physical_meaning\": \"incompressible viscous flow driven by prescribed inlet flux and outlet pressure\",\n"
               << "  \"domain\": [[0,0,0],[1,1,1]],\n"
               << "  \"n_cell\": [" << m_geom.Domain().length(0) << ", "
               << m_geom.Domain().length(1) << ", " << m_geom.Domain().length(2) << "],\n"
               << "  \"dx\": [" << m_dx[0] << ", " << m_dx[1] << ", " << m_dx[2] << "],\n"
               << "  \"dt\": " << dt << ",\n  \"nu\": " << viscosity
               << ",\n  \"Re_inlet\": " << (m_boundary.inlet_target_flux / (m_geom.ProbLength(1) * m_geom.ProbLength(2))) / viscosity
               << ",\n  \"steps\": " << steps << ",\n  \"final_time\": " << dt * steps << ",\n"
               << "  \"initial_condition\": \"fluid at rest; inlet/outlet BC applied at t=0\",\n"
               << "  \"driving\": \"xlo uniform inlet target flux; xhi fixed pressure outlet\",\n"
               << "  \"boundary_conditions\": {\"xlo\":\"inflow\",\"xhi\":\"outflow pressure=0\",\"ylo\":\"no-slip\",\"yhi\":\"no-slip\",\"zlo\":\"periodic\",\"zhi\":\"periodic\"},\n"
               << "  \"pressure_datum\": \"xhi outlet pressure=0; projection correction Dirichlet there\",\n"
               << "  \"records\": [\n";
    }

    const amrex::Box& domain = m_geom.Domain();
    const int xlo = domain.smallEnd(0);
    const int xhi = domain.bigEnd(0);
    const amrex::Real cross_section = m_geom.ProbLength(1) * m_geom.ProbLength(2);
    const amrex::Real inlet_speed = m_boundary.inlet_target_flux / cross_section;
    amrex::Real total_step_seconds = 0.0;
    UniformFlowDiagnostics time_sum{};
    amrex::Real time_sum_section_u_in = 0.0;
    amrex::Real time_sum_section_u_out = 0.0;
    amrex::Real time_sum_center_u = 0.0;
    amrex::Real time_sum_pressure_drop = 0.0;
    for (int step = 0; step < steps; ++step) {
        advance_one_step(dt, viscosity);
        total_step_seconds += m_last_advance_seconds;
        const UniformFlowDiagnostics flow = uniform_flow_diagnostics();
        const UniformPlaneStatistics inlet_section = uniform_plane_statistics(0, xlo);
        const UniformPlaneStatistics outlet_section = uniform_plane_statistics(0, xhi);
        amrex::Array<amrex::Real, AMREX_SPACEDIM> center_position{};
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int center_index = domain.smallEnd(dir) + domain.length(dir) / 2;
            center_position[dir] = m_geom.ProbLo(dir) +
                (static_cast<amrex::Real>(center_index - domain.smallEnd(dir)) + 0.5) * m_dx[dir];
        }
        const UniformPointSample center = sample_uniform_point(center_position);
        const amrex::Real mean_u_in = inlet_section.mean_velocity[0];
        const amrex::Real mean_u_out = outlet_section.mean_velocity[0];
        const amrex::Real center_u = center.velocity[0];
        const amrex::Real pressure_drop = inlet_section.mean_pressure - outlet_section.mean_pressure;
        time_sum.integrated_divergence += flow.integrated_divergence;
        time_sum.max_abs_divergence += flow.max_abs_divergence;
        time_sum.net_mass_flux += flow.net_mass_flux;
        time_sum.outlet_flow += flow.outlet_flow;
        time_sum.kinetic_energy += flow.kinetic_energy;
        time_sum.pressure_mean += flow.pressure_mean;
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) time_sum.momentum[comp] += flow.momentum[comp];
        time_sum_section_u_in += mean_u_in;
        time_sum_section_u_out += mean_u_out;
        time_sum_center_u += center_u;
        time_sum_pressure_drop += pressure_drop;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (step != 0) output << ",\n";
            output << "    {\"step\": " << step + 1 << ", \"time\": " << m_time
                   << ", \"step_seconds\": " << m_last_advance_seconds
                   << ", \"integrated_divergence\": " << flow.integrated_divergence
                   << ", \"post_projection_max_abs_divergence\": " << flow.max_abs_divergence
                   << ", \"net_mass_flux\": " << flow.net_mass_flux
                   << ", \"outlet_flow\": " << flow.outlet_flow
                   << ", \"section_mean_u_in\": " << mean_u_in
                   << ", \"section_mean_u_out\": " << mean_u_out
                   << ", \"centerline_u\": " << center_u
                   << ", \"pressure_drop\": " << pressure_drop
                   << ", \"pressure_mean\": " << flow.pressure_mean
                   << ", \"pressure_min\": " << flow.pressure_min
                   << ", \"pressure_max\": " << flow.pressure_max
                   << ", \"momentum\": [" << flow.momentum[0] << ", " << flow.momentum[1] << ", " << flow.momentum[2]
                   << "], \"kinetic_energy\": " << flow.kinetic_energy << "}";
        }
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        const amrex::Real inverse_samples = 1.0 / static_cast<amrex::Real>(steps);
        output << "\n  ],\n  \"time_average_method\": \"arithmetic mean of equally spaced post-step samples\",\n"
               << "  \"time_averages\": {\"sample_count\": " << steps
               << ", \"integrated_divergence\": " << time_sum.integrated_divergence * inverse_samples
               << ", \"max_abs_divergence\": " << time_sum.max_abs_divergence * inverse_samples
               << ", \"net_mass_flux\": " << time_sum.net_mass_flux * inverse_samples
               << ", \"outlet_flow\": " << time_sum.outlet_flow * inverse_samples
               << ", \"momentum\": [" << time_sum.momentum[0] * inverse_samples << ", "
               << time_sum.momentum[1] * inverse_samples << ", " << time_sum.momentum[2] * inverse_samples
               << "], \"kinetic_energy\": " << time_sum.kinetic_energy * inverse_samples
               << ", \"pressure_mean\": " << time_sum.pressure_mean * inverse_samples
               << ", \"section_mean_u_in\": " << time_sum_section_u_in * inverse_samples
               << ", \"section_mean_u_out\": " << time_sum_section_u_out * inverse_samples
               << ", \"centerline_u\": " << time_sum_center_u * inverse_samples
               << ", \"pressure_drop\": " << time_sum_pressure_drop * inverse_samples << "},\n"
               << "  \"total_step_seconds\": " << total_step_seconds
               << ",\n  \"validation_note\": \"physical run / not yet validated; no legacy or literature reference data available\"\n}\n";
    }
    amrex::Print() << "VWiS physical Cartesian channel run complete: report=" << report_path
                   << " steps=" << steps << " final_time=" << m_time << "\n";
}
