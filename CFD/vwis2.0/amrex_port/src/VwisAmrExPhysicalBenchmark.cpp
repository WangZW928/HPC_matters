#include "VwisAmrExSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
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

amrex::Real cell_plane_average(amrex::MultiFab const& field, amrex::Geometry const& geom,
                               int direction, int index, int component)
{
    amrex::MultiFab plane(field.boxArray(), field.DistributionMap(), 1, 0);
    plane.setVal(0.0);
    for (amrex::MFIter mfi(plane, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const dst = plane.array(mfi);
        auto const src = field.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int coordinate = direction == 0 ? i : (direction == 1 ? j : k);
            if (coordinate == index) dst(i,j,k) = src(i,j,k,component);
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real sum = global_sum(plane.sum(0, true));
    const amrex::Box& domain = geom.Domain();
    const int count = domain.length((direction + 1) % AMREX_SPACEDIM) *
                      domain.length((direction + 2) % AMREX_SPACEDIM);
    return sum / static_cast<amrex::Real>(count);
}

amrex::Real center_value(amrex::MultiFab const& field, amrex::Box const& domain,
                         int component)
{
    const int i = domain.smallEnd(0) + domain.length(0) / 2;
    const int j = domain.smallEnd(1) + domain.length(1) / 2;
    const int k = domain.smallEnd(2) + domain.length(2) / 2;
    amrex::Real value = 0.0;
    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(amrex::IntVect(AMREX_D_DECL(i,j,k)))) {
            value = field.const_array(mfi)(i,j,k,component);
        }
    }
    return global_sum(value);
}
} // namespace

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
    for (int step = 0; step < steps; ++step) {
        advance_one_step(dt, viscosity);
        total_step_seconds += m_last_advance_seconds;
        amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
        compute_cartesian_divergence(divergence);
        amrex::Real max_div = divergence.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(max_div);
        amrex::Real momentum[AMREX_SPACEDIM]{};
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp)
            momentum[comp] = global_sum(m_ucat.sum(comp, true) * m_cell_volume);
        amrex::MultiFab kinetic(m_ba, m_dm, 1, 0);
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi); auto const e = kinetic.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                e(i,j,k) = 0.5 * (u(i,j,k,0)*u(i,j,k,0) + u(i,j,k,1)*u(i,j,k,1) + u(i,j,k,2)*u(i,j,k,2));
            });
        }
        amrex::Gpu::streamSynchronize();
        const amrex::Real energy = global_sum(kinetic.sum(0, true) * m_cell_volume);
        const amrex::Real net_mass_flux = boundary_flux(0, false) + boundary_flux(0, true) +
                                           boundary_flux(1, false) + boundary_flux(1, true);
        const amrex::Real outlet_flow = boundary_flux(0, true);
        const amrex::Real mean_u_in = cell_plane_average(m_ucat, m_geom, 0, xlo, 0);
        const amrex::Real mean_u_out = cell_plane_average(m_ucat, m_geom, 0, xhi, 0);
        const amrex::Real center_u = center_value(m_ucat, domain, 0);
        const amrex::Real pressure_drop = cell_plane_average(m_p, m_geom, 0, xlo, 0) -
                                           cell_plane_average(m_p, m_geom, 0, xhi, 0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (step != 0) output << ",\n";
            output << "    {\"step\": " << step + 1 << ", \"time\": " << m_time
                   << ", \"step_seconds\": " << m_last_advance_seconds
                   << ", \"post_projection_max_abs_divergence\": " << max_div
                   << ", \"net_mass_flux\": " << net_mass_flux
                   << ", \"outlet_flow\": " << outlet_flow
                   << ", \"section_mean_u_in\": " << mean_u_in
                   << ", \"section_mean_u_out\": " << mean_u_out
                   << ", \"centerline_u\": " << center_u
                   << ", \"pressure_drop\": " << pressure_drop
                   << ", \"momentum\": [" << momentum[0] << ", " << momentum[1] << ", " << momentum[2]
                   << "], \"kinetic_energy\": " << energy << "}";
        }
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output << "\n  ],\n  \"total_step_seconds\": " << total_step_seconds
               << ",\n  \"validation_note\": \"physical run / not yet validated; no legacy or literature reference data available\"\n}\n";
    }
    amrex::Print() << "VWiS physical Cartesian channel run complete: report=" << report_path
                   << " steps=" << steps << " final_time=" << m_time << "\n";
}
