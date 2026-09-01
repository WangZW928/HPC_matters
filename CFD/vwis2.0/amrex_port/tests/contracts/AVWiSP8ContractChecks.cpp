#include "AVWiSContractTestAccess.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void require_p8_restart_scope()
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::runtime_error(
            "P8 checkpoint rejected: checkpoint/restart supports exactly one MPI rank");
    }
#ifdef AMREX_USE_GPU
    throw std::runtime_error(
        "P8 checkpoint rejected: checkpoint/restart is validated only for an AMReX CPU build");
#endif
}

amrex::Real max_difference(amrex::MultiFab const& lhs, amrex::MultiFab const& rhs,
                           int components, int nghost = 0)
{
    amrex::MultiFab difference(lhs.boxArray(), lhs.DistributionMap(), components, nghost);
    amrex::MultiFab::Copy(difference, lhs, 0, 0, components, nghost);
    amrex::MultiFab::Subtract(difference, rhs, 0, 0, components, nghost);
    amrex::Real result = 0.0;
    for (int comp = 0; comp < components; ++comp) {
        result = amrex::max(result, difference.norm0(comp, nghost, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}

void require_p8_diagnostics_scope()
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::runtime_error(
            "P8 diagnostics rejected: sampling/section output supports exactly one MPI rank");
    }
#ifdef AMREX_USE_GPU
    throw std::runtime_error(
        "P8 diagnostics rejected: sampling/section output is validated only for an AMReX CPU build");
#endif
}
} // namespace

void AVWiSContractTestAccess::run_p8_restart_contract_checks(
    std::string const& path, amrex::Real dt, int total_steps,
    int checkpoint_step, amrex::Real viscosity)
{
    require_p8_restart_scope();
    if (total_steps < 2 || checkpoint_step <= 0 || checkpoint_step >= total_steps) {
        throw std::runtime_error("P8 restart contract requires 0 < checkpoint_step < total_steps");
    }

    const amrex::Real xlo = m_geom.ProbLo(0);
    const amrex::Real length = m_geom.ProbLength(0);
    const amrex::Real dx0 = m_dx[0];
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto u = m_ucat.array(mfi);
        auto p = m_p.array(mfi);
        auto phi = m_phi.array(mfi);
        auto nvert = m_nvert.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx0;
            u(i,j,k,0) = 0.0;
            u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
            u(i,j,k,2) = 0.0;
            p(i,j,k) = 0.01 * (i + 2*j + 3*k);
            phi(i,j,k) = -0.02 * (2*i - j + k);
            nvert(i,j,k) = ((i + j + k) % 7 == 0) ? 1.0 : 0.0;
        });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();
    amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    amrex::MultiFab::Copy(m_ucat_older, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    m_ucat_old.mult(0.9, 0, AMREX_SPACEDIM, m_nghost);
    m_ucat_older.mult(0.8, 0, AMREX_SPACEDIM, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont[dir], 0, 0, 1, m_nghost);
        m_ucont_old[dir].mult(0.9, 0, 1, m_nghost);
        m_ucont_older[dir].mult(0.8, 0, 1, m_nghost);
    }
    m_time = 2.0 * dt;
    m_step = 2;
    m_history_depth = 3;
    fill_ghost_cells();

    auto cell_copy = [&](amrex::MultiFab const& source, int components) {
        amrex::MultiFab result(source.boxArray(), source.DistributionMap(), components, m_nghost);
        amrex::MultiFab::Copy(result, source, 0, 0, components, m_nghost);
        return result;
    };
    amrex::MultiFab initial_p = cell_copy(m_p, 1);
    amrex::MultiFab initial_phi = cell_copy(m_phi, 1);
    amrex::MultiFab initial_nvert = cell_copy(m_nvert, 1);
    amrex::MultiFab initial_ucat = cell_copy(m_ucat, AMREX_SPACEDIM);
    amrex::MultiFab initial_ucat_old = cell_copy(m_ucat_old, AMREX_SPACEDIM);
    amrex::MultiFab initial_ucat_older = cell_copy(m_ucat_older, AMREX_SPACEDIM);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> initial_ucont, initial_ucont_old, initial_ucont_older;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        initial_ucont[dir] = cell_copy(m_ucont[dir], 1);
        initial_ucont_old[dir] = cell_copy(m_ucont_old[dir], 1);
        initial_ucont_older[dir] = cell_copy(m_ucont_older[dir], 1);
    }

    for (int step = 0; step < total_steps; ++step) advance_one_step(dt, viscosity);
    amrex::MultiFab final_p = cell_copy(m_p, 1);
    amrex::MultiFab final_phi = cell_copy(m_phi, 1);
    amrex::MultiFab final_nvert = cell_copy(m_nvert, 1);
    amrex::MultiFab final_ucat = cell_copy(m_ucat, AMREX_SPACEDIM);
    amrex::MultiFab final_ucat_old = cell_copy(m_ucat_old, AMREX_SPACEDIM);
    amrex::MultiFab final_ucat_older = cell_copy(m_ucat_older, AMREX_SPACEDIM);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> final_ucont, final_ucont_old, final_ucont_older;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        final_ucont[dir] = cell_copy(m_ucont[dir], 1);
        final_ucont_old[dir] = cell_copy(m_ucont_old[dir], 1);
        final_ucont_older[dir] = cell_copy(m_ucont_older[dir], 1);
    }
    amrex::Real final_time = m_time;
    std::uint64_t final_step = m_step;

    auto restore_initial = [&]() {
        amrex::MultiFab::Copy(m_p, initial_p, 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_phi, initial_phi, 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_nvert, initial_nvert, 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_ucat, initial_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab::Copy(m_ucat_old, initial_ucat_old, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab::Copy(m_ucat_older, initial_ucat_older, 0, 0, AMREX_SPACEDIM, m_nghost);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::MultiFab::Copy(m_ucont[dir], initial_ucont[dir], 0, 0, 1, m_nghost);
            amrex::MultiFab::Copy(m_ucont_old[dir], initial_ucont_old[dir], 0, 0, 1, m_nghost);
            amrex::MultiFab::Copy(m_ucont_older[dir], initial_ucont_older[dir], 0, 0, 1, m_nghost);
        }
        m_time = 2.0 * dt; m_step = 2; m_history_depth = 3;
        mark_valid_modified(); fill_ghost_cells();
    };
    restore_initial();
    for (int step = 0; step < checkpoint_step; ++step) advance_one_step(dt, viscosity);

    amrex::MultiFab disk_p = cell_copy(m_p, 1);
    amrex::MultiFab disk_phi = cell_copy(m_phi, 1);
    amrex::MultiFab disk_nvert = cell_copy(m_nvert, 1);
    amrex::MultiFab disk_ucat = cell_copy(m_ucat, AMREX_SPACEDIM);
    amrex::MultiFab disk_ucat_old = cell_copy(m_ucat_old, AMREX_SPACEDIM);
    amrex::MultiFab disk_ucat_older = cell_copy(m_ucat_older, AMREX_SPACEDIM);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> disk_ucont, disk_ucont_old, disk_ucont_older;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        disk_ucont[dir] = cell_copy(m_ucont[dir], 1);
        disk_ucont_old[dir] = cell_copy(m_ucont_old[dir], 1);
        disk_ucont_older[dir] = cell_copy(m_ucont_older[dir], 1);
    }
    amrex::Real disk_time = m_time;
    std::uint64_t disk_step = m_step;
    int disk_history = m_history_depth;
    write_checkpoint(path);

    m_p.setVal(91.0); m_phi.setVal(92.0); m_nvert.setVal(93.0);
    m_ucat.setVal(94.0); m_ucat_old.setVal(95.0); m_ucat_older.setVal(96.0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].setVal(97.0); m_ucont_old[dir].setVal(98.0); m_ucont_older[dir].setVal(99.0);
    }
    m_time = -1.0; m_step = 0; m_history_depth = 1;
    read_checkpoint(path);

    amrex::Real roundtrip_error = 0.0;
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_p, disk_p, 1, m_nghost));
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_phi, disk_phi, 1, m_nghost));
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_nvert, disk_nvert, 1, m_nghost));
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucat, disk_ucat, AMREX_SPACEDIM, m_nghost));
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucat_old, disk_ucat_old, AMREX_SPACEDIM, m_nghost));
    roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucat_older, disk_ucat_older, AMREX_SPACEDIM, m_nghost));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucont[dir], disk_ucont[dir], 1, m_nghost));
        roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucont_old[dir], disk_ucont_old[dir], 1, m_nghost));
        roundtrip_error = amrex::max(roundtrip_error, max_difference(m_ucont_older[dir], disk_ucont_older[dir], 1, m_nghost));
    }
    if (roundtrip_error != 0.0 || m_time != disk_time || m_step != disk_step ||
        m_history_depth != disk_history) {
        throw std::runtime_error("P8 VisMF round-trip changed persistent state");
    }

    for (int step = checkpoint_step; step < total_steps; ++step) advance_one_step(dt, viscosity);
    amrex::Real continuation_error = 0.0;
    continuation_error = amrex::max(continuation_error, max_difference(m_p, final_p, 1));
    continuation_error = amrex::max(continuation_error, max_difference(m_phi, final_phi, 1));
    continuation_error = amrex::max(continuation_error, max_difference(m_nvert, final_nvert, 1));
    continuation_error = amrex::max(continuation_error, max_difference(m_ucat, final_ucat, AMREX_SPACEDIM));
    continuation_error = amrex::max(continuation_error, max_difference(m_ucat_old, final_ucat_old, AMREX_SPACEDIM));
    continuation_error = amrex::max(continuation_error, max_difference(m_ucat_older, final_ucat_older, AMREX_SPACEDIM));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        continuation_error = amrex::max(continuation_error, max_difference(m_ucont[dir], final_ucont[dir], 1));
        continuation_error = amrex::max(continuation_error, max_difference(m_ucont_old[dir], final_ucont_old[dir], 1));
        continuation_error = amrex::max(continuation_error, max_difference(m_ucont_older[dir], final_ucont_older[dir], 1));
    }
    if (continuation_error != 0.0 || m_time != final_time || m_step != final_step || m_history_depth != 3) {
        throw std::runtime_error("P8 uninterrupted and checkpoint/restart trajectories differ");
    }
    amrex::Print() << "AVWiS P8-001/P8-002: PASS (VisMF all histories/state; strict Header; "
                   << "roundtrip_error=" << roundtrip_error
                   << ", continuation_error=" << continuation_error << ")\n";
}

void AVWiSContractTestAccess::run_p8_sampling_statistics_contract(
    std::string const& report_path, std::string const& plane_path)
{
    require_p8_diagnostics_scope();
    const amrex::Box& domain = m_geom.Domain();
    if (domain.length(0) != 4 || domain.length(1) != 3 || domain.length(2) != 2 ||
        !m_geom.isAllPeriodic()) {
        throw std::runtime_error("P8 sampling contract requires periodic n_cell=4 3 2");
    }
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        auto const p = m_p.array(mfi);
        const amrex::Box& box = mfi.validbox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    u(i,j,k,0) = 1.0;
                    u(i,j,k,1) = 2.0;
                    u(i,j,k,2) = 3.0;
                    p(i,j,k) = static_cast<amrex::Real>(100*i + 10*j + k);
                }
            }
        }
    }
    mark_valid_modified();
    sync_ucont_from_ucat();
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probe_position{
        AMREX_D_DECL(0.625, 0.5, 0.75)};
    const UniformPointSample probe = sample_uniform_point(probe_position);
    const UniformPlaneStatistics plane = uniform_plane_statistics(0, 1);
    const UniformFlowDiagnostics flow = uniform_flow_diagnostics();
    write_uniform_plane_csv(plane_path, 0, 1);

    const amrex::Real tolerance = 64.0 * std::numeric_limits<amrex::Real>::epsilon();
    auto close = [=](amrex::Real lhs, amrex::Real rhs) {
        return std::abs(lhs-rhs) <= tolerance * amrex::max(1.0, std::abs(rhs));
    };
    if (probe.cell[0] != 2 || probe.cell[1] != 1 || probe.cell[2] != 1 ||
        !close(probe.pressure, 211.0) || !close(probe.divergence, 0.0) ||
        plane.cell_count != 6 || !close(plane.mean_pressure, 110.5) ||
        !close(plane.normal_flow, 1.0) || !close(flow.integrated_divergence, 0.0) ||
        !close(flow.max_abs_divergence, 0.0) || !close(flow.momentum[0], 1.0) ||
        !close(flow.momentum[1], 2.0) || !close(flow.momentum[2], 3.0) ||
        !close(flow.kinetic_energy, 7.0) || !close(flow.pressure_mean, 160.5) ||
        !close(flow.pressure_min, 0.0) || !close(flow.pressure_max, 321.0)) {
        throw std::runtime_error("P8 sampling/statistics manufactured values failed");
    }

    std::ofstream output(report_path);
    if (!output) throw std::runtime_error("cannot write P8 sampling report: " + report_path);
    output.precision(std::numeric_limits<amrex::Real>::max_digits10);
    output << "{\n  \"schema\": \"avwis-uniform-diagnostics-v1\",\n"
           << "  \"status\": \"PASS\",\n  \"case_type\": \"manufactured contract test\",\n"
           << "  \"scope\": \"single-level uniform Cartesian CPU single-rank\",\n"
           << "  \"plotfile_compatible\": false,\n"
           << "  \"probe\": {\"cell\": [" << probe.cell[0] << ',' << probe.cell[1] << ',' << probe.cell[2]
           << "], \"velocity\": [" << probe.velocity[0] << ',' << probe.velocity[1] << ',' << probe.velocity[2]
           << "], \"pressure\": " << probe.pressure << ", \"divergence\": " << probe.divergence << "},\n"
           << "  \"plane\": {\"direction\": 0, \"cell_index\": 1, \"cell_count\": " << plane.cell_count
           << ", \"mean_velocity\": [" << plane.mean_velocity[0] << ',' << plane.mean_velocity[1] << ',' << plane.mean_velocity[2]
           << "], \"mean_pressure\": " << plane.mean_pressure << ", \"normal_flow\": " << plane.normal_flow << "},\n"
           << "  \"flow\": {\"integrated_divergence\": " << flow.integrated_divergence
           << ", \"max_abs_divergence\": " << flow.max_abs_divergence
           << ", \"net_mass_flux\": " << flow.net_mass_flux << ", \"outlet_flow\": " << flow.outlet_flow
           << ", \"momentum\": [" << flow.momentum[0] << ',' << flow.momentum[1] << ',' << flow.momentum[2]
           << "], \"kinetic_energy\": " << flow.kinetic_energy
           << ", \"pressure_mean\": " << flow.pressure_mean << ", \"pressure_min\": " << flow.pressure_min
           << ", \"pressure_max\": " << flow.pressure_max << "}\n}\n";
    amrex::Print() << "AVWiS P8-003 sampling/statistics: PASS report=" << report_path
                   << " plane=" << plane_path << "\n";
}
