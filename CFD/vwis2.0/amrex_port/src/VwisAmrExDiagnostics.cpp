#include "VwisAmrExSolver.H"

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
#include <tuple>
#include <vector>

namespace {

void require_p8_diagnostics_scope()
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::runtime_error("P8 diagnostics rejected: sampling/section output supports exactly one MPI rank");
    }
#ifdef AMREX_USE_GPU
    throw std::runtime_error("P8 diagnostics rejected: sampling/section output is validated only for an AMReX CPU build");
#endif
}

struct PlaneRow {
    int i;
    int j;
    int k;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> position;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> velocity;
    amrex::Real pressure;
    amrex::Real divergence;
};

} // namespace

UniformPointSample VwisAmrExSolver::sample_uniform_point(
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& position) const
{
    require_p8_diagnostics_scope();
    UniformPointSample result;
    const amrex::Box& domain = m_geom.Domain();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!(position[dir] >= m_geom.ProbLo(dir) && position[dir] < m_geom.ProbHi(dir))) {
            throw std::runtime_error("P8 point probe must lie in the half-open physical domain");
        }
        result.cell[dir] = domain.smallEnd(dir) + static_cast<int>(
            std::floor((position[dir] - m_geom.ProbLo(dir)) / m_dx[dir]));
        result.cell_center[dir] = m_geom.ProbLo(dir) +
            (static_cast<amrex::Real>(result.cell[dir] - domain.smallEnd(dir)) + 0.5) * m_dx[dir];
    }

    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    compute_cartesian_divergence(divergence);
    const amrex::IntVect cell(AMREX_D_DECL(result.cell[0], result.cell[1], result.cell[2]));
    int found = 0;
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        if (!mfi.validbox().contains(cell)) continue;
        auto const u = m_ucat.const_array(mfi);
        auto const p = m_p.const_array(mfi);
        auto const div = divergence.const_array(mfi);
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            result.velocity[comp] = u(result.cell[0], result.cell[1], result.cell[2], comp);
        }
        result.pressure = p(result.cell[0], result.cell[1], result.cell[2]);
        result.divergence = div(result.cell[0], result.cell[1], result.cell[2]);
        ++found;
    }
    if (found != 1) throw std::runtime_error("P8 point probe did not resolve to exactly one valid cell");
    return result;
}

UniformPlaneStatistics VwisAmrExSolver::uniform_plane_statistics(
    int direction, int cell_index) const
{
    require_p8_diagnostics_scope();
    const amrex::Box& domain = m_geom.Domain();
    if (direction < 0 || direction >= AMREX_SPACEDIM ||
        cell_index < domain.smallEnd(direction) || cell_index > domain.bigEnd(direction)) {
        throw std::runtime_error("P8 plane direction/index is outside the Cartesian domain");
    }

    UniformPlaneStatistics result;
    result.direction = direction;
    result.cell_index = cell_index;
    result.coordinate = m_geom.ProbLo(direction) +
        (static_cast<amrex::Real>(cell_index - domain.smallEnd(direction)) + 0.5) * m_dx[direction];
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const p = m_p.const_array(mfi);
        const amrex::Box& box = mfi.validbox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
                    if (index[direction] != cell_index) continue;
                    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                        result.mean_velocity[comp] += u(i,j,k,comp);
                    }
                    result.mean_pressure += p(i,j,k);
                    ++result.cell_count;
                }
            }
        }
    }
    if (result.cell_count <= 0) throw std::runtime_error("P8 plane contains no cells");
    for (auto& value : result.mean_velocity) value /= static_cast<amrex::Real>(result.cell_count);
    result.mean_pressure /= static_cast<amrex::Real>(result.cell_count);

    amrex::MultiFab selected(m_ucont[direction].boxArray(), m_dm, 1, 0);
    selected.setVal(0.0);
    for (amrex::MFIter mfi(selected, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const dst = selected.array(mfi);
        auto const src = m_ucont[direction].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int coordinate = direction == 0 ? i : (direction == 1 ? j : k);
            if (coordinate == cell_index) dst(i,j,k) = src(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    result.normal_flow = selected.sum_unique(0, false, m_geom.periodicity());
    return result;
}

UniformFlowDiagnostics VwisAmrExSolver::uniform_flow_diagnostics() const
{
    require_p8_diagnostics_scope();
    UniformFlowDiagnostics result;
    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    compute_cartesian_divergence(divergence);
    result.pressure_min = std::numeric_limits<amrex::Real>::max();
    result.pressure_max = std::numeric_limits<amrex::Real>::lowest();
    amrex::Real pressure_sum = 0.0;
    amrex::Long count = 0;
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const p = m_p.const_array(mfi);
        auto const div = divergence.const_array(mfi);
        const amrex::Box& box = mfi.validbox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const amrex::Real divergence_value = div(i,j,k);
                    result.integrated_divergence += divergence_value * m_cell_volume;
                    result.max_abs_divergence = amrex::max(result.max_abs_divergence,
                                                           std::abs(divergence_value));
                    amrex::Real speed_squared = 0.0;
                    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                        const amrex::Real velocity = u(i,j,k,comp);
                        result.momentum[comp] += velocity * m_cell_volume;
                        speed_squared += velocity * velocity;
                    }
                    result.kinetic_energy += 0.5 * speed_squared * m_cell_volume;
                    const amrex::Real pressure = p(i,j,k);
                    pressure_sum += pressure;
                    result.pressure_min = amrex::min(result.pressure_min, pressure);
                    result.pressure_max = amrex::max(result.pressure_max, pressure);
                    ++count;
                }
            }
        }
    }
    result.pressure_mean = pressure_sum / static_cast<amrex::Real>(count);
    if (m_boundary.enabled) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (m_geom.isPeriodic(dir)) continue;
            for (int side = 0; side < 2; ++side) {
                const amrex::Real flux = boundary_flux(dir, side == 1);
                result.net_mass_flux += flux;
                if (m_boundary.sides[2 * dir + side].velocity == CartesianBC::Outflow) {
                    result.outlet_flow += flux;
                }
            }
        }
    }
    return result;
}

void VwisAmrExSolver::write_uniform_plane_csv(
    std::string const& path, int direction, int cell_index) const
{
    require_p8_diagnostics_scope();
    (void)uniform_plane_statistics(direction, cell_index); // validates the request
    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    compute_cartesian_divergence(divergence);
    std::vector<PlaneRow> rows;
    const amrex::Box& domain = m_geom.Domain();
    for (amrex::MFIter mfi(m_ucat); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const p = m_p.const_array(mfi);
        auto const div = divergence.const_array(mfi);
        const amrex::Box& box = mfi.validbox();
        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
                    if (index[direction] != cell_index) continue;
                    PlaneRow row{i, j, k, {}, {}, p(i,j,k), div(i,j,k)};
                    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                        row.position[dir] = m_geom.ProbLo(dir) +
                            (static_cast<amrex::Real>(index[dir] - domain.smallEnd(dir)) + 0.5) * m_dx[dir];
                        row.velocity[dir] = u(i,j,k,dir);
                    }
                    rows.push_back(row);
                }
            }
        }
    }
    std::sort(rows.begin(), rows.end(), [](PlaneRow const& lhs, PlaneRow const& rhs) {
        return std::tie(lhs.i, lhs.j, lhs.k) < std::tie(rhs.i, rhs.j, rhs.k);
    });
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write P8 plane CSV: " + path);
    output.precision(std::numeric_limits<amrex::Real>::max_digits10);
    output << "i,j,k,x,y,z,u,v,w,pressure,divergence\n";
    for (auto const& row : rows) {
        output << row.i << ',' << row.j << ',' << row.k << ','
               << row.position[0] << ',' << row.position[1] << ',' << row.position[2] << ','
               << row.velocity[0] << ',' << row.velocity[1] << ',' << row.velocity[2] << ','
               << row.pressure << ',' << row.divergence << '\n';
    }
}

void VwisAmrExSolver::run_p8_sampling_statistics_contract(
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
    output << "{\n  \"schema\": \"vwis-uniform-diagnostics-v1\",\n"
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
    amrex::Print() << "VWiS AMReX P8-003 sampling/statistics: PASS report=" << report_path
                   << " plane=" << plane_path << "\n";
}
