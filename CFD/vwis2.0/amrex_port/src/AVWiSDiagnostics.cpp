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

char const* location_name(FieldLocation location)
{
    switch (location) {
    case FieldLocation::Cell: return "cell";
    case FieldLocation::XFace: return "x-face";
    case FieldLocation::YFace: return "y-face";
    case FieldLocation::ZFace: return "z-face";
    }
    return "unknown";
}

} // namespace

P3Diagnostics AVWiSSolver::p3_diagnostics(char const* stage, bool require_fresh) const
{
    if (!m_boundary.enabled) throw std::runtime_error("P3 diagnostics require explicit boundary configuration");
    if (require_fresh) require_ghosts_fresh(stage);
    P3Diagnostics result;
    result.valid_epoch = m_valid_epoch;
    result.halo_epoch = m_halo_epoch;
    result.physical_epoch = m_physical_epoch;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            result.outward_boundary_flux += boundary_flux(dir, false) + boundary_flux(dir, true);
        }
    }

    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const div = divergence.array(mfi);
        auto const fx = m_ucont[0].const_array(mfi);
        auto const fy = m_ucont[1].const_array(mfi);
        auto const fz = m_ucont[2].const_array(mfi);
        const amrex::Real inverse_volume = 1.0 / m_cell_volume;
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            div(i,j,k) = ((fx(i+1,j,k)-fx(i,j,k)) + (fy(i,j+1,k)-fy(i,j,k))
                         + (fz(i,j,k+1)-fz(i,j,k))) * inverse_volume;
        });
    }
    amrex::Gpu::streamSynchronize();
    amrex::Real local_integral = divergence.sum(0, true) * m_cell_volume;
    amrex::Real local_max = divergence.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealSum(local_integral);
    amrex::ParallelDescriptor::ReduceRealMax(local_max);
    result.integrated_divergence = local_integral;
    result.max_abs_divergence = local_max;
    amrex::Print() << "P3 stage=" << stage << " freshness=" << result.valid_epoch << "/"
                   << result.halo_epoch << "/" << result.physical_epoch
                   << " outward_flux=" << result.outward_boundary_flux
                   << " integral_div=" << result.integrated_divergence
                   << " max_abs_div=" << result.max_abs_divergence << "\n";
    return result;
}

void AVWiSSolver::diagnostics() const
{
    amrex::Print() << "AVWiS P5 "
                   << coordinate_system_mode_name(m_mapping_operator.coordinates)
                   << " sub-contract: boxes=" << m_ba.size()
                   << ", ranks=" << amrex::ParallelDescriptor::NProcs()
                   << ", ghosts=" << m_nghost
                   << ", dx=" << m_dx[0] << "," << m_dx[1] << "," << m_dx[2]
                   << ", cell_volume=" << m_cell_volume
                   << ", max(|P|)=" << m_p.norm0(0, 0, true)
                   << ", time=" << m_time << ", step=" << m_step
                   << ", history_depth=" << m_history_depth
                   << ", init_s=" << m_initialize_seconds
                   << ", advance_s=" << m_last_advance_seconds << "\n";
    for (auto const& field : m_fields) {
        amrex::Print() << "  field " << field.name << " location=" << location_name(field.location)
                       << " nComp=" << field.components << " nGrow=" << field.ghost_cells
                       << " layer=" << field.time_layer << " units=" << field.units << "\n";
    }
    if (m_boundary.enabled) {
        amrex::Print() << "  boundary geometry="
                       << boundary_geometry_mode_name(m_boundary.geometry) << "\n";
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::Print() << "  boundary dir=" << dir << " lo="
                           << cartesian_bc_name(m_boundary.sides[2 * dir].velocity) << " hi="
                           << cartesian_bc_name(m_boundary.sides[2 * dir + 1].velocity) << "\n";
        }
    }
}

void AVWiSSolver::write_metadata_manifest(std::string const& path) const
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        throw std::runtime_error(
            "C2.2 mapped mode has no mapping-provenance metadata/checkpoint schema");
    }
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write P5 metadata manifest: " + path);
    output << "{\n  \"schema\": \"avwis-amrex-p5-cartesian-contract-v1\",\n"
           << "  \"payload_written\": false,\n"
           << "  \"note\": \"Not a plotfile or checkpoint; P5-004 history cannot be restarted from this metadata.\",\n"
           << "  \"fields\": [\n";
    for (std::size_t i = 0; i < m_fields.size(); ++i) {
        auto const& field = m_fields[i];
        output << "    {\"name\": \"" << field.name << "\", \"location\": \""
               << location_name(field.location) << "\", \"components\": " << field.components
               << ", \"ngrow\": " << field.ghost_cells << ", \"units\": \"" << field.units
               << "\", \"time_layer\": \"" << field.time_layer
               << "\", \"component_names\": \"" << field.component_names
               << "\", \"ownership\": \"" << field.ownership << "\"}"
               << (i + 1 == m_fields.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"cartesian\": {\"dx\": [" << m_dx[0] << ", " << m_dx[1] << ", " << m_dx[2]
           << "], \"cell_volume\": " << m_cell_volume << "},\n  \"metrics\": [\n";
    for (std::size_t i = 0; i < m_metrics.size(); ++i) {
        auto const& metric = m_metrics[i];
        output << "    {\"name\": \"" << metric.name << "\", \"location\": \"" << metric.location
               << "\", \"value\": \"" << metric.value << "\", \"meaning\": \"" << metric.meaning << "\"}"
               << (i + 1 == m_metrics.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"time_state\": {\"time\": " << m_time << ", \"step\": " << m_step
           << ", \"history_depth\": " << m_history_depth << "},\n"
           << "  \"advance_one_step\": \"provisional explicit Euler RHS plus Cartesian projection; not legacy SNES\",\n"
           << "  \"projection\": \"single-level Cartesian MLPoisson/MLMG; used by the provisional explicit baseline\"\n}\n";
}

UniformPointSample AVWiSSolver::sample_uniform_point(
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

UniformPlaneStatistics AVWiSSolver::uniform_plane_statistics(
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

UniformFlowDiagnostics AVWiSSolver::uniform_flow_diagnostics() const
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

void AVWiSSolver::write_uniform_plane_csv(
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
