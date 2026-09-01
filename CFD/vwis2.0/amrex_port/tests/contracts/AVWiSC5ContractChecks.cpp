#include "AVWiSContractTestAccess.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_velocity(int i, int j, int k, int comp) noexcept
{
    return 0.75 + 0.125 * static_cast<amrex::Real>(comp)
         + 0.03125 * static_cast<amrex::Real>(i)
         - 0.015625 * static_cast<amrex::Real>(j)
         + 0.0078125 * static_cast<amrex::Real>(k);
}
} // namespace

void AVWiSContractTestAccess::run_p5_mapped_boundary_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    validate_boundary_config();
    if (!m_boundary.enabled ||
        m_boundary.geometry != BoundaryGeometryMode::MappedOrthogonal ||
        m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_mapping_operator.mapping_type != "analytic_orthogonal" ||
        m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG) {
        throw std::runtime_error("C5.1 contract requires explicit mapped_orthogonal boundary mode");
    }
    if (m_ba.size() < 2) {
        throw std::runtime_error("C5.1 contract requires a multi-Box layout");
    }

    const amrex::Box domain = m_geom.Domain();
    const auto lo = domain.smallEnd();
    const auto hi = domain.bigEnd();
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) periodic[dir] = m_geom.isPeriodic(dir);
    constexpr amrex::Real geometric_tolerance = 2.0e-12;
    constexpr amrex::Real velocity_tolerance = 2.0e-12;

    // Every non-periodic logical side must resolve to a positive, axis-aligned
    // physical area vector for this deliberately diagonal C5.1 path.
    amrex::Gpu::DeviceScalar<int> geometry_errors_device(0);
    int* const geometry_errors = geometry_errors_device.dataPtr();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const& area_field = m_metric_data.face_area_vector_fc(dir);
        const amrex::Box face_domain = amrex::convert(
            domain, amrex::IntVect::TheDimensionVector(dir));
        const int low_face = face_domain.smallEnd(dir);
        const int high_face = face_domain.bigEnd(dir);
        for (amrex::MFIter mfi(area_field, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const area = area_field.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                if (periodic[dir] || (index[dir] != low_face && index[dir] != high_face)) return;
                const amrex::Real axial = area(i,j,k,dir);
                if (!(axial > 0.0)) amrex::Gpu::Atomic::Add(geometry_errors, 1);
                for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                    if (comp != dir && std::abs(area(i,j,k,comp)) >
                        geometric_tolerance * amrex::max(amrex::Real(1.0), std::abs(axial))) {
                        amrex::Gpu::Atomic::Add(geometry_errors, 1);
                    }
                }
            });
        }
    }
    amrex::Gpu::streamSynchronize();
    int geometry_error_count = geometry_errors_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(geometry_error_count);
    if (geometry_error_count != 0) {
        throw std::runtime_error("C5.1 logical-face to physical-normal/area contract failed");
    }

    // Periodic coordinates translate physically while their metric repeats.
    amrex::Gpu::DeviceScalar<int> translation_errors_device(0);
    int* const translation_errors = translation_errors_device.dataPtr();
    auto const logical = m_metric_data.logical_grid();
    auto const scale = m_mapping_operator.analytic_parameters.scale;
    auto const stretch = m_mapping_operator.analytic_parameters.stretch;
    auto const& nodes = m_metric_data.node_coordinates_nd();
    for (amrex::MFIter mfi(nodes, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = nodes.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                if (!periodic[dir] ||
                    (index[dir] != lo[dir] && index[dir] != hi[dir] + 1)) continue;
                const amrex::Real logical_length = logical.spacing[dir] * domain.length(dir);
                const amrex::Real coordinate = logical.lower[dir] +
                    static_cast<amrex::Real>(index[dir] - lo[dir]) * logical.spacing[dir];
                const amrex::Real expected = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                    coordinate, logical.lower[dir], logical_length, scale[dir], stretch[dir]);
                if (std::abs(xyz(i,j,k,dir) - expected) >
                    geometric_tolerance * amrex::max(amrex::Real(1.0), std::abs(expected))) {
                    amrex::Gpu::Atomic::Add(translation_errors, 1);
                }
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    int translation_error_count = translation_errors_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(translation_error_count);
    if (translation_error_count != 0) {
        throw std::runtime_error("C5.1 periodic physical-translation contract failed");
    }

    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                u(i,j,k,comp) = manufactured_velocity(i,j,k,comp);
            });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();

    // Cell ghosts: periodic directions wrap first; if one or more physical
    // sides meet, the first physical direction in x,y,z owns the corner rule.
    amrex::Gpu::DeviceScalar<int> ghost_errors_device(0);
    int* const ghost_errors = ghost_errors_device.dataPtr();
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
                if (domain.contains(iv)) return;
                int source[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                bool physical = false;
                for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                    while (source[dir] < lo[dir] && periodic[dir]) source[dir] += domain.length(dir);
                    while (source[dir] > hi[dir] && periodic[dir]) source[dir] -= domain.length(dir);
                    if (source[dir] < lo[dir]) { source[dir] = lo[dir]; physical = true; }
                    if (source[dir] > hi[dir]) { source[dir] = hi[dir]; physical = true; }
                }
                const amrex::Real interior = manufactured_velocity(
                    source[0], source[1], source[2], comp);
                const amrex::Real expected = physical ? -interior : interior;
                if (std::abs(u(i,j,k,comp) - expected) >
                    velocity_tolerance * amrex::max(amrex::Real(1.0), std::abs(expected))) {
                    amrex::Gpu::Atomic::Add(ghost_errors, 1);
                }
            });
    }
    amrex::Gpu::streamSynchronize();
    int ghost_error_count = ghost_errors_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(ghost_error_count);
    if (ghost_error_count != 0) {
        throw std::runtime_error("C5.1 wall/periodic/corner ghost contract failed: errors=" +
                                 std::to_string(ghost_error_count));
    }

    // At every physical wall, the boundary average has zero normal and
    // tangential velocity and Ucont is exactly the already-integrated u.S.
    amrex::Gpu::DeviceScalar<int> wall_errors_device(0);
    int* const wall_errors = wall_errors_device.dataPtr();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) continue;
        const amrex::Box face_domain = amrex::convert(
            domain, amrex::IntVect::TheDimensionVector(dir));
        for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const flux = m_ucont[dir].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                if ((index[dir] == face_domain.smallEnd(dir) ||
                     index[dir] == face_domain.bigEnd(dir)) && flux(i,j,k) != 0.0) {
                    amrex::Gpu::Atomic::Add(wall_errors, 1);
                }
            });
        }
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                    if (index[dir] != lo[dir] && index[dir] != hi[dir]) return;
                    int ghost[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                    ghost[dir] += index[dir] == lo[dir] ? -1 : 1;
                    if (std::abs(u(i,j,k,comp) + u(ghost[0],ghost[1],ghost[2],comp)) >
                        velocity_tolerance * amrex::max(amrex::Real(1.0), std::abs(u(i,j,k,comp)))) {
                        amrex::Gpu::Atomic::Add(wall_errors, 1);
                    }
                });
        }
    }
    amrex::Gpu::streamSynchronize();
    int wall_error_count = wall_errors_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(wall_error_count);
    if (wall_error_count != 0) {
        throw std::runtime_error("C5.1 no-penetration/no-slip wall contract failed");
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Long expected_unique_faces = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            expected_unique_faces *= domain.length(d) + ((d == dir && !periodic[d]) ? 1 : 0);
        }
        auto owner = m_ucont[dir].OwnerMask(m_geom.periodicity());
        if (owner->sum(0, 0, false) != expected_unique_faces) {
            throw std::runtime_error("C5.1 multi-Box face-owner contract failed");
        }
    }

    // Error injection: boundary consumers must reject stale flow ghosts,
    // metric epochs, and a solver/metric layout mismatch.
    mark_valid_modified();
    bool stale_ghost_rejected = false;
    try { fill_physical_ghost_cells(); }
    catch (std::runtime_error const&) { stale_ghost_rejected = true; }
    if (!stale_ghost_rejected) throw std::runtime_error("C5.1 stale flow ghost was accepted");
    apply_boundary_pipeline("c5.1-after-stale-ghost-probe");

    const std::uint64_t saved_metric_epoch = m_metric_epoch;
    ++m_metric_epoch;
    bool stale_metric_rejected = false;
    try { fill_physical_ghost_cells(); }
    catch (std::runtime_error const&) { stale_metric_rejected = true; }
    m_metric_epoch = saved_metric_epoch;
    if (!stale_metric_rejected) throw std::runtime_error("C5.1 stale metric epoch was accepted");

    const amrex::BoxArray saved_boxes = m_ba;
    m_ba = amrex::BoxArray(domain);
    bool layout_rejected = false;
    try { validate_boundary_config(); }
    catch (std::runtime_error const&) { layout_rejected = true; }
    m_ba = saved_boxes;
    if (!layout_rejected) throw std::runtime_error("C5.1 metric/layout mismatch was accepted");

    // A telescoping integrated face flux is compatible with periodic/Neumann
    // pressure BCs. Physical end faces vanish; no metric area is multiplied twice.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) m_ucont[dir].setVal(0.0);
    int manufactured_dir = 0;
    const amrex::Box xfaces = amrex::convert(
        domain, amrex::IntVect::TheDimensionVector(manufactured_dir));
    const int face_lo = xfaces.smallEnd(manufactured_dir);
    const amrex::Real count = static_cast<amrex::Real>(domain.length(manufactured_dir));
    auto const& area_x = m_metric_data.face_area_vector_fc(manufactured_dir);
    for (amrex::MFIter mfi(m_ucont[manufactured_dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const flux = m_ucont[manufactured_dir].array(mfi);
        auto const area = area_x.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real phase = static_cast<amrex::Real>(i - face_lo) / count;
            const amrex::Real wave = periodic[manufactured_dir]
                ? amrex::Math::sinpi(2.0 * phase) : amrex::Math::sinpi(phase);
            flux(i,j,k) = wave * area(i,j,k,manufactured_dir);
        });
    }
    mark_valid_modified();
    const ProjectionDiagnostics report = project_orthogonal(dt, time_coefficient);
    const amrex::Real reduction_target = amrex::max(
        amrex::Real(1.0e-9), amrex::Real(1.0e-7) * report.max_divergence_before);
    if (!(report.max_divergence_before > 1.0e-8) ||
        !(report.max_divergence_after < reduction_target)) {
        throw std::runtime_error("C5.1 mapped wall projection did not reduce divergence");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) continue;
        if (std::abs(boundary_flux(dir, false)) > 2.0e-12 ||
            std::abs(boundary_flux(dir, true)) > 2.0e-12) {
            throw std::runtime_error("C5.1 projection changed a wall-normal integrated flux");
        }
    }

    amrex::Print() << "AVWiS P5-003 C5.1 mapped boundary contract: PASS "
                   << "mapping=" << m_metric_data.mapping_id()
                   << " max_div_before=" << report.max_divergence_before
                   << " max_div_after=" << report.max_divergence_after << "\n";
}

void AVWiSContractTestAccess::run_p5_mapped_boundary_c52_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    validate_boundary_config();
    if (!m_boundary.enabled ||
        m_boundary.geometry != BoundaryGeometryMode::MappedOrthogonal ||
        m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_mapping_operator.mapping_type != "analytic_orthogonal" ||
        m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG) {
        throw std::runtime_error("C5.2 contract requires explicit analytic mapped_orthogonal mode");
    }
    if (m_ba.size() < 2) {
        throw std::runtime_error("C5.2 contract requires a multi-Box layout");
    }

    const amrex::Box domain = m_geom.Domain();
    const auto lo = domain.smallEnd();
    const auto hi = domain.bigEnd();
    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    amrex::GpuArray<int, 2 * AMREX_SPACEDIM> kinds{};
    int inflows = 0;
    int outflows = 0;
    int moving_walls = 0;
    int slip_or_symmetry = 0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        periodic[dir] = m_geom.isPeriodic(dir);
        for (int side = 0; side < 2; ++side) {
            const CartesianBC kind = m_boundary.sides[2 * dir + side].velocity;
            kinds[2 * dir + side] = static_cast<int>(kind);
            inflows += kind == CartesianBC::Inflow;
            outflows += kind == CartesianBC::Outflow;
            moving_walls += kind == CartesianBC::MovingWall;
            slip_or_symmetry += kind == CartesianBC::SlipWall || kind == CartesianBC::Symmetry;
        }
    }
    if (!((inflows == 1 && outflows == 1) ||
          (moving_walls > 0 && slip_or_symmetry > 0 && inflows == 0 && outflows == 0))) {
        throw std::runtime_error(
            "C5.2 focused inputs must exercise inlet/outlet or moving/slip/symmetry modes");
    }

    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                u(i,j,k,comp) = manufactured_velocity(i,j,k,comp);
            });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();

    constexpr amrex::Real tolerance = 4.0e-12;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> wall_velocity{};
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        wall_velocity[comp] = m_boundary.moving_wall_velocity[comp];
    }

    // Check physical-vector ghost states away from multi-normal corners. The
    // boundary face average is the prescribed vector for Dirichlet modes, the
    // interior tangential projection for slip/symmetry, and the interior state
    // for outflow. Inlet expected velocity is recovered from authoritative
    // integrated u.S and the same stored physical area vector.
    amrex::Gpu::DeviceScalar<int> state_errors_device(0);
    int* const state_errors = state_errors_device.dataPtr();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) continue;
        auto const& area_field = m_metric_data.face_area_vector_fc(dir);
        for (int side = 0; side < 2; ++side) {
            const int boundary_cell = side == 0 ? lo[dir] : hi[dir];
            for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const u = m_ucat.const_array(mfi);
                auto const flux = m_ucont[dir].const_array(mfi);
                auto const area = area_field.const_array(mfi);
                amrex::ParallelFor(
                    mfi.validbox(), AMREX_SPACEDIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                        const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                        if (index[dir] != boundary_cell) return;
                        for (int other = 0; other < AMREX_SPACEDIM; ++other) {
                            if (other != dir && !periodic[other] &&
                                (index[other] == lo[other] || index[other] == hi[other])) return;
                        }
                        int ghost[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                        ghost[dir] += side == 0 ? -1 : 1;
                        int face[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                        if (side != 0) ++face[dir];
                        amrex::Real area_squared = 0.0;
                        for (int physical = 0; physical < AMREX_SPACEDIM; ++physical) {
                            const amrex::Real s = area(face[0],face[1],face[2],physical);
                            area_squared += s * s;
                        }
                        const amrex::Real physical_area = std::sqrt(area_squared);
                        const amrex::Real normal_component =
                            area(face[0],face[1],face[2],comp) / physical_area;
                        amrex::Real interior_normal = 0.0;
                        for (int physical = 0; physical < AMREX_SPACEDIM; ++physical) {
                            interior_normal += u(i,j,k,physical) *
                                area(face[0],face[1],face[2],physical) / physical_area;
                        }
                        const amrex::Real average = 0.5 *
                            (u(i,j,k,comp) + u(ghost[0],ghost[1],ghost[2],comp));
                        const int kind = kinds[2 * dir + side];
                        amrex::Real expected = 0.0;
                        if (kind == static_cast<int>(CartesianBC::MovingWall)) {
                            expected = wall_velocity[comp];
                        } else if (kind == static_cast<int>(CartesianBC::SlipWall) ||
                                   kind == static_cast<int>(CartesianBC::Symmetry)) {
                            expected = u(i,j,k,comp) - interior_normal * normal_component;
                        } else if (kind == static_cast<int>(CartesianBC::Inflow)) {
                            expected = flux(face[0],face[1],face[2]) /
                                physical_area * normal_component;
                        } else if (kind == static_cast<int>(CartesianBC::Outflow)) {
                            expected = u(i,j,k,comp);
                        }
                        if (std::abs(average - expected) >
                            tolerance * amrex::max(amrex::Real(1.0), std::abs(expected))) {
                            amrex::Gpu::Atomic::Add(state_errors, 1);
                        }
                    });
            }
        }
    }
    amrex::Gpu::streamSynchronize();
    int state_error_count = state_errors_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(state_error_count);
    if (state_error_count != 0) {
        throw std::runtime_error("C5.2 mapped physical-vector ghost contract failed: errors=" +
                                 std::to_string(state_error_count));
    }

    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> flux_before{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) continue;
        for (int side = 0; side < 2; ++side) {
            const int slot = 2 * dir + side;
            flux_before[slot] = boundary_flux(dir, side != 0);
            const CartesianBC kind = m_boundary.sides[slot].velocity;
            const amrex::Real expected = kind == CartesianBC::Inflow
                ? -m_boundary.inlet_target_flux
                : (kind == CartesianBC::Outflow && m_boundary.constrain_outlet_flux
                    ? m_boundary.inlet_target_flux : 0.0);
            if (kind != CartesianBC::Outflow || m_boundary.constrain_outlet_flux) {
                if (std::abs(flux_before[slot] - expected) > tolerance) {
                    throw std::runtime_error("C5.2 physical normal-flux normalization failed");
                }
            }
        }
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Long expected_unique_faces = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            expected_unique_faces *= domain.length(d) + ((d == dir && !periodic[d]) ? 1 : 0);
        }
        auto owner = m_ucont[dir].OwnerMask(m_geom.periodicity());
        if (owner->sum(0, 0, false) != expected_unique_faces) {
            throw std::runtime_error("C5.2 multi-Box face-owner contract failed");
        }
    }

    // Repeating the ordered halo -> physical -> owner pipeline must reproduce
    // every valid and ghost Cartesian state, including mixed-BC corners.
    amrex::MultiFab snapshot(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    amrex::MultiFab::Copy(snapshot, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    apply_boundary_pipeline("c5.2-deterministic-repeat");
    amrex::MultiFab::Subtract(snapshot, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    amrex::Real repeat_error = snapshot.norm0(0, m_nghost, true);
    for (int comp = 1; comp < AMREX_SPACEDIM; ++comp) {
        repeat_error = amrex::max(repeat_error, snapshot.norm0(comp, m_nghost, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(repeat_error);
    if (repeat_error != 0.0) {
        throw std::runtime_error("C5.2 mixed-boundary corner/ghost ordering is nondeterministic");
    }

    const ProjectionDiagnostics report = project_orthogonal(dt, time_coefficient);
    const amrex::Real reduction_target = amrex::max(
        amrex::Real(1.0e-9), amrex::Real(1.0e-7) * report.max_divergence_before);
    if (!(report.max_divergence_before > 1.0e-8) ||
        !(report.max_divergence_after < reduction_target)) {
        throw std::runtime_error("C5.2 mapped boundary projection did not reduce divergence");
    }
    if (report.singular != (outflows == 0)) {
        throw std::runtime_error("C5.2 mapped outlet pressure null-space classification failed");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (periodic[dir]) continue;
        for (int side = 0; side < 2; ++side) {
            const int slot = 2 * dir + side;
            if (m_boundary.sides[slot].velocity == CartesianBC::Outflow) continue;
            if (std::abs(boundary_flux(dir, side != 0) - flux_before[slot]) > tolerance) {
                throw std::runtime_error("C5.2 projection changed a prescribed physical normal flux");
            }
        }
    }

    amrex::Print() << "AVWiS P5-003 C5.2 mapped boundary contract: PASS "
                   << "mapping=" << m_metric_data.mapping_id()
                   << " singular=" << report.singular
                   << " max_div_before=" << report.max_divergence_before
                   << " max_div_after=" << report.max_divergence_after << "\n";
}
