#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <stdexcept>

#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <fstream>
#include <limits>

namespace {
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real manufactured_velocity(int i, int j, int k, int comp)
{
    return static_cast<amrex::Real>(10 * comp + (comp + 1) * (i + 2 * j + 3 * k));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int wrap_index(int index, int lo, int length)
{
    const int translated = index - lo;
    return lo + (translated % length + length) % length;
}
} // namespace

void VwisAmrExSolver::run_p2_transform_layout_checks()
{
    const amrex::Box& domain = m_geom.Domain();
    const int lo[AMREX_SPACEDIM] = {AMREX_D_DECL(domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2))};
    const int hi[AMREX_SPACEDIM] = {AMREX_D_DECL(domain.bigEnd(0), domain.bigEnd(1), domain.bigEnd(2))};
    const int length[AMREX_SPACEDIM] = {AMREX_D_DECL(domain.length(0), domain.length(1), domain.length(2))};
    const int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(m_geom.isPeriodic(0), m_geom.isPeriodic(1), m_geom.isPeriodic(2))};
    if (m_metrics.size() != 7 || m_cell_volume <= 0.0) {
        throw std::runtime_error("vwis: Cartesian metric metadata contract failed");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (m_dx[dir] != m_geom.CellSize()[dir] || m_dx[dir] <= 0.0) {
            throw std::runtime_error("vwis: Geometry dx contract failed");
        }
        const amrex::Real metric_error = m_face_area[dir] * m_dx[dir] - m_cell_volume;
        const amrex::Real metric_tolerance = 16.0 * std::numeric_limits<amrex::Real>::epsilon()
                                           * amrex::max(1.0, std::abs(m_cell_volume));
        if (std::abs(metric_error) > metric_tolerance) {
            throw std::runtime_error("vwis: Cartesian face-area/cell-volume contract failed");
        }
        auto expected_face_ba = amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(dir));
        if (m_ucont[dir].nComp() != 1 || m_ucont[dir].nGrow() != m_nghost ||
            !m_ucont[dir].boxArray().ixType().nodeCentered(dir) ||
            m_ucont[dir].boxArray() != expected_face_ba) {
            throw std::runtime_error("vwis: P2 face IndexType/ownership contract failed");
        }
    }

    // A constant velocity distinguishes integrated face flux from face speed
    // and makes shared-face ownership/counting independently checkable.
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            u(i,j,k,0) = 1.0; u(i,j,k,1) = 2.0; u(i,j,k,2) = 3.0;
        });
    }
    sync_ucont_from_ucat();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::Real expected_norm = static_cast<amrex::Real>(dir + 1) * m_face_area[dir];
        const amrex::Real value_tolerance = 32.0 * std::numeric_limits<amrex::Real>::epsilon()
                                           * amrex::max(1.0, std::abs(expected_norm));
        if (std::abs(m_ucont[dir].norm0(0, 1, true) - expected_norm) > value_tolerance) {
            throw std::runtime_error("vwis: constant Ucat/Ucont volume-flux contract failed");
        }
        amrex::Long expected_unique_faces = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            expected_unique_faces *= domain.length(d) + ((d == dir && !m_geom.isPeriodic(d)) ? 1 : 0);
        }
        auto owner_mask = m_ucont[dir].OwnerMask(m_geom.periodicity());
        if (owner_mask->sum(0, 0, false) != expected_unique_faces) {
            throw std::runtime_error("vwis: face OwnerMask unique-count contract failed");
        }
        const amrex::Real expected_flux_sum = expected_norm * static_cast<amrex::Real>(expected_unique_faces);
        const amrex::Real flux_sum = m_ucont[dir].sum_unique(0, false, m_geom.periodicity());
        const amrex::Real sum_tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon()
                                        * amrex::max(1.0, std::abs(expected_flux_sum));
        if (std::abs(flux_sum - expected_flux_sum) > sum_tolerance) {
            throw std::runtime_error("vwis: unique face-flux sum contract failed");
        }
    }

    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const ucat = m_ucat.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                ucat(i, j, k, comp) = manufactured_velocity(i, j, k, comp);
            }
        });
    }
    sync_ucont_from_ucat();

    amrex::Gpu::streamSynchronize();
    amrex::Gpu::DeviceScalar<int> face_error_device(0);
    int* const face_errors = face_error_device.dataPtr();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::Real face_area = m_face_area[dir];
        for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const ucont = m_ucont[dir].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int face = dir == 0 ? i : (dir == 1 ? j : k);
                int il = i - (dir == 0); int jl = j - (dir == 1); int kl = k - (dir == 2);
                int ir = i;               int jr = j;               int kr = k;
                if (periodic[dir]) {
                    il = wrap_index(il, lo[0], length[0]); jl = wrap_index(jl, lo[1], length[1]); kl = wrap_index(kl, lo[2], length[2]);
                    ir = wrap_index(ir, lo[0], length[0]); jr = wrap_index(jr, lo[1], length[1]); kr = wrap_index(kr, lo[2], length[2]);
                }
                amrex::Real expected;
                if (!periodic[dir] && face == lo[dir]) expected = manufactured_velocity(ir, jr, kr, dir) * face_area;
                else if (!periodic[dir] && face == hi[dir] + 1) expected = manufactured_velocity(il, jl, kl, dir) * face_area;
                else expected = 0.5 * (manufactured_velocity(il, jl, kl, dir) + manufactured_velocity(ir, jr, kr, dir)) * face_area;
                const amrex::Real tolerance = 64.0 * std::numeric_limits<amrex::Real>::epsilon()
                                            * amrex::max(1.0, expected > 0.0 ? expected : -expected);
                const amrex::Real error = ucont(i, j, k) - expected;
                if (error > tolerance || error < -tolerance) amrex::Gpu::Atomic::Add(face_errors, 1);
            });
        }
    }
    amrex::Gpu::streamSynchronize();
    int face_error_count = face_error_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(face_error_count);
    if (face_error_count != 0) throw std::runtime_error("vwis: Ucat-to-Ucont linear interpolation/face ownership failed");

    sync_ucat_from_ucont();
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::DeviceScalar<int> derived_error_device(0);
    int* const derived_errors = derived_error_device.dataPtr();
    const amrex::Real dx0 = m_dx[0];
    const amrex::Real dx1 = m_dx[1];
    const amrex::Real dx2 = m_dx[2];
    const amrex::Real cell_volume = m_cell_volume;
    const amrex::Real expected_divergence = 1.0 / dx0 + 4.0 / dx1 + 9.0 / dx2;
    const amrex::Real divergence_tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon()
                                           * amrex::max(1.0, std::abs(expected_divergence));
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const ucat = m_ucat.const_array(mfi);
        auto const divergence = m_phi.array(mfi); // Phi remains a workspace; this is a regression-only derived field.
        auto const ux = m_ucont[0].const_array(mfi);
        auto const uy = m_ucont[1].const_array(mfi);
        auto const uz = m_ucont[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const bool interior = i > lo[0] && i < hi[0] && j > lo[1] && j < hi[1] && k > lo[2] && k < hi[2];
            if (interior) {
                for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                    const amrex::Real expected_velocity = manufactured_velocity(i, j, k, comp);
                    const amrex::Real velocity_error = ucat(i, j, k, comp) - expected_velocity;
                    const amrex::Real velocity_tolerance = 128.0 * std::numeric_limits<amrex::Real>::epsilon()
                                                         * amrex::max(1.0, expected_velocity > 0.0 ? expected_velocity : -expected_velocity);
                    if (velocity_error > velocity_tolerance || velocity_error < -velocity_tolerance) amrex::Gpu::Atomic::Add(derived_errors, 1);
                }
                const amrex::Real div = ((ux(i + 1, j, k) - ux(i, j, k))
                                       + (uy(i, j + 1, k) - uy(i, j, k))
                                       + (uz(i, j, k + 1) - uz(i, j, k))) / cell_volume;
                divergence(i, j, k) = div;
                const amrex::Real error = div - expected_divergence;
                if (error > divergence_tolerance || error < -divergence_tolerance) amrex::Gpu::Atomic::Add(derived_errors, 1);
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    int derived_error_count = derived_error_device.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(derived_error_count);
    if (derived_error_count != 0) throw std::runtime_error("vwis: Ucont-to-Ucat/derived divergence regression failed");

    // Check face grow cells separately.  Inter-box/MPI and periodic images are
    // filled; out-of-domain non-periodic ghosts remain untouched for P3.
    constexpr amrex::Real face_sentinel = -765432.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab face_probe(m_ucont[dir].boxArray(), m_dm, 1, m_nghost);
        face_probe.setVal(face_sentinel);
        for (amrex::MFIter mfi(face_probe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const probe = face_probe.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (periodic[d]) index[d] = wrap_index(index[d], lo[d], length[d]);
                }
                probe(i, j, k) = static_cast<amrex::Real>(13 * dir + index[0] + 101 * index[1] + 10007 * index[2]);
            });
        }
        face_probe.FillBoundary(m_geom.periodicity());
        amrex::Gpu::streamSynchronize();
        amrex::Gpu::DeviceScalar<int> face_ghost_error_device(0);
        int* const face_ghost_errors = face_ghost_error_device.dataPtr();
        const amrex::Box face_domain = amrex::convert(domain, amrex::IntVect::TheDimensionVector(dir));
        for (amrex::MFIter mfi(face_probe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const probe = face_probe.const_array(mfi);
            const amrex::Box grown = mfi.fabbox();
            amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
                bool has_source = true;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (periodic[d]) index[d] = wrap_index(index[d], lo[d], length[d]);
                    else if (index[d] < face_domain.smallEnd(d) || index[d] > face_domain.bigEnd(d)) has_source = false;
                }
                const amrex::Real expected = has_source
                    ? static_cast<amrex::Real>(13 * dir + index[0] + 101 * index[1] + 10007 * index[2])
                    : face_sentinel;
                if (probe(i, j, k) != expected) amrex::Gpu::Atomic::Add(face_ghost_errors, 1);
            });
        }
        amrex::Gpu::streamSynchronize();
        int face_ghost_error_count = face_ghost_error_device.dataValue();
        amrex::ParallelDescriptor::ReduceIntSum(face_ghost_error_count);
        if (face_ghost_error_count != 0) throw std::runtime_error("vwis: face ghost/physical-boundary limitation contract failed");
    }
    amrex::Print() << "VWiS AMReX P2-003/004/005: PASS (Cartesian transforms/metrics/shared faces/derived divergence)\n";
}

void VwisAmrExSolver::run_runtime_contract_checks()
{
    constexpr amrex::Real sentinel = -987654.0;
    if (m_p.nComp() != 1 || m_p.nGrow() != m_nghost ||
        m_ucat.nComp() != AMREX_SPACEDIM || m_ucat.nGrow() != m_nghost ||
        !m_p.boxArray().ixType().cellCentered() ||
        !m_ucat.boxArray().ixType().cellCentered() || m_cell_volume <= 0.0) {
        throw std::runtime_error("vwis: base cell layout/geometry contract failed");
    }
    if (m_p.norm0(0, 0, true) != 0.0 || m_ucat.norm0(0, 0, true) != 0.0) {
        throw std::runtime_error("vwis: initialization contract failed");
    }
    const amrex::Box& domain = m_geom.Domain();

    // A global-index manufactured field makes every periodic/inter-box ghost value checkable.
    m_p.setVal(sentinel);
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    for (amrex::MFIter mfi(m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = m_p.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = static_cast<amrex::Real>(i + nx * (j + ny * k));
        });
    }
    m_p.FillBoundary(m_geom.periodicity());
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::DeviceScalar<int> device_errors(0);
    int* const errors = device_errors.dataPtr();
    for (amrex::MFIter mfi(m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = m_p.const_array(mfi);
        const amrex::Box grown = mfi.fabbox();
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int wi = (i % nx + nx) % nx;
            const int wj = (j % ny + ny) % ny;
            const int wk = (k % nz + nz) % nz;
            const amrex::Real expected = static_cast<amrex::Real>(wi + nx * (wj + ny * wk));
            if (p(i, j, k) != expected) amrex::Gpu::Atomic::Add(errors, 1);
        });
    }
    amrex::Gpu::streamSynchronize();
    int halo_errors = device_errors.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(halo_errors);
    if (halo_errors != 0) throw std::runtime_error("vwis: periodic/inter-box halo contract failed");

    // FillBoundary must leave non-periodic physical ghosts untouched until a BC functor owns them.
    amrex::MultiFab physical_probe(m_ba, m_dm, 1, m_nghost);
    physical_probe.setVal(sentinel);
    for (amrex::MFIter mfi(physical_probe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const probe = physical_probe.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            probe(i, j, k) = 1.0;
        });
    }
    physical_probe.FillBoundary(amrex::Periodicity::NonPeriodic());
    amrex::Gpu::streamSynchronize();
    amrex::Gpu::DeviceScalar<int> physical_errors(0);
    int* const physical_error_ptr = physical_errors.dataPtr();
    for (amrex::MFIter mfi(physical_probe, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const probe = physical_probe.const_array(mfi);
        const amrex::Box grown = mfi.fabbox();
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (!domain.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))) && probe(i, j, k) != sentinel) {
                amrex::Gpu::Atomic::Add(physical_error_ptr, 1);
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    int physical_errors_host = physical_errors.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(physical_errors_host);
    if (physical_errors_host != 0) throw std::runtime_error("vwis: physical-boundary limitation contract failed");
    amrex::Print() << "VWiS AMReX base runtime contract: PASS (cell layout/initialization/halo/physical-BC limitation)\n";
}

P3Diagnostics VwisAmrExSolver::p3_diagnostics(char const* stage, bool require_fresh) const
{
    if (!m_boundary.enabled) throw std::runtime_error("P3 diagnostics require explicit Cartesian boundary configuration");
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

void VwisAmrExSolver::run_p3_boundary_contract_checks()
{
    if (!m_boundary.enabled) throw std::runtime_error("run_p3_boundary_checks requires vwisbcs.enabled=1");
    // Error injection: a valid-region epoch change must be rejected until both
    // halo and physical fills have run in the documented order.
    mark_valid_modified();
    bool stale_detected = false;
    try { require_ghosts_fresh("P3 stale-error injection"); }
    catch (std::runtime_error const&) { stale_detected = true; }
    if (!stale_detected) throw std::runtime_error("P3 stale ghost error injection was not detected");

    constexpr amrex::Real sentinel = -876543.0;
    const amrex::Box domain = m_geom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    m_p.setVal(sentinel);
    for (amrex::MFIter mfi(m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = m_p.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i,j,k) = static_cast<amrex::Real>(i + nx * (j + ny * k));
        });
    }
    mark_valid_modified();
    apply_boundary_pipeline("p3-contract");

    amrex::Gpu::DeviceScalar<int> device_errors(0);
    int* const errors = device_errors.dataPtr();
    for (amrex::MFIter mfi(m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = m_p.const_array(mfi);
        const amrex::Box grown = mfi.fabbox();
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::IntVect iv(AMREX_D_DECL(i,j,k));
            if (domain.contains(iv)) {
                const amrex::Real expected = static_cast<amrex::Real>(i + nx * (j + ny * k));
                if (p(i,j,k) != expected) amrex::Gpu::Atomic::Add(errors, 1);
            } else if (p(i,j,k) == sentinel) {
                amrex::Gpu::Atomic::Add(errors, 1);
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    int error_count = device_errors.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(error_count);
    if (error_count != 0) throw std::runtime_error("P3 cell halo/physical ghost contract failed");

    amrex::Real inlet_outward = 0.0;
    amrex::Real outlet_outward = 0.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side = 0; side < 2; ++side) {
            auto const kind = m_boundary.sides[2 * dir + side].velocity;
            if (kind == CartesianBC::Inflow) inlet_outward += boundary_flux(dir, side != 0);
            if (kind == CartesianBC::Outflow) outlet_outward += boundary_flux(dir, side != 0);
        }
    }
    const amrex::Real tolerance = 512.0 * std::numeric_limits<amrex::Real>::epsilon()
                                * amrex::max(1.0, m_boundary.inlet_target_flux);
    if (std::abs(inlet_outward + m_boundary.inlet_target_flux) > tolerance ||
        (m_boundary.constrain_outlet_flux && std::abs(outlet_outward - m_boundary.inlet_target_flux) > tolerance)) {
        throw std::runtime_error("P3 inlet profile/outlet flux constraint failed");
    }
    auto const report = p3_diagnostics("p3-contract-final", true);
    if (m_boundary.constrain_outlet_flux && (std::abs(report.outward_boundary_flux) > tolerance ||
        std::abs(report.integrated_divergence - report.outward_boundary_flux) > tolerance)) {
        throw std::runtime_error("P3 global boundary-flux/divergence identity failed");
    }
    amrex::Print() << "VWiS AMReX P3-001/002/003/004: PASS "
                   << "(explicit BC/physical ghosts/plane flux/freshness+MPI diagnostics)\n";
}

void VwisAmrExSolver::diagnostics() const
{
    amrex::Print() << "VWiS AMReX P5 Cartesian sub-contract: boxes=" << m_ba.size()
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
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::Print() << "  boundary dir=" << dir << " lo="
                           << cartesian_bc_name(m_boundary.sides[2 * dir].velocity) << " hi="
                           << cartesian_bc_name(m_boundary.sides[2 * dir + 1].velocity) << "\n";
        }
    }
}

void VwisAmrExSolver::write_metadata_manifest(std::string const& path) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write P5 metadata manifest: " + path);
    output << "{\n  \"schema\": \"vwis-amrex-p5-cartesian-contract-v1\",\n"
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
