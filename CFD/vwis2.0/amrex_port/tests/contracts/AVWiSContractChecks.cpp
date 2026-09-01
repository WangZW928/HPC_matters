#include "AVWiSSolver.H"
#include "AVWiSContractTestAccess.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
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

void AVWiSContractTestAccess::run_p2_transform_layout_checks()
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
    amrex::Print() << "AVWiS P2-003/004/005: PASS (Cartesian transforms/metrics/shared faces/derived divergence)\n";
}

void AVWiSContractTestAccess::run_runtime_contract_checks()
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
    amrex::Print() << "AVWiS base runtime contract: PASS (cell layout/initialization/halo/physical-BC limitation)\n";
}

void AVWiSContractTestAccess::run_p3_boundary_contract_checks()
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
    amrex::Print() << "AVWiS P3-001/002/003/004: PASS "
                   << "(explicit BC/physical ghosts/plane flux/freshness+MPI diagnostics)\n";
}

void AVWiSContractTestAccess::run_p4_projection_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    validate_projection_boundary_policy();

    bool has_outflow = false;
    if (m_boundary.enabled) {
        for (auto const& side : m_boundary.sides) {
            has_outflow = has_outflow || side.velocity == CartesianBC::Outflow;
        }
    }

    // A constant accumulated pressure with zero face flux must produce a zero
    // correction in the singular cases. (The inflow/outflow test necessarily
    // imposes nonzero P3 boundary flux before each projection.)
    if (!has_outflow) {
        m_p.setVal(3.25);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) m_ucont[dir].setVal(0.0);
        mark_valid_modified();
        auto const constant_report = project_cartesian(dt, time_coefficient);
        const amrex::Real zero_tolerance = 256.0 * std::numeric_limits<amrex::Real>::epsilon();
        if (constant_report.max_divergence_after > zero_tolerance) {
            throw std::runtime_error("P4 constant-pressure/zero-correction contract failed");
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::Real norm = m_ucont[dir].norm0(0, 0, true);
            amrex::ParallelDescriptor::ReduceRealMax(norm);
            if (norm > zero_tolerance) {
                throw std::runtime_error("P4 constant pressure changed face flux");
            }
        }
    }

    // Manufacture a compatible divergence from integrated face flux. For an
    // inflow/outflow case the existing boundary pipeline supplies equal total
    // inlet/outlet flux and zero wall flux; for a singular case the sine mode
    // is periodic or vanishes at closed boundary faces.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) m_ucont[dir].setVal(0.0);
    if (m_boundary.enabled) {
        mark_valid_modified();
        apply_boundary_pipeline("p4-manufactured-boundary");
    }

    if (!has_outflow) {
        const amrex::Box face_domain = amrex::convert(
            m_geom.Domain(), amrex::IntVect::TheDimensionVector(0));
        const int lo = face_domain.smallEnd(0);
        const amrex::Real length = static_cast<amrex::Real>(m_geom.Domain().length(0));
        const amrex::Real area = m_face_area[0];
        for (amrex::MFIter mfi(m_ucont[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const flux = m_ucont[0].array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                flux(i,j,k) = area * amrex::Math::sinpi(
                    2.0 * static_cast<amrex::Real>(i-lo) / length);
            });
        }
    }
    mark_valid_modified();

    // Explicitly prove incompatible singular data are rejected; production
    // code never repairs them by silently subtracting a mean.
    if (!has_outflow) {
        if (m_boundary.enabled) apply_boundary_pipeline("p4-compatibility-probe");
        else fill_ghost_cells();
        compute_cartesian_divergence(m_projection_rhs);
        m_projection_rhs.plus(1.0, 0, 1, 0);
        bool rejected = false;
        try { (void)validate_singular_rhs_compatibility(m_projection_rhs); }
        catch (std::runtime_error const&) { rejected = true; }
        if (!rejected) throw std::runtime_error("P4 incompatible singular RHS was not rejected");
    }

    auto const report = project_cartesian(dt, time_coefficient);
    const amrex::Real reduction_target = amrex::max(1.0e-9, 1.0e-7 * report.max_divergence_before);
    if (!(report.max_divergence_before > 1.0e-8) ||
        !(report.max_divergence_after < reduction_target)) {
        throw std::runtime_error("P4 manufactured projection did not reduce divergence to tolerance");
    }
    amrex::Print() << "AVWiS P4 Cartesian projection contract: PASS "
                   << "(integrated-flux RHS/BC+datum/MLMG/correction/Ucat sync)\n";
}


void AVWiSContractTestAccess::run_p5_advection_contract_checks()
{
    amrex::MultiFab rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    const amrex::Real roundoff = 4096.0 * std::numeric_limits<amrex::Real>::epsilon();

    if (m_boundary.enabled) {
        int inflow_dir = -1;
        int inflow_side = 0;
        int outflow_dir = -1;
        int outflow_side = 0;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            for (int side = 0; side < 2; ++side) {
                const auto kind = m_boundary.sides[2 * dir + side].velocity;
                if (kind == CartesianBC::Inflow) { inflow_dir = dir; inflow_side = side; }
                if (kind == CartesianBC::Outflow) { outflow_dir = dir; outflow_side = side; }
            }
        }
        if (inflow_dir < 0 || outflow_dir != inflow_dir || outflow_side == inflow_side ||
            !m_boundary.constrain_outlet_flux) {
            throw std::runtime_error(
                "P5 boundary contract requires opposite inflow/constrained-outflow faces");
        }
        amrex::Real cross_section = 1.0;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (dir != inflow_dir) cross_section *= m_geom.ProbLength(dir);
        }
        const amrex::Real speed = (inflow_side == 0 ? 1.0 : -1.0) *
                                  m_boundary.inlet_target_flux / cross_section;
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    u(i,j,k,comp) = comp == inflow_dir ? speed : 0.0;
                });
        }
        mark_valid_modified();
        sync_ucont_from_ucat();
        compute_cartesian_advection_rhs(rhs);
        amrex::Real max_error = 0.0;
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            max_error = amrex::max(max_error, rhs.norm0(comp, 0, true));
        }
        amrex::ParallelDescriptor::ReduceRealMax(max_error);
        if (max_error > roundoff * amrex::max(1.0, speed * speed / m_dx[inflow_dir])) {
            throw std::runtime_error("P5 constant boundary advection RHS is not zero");
        }
        amrex::Print() << "AVWiS P5-001 boundary/multi-Box advection: PASS max_error="
                       << max_error << "\n";
        return;
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5 manufactured advection contract requires a fully periodic domain");
        }
    }

    constexpr amrex::Real advecting_speed = 0.75;
    const amrex::Real xlo = m_geom.ProbLo(0);
    const amrex::Real length = m_geom.ProbLength(0);
    const amrex::Real dx = m_dx[0];
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
            u(i,j,k,0) = advecting_speed;
            u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
            u(i,j,k,2) = 0.0;
        });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();
    compute_cartesian_advection_rhs(rhs);

    amrex::MultiFab discrete_error(m_ba, m_dm, 1, 0);
    amrex::MultiFab continuous_error(m_ba, m_dm, 1, 0);
    const amrex::Real pi = 3.141592653589793238462643383279502884;
    const amrex::Real discrete_wavenumber = std::sin(2.0 * pi * dx / length) / dx;
    const amrex::Real exact_wavenumber = 2.0 * pi / length;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = rhs.const_array(mfi);
        auto const disc = discrete_error.array(mfi);
        auto const cont = continuous_error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
            const amrex::Real cosine = amrex::Math::cospi(2.0 * (x-xlo) / length);
            disc(i,j,k) = adv(i,j,k,1) + advecting_speed * discrete_wavenumber * cosine;
            cont(i,j,k) = adv(i,j,k,1) + advecting_speed * exact_wavenumber * cosine;
        });
    }
    amrex::Real stencil_error = discrete_error.norm0(0, 0, true);
    amrex::Real continuous_linf = continuous_error.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
    amrex::ParallelDescriptor::ReduceRealMax(continuous_linf);
    const amrex::Real scale = std::abs(advecting_speed * exact_wavenumber);
    const amrex::Real second_order_bound = std::abs(advecting_speed) *
        exact_wavenumber * exact_wavenumber * exact_wavenumber * dx * dx / 6.0;
    if (stencil_error > roundoff * amrex::max(1.0, scale) ||
        continuous_linf > second_order_bound) {
        throw std::runtime_error("P5 periodic manufactured advection stencil/accuracy failed");
    }
    for (int comp : {0, 2}) {
        amrex::Real norm = rhs.norm0(comp, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(norm);
        if (norm > roundoff) throw std::runtime_error("P5 constant advected component RHS is not zero");
    }
    amrex::Print() << "AVWiS P5-001 periodic manufactured advection: PASS dx=" << dx
                   << " stencil_error=" << stencil_error
                   << " continuous_Linf=" << continuous_linf
                   << " error_over_dx2=" << continuous_linf / (dx * dx) << "\n";
}


void AVWiSContractTestAccess::run_p5_viscous_contract_checks(amrex::Real viscosity)
{
    if (!std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::runtime_error("P5 viscous contract requires finite positive avwis.viscosity");
    }
    amrex::MultiFab rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab error(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab work(m_ba, m_dm, AMREX_SPACEDIM, 0);
    const amrex::Real roundoff = 8192.0 * std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real amplitudes[AMREX_SPACEDIM] = {AMREX_D_DECL(1.0, -0.5, 0.25)};

    if (m_boundary.enabled) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            for (int side = 0; side < 2; ++side) {
                if (m_boundary.sides[2 * dir + side].velocity != CartesianBC::NoSlipWall) {
                    throw std::runtime_error(
                        "P5 physical viscous contract requires all non-periodic faces to be noslip");
                }
            }
        }
        const auto problo = m_geom.ProbLoArray();
        const auto dx = m_geom.CellSizeArray();
        const amrex::Real lx = m_geom.ProbLength(0);
        const amrex::Real ly = m_geom.ProbLength(1);
        const amrex::Real lz = m_geom.ProbLength(2);
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
                    const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
                    u(i,j,k,comp) = amplitudes[comp] *
                        amrex::Math::sinpi((x-problo[0]) / lx) *
                        amrex::Math::sinpi((y-problo[1]) / ly) *
                        amrex::Math::sinpi((z-problo[2]) / lz);
                });
        }
        mark_valid_modified();
        compute_cartesian_viscous_rhs(rhs, viscosity);

        const amrex::Real eigenvalue = -4.0 * viscosity * (
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[0] / lx) / m_dx[0], 2) +
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[1] / ly) / m_dx[1], 2) +
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[2] / lz) / m_dx[2], 2));
        const auto lo = m_geom.Domain().smallEnd();
        const auto hi = m_geom.Domain().bigEnd();
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi);
            auto const visc = rhs.const_array(mfi);
            auto const err = error.array(mfi);
            auto const wall = work.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    err(i,j,k,comp) = visc(i,j,k,comp) - eigenvalue * u(i,j,k,comp);
                    amrex::Real flux = 0.0;
                    if (i == lo[0] || i == hi[0]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[1]*dx[2] / dx[0];
                    if (j == lo[1] || j == hi[1]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[0]*dx[2] / dx[1];
                    if (k == lo[2] || k == hi[2]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[0]*dx[1] / dx[2];
                    wall(i,j,k,comp) = flux;
                });
        }
        amrex::Real max_error = 0.0;
        amrex::Real max_balance_error = 0.0;
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            amrex::Real stencil_error = error.norm0(comp, 0, true);
            amrex::Real volume_rhs = rhs.sum(comp, true) * m_cell_volume;
            amrex::Real wall_flux = work.sum(comp, true);
            amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
            amrex::ParallelDescriptor::ReduceRealSum(volume_rhs);
            amrex::ParallelDescriptor::ReduceRealSum(wall_flux);
            max_error = amrex::max(max_error, stencil_error);
            max_balance_error = amrex::max(max_balance_error, std::abs(volume_rhs-wall_flux));
        }
        const amrex::Real scale = std::abs(eigenvalue);
        if (max_error > roundoff * amrex::max(1.0, scale) ||
            max_balance_error > roundoff * amrex::max(1.0, scale)) {
            throw std::runtime_error("P5 no-slip viscous stencil/boundary-flux balance failed");
        }
        amrex::Print() << "AVWiS P5-002 boundary/multi-Box viscosity: PASS max_error="
                       << max_error << " boundary_balance_error=" << max_balance_error << "\n";
        return;
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5 manufactured viscous contract requires a fully periodic domain");
        }
    }
    const amrex::Real xlo = m_geom.ProbLo(0);
    const amrex::Real length = m_geom.ProbLength(0);
    const amrex::Real dx = m_dx[0];
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                u(i,j,k,comp) = amplitudes[comp] * amrex::Math::sinpi(2.0 * (x-xlo) / length);
            });
    }
    mark_valid_modified();
    compute_cartesian_viscous_rhs(rhs, viscosity);

    const amrex::Real pi = 3.141592653589793238462643383279502884;
    const amrex::Real exact_wavenumber = 2.0 * pi / length;
    const amrex::Real discrete_eigenvalue = -4.0 * viscosity *
        std::pow(std::sin(pi * dx / length) / dx, 2);
    const amrex::Real exact_eigenvalue = -viscosity * exact_wavenumber * exact_wavenumber;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const visc = rhs.const_array(mfi);
        auto const err = error.array(mfi);
        auto const energy = work.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                err(i,j,k,comp) = visc(i,j,k,comp) - discrete_eigenvalue * u(i,j,k,comp);
                energy(i,j,k,comp) = u(i,j,k,comp) * visc(i,j,k,comp);
            });
    }
    amrex::Real max_error = 0.0;
    amrex::Real max_momentum = 0.0;
    amrex::Real energy = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        amrex::Real stencil_error = error.norm0(comp, 0, true);
        amrex::Real momentum = rhs.sum(comp, true) * m_cell_volume;
        amrex::Real component_energy = work.sum(comp, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
        amrex::ParallelDescriptor::ReduceRealSum(momentum);
        amrex::ParallelDescriptor::ReduceRealSum(component_energy);
        max_error = amrex::max(max_error, stencil_error);
        max_momentum = amrex::max(max_momentum, std::abs(momentum));
        energy += component_energy;
    }
    amrex::MultiFab::Copy(error, rhs, 0, 0, AMREX_SPACEDIM, 0);
    for (amrex::MFIter mfi(error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const cont = error.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                cont(i,j,k,comp) -= exact_eigenvalue * u(i,j,k,comp);
            });
    }
    amrex::Real continuous_linf = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        amrex::Real norm = error.norm0(comp, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(norm);
        continuous_linf = amrex::max(continuous_linf, norm);
    }
    const amrex::Real accuracy_bound = viscosity * std::pow(exact_wavenumber, 4) * dx * dx / 12.0;
    const amrex::Real scale = std::abs(exact_eigenvalue);
    if (max_error > roundoff * amrex::max(1.0, scale) ||
        max_momentum > roundoff * amrex::max(1.0, scale) ||
        !(energy < 0.0) || continuous_linf > accuracy_bound) {
        throw std::runtime_error("P5 periodic viscous stencil/conservation/dissipation failed");
    }
    amrex::Print() << "AVWiS P5-002 periodic manufactured viscosity: PASS dx=" << dx
                   << " stencil_error=" << max_error
                   << " continuous_Linf=" << continuous_linf
                   << " momentum_error=" << max_momentum
                   << " energy_rate=" << energy << "\n";
}


void AVWiSContractTestAccess::run_p5_time_contract_checks(
    amrex::Real dt, amrex::Real final_time, amrex::Real viscosity)
{
    if (!std::isfinite(final_time) || final_time <= 0.0 ||
        !std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::runtime_error("P5-004 time contract requires positive final_time and viscosity");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5-004 manufactured time contract requires a fully periodic domain");
        }
    }
    const amrex::Real coarse_steps_real = final_time / dt;
    const int coarse_steps = static_cast<int>(std::llround(coarse_steps_real));
    if (coarse_steps < 2 ||
        std::abs(coarse_steps_real - coarse_steps) > 128.0 * std::numeric_limits<amrex::Real>::epsilon() * coarse_steps) {
        throw std::runtime_error("P5-004 final_time must be an integer multiple of dt with at least two steps");
    }

    struct SequenceResult {
        amrex::Real error = 0.0;
        amrex::Real momentum_drift = 0.0;
        amrex::Real max_divergence = 0.0;
        amrex::Real history_error = 0.0;
        TimeStepDiagnostics stability;
    };

    auto max_difference = [&](amrex::MultiFab const& lhs, amrex::MultiFab const& rhs,
                              int components) {
        amrex::MultiFab difference(lhs.boxArray(), lhs.DistributionMap(), components, 0);
        amrex::MultiFab::Copy(difference, lhs, 0, 0, components, 0);
        amrex::MultiFab::Subtract(difference, rhs, 0, 0, components, 0);
        amrex::Real result = 0.0;
        for (int comp = 0; comp < components; ++comp) {
            result = amrex::max(result, difference.norm0(comp, 0, true));
        }
        amrex::ParallelDescriptor::ReduceRealMax(result);
        return result;
    };

    auto run_sequence = [&](amrex::Real step_dt, int steps) {
        SequenceResult result;
        const amrex::Real xlo = m_geom.ProbLo(0);
        const amrex::Real length = m_geom.ProbLength(0);
        const amrex::Real dx = m_dx[0];
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                u(i,j,k,0) = 0.0;
                u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
                u(i,j,k,2) = 0.0;
            });
        }
        mark_valid_modified();
        sync_ucont_from_ucat();
        amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab::Copy(m_ucat_older, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
        amrex::MultiFab initial_cells(m_ba, m_dm, AMREX_SPACEDIM, 0);
        amrex::MultiFab::Copy(initial_cells, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> initial_faces;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            initial_faces[dir].define(m_ucont[dir].boxArray(), m_dm, 1, 0);
            amrex::MultiFab::Copy(initial_faces[dir], m_ucont[dir], 0, 0, 1, 0);
            amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
            amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont[dir], 0, 0, 1, m_nghost);
        }
        m_time = 0.0;
        m_step = 0;
        m_history_depth = 1;
        result.stability = time_step_diagnostics(step_dt, viscosity);

        amrex::Real initial_momentum = m_ucat.sum(1, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealSum(initial_momentum);
        amrex::MultiFab previous(m_ba, m_dm, AMREX_SPACEDIM, 0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> previous_faces;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            previous_faces[dir].define(m_ucont[dir].boxArray(), m_dm, 1, 0);
        }

        for (int step = 0; step < steps; ++step) {
            amrex::MultiFab::Copy(previous, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                amrex::MultiFab::Copy(previous_faces[dir], m_ucont[dir], 0, 0, 1, 0);
            }
            advance_one_step(step_dt, viscosity);
            result.history_error = amrex::max(
                result.history_error, max_difference(m_ucat_old, previous, AMREX_SPACEDIM));
            if (step == 1) {
                result.history_error = amrex::max(
                    result.history_error, max_difference(m_ucat_older, initial_cells, AMREX_SPACEDIM));
            }
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                result.history_error = amrex::max(
                    result.history_error, max_difference(m_ucont_old[dir], previous_faces[dir], 1));
                if (step == 1) {
                    result.history_error = amrex::max(
                        result.history_error, max_difference(m_ucont_older[dir], initial_faces[dir], 1));
                }
            }
        }

        const amrex::Real pi = 3.141592653589793238462643383279502884;
        const amrex::Real eigenvalue = -4.0 * viscosity *
            std::pow(std::sin(pi * dx / length) / dx, 2);
        const amrex::Real exact_factor = std::exp(eigenvalue * final_time);
        amrex::MultiFab error(m_ba, m_dm, 1, 0);
        for (amrex::MFIter mfi(error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi);
            auto const err = error.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                const amrex::Real exact = exact_factor * amrex::Math::sinpi(2.0 * (x-xlo) / length);
                err(i,j,k) = u(i,j,k,1) - exact;
            });
        }
        result.error = error.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(result.error);
        amrex::Real final_momentum = m_ucat.sum(1, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealSum(final_momentum);
        result.momentum_drift = std::abs(final_momentum - initial_momentum);
        amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
        compute_cartesian_divergence(divergence);
        result.max_divergence = divergence.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(result.max_divergence);
        return result;
    };

    const SequenceResult coarse = run_sequence(dt, coarse_steps);
    const SequenceResult fine = run_sequence(0.5 * dt, 2 * coarse_steps);
    const amrex::Real order_ratio = coarse.error / fine.error;
    const amrex::Real roundoff = 32768.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (!(order_ratio > 1.8 && order_ratio < 2.2) ||
        fine.momentum_drift > roundoff || fine.max_divergence > roundoff ||
        fine.history_error > roundoff || m_step != static_cast<std::uint64_t>(2 * coarse_steps) ||
        m_history_depth != 3 || std::abs(m_time - final_time) > roundoff ||
        coarse.stability.advective_cfl > 1.0 || coarse.stability.diffusive_number > 1.0) {
        throw std::runtime_error("P5-004 explicit temporal order/conservation/history contract failed");
    }

    amrex::Print() << "AVWiS P5-004 explicit Euler time contract: PASS"
                   << " coarse_error=" << coarse.error
                   << " fine_error=" << fine.error
                   << " ratio=" << order_ratio
                   << " advective_CFL=" << coarse.stability.advective_cfl
                   << " diffusive_number=" << coarse.stability.diffusive_number
                   << " momentum_drift=" << fine.momentum_drift
                   << " max_divergence=" << fine.max_divergence
                   << " history_error=" << fine.history_error
                   << " projection_time_coefficient="
                   << fine.stability.projection_time_coefficient << "\n";
}
