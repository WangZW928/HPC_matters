#include "AVWiSMetricData.H"

#include <AMReX.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
static_assert(std::is_same_v<decltype(std::declval<MetricData const&>().cell_volume_cc()),
                             amrex::MultiFab const&>,
              "MetricData production accessors must remain const-only");
static_assert(std::is_same_v<decltype(std::declval<MetricData const&>().face_area_vector_fc(0)),
                             amrex::MultiFab const&>,
              "MetricData face accessors must remain const-only");

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool differs(amrex::Real value, amrex::Real expected, amrex::Real scale = 1.0) noexcept
{
    amrex::Real const tolerance = 256.0 * std::numeric_limits<amrex::Real>::epsilon()
                                * amrex::max(amrex::Real(1.0), scale);
    return std::abs(value - expected) > tolerance;
}

void check_layout(MetricData const& metric, amrex::BoxArray const& boxes, int nghost)
{
    if (!metric.is_defined() || metric.epoch() != 1 || metric.mapping_id() != "identity" ||
        metric.nghost() != nghost || metric.node_coordinates_nd().nComp() != 3 ||
        metric.cell_center_coordinates_cc().nComp() != 3 ||
        metric.mapping_jacobian_cc().nComp() != 1 ||
        metric.inverse_mapping_jacobian_cc().nComp() != 1 ||
        metric.grad_xi_cc().nComp() != 9 || metric.area_cofactor_cc().nComp() != 9 ||
        metric.cell_volume_cc().nComp() != 1 ||
        metric.node_coordinates_nd().nGrow() < nghost ||
        metric.cell_volume_cc().nGrow() != nghost) {
        throw std::runtime_error("P5-003 MetricData field/epoch/read-only layout contract failed");
    }
    if (metric.node_coordinates_nd().boxArray() !=
        amrex::convert(boxes, amrex::IntVect::TheNodeVector())) {
        throw std::runtime_error("P5-003 nodal BoxArray contract failed");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const expected = amrex::convert(boxes, amrex::IntVect::TheDimensionVector(dir));
        if (metric.face_area_vector_fc(dir).boxArray() != expected ||
            metric.face_area_vector_fc(dir).nComp() != 3 ||
            metric.face_area_vector_fc(dir).nGrow() != nghost ||
            metric.face_gradient_metric_fc(dir).boxArray() != expected ||
            metric.face_gradient_metric_fc(dir).nComp() != 3 ||
            metric.projection_beta_fc(dir).nComp() != 1 ||
            metric.projection_beta_fc(dir).nGrow() != 0) {
            throw std::runtime_error("P5-003 face metric IndexType/layout contract failed");
        }
    }
}

void check_identity_values(MetricData const& metric, LogicalGrid const& logical,
                           amrex::Geometry const& geometry)
{
    const amrex::Real h0 = logical.spacing[0];
    const amrex::Real h1 = logical.spacing[1];
    const amrex::Real h2 = logical.spacing[2];
    const amrex::Real expected_volume = h0 * h1 * h2;
    const amrex::Real area[3] = {h1 * h2, h0 * h2, h0 * h1};
    const int ilo = logical.cell_domain.smallEnd(0);
    const int jlo = logical.cell_domain.smallEnd(1);
    const int klo = logical.cell_domain.smallEnd(2);
    const amrex::Real xlo = logical.lower[0];
    const amrex::Real ylo = logical.lower[1];
    const amrex::Real zlo = logical.lower[2];

    amrex::Gpu::DeviceScalar<int> device_errors(0);
    int* const errors = device_errors.dataPtr();
    for (amrex::MFIter mfi(metric.node_coordinates_nd(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = metric.node_coordinates_nd().const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const expected[3] = {
                xlo + static_cast<amrex::Real>(i - ilo) * h0,
                ylo + static_cast<amrex::Real>(j - jlo) * h1,
                zlo + static_cast<amrex::Real>(k - klo) * h2};
            for (int comp = 0; comp < 3; ++comp) {
                if (differs(xyz(i, j, k, comp), expected[comp], std::abs(expected[comp]))) {
                    amrex::Gpu::Atomic::Add(errors, 1);
                }
            }
        });
    }
    for (amrex::MFIter mfi(metric.cell_volume_cc(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const center = metric.cell_center_coordinates_cc().const_array(mfi);
        auto const jx = metric.mapping_jacobian_cc().const_array(mfi);
        auto const jxi = metric.inverse_mapping_jacobian_cc().const_array(mfi);
        auto const grad = metric.grad_xi_cc().const_array(mfi);
        auto const cofactor = metric.area_cofactor_cc().const_array(mfi);
        auto const volume = metric.cell_volume_cc().const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const expected_center[3] = {
                xlo + (static_cast<amrex::Real>(i - ilo) + 0.5) * h0,
                ylo + (static_cast<amrex::Real>(j - jlo) + 0.5) * h1,
                zlo + (static_cast<amrex::Real>(k - klo) + 0.5) * h2};
            if (differs(jx(i, j, k), 1.0) || differs(jxi(i, j, k), 1.0) ||
                differs(volume(i, j, k), expected_volume, std::abs(expected_volume))) {
                amrex::Gpu::Atomic::Add(errors, 1);
            }
            for (int m = 0; m < 3; ++m) {
                if (differs(center(i, j, k, m), expected_center[m], std::abs(expected_center[m]))) {
                    amrex::Gpu::Atomic::Add(errors, 1);
                }
                for (int l = 0; l < 3; ++l) {
                    amrex::Real const expected = m == l ? 1.0 : 0.0;
                    if (differs(grad(i, j, k, 3 * m + l), expected) ||
                        differs(cofactor(i, j, k, 3 * m + l), expected)) {
                        amrex::Gpu::Atomic::Add(errors, 1);
                    }
                }
            }
        });
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::Real expected_area = area[dir];
        for (amrex::MFIter mfi(metric.face_area_vector_fc(dir), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const s = metric.face_area_vector_fc(dir).const_array(mfi);
            auto const q = metric.face_gradient_metric_fc(dir).const_array(mfi);
            auto const beta = metric.projection_beta_fc(dir).const_array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                for (int comp = 0; comp < 3; ++comp) {
                    amrex::Real const expected = comp == dir ? expected_area : 0.0;
                    if (differs(s(i, j, k, comp), expected, std::abs(expected_area)) ||
                        differs(q(i, j, k, comp), expected, std::abs(expected_area))) {
                        amrex::Gpu::Atomic::Add(errors, 1);
                    }
                }
            });
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (beta(i, j, k) != 1.0) amrex::Gpu::Atomic::Add(errors, 1);
            });
        }

        amrex::Long expected_unique_faces = 1;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            expected_unique_faces *= logical.cell_domain.length(d) + (d == dir ? 1 : 0);
        }
        auto owner_mask = metric.face_area_vector_fc(dir).OwnerMask(geometry.periodicity());
        if (owner_mask->sum(0, 0, false) != expected_unique_faces) {
            throw std::runtime_error("P5-003 shared-face owner count contract failed");
        }
    }
    amrex::Gpu::streamSynchronize();
    int error_count = device_errors.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(error_count);
    if (error_count != 0) {
        throw std::runtime_error("P5-003 identity Cartesian/ghost/shared-face value regression failed");
    }
}

void check_constant_flux(MetricData const& metric, amrex::BoxArray const& boxes,
                         amrex::DistributionMapping const& distribution)
{
    constexpr amrex::Real u0 = 1.25;
    constexpr amrex::Real u1 = -0.5;
    constexpr amrex::Real u2 = 2.0;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        flux[dir].define(metric.face_area_vector_fc(dir).boxArray(), distribution, 1, 1);
        for (amrex::MFIter mfi(flux[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const s = metric.face_area_vector_fc(dir).const_array(mfi);
            auto const ucont = flux[dir].array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                ucont(i, j, k) = u0 * s(i, j, k, 0)
                               + u1 * s(i, j, k, 1)
                               + u2 * s(i, j, k, 2);
            });
        }
        flux[dir].OverrideSync(amrex::Periodicity::NonPeriodic());
        flux[dir].FillBoundary(amrex::Periodicity::NonPeriodic());
    }

    amrex::MultiFab divergence(boxes, distribution, 1, 0);
    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const div = divergence.array(mfi);
        auto const volume = metric.cell_volume_cc().const_array(mfi);
        auto const fx = flux[0].const_array(mfi);
        auto const fy = flux[1].const_array(mfi);
        auto const fz = flux[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            div(i, j, k) = ((fx(i + 1, j, k) - fx(i, j, k))
                          + (fy(i, j + 1, k) - fy(i, j, k))
                          + (fz(i, j, k + 1) - fz(i, j, k))) / volume(i, j, k);
        });
    }
    amrex::Gpu::streamSynchronize();
    amrex::Real max_divergence = divergence.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(max_divergence);
    if (max_divergence > 1.0e-12) {
        throw std::runtime_error("P5-003 Ucont=u dot S / constant free-stream GCL contract failed");
    }
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        amrex::IntVect const lo(AMREX_D_DECL(0, 0, 0));
        amrex::IntVect const hi(AMREX_D_DECL(11, 9, 7));
        amrex::Box const domain(lo, hi);
        amrex::RealBox const physical_domain(
            {AMREX_D_DECL(0.25, -1.0, 2.0)},
            {AMREX_D_DECL(2.25, 2.0, 6.0)});
        amrex::Array<int, AMREX_SPACEDIM> const periodic{AMREX_D_DECL(0, 0, 0)};
        amrex::Geometry geometry(domain, &physical_domain, 0, periodic.data());
        amrex::BoxArray boxes(domain);
        boxes.maxSize(4);
        amrex::DistributionMapping distribution(boxes);

        LogicalGrid const logical = LogicalGrid::from_cartesian_geometry(geometry);
        IdentityCoordinateMapping mapping;
        MetricData metric;
        metric.define(boxes, distribution, 1);
        metric.build(mapping, logical, geometry);
        check_layout(metric, boxes, 1);

        MetricDiagnostics const diagnostics = metric.validate();
        if (!diagnostics.passed || diagnostics.minimum_mapping_jacobian <= 0.0 ||
            diagnostics.minimum_cell_volume <= 0.0 || diagnostics.minimum_face_area <= 0.0 ||
            diagnostics.maximum_reciprocal_relative_error > 5.0e-13 ||
            diagnostics.maximum_gcl_relative_error > 1.0e-12 ||
            diagnostics.maximum_reciprocity_relative_error > 1.0e-11) {
            throw std::runtime_error("P5-003 positive Jacobian/volume/GCL/reciprocity contract failed");
        }
        check_identity_values(metric, logical, geometry);
        check_constant_flux(metric, boxes, distribution);
        metric.rebuild(mapping, logical, geometry);
        if (metric.epoch() != 2 || !metric.validate().passed) {
            throw std::runtime_error("P5-003 explicit rebuild/epoch contract failed");
        }

        amrex::Print() << "AVWiS P5-003 G0.1 metric contract: PASS "
                       << "(identity/Jx/Jxi/volume/shared-face/GCL/Ucont/multi-Box/ghost)\n";
    } catch (std::exception const& error) {
        amrex::Print() << "AVWiS P5-003 metric contract error: " << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
