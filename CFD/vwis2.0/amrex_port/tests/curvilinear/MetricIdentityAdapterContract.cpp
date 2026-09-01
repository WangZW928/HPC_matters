#include "AVWiSMetricAdapter.H"

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void run_case(int max_grid_size)
{
    amrex::IntVect const lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect const hi(AMREX_D_DECL(7, 5, 3));
    amrex::Box const domain(lo, hi);
    amrex::RealBox const physical({AMREX_D_DECL(-0.25, 0.5, 1.25)},
                                 {AMREX_D_DECL(1.75, 3.5, 2.25)});
    amrex::Array<int, AMREX_SPACEDIM> const periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geometry(domain, &physical, 0, periodic.data());
    amrex::BoxArray boxes(domain);
    boxes.maxSize(max_grid_size);
    amrex::DistributionMapping distribution(boxes);

    MetricData metric;
    metric.define(boxes, distribution, 1);
    IdentityCoordinateMapping mapping;
    metric.build(mapping, LogicalGrid::from_cartesian_geometry(geometry), geometry);
    std::uint64_t const epoch = metric.epoch();

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> flux_view{};
    amrex::Real const velocity[AMREX_SPACEDIM] = {AMREX_D_DECL(0.75, -1.25, 2.5)};
    auto const* dx = geometry.CellSize();
    amrex::Real const cartesian_area[AMREX_SPACEDIM] = {
        AMREX_D_DECL(dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1])};
    amrex::Real const cartesian_volume = dx[0] * dx[1] * dx[2];
    amrex::Real max_constant_flux_error = 0.0;

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        flux[dir].define(metric.face_area_vector_fc(dir).boxArray(), distribution, 1, 1);
        amrex::MultiFab difference(flux[dir].boxArray(), distribution, 1, 0);
        for (amrex::MFIter mfi(flux[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const s = metric.face_area_vector_fc(dir).const_array(mfi);
            auto const q = flux[dir].array(mfi);
            auto const delta = difference.array(mfi);
            amrex::Real const expected = velocity[dir] * cartesian_area[dir];
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                q(i,j,k) = velocity[0] * s(i,j,k,0) + velocity[1] * s(i,j,k,1)
                           + velocity[2] * s(i,j,k,2);
                delta(i,j,k) = q(i,j,k) - expected;
            });
        }
        flux[dir].OverrideSync(geometry.periodicity());
        flux[dir].FillBoundary(geometry.periodicity());
        max_constant_flux_error = amrex::max(max_constant_flux_error,
                                              difference.norm0(0, 0, true));
        flux_view[dir] = &flux[dir];
    }
    amrex::ParallelDescriptor::ReduceRealMax(max_constant_flux_error);

    amrex::MultiFab mapped_constant(boxes, distribution, 1, 0);
    compute_identity_metric_divergence(flux_view, metric, epoch, mapped_constant);
    amrex::Real max_constant_divergence = mapped_constant.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(max_constant_divergence);

    // Give each integrated face flux an affine normal-index increment. This
    // exercises a nonzero divergence independently of the constant-flux case.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Real const area = cartesian_area[dir];
        amrex::Real const slope = static_cast<amrex::Real>(dir + 1);
        for (amrex::MFIter mfi(flux[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const q = flux[dir].array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                int const face_index = dir == 0 ? i : (dir == 1 ? j : k);
                q(i,j,k) = slope * static_cast<amrex::Real>(face_index) * area;
            });
        }
        flux[dir].OverrideSync(geometry.periodicity());
        flux[dir].FillBoundary(geometry.periodicity());
    }

    amrex::MultiFab mapped(boxes, distribution, 1, 0);
    amrex::MultiFab cartesian_reference(boxes, distribution, 1, 0);
    compute_identity_metric_divergence(flux_view, metric, epoch, mapped);
    for (amrex::MFIter mfi(cartesian_reference, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const ref = cartesian_reference.array(mfi);
        auto const fx = flux[0].const_array(mfi);
        auto const fy = flux[1].const_array(mfi);
        auto const fz = flux[2].const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            ref(i,j,k) = ((fx(i+1,j,k)-fx(i,j,k)) +
                          (fy(i,j+1,k)-fy(i,j,k)) +
                          (fz(i,j,k+1)-fz(i,j,k))) / cartesian_volume;
        });
    }
    amrex::MultiFab::Subtract(mapped, cartesian_reference, 0, 0, 1, 0);
    amrex::Real max_divergence_difference = mapped.norm0(0, 0, true);
    amrex::Real reference_scale = cartesian_reference.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(max_divergence_difference);
    amrex::ParallelDescriptor::ReduceRealMax(reference_scale);
    amrex::Real const tolerance = 512.0 * std::numeric_limits<amrex::Real>::epsilon()
                                * amrex::max(amrex::Real(1.0), reference_scale);
    if (max_constant_flux_error > tolerance || max_constant_divergence > tolerance ||
        max_divergence_difference > tolerance) {
        throw std::runtime_error("P5-003 C1 identity adapter differs from Cartesian reference");
    }

    metric.rebuild(mapping, LogicalGrid::from_cartesian_geometry(geometry), geometry);
    bool rejected_stale_epoch = false;
    try {
        compute_identity_metric_divergence(flux_view, metric, epoch, mapped_constant);
    } catch (std::runtime_error const&) {
        rejected_stale_epoch = true;
    }
    if (!rejected_stale_epoch) {
        throw std::runtime_error("P5-003 C1 adapter accepted a stale metric epoch");
    }
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        run_case(64); // one Box
        run_case(3);  // many Boxes and overlapping faces
        amrex::Print() << "AVWiS P5-003 C1 identity metric adapter contract: PASS\n";
    } catch (std::exception const& error) {
        amrex::Print() << "AVWiS P5-003 C1 adapter contract error: " << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
