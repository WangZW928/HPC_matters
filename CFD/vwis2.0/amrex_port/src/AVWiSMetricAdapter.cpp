#include "AVWiSMetricAdapter.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>

#include <stdexcept>

void compute_identity_metric_divergence(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    amrex::MultiFab& divergence)
{
    if (metric.mapping_id() != "identity") {
        throw std::runtime_error("AVWiS C1 divergence adapter accepts identity MetricData only");
    }
    if (expected_metric_epoch == 0 || metric.epoch() != expected_metric_epoch) {
        throw std::runtime_error("AVWiS C1 divergence adapter detected a stale metric epoch");
    }

    amrex::MultiFab const& volume = metric.cell_volume_cc();
    if (!divergence.boxArray().ixType().cellCentered() || divergence.nComp() != 1 ||
        divergence.boxArray() != volume.boxArray() ||
        divergence.DistributionMap() != volume.DistributionMap()) {
        throw std::runtime_error("AVWiS C1 divergence/metric cell layouts differ");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (ucont[dir] == nullptr || ucont[dir]->nComp() != 1 ||
            ucont[dir]->boxArray() != metric.face_area_vector_fc(dir).boxArray() ||
            ucont[dir]->DistributionMap() != volume.DistributionMap()) {
            throw std::runtime_error("AVWiS C1 Ucont/metric face layouts differ");
        }
    }

    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const div = divergence.array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const fx = ucont[0]->const_array(mfi);
        auto const fy = ucont[1]->const_array(mfi);
        auto const fz = ucont[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Ucont already contains u dot S. Do not apply face area a second time.
            div(i,j,k) = ((fx(i+1,j,k)-fx(i,j,k)) +
                          (fy(i,j+1,k)-fy(i,j,k)) +
                          (fz(i,j,k+1)-fz(i,j,k))) / cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
}
