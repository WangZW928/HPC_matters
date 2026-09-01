#ifndef AVWIS_METRIC_ADAPTER_H_
#define AVWIS_METRIC_ADAPTER_H_

#include "AVWiSMetricData.H"

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>

#include <cstdint>

/**
 * C1/G0.2 adapter for the existing integrated-face-flux divergence path.
 *
 * This boundary deliberately accepts identity metrics only. Ucont remains an
 * integrated volume flux on each logical face, so the adapter applies exactly
 * one metric factor: division by the cell's discrete physical volume.
 */
void compute_identity_metric_divergence(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    amrex::MultiFab& divergence);

#endif
