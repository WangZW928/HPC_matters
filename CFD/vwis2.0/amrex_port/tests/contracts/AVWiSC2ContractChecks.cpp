#include "AVWiSContractTestAccess.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <stdexcept>

void AVWiSContractTestAccess::run_p5_orthogonal_projection_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG ||
        m_metric_data.mapping_id() != "analytic_orthogonal") {
        throw std::runtime_error("C2.2 solver contract did not receive the explicit mapped configuration");
    }

    auto const domain = m_geom.Domain();
    int const ilo = domain.smallEnd(0);
    int const nx = domain.length(0);
    constexpr amrex::Real two_pi = 6.283185307179586476925286766559005768;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) m_ucont[dir].setVal(0.0);
    for (amrex::MFIter mfi(m_ucont[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const flux = m_ucont[0].array(mfi);
        auto const area = m_metric_data.face_area_vector_fc(0).const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const phase = two_pi * static_cast<amrex::Real>(i - ilo) /
                                      static_cast<amrex::Real>(nx);
            flux(i,j,k) = std::sin(phase) * area(i,j,k,0);
        });
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
    }
    mark_valid_modified();

    ProjectionDiagnostics const report = project_orthogonal(dt, time_coefficient);
    if (!(report.max_divergence_before > 1.0e-3) ||
        report.max_divergence_after > 1.0e-9 ||
        report.max_divergence_before /
            amrex::max(report.max_divergence_after, amrex::Real(1.0e-30)) < 1.0e8) {
        throw std::runtime_error("C2.2 orthogonal solver projection did not reduce divergence");
    }

    amrex::Print() << "AVWiS P5-003 C2.2 orthogonal solver projection contract: PASS "
                   << "before=" << report.max_divergence_before
                   << " after=" << report.max_divergence_after << "\n";
}
