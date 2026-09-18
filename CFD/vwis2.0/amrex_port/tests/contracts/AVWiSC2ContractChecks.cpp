#include "AVWiSContractTestAccess.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <limits>
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

void AVWiSContractTestAccess::run_p5_orthogonal_time_contract_checks(
    std::string const& report_path, amrex::Real dt, amrex::Real viscosity)
{
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG ||
        !m_geom.isAllPeriodic() || !(dt > 0.0) || !(viscosity > 0.0) ||
        report_path.empty()) {
        throw std::runtime_error(
            "C2.3 mapped time contract requires periodic orthogonal mapping, positive dt/viscosity, and report path");
    }

    amrex::MultiFab advection(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab viscous(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab error_work(m_ba, m_dm, 2, 0);
    auto const logical = m_metric_data.logical_grid();
    auto const parameters = m_mapping_operator.analytic_parameters;
    auto const domain = m_geom.Domain();
    const amrex::Real logical_length = logical.spacing[0] * domain.length(0);
    const amrex::Real wave_number = 2.0 * 3.141592653589793238462643383279502884 /
                                    logical_length;
    constexpr amrex::Real transport_speed = 0.35;

    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        const amrex::Real xi_lo = logical.lower[0];
        const amrex::Real h = logical.spacing[0];
        const int ilo = domain.smallEnd(0);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real xi = xi_lo + (static_cast<amrex::Real>(i-ilo) + 0.5) * h;
            u(i,j,k,0) = transport_speed;
            u(i,j,k,1) = std::sin(wave_number * (xi-xi_lo));
            u(i,j,k,2) = 0.0;
        });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();
    compute_advection_rhs(advection);
    compute_viscous_rhs(viscous, viscosity);

    for (amrex::MFIter mfi(error_work, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = advection.const_array(mfi);
        auto const visc = viscous.const_array(mfi);
        auto const error = error_work.array(mfi);
        auto const volume = m_metric_data.cell_volume_cc().const_array(mfi);
        const amrex::Real xi_lo = logical.lower[0];
        const amrex::Real h = logical.spacing[0];
        const int ilo = domain.smallEnd(0);
        const amrex::Real scale = parameters.scale[0];
        const amrex::Real stretch = parameters.stretch[0];
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real xi = xi_lo + (static_cast<amrex::Real>(i-ilo) + 0.5) * h;
            const amrex::Real phase = wave_number * (xi-xi_lo);
            const amrex::Real map_prime = scale * (1.0 + stretch * std::cos(phase));
            const amrex::Real map_second = -scale * stretch * wave_number * std::sin(phase);
            const amrex::Real exact_adv = -transport_speed * wave_number *
                                           std::cos(phase) / map_prime;
            const amrex::Real exact_visc = viscosity *
                (-wave_number * wave_number * std::sin(phase) /
                     (map_prime * map_prime)
                 - wave_number * std::cos(phase) * map_second /
                     (map_prime * map_prime * map_prime));
            const amrex::Real adv_error = adv(i,j,k,1) - exact_adv;
            const amrex::Real visc_error = visc(i,j,k,1) - exact_visc;
            error(i,j,k,0) = adv_error * adv_error * volume(i,j,k);
            error(i,j,k,1) = visc_error * visc_error * volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
    amrex::Real physical_volume = m_metric_data.cell_volume_cc().sum(0, true);
    amrex::Real advection_l2 = error_work.sum(0, true);
    amrex::Real viscous_l2 = error_work.sum(1, true);
    amrex::ParallelDescriptor::ReduceRealSum(physical_volume);
    amrex::ParallelDescriptor::ReduceRealSum(advection_l2);
    amrex::ParallelDescriptor::ReduceRealSum(viscous_l2);
    advection_l2 = std::sqrt(advection_l2 / physical_volume);
    viscous_l2 = std::sqrt(viscous_l2 / physical_volume);

    // This transverse shear is discretely divergence-free. Its non-trivial
    // mapped advection/diffusion predictor should therefore pass through the
    // matched projection without changing the explicit Euler result.
    amrex::MultiFab expected_step(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(expected_step, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
    for (amrex::MFIter mfi(expected_step, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const expected = expected_step.array(mfi);
        auto const adv = advection.const_array(mfi);
        auto const visc = viscous.const_array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                expected(i,j,k,comp) += dt * (adv(i,j,k,comp) + visc(i,j,k,comp));
            });
    }
    amrex::Gpu::streamSynchronize();
    advance_one_step(dt, viscosity);
    amrex::MultiFab::Subtract(expected_step, m_ucat, 0, 0, AMREX_SPACEDIM, 0);
    amrex::Real manufactured_step_error = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        manufactured_step_error = amrex::max(
            manufactured_step_error, expected_step.norm0(comp, 0, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(manufactured_step_error);
    if (manufactured_step_error > 1.0e-9) {
        throw std::runtime_error(
            "C2.3 mapped time step changed a divergence-free manufactured predictor");
    }

    // A uniform physical velocity is preserved by mapped advection, mapped
    // diffusion, and the matched orthogonal projection/time path.
    constexpr amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> free_stream{
        0.2, -0.1, 0.05};
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                u(i,j,k,comp) = free_stream[comp];
            });
    }
    m_p.setVal(0.0);
    m_phi.setVal(0.0);
    mark_valid_modified();
    sync_ucont_from_ucat();
    advance_one_step(dt, viscosity);
    amrex::Real free_stream_error = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        m_ucat.plus(-free_stream[comp], comp, 1, 0);
        free_stream_error = amrex::max(
            free_stream_error, m_ucat.norm0(comp, 0, true));
        m_ucat.plus(free_stream[comp], comp, 1, 0);
    }
    amrex::ParallelDescriptor::ReduceRealMax(free_stream_error);
    const amrex::Real free_stream_tolerance =
        32768.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (free_stream_error > free_stream_tolerance) {
        throw std::runtime_error("C2.3 mapped time step did not preserve free stream");
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(report_path);
        if (!output) throw std::runtime_error("cannot write C2.3 report: " + report_path);
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n  \"schema\": \"avwis-c2.3-orthogonal-mms-v1\",\n"
               << "  \"status\": \"PASS\",\n  \"n_cell\": " << domain.length(0) << ",\n"
               << "  \"logical_dx\": " << logical.spacing[0] << ",\n"
               << "  \"advection_l2\": " << advection_l2 << ",\n"
               << "  \"viscous_l2\": " << viscous_l2 << ",\n"
               << "  \"manufactured_step_error\": " << manufactured_step_error << ",\n"
               << "  \"free_stream_step_error\": " << free_stream_error << "\n}\n";
    }
    amrex::Print() << "AVWiS P5-003 C2.3 mapped RHS/time: PASS n="
                   << domain.length(0) << " advection_L2=" << advection_l2
                   << " viscous_L2=" << viscous_l2
                   << " manufactured_step_error=" << manufactured_step_error
                   << " free_stream_step_error=" << free_stream_error << "\n";
}
