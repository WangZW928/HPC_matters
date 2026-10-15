#include "AVWiSSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

TimeStepDiagnostics AVWiSSolver::time_step_diagnostics(
    amrex::Real dt, amrex::Real viscosity) const
{
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error("P5-004 explicit dt must be finite and positive");
    }
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
        throw std::runtime_error("P5-004 viscosity must be finite and non-negative");
    }

    TimeStepDiagnostics result;
    if (m_mapping_operator.mapping_type == "analytic_shear" && m_mapping_operator.shear_epsilon != 0.0) {
        auto const h = m_metric_data.logical_grid().spacing;
        amrex::Real total_inverse_length = 0.0;
        for (int d=0;d<3;++d) {
            amrex::Real row_bound = 0.0;
            for (int c=0;c<3;++c) {
                amrex::Real g = m_metric_data.grad_xi_cc().norm0(3*d+c,0,true);
                amrex::Real u = m_ucat.norm0(c,0,true);
                amrex::ParallelDescriptor::ReduceRealMax(g);
                amrex::ParallelDescriptor::ReduceRealMax(u);
                row_bound += g;
                result.advective_cfl += dt*g*u/h[d];
            }
            total_inverse_length += row_bound/h[d];
        }
        // Includes cross coefficients via an absolute row-sum upper bound.
        result.diffusive_number = 2.0*viscosity*dt*total_inverse_length*total_inverse_length;
        return result;
    }
    auto const logical_spacing = m_metric_data.logical_grid().spacing;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Real max_velocity = m_ucat.norm0(dir, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(max_velocity);
        amrex::Real max_inverse_scale = 1.0;
        if (m_mapping_operator.coordinates == CoordinateSystemMode::Mapped) {
            max_inverse_scale = m_metric_data.grad_xi_cc().norm0(3 * dir + dir, 0, true);
            amrex::ParallelDescriptor::ReduceRealMax(max_inverse_scale);
        }
        const amrex::Real inverse_physical_spacing =
            max_inverse_scale / logical_spacing[dir];
        result.advective_cfl += dt * max_velocity * inverse_physical_spacing;
        result.diffusive_number += 2.0 * viscosity * dt *
            inverse_physical_spacing * inverse_physical_spacing;
    }
    return result;
}

void AVWiSSolver::advance_one_step(
    amrex::Real dt, amrex::Real viscosity,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& body_force,
    amrex::MultiFab const* spatial_body_force)
{
    // Reject invalid input before touching any velocity, history, or epoch.
    if (spatial_body_force &&
        (spatial_body_force->boxArray() != m_ba ||
         spatial_body_force->DistributionMap() != m_dm ||
         spatial_body_force->nComp() != AMREX_SPACEDIM ||
         !spatial_body_force->is_finite(0, AMREX_SPACEDIM, 0))) {
        throw std::runtime_error(
            "spatial body force must be finite cell acceleration on the solver layout with 3 components");
    }
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> force{};
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        if (!std::isfinite(body_force[comp])) {
            throw std::runtime_error("P5-004 body force must contain finite values");
        }
        force[comp] = body_force[comp];
    }
    const TimeStepDiagnostics stability = time_step_diagnostics(dt, viscosity);
    if (stability.advective_cfl > 1.0 + 64.0 * std::numeric_limits<amrex::Real>::epsilon() ||
        stability.diffusive_number > 1.0 + 64.0 * std::numeric_limits<amrex::Real>::epsilon()) {
        throw std::runtime_error(
            "P5-004 explicit step rejected: advective CFL and diffusive number must each be <= 1");
    }

    if (m_mapping_operator.pressure_contract == "incremental_pressure") {
        advance_incremental_pressure(dt, viscosity, force, spatial_body_force);
        return;
    }
    const amrex::Real start = amrex::second();
    compute_advection_rhs(m_advection_rhs);
    compute_viscous_rhs(m_viscous_rhs, viscosity);

    // After the step these layers are (n+1,n,n-1).  The explicit scheme reads
    // only layer n; retaining both older layers fixes restart/BDF2 data shape
    // without claiming that either an implicit or BDF2 solve exists.
    amrex::MultiFab::Copy(m_ucat_older, m_ucat_old, 0, 0, AMREX_SPACEDIM, m_nghost);
    amrex::MultiFab::Copy(m_ucat_old, m_ucat, 0, 0, AMREX_SPACEDIM, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab::Copy(m_ucont_older[dir], m_ucont_old[dir], 0, 0, 1, m_nghost);
        amrex::MultiFab::Copy(m_ucont_old[dir], m_ucont[dir], 0, 0, 1, m_nghost);
    }

    bool const face_authority = m_mapping_operator.state_contract == "face_authoritative";
    if (face_authority) {
        // Preserve Ucont^n. Interpolate only the acceleration, never overwrite
        // the mass-authoritative state by I(R(Ucont^n)). The reconstructed Ucat
        // supplies transport/diffusion at n; pressure is the current DG multiplier.
        auto& rate = m_deferred_workspace->predictor_rate;
        for (amrex::MFIter mfi(rate, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const r=rate.array(mfi);
            auto const adv=m_advection_rhs.const_array(mfi);
            auto const visc=m_viscous_rhs.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),3,
                [=] AMREX_GPU_DEVICE(int i,int j,int k,int c) noexcept {
                    r(i,j,k,c)=adv(i,j,k,c)+visc(i,j,k,c)+force[c];
                });
        }
        if (spatial_body_force)
            amrex::MultiFab::Add(rate, *spatial_body_force, 0, 0, 3, 0);
        rate.FillBoundary(m_geom.periodicity());
        amrex::Array<amrex::MultiFab*,3> rate_faces{
            &m_deferred_workspace->face_candidate[0],
            &m_deferred_workspace->face_candidate[1],
            &m_deferred_workspace->face_candidate[2]};
        sync_metric_ucont_from_ucat(rate,rate_faces,m_metric_data,m_metric_epoch,
                                   m_mapping_operator,m_geom.periodicity());
        for (int d=0;d<3;++d)
            amrex::MultiFab::Saxpy(m_ucont[d],dt,*rate_faces[d],0,0,1,0);
        amrex::Gpu::streamSynchronize();
        mark_valid_modified();fill_ghost_cells();
    } else {
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            auto const adv = m_advection_rhs.const_array(mfi);
            auto const visc = m_viscous_rhs.const_array(mfi);
            amrex::ParallelFor(
                mfi.tilebox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    u(i,j,k,comp) += dt *
                        (adv(i,j,k,comp) + visc(i,j,k,comp) + force[comp]);
                });
        }
        if (spatial_body_force)
            amrex::MultiFab::Saxpy(m_ucat, dt, *spatial_body_force, 0, 0, 3, 0);
        amrex::Gpu::streamSynchronize();
        mark_valid_modified();
        sync_ucont_from_ucat();
    }
    ProjectionDiagnostics projection;
    if (m_mapping_operator.coordinates == CoordinateSystemMode::Mapped) {
        projection=project_mapped(dt, stability.projection_time_coefficient);
    } else {
        projection=project_cartesian(dt, stability.projection_time_coefficient);
    }
    m_last_step_projection=projection;

    m_time += dt;
    ++m_step;
    m_history_depth = std::min(3, m_history_depth + 1);
    m_last_advance_seconds = amrex::second() - start;
}

void AVWiSSolver::advance_incremental_pressure(
    amrex::Real dt, amrex::Real viscosity,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& force,
    amrex::MultiFab const* spatial_body_force)
{
    auto& pressure = *m_pressure_increment;
    if (pressure.phase != PressureIncrementPhase::Ready)
        throw std::runtime_error("M2 pressure increment already in progress");
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    auto& work = *m_deferred_workspace;
    amrex::Real const start = amrex::second();
    // All predictor/correction work is provisional. Neither histories nor the
    // last accepted Phi/P/velocity/time/epochs are published on solve failure.
    pressure.phase = PressureIncrementPhase::Predictor;
    try {
        fill_ghost_cells();
        if (!m_p.is_finite(0,1,0))
            throw std::runtime_error("M2 incremental predictor pressure is nonfinite");
        compute_advection_rhs(m_advection_rhs);
        compute_viscous_rhs(m_viscous_rhs, viscosity);
        auto& rate = work.predictor_rate;
        amrex::MultiFab::Copy(rate,m_advection_rhs,0,0,3,0);
        amrex::MultiFab::Add(rate,m_viscous_rhs,0,0,3,0);
        for (int c=0;c<3;++c) rate.plus(force[c],c,1,0);
        if (spatial_body_force)
            amrex::MultiFab::Add(rate, *spatial_body_force, 0, 0, 3, 0);
        rate.FillBoundary(m_geom.periodicity());
        amrex::Array<amrex::MultiFab*,3> rate_faces{
            &m_phi_operator_flux[0], &m_phi_operator_flux[1], &m_phi_operator_flux[2]};
        sync_metric_ucont_from_ucat(rate, rate_faces, m_metric_data, m_metric_epoch,
                                   m_mapping_operator, m_geom.periodicity());
        amrex::Array<amrex::MultiFab const*,3> candidates{}, predictor{};
        for (int d=0;d<3;++d) {
            auto& face=work.face_candidate[d];
            amrex::MultiFab::Copy(face,m_ucont[d],0,0,1,0);
            amrex::MultiFab::Saxpy(face,dt,m_phi_operator_flux[d],0,0,1,0);
            // Keep B=F^n+dt I(A^n) in candidate faces; materialize the
            // old-P predictor F*=B-dt G_f(P^n) separately for diagnostics.
            compute_metric_face_gradient_flux(m_p,d,m_metric_data,m_metric_epoch,
                m_mapping_operator,m_phi_operator_flux[d]);
            m_phi_operator_flux[d].mult(-dt,0,1,0);
            amrex::MultiFab::Add(m_phi_operator_flux[d],face,0,0,1,0);
            predictor[d]=&m_phi_operator_flux[d];
            face.OverrideSync(m_geom.periodicity());
            face.FillBoundary(m_geom.periodicity());
            candidates[d]=&face;
            if (!face.is_finite(0,1,0) || !predictor[d]->is_finite(0,1,0))
                throw std::runtime_error("M2 incremental predictor face is nonfinite");
        }
        ProjectionDiagnostics report;
        report.singular=true;
        compute_metric_divergence(predictor,m_metric_data,m_metric_epoch,
            m_mapping_operator,m_projection_rhs);
        report.max_divergence_before=m_projection_rhs.norm0(0,0,true);
        amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_before);
        // Shift the unknown Q=P^n+Phi: (-N G_f)Q=-N(B)/dt is
        // algebraically identical to (-N G_f)Phi=-N(F*)/dt. Solving
        // for Q avoids a small, cancellation-dominated incremental RHS.
        // No mean repair or relaxed inner/full stopping criterion is used.
        compute_metric_net_flux(candidates,m_metric_data,m_metric_epoch,
            m_mapping_operator,m_projection_rhs);
        m_projection_rhs.mult(-1.0/dt,0,1,0);
        auto const solved=solve_incremental_pressure(pressure.pressure,m_projection_rhs);
        amrex::MultiFab::Copy(pressure.phi,pressure.pressure,0,0,1,0);
        amrex::MultiFab::Subtract(pressure.phi,m_p,0,0,1,0);
        pressure.phi.FillBoundary(m_geom.periodicity());
        // Publish an exact once update P_old+Phi, and correct faces with
        // that same rounded combined multiplier. Periodic P uses zero datum.
        amrex::MultiFab::Copy(pressure.pressure,m_p,0,0,1,0);
        amrex::MultiFab::Add(pressure.pressure,pressure.phi,0,0,1,0);
        pressure.pressure.FillBoundary(m_geom.periodicity());
        report.outer_iterations=solved.iterations;
        report.inner_iterations=solved.inner_iterations;
        report.full_relative_residual=solved.full_relative_residual;
        report.final_residual=solved.full_max_residual;
        for (int d=0;d<3;++d) {
            compute_metric_face_gradient_flux(pressure.pressure,d,m_metric_data,m_metric_epoch,
                m_mapping_operator,m_phi_operator_flux[d]);
            auto& face=work.face_candidate[d];
            amrex::MultiFab::Saxpy(face,-dt,m_phi_operator_flux[d],0,0,1,0);
            face.OverrideSync(m_geom.periodicity());
            face.FillBoundary(m_geom.periodicity());
            if (!face.is_finite(0,1,0))
                throw std::runtime_error("M2 incremental projected face is nonfinite");
        }
        // Re-audit the actual rounded P_old+Phi used for face correction.
        // The solve's full residual alone does not cover subtraction/addition.
        amrex::Array<amrex::MultiFab const*,3> pressure_flux{
            &m_phi_operator_flux[0],&m_phi_operator_flux[1],&m_phi_operator_flux[2]};
        compute_metric_divergence(pressure_flux,m_metric_data,m_metric_epoch,
            m_mapping_operator,work.residual);
        amrex::MultiFab::Add(work.residual,work.physical_rhs,0,0,1,0);
        report.final_residual=work.residual.norm0(0,0,true);
        auto physical_scale=work.physical_rhs.norm0(0,0,true);
        amrex::ParallelDescriptor::ReduceRealMax(report.final_residual);
        amrex::ParallelDescriptor::ReduceRealMax(physical_scale);
        report.full_relative_residual=physical_scale>0 ? report.final_residual/physical_scale : report.final_residual;
        if (!std::isfinite(report.full_relative_residual) || report.full_relative_residual>1.0e-11)
            throw std::runtime_error("M2 incremental combined pressure failed full residual gate <= 1e-11");
        sync_metric_ucat_from_ucont(candidates,m_metric_data,m_metric_epoch,
            m_mapping_operator,work.reconstructed_cell);
        if (!pressure.pressure.is_finite(0,1,0) || !pressure.phi.is_finite(0,1,0) ||
            !work.reconstructed_cell.is_finite(0,3,0))
            throw std::runtime_error("M2 incremental pressure/cell candidate is nonfinite");
        compute_metric_divergence(candidates,m_metric_data,m_metric_epoch,
            m_mapping_operator,m_projection_rhs);
        report.max_divergence_after=m_projection_rhs.norm0(0,0,true);
        amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_after);
        if (!std::isfinite(report.max_divergence_after) || report.max_divergence_after>1.0e-10)
            throw std::runtime_error("M2 incremental candidate failed projection divergence gate <= 1e-10");
        pressure.phase=PressureIncrementPhase::CorrectionReady;
        // Commit once per accepted step. Allocation/communication/device faults
        // during publication are fatal; recoverable numerical checks end here.
        amrex::MultiFab::Copy(m_ucat_older,m_ucat_old,0,0,3,m_nghost);
        amrex::MultiFab::Copy(m_ucat_old,m_ucat,0,0,3,m_nghost);
        for (int d=0;d<3;++d) {
            amrex::MultiFab::Copy(m_ucont_older[d],m_ucont_old[d],0,0,1,m_nghost);
            amrex::MultiFab::Copy(m_ucont_old[d],m_ucont[d],0,0,1,m_nghost);
            amrex::MultiFab::Copy(m_ucont[d],*candidates[d],0,0,1,0);
        }
        amrex::MultiFab::Copy(m_ucat,work.reconstructed_cell,0,0,3,0);
        amrex::MultiFab::Copy(m_phi,pressure.phi,0,0,1,0);
        amrex::MultiFab::Copy(m_p,pressure.pressure,0,0,1,0);
        mark_valid_modified();fill_ghost_cells();
        m_last_step_projection=report;
        m_time+=dt;++m_step;m_history_depth=std::min(3,m_history_depth+1);
        m_last_advance_seconds=amrex::second()-start;
        pressure.phase=PressureIncrementPhase::Ready;
        amrex::Print()<<"M2 incremental pressure: accepted_step="<<m_step
            <<" alpha=1 predictor=old_P Phi=current_increment P=P_old+Phi"
            <<" solve_unknown=P_old+Phi datum=zero_volume_mean"
            <<" physical_pressure_convergence=UNVERIFIED iterations="<<report.outer_iterations
            <<" inner_iterations="<<report.inner_iterations
            <<" full_relative_residual="<<report.full_relative_residual
            <<" max_div_after="<<report.max_divergence_after<<"\n";
    } catch (...) {
        pressure.phase=PressureIncrementPhase::Ready;
        throw;
    }
}
