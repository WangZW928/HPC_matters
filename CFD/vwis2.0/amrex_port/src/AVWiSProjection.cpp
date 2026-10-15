#include "AVWiSSolver.H"
#include "AVWiSMetricAdapter.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
amrex::Long global_cell_count(amrex::Box const& domain)
{
    amrex::Long result = 1;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) result *= domain.length(dir);
    return result;
}
} // namespace

void AVWiSSolver::validate_projection_boundary_policy() const
{
    if (!m_boundary.enabled) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P4 projection requires explicit vwisbcs on every non-periodic direction");
            }
        }
        return;
    }

    int inflows = 0;
    int outflows = 0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side = 0; side < 2; ++side) {
            auto const kind = m_boundary.sides[2 * dir + side].velocity;
            inflows += kind == CartesianBC::Inflow;
            outflows += kind == CartesianBC::Outflow;
        }
    }
    if (!((inflows == 1 && outflows == 1) || (inflows == 0 && outflows == 0))) {
        throw std::runtime_error(
            "P4 supports one pressure-Dirichlet outflow paired with one inflow, "
            "or a singular periodic/closed no-penetration domain");
    }
}

void AVWiSSolver::compute_cartesian_divergence(amrex::MultiFab& divergence) const
{
    if (divergence.boxArray() != m_ba || divergence.DistributionMap() != m_dm ||
        divergence.nComp() != 1) {
        throw std::runtime_error("P4 divergence destination must be one cell-centred component");
    }
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> flux{
        AMREX_D_DECL(&m_ucont[0], &m_ucont[1], &m_ucont[2])};
    compute_metric_divergence(flux, m_metric_data, m_metric_epoch,
                              m_mapping_operator, divergence);
}

amrex::Real
AVWiSSolver::validate_singular_rhs_compatibility(amrex::MultiFab const& rhs) const
{
    amrex::Real sum = rhs.sum(0, true);
    amrex::ParallelDescriptor::ReduceRealSum(sum);
    const amrex::Real mean = sum / static_cast<amrex::Real>(global_cell_count(m_geom.Domain()));
    amrex::Real scale = rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(scale);
    const amrex::Real tolerance = 4096.0 * std::numeric_limits<amrex::Real>::epsilon()
                                * amrex::max(1.0, scale);
    if (std::abs(mean) > tolerance) {
        throw std::runtime_error(
            "P4 singular periodic/Neumann pressure RHS is incompatible: mean=" +
            std::to_string(mean) + "; no automatic mean subtraction is permitted");
    }
    return mean;
}

ProjectionDiagnostics
AVWiSSolver::project_cartesian(amrex::Real dt, amrex::Real time_coefficient)
{
    if (m_mapping_operator.pressure_contract == "incremental_pressure")
        throw std::runtime_error("M2 incremental_pressure requires advance_one_step; standalone projection cannot commit a pressure increment");
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian ||
        m_mapping_operator.projection != ProjectionOperatorMode::CartesianMLMG) {
        throw std::runtime_error(
            "P4 Cartesian projection rejected a non-Cartesian mapping/operator configuration");
    }
    if (!(dt > 0.0) || !(time_coefficient > 0.0)) {
        throw std::runtime_error("P4 projection requires positive dt and time_coefficient");
    }
    validate_projection_boundary_policy();

    if (m_boundary.enabled) apply_boundary_pipeline("pre-p4-projection");
    else fill_ghost_cells();

    compute_cartesian_divergence(m_projection_rhs);
    ProjectionDiagnostics report;
    report.max_divergence_before = m_projection_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_before);

    const amrex::Real rhs_scale = time_coefficient / dt;
    m_projection_rhs.mult(rhs_scale, 0, 1, 0);

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc{};
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc{};
    bool has_pressure_dirichlet = false;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (m_geom.isPeriodic(dir)) {
            lobc[dir] = amrex::LinOpBCType::Periodic;
            hibc[dir] = amrex::LinOpBCType::Periodic;
        } else {
            auto const lo_kind = m_boundary.sides[2 * dir].velocity;
            auto const hi_kind = m_boundary.sides[2 * dir + 1].velocity;
            lobc[dir] = lo_kind == CartesianBC::Outflow
                      ? amrex::LinOpBCType::Dirichlet : amrex::LinOpBCType::Neumann;
            hibc[dir] = hi_kind == CartesianBC::Outflow
                      ? amrex::LinOpBCType::Dirichlet : amrex::LinOpBCType::Neumann;
            has_pressure_dirichlet = has_pressure_dirichlet ||
                lo_kind == CartesianBC::Outflow || hi_kind == CartesianBC::Outflow;
        }
    }
    report.singular = !has_pressure_dirichlet;

    amrex::Real rhs_sum = m_projection_rhs.sum(0, true);
    amrex::ParallelDescriptor::ReduceRealSum(rhs_sum);
    report.rhs_mean = rhs_sum /
        static_cast<amrex::Real>(global_cell_count(m_geom.Domain()));
    if (report.singular) report.rhs_mean = validate_singular_rhs_compatibility(m_projection_rhs);

    // Phi is an incremental pressure correction. Thus a fixed-pressure outflow
    // is homogeneous Dirichlet for Phi; walls/inflow are homogeneous Neumann.
    // Periodic/all-Neumann systems retain their constant null space, for which
    // compatibility is checked above and a zero-mean gauge is imposed below.
    m_phi.setVal(0.0);
    m_projection_bc.setVal(0.0);
    amrex::LPInfo info;
    amrex::MLPoisson poisson({m_geom}, {m_ba}, {m_dm}, info);
    poisson.setMaxOrder(2);
    poisson.setDomainBC(lobc, hibc);
    poisson.setEnforceSingularSolvable(false);
    poisson.setLevelBC(0, &m_projection_bc);

    amrex::MLMG mlmg(poisson);
    mlmg.setMaxIter(200);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);
    report.final_residual = mlmg.solve({&m_phi}, {&m_projection_rhs}, 1.0e-11, 0.0);

    if (report.singular) {
        amrex::Real phi_sum = m_phi.sum(0, true);
        amrex::ParallelDescriptor::ReduceRealSum(phi_sum);
        const amrex::Real phi_mean = phi_sum /
            static_cast<amrex::Real>(global_cell_count(m_geom.Domain()));
        m_phi.plus(-phi_mean, 0, 1, 0);
    }

    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> operator_flux{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        operator_flux[dir] = &m_phi_operator_flux[dir];
    }
    mlmg.getFluxes({operator_flux}, amrex::MLMG::Location::FaceCenter);

    const amrex::Real correction_scale = dt / time_coefficient;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_phi_operator_flux[dir].OverrideSync(m_geom.periodicity());
        m_phi_operator_flux[dir].FillBoundary(m_geom.periodicity());
        const amrex::Real integrated_scale = correction_scale * m_face_area[dir];
        for (amrex::MFIter mfi(m_ucont[dir], false); mfi.isValid(); ++mfi) {
            auto const flux = m_ucont[dir].array(mfi);
            auto const op_flux = m_phi_operator_flux[dir].const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                // MLMG getFluxes supplies the operator flux -grad(Phi). The
                // physical correction is -dt/alpha*grad(Phi), so add the
                // operator flux after converting it to integrated flux with A_d.
                flux(i,j,k) += integrated_scale * op_flux(i,j,k);
            });
        }
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const velocity = m_ucat.array(mfi);
            auto const op_flux = m_phi_operator_flux[dir].const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int ip = i + (dir == 0);
                const int jp = j + (dir == 1);
                const int kp = k + (dir == 2);
                velocity(i,j,k,dir) += correction_scale * 0.5
                    * (op_flux(i,j,k) + op_flux(ip,jp,kp));
            });
        }
    }

    amrex::MultiFab::Add(m_p, m_phi, 0, 0, 1, 0);
    mark_valid_modified();
    // Update Ucat by the pressure correction itself. Reconstructing all of
    // Ucat from adjacent faces here would apply an unphysical averaging filter
    // on every time step. Do not re-impose P3's pre-projection outlet flux: the
    // pressure-Dirichlet outlet is allowed to change it.
    fill_ghost_cells();
    if (m_boundary.enabled) fill_physical_ghost_cells_impl(false);

    compute_cartesian_divergence(m_projection_rhs);
    report.max_divergence_after = m_projection_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_after);
    amrex::Print() << "P4 Cartesian projection: singular=" << report.singular
                   << " rhs_mean=" << report.rhs_mean
                   << " residual=" << report.final_residual
                   << " max_div_before=" << report.max_divergence_before
                   << " max_div_after=" << report.max_divergence_after << "\n";
    return report;
}

ProjectionDiagnostics
AVWiSSolver::project_orthogonal(amrex::Real dt, amrex::Real time_coefficient)
{
    if (m_mapping_operator.pressure_contract == "incremental_pressure")
        throw std::runtime_error("M2 incremental_pressure requires advance_one_step; standalone projection cannot commit a pressure increment");
    if (!(dt > 0.0) || !(time_coefficient > 0.0)) {
        throw std::runtime_error("C2.2 orthogonal projection requires positive dt and time_coefficient");
    }
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        (m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG &&
        !(m_mapping_operator.mapping_type == "analytic_shear" && m_mapping_operator.shear_epsilon == 0.0))) {
        throw std::runtime_error(
            "C2.2 orthogonal projection requires mapped coordinates and orthogonal_mlmg");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir) && !m_boundary.enabled) {
            throw std::runtime_error(
                "C5.2 orthogonal projection requires explicit mapped physical boundaries");
        }
    }

    validate_projection_boundary_policy();
    if (m_boundary.enabled) apply_boundary_pipeline("pre-c5.2-orthogonal-projection");
    else fill_ghost_cells();
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> velocity_flux{
        AMREX_D_DECL(&m_ucont[0], &m_ucont[1], &m_ucont[2])};
    compute_metric_divergence(velocity_flux, m_metric_data, m_metric_epoch,
                              m_mapping_operator, m_projection_rhs);

    ProjectionDiagnostics report;
    bool has_pressure_dirichlet = false;
    report.max_divergence_before = m_projection_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_before);

    amrex::Real divergence_scale = 1.0;
    auto const logical_spacing = m_metric_data.logical_grid().spacing;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Real velocity_scale = m_ucat.norm0(dir, 0, true);
        amrex::Real inverse_map_scale =
            m_metric_data.grad_xi_cc().norm0(3*dir+dir, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(velocity_scale);
        amrex::ParallelDescriptor::ReduceRealMax(inverse_map_scale);
        divergence_scale += velocity_scale * inverse_map_scale / logical_spacing[dir];
    }
    const amrex::Real zero_divergence_tolerance = 4096.0
        * std::numeric_limits<amrex::Real>::epsilon() * divergence_scale;
    if (report.max_divergence_before <= zero_divergence_tolerance) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (m_geom.isPeriodic(dir)) continue;
            has_pressure_dirichlet = has_pressure_dirichlet ||
                m_boundary.sides[2*dir].velocity == CartesianBC::Outflow ||
                m_boundary.sides[2*dir+1].velocity == CartesianBC::Outflow;
        }
        m_phi.setVal(0.0);
        report.singular = !has_pressure_dirichlet;
        report.max_divergence_after = report.max_divergence_before;
        amrex::Print() << "C2.2 orthogonal projection: skipped roundoff-scale RHS; max_div="
                       << report.max_divergence_before << "\n";
        return report;
    }

    // MLABecLaplacian represents -div_xi(b grad_xi). With b_f=beta Q_mm h_m,
    // its cell action is the negative integrated physical operator. Therefore
    // the matching RHS is -alpha/dt times the net integrated predicted flux.
    auto const& volume = m_metric_data.cell_volume_cc();
    amrex::Real const rhs_scale = -time_coefficient / dt;
    for (amrex::MFIter mfi(m_projection_rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const rhs = m_projection_rhs.array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            rhs(i,j,k) *= rhs_scale * cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();

    amrex::Real rhs_sum = m_projection_rhs.sum(0, true);
    amrex::ParallelDescriptor::ReduceRealSum(rhs_sum);
    amrex::Real rhs_norm = m_projection_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(rhs_norm);
    amrex::Real volume_sum = volume.sum(0, true);
    amrex::ParallelDescriptor::ReduceRealSum(volume_sum);
    report.rhs_mean = rhs_sum / volume_sum;
    amrex::Real const compatibility_tolerance =
        4096.0 * std::numeric_limits<amrex::Real>::epsilon() *
        amrex::max(amrex::Real(1.0), rhs_norm) *
        static_cast<amrex::Real>(global_cell_count(m_geom.Domain()));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (m_geom.isPeriodic(dir)) continue;
        has_pressure_dirichlet = has_pressure_dirichlet ||
            m_boundary.sides[2 * dir].velocity == CartesianBC::Outflow ||
            m_boundary.sides[2 * dir + 1].velocity == CartesianBC::Outflow;
    }
    report.singular = !has_pressure_dirichlet;
    if (report.singular && std::abs(rhs_sum) > compatibility_tolerance) {
        throw std::runtime_error(
            "C2.2 singular orthogonal pressure RHS is volume-weighted incompatible");
    }

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> b_coefficient;
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> b_view{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const& q_metric = m_metric_data.face_gradient_metric_fc(dir);
        b_coefficient[dir].define(q_metric.boxArray(), m_dm, 1, 0);
        amrex::Real const h = m_metric_data.logical_grid().spacing[dir];
        for (amrex::MFIter mfi(b_coefficient[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const b = b_coefficient[dir].array(mfi);
            auto const q = q_metric.const_array(mfi);
            auto const beta = m_metric_data.projection_beta_fc(dir).const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                b(i,j,k) = beta(i,j,k) * q(i,j,k,dir) * h;
            });
        }
        b_coefficient[dir].OverrideSync(m_geom.periodicity());
        b_view[dir] = &b_coefficient[dir];
    }
    amrex::Gpu::streamSynchronize();

    m_phi.setVal(0.0);
    m_projection_bc.setVal(0.0);
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc{};
    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (m_geom.isPeriodic(dir)) {
            lobc[dir] = amrex::LinOpBCType::Periodic;
            hibc[dir] = amrex::LinOpBCType::Periodic;
        } else {
            lobc[dir] = m_boundary.sides[2 * dir].velocity == CartesianBC::Outflow
                ? amrex::LinOpBCType::Dirichlet : amrex::LinOpBCType::Neumann;
            hibc[dir] = m_boundary.sides[2 * dir + 1].velocity == CartesianBC::Outflow
                ? amrex::LinOpBCType::Dirichlet : amrex::LinOpBCType::Neumann;
        }
    }
    amrex::LPInfo info;
    amrex::MLABecLaplacian operator_({m_geom}, {m_ba}, {m_dm}, info);
    operator_.setMaxOrder(2);
    operator_.setDomainBC(lobc, hibc);
    operator_.setScalars(0.0, 1.0);
    operator_.setACoeffs(0, 0.0);
    operator_.setBCoeffs(0, b_view);
    operator_.setLevelBC(0, &m_projection_bc);

    amrex::MLMG mlmg(operator_);
    mlmg.setMaxIter(200);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);
    report.final_residual = mlmg.solve({&m_phi}, {&m_projection_rhs}, 1.0e-11, 0.0);

    if (report.singular) {
        amrex::MultiFab weighted_phi(m_ba, m_dm, 1, 0);
        amrex::MultiFab::Copy(weighted_phi, m_phi, 0, 0, 1, 0);
        amrex::MultiFab::Multiply(weighted_phi, volume, 0, 0, 1, 0);
        amrex::Real weighted_phi_sum = weighted_phi.sum(0, true);
        amrex::ParallelDescriptor::ReduceRealSum(weighted_phi_sum);
        m_phi.plus(-weighted_phi_sum / volume_sum, 0, 1, 0);
    }
    m_phi.FillBoundary(m_geom.periodicity());
    if (m_boundary.enabled) {
        // MLMG owns Phi valid cells; C5.2 supplies matching homogeneous
        // Neumann ghosts or odd homogeneous-Dirichlet outlet ghosts before
        // the shared explicit face correction.
        fill_physical_ghost_cells_impl(false);
    }

    amrex::Real const correction_scale = dt / time_coefficient;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        compute_orthogonal_face_gradient_flux(
            m_phi, dir, m_metric_data, m_metric_epoch, m_mapping_operator,
            m_phi_operator_flux[dir]);
        for (amrex::MFIter mfi(m_ucont[dir], false); mfi.isValid(); ++mfi) {
            auto const flux = m_ucont[dir].array(mfi);
            auto const pressure_flux = m_phi_operator_flux[dir].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                flux(i,j,k) -= correction_scale * pressure_flux(i,j,k);
            });
        }
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
    }

    amrex::MultiFab::Add(m_p, m_phi, 0, 0, 1, 0);
    amrex::MultiFab pressure_gradient(m_ba, m_dm, AMREX_SPACEDIM, 0);
    compute_metric_cell_gradient(
        m_phi, m_metric_data, m_metric_epoch, m_mapping_operator, pressure_gradient);
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const velocity = m_ucat.array(mfi);
        auto const gradient = pressure_gradient.const_array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                velocity(i,j,k,comp) -= correction_scale * gradient(i,j,k,comp);
            });
    }
    amrex::Gpu::streamSynchronize();
    mark_valid_modified();
    fill_ghost_cells();
    if (m_boundary.enabled) fill_physical_ghost_cells_impl(false);

    compute_metric_divergence(velocity_flux, m_metric_data, m_metric_epoch,
                              m_mapping_operator, m_projection_rhs);
    report.max_divergence_after = m_projection_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_after);
    amrex::Print() << "C2.2/C5.2 orthogonal projection: mapping="
                   << m_metric_data.mapping_id()
                   << " residual=" << report.final_residual
                   << " max_div_before=" << report.max_divergence_before
                   << " max_div_after=" << report.max_divergence_after << "\n";
    return report;
}

DeferredProjectionResult AVWiSSolver::solve_incremental_pressure(
    amrex::MultiFab& pressure, amrex::MultiFab const& integrated_rhs)
{
    if (m_mapping_operator.mapping_type == "analytic_shear")
        return solve_deferred_metric_poisson(pressure, integrated_rhs, m_metric_data,
            m_metric_epoch, m_mapping_operator, m_geom, *m_deferred_workspace);

    // Both diagonal operators solve A Q = rhs with A=-N G_f. The
    // Cartesian MLPoisson convention is +Laplacian, so its RHS is -rhs/V;
    // mapped MLABecLaplacian already represents the integrated negative action.
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    auto& work = *m_deferred_workspace;
    auto const& volume = m_metric_data.cell_volume_cc();
    auto rhs_sum = integrated_rhs.sum(0, true);
    auto rhs_norm = integrated_rhs.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealSum(rhs_sum);
    amrex::ParallelDescriptor::ReduceRealMax(rhs_norm);
    auto const cells = static_cast<amrex::Real>(m_geom.Domain().numPts());
    if (!std::isfinite(rhs_sum) || !std::isfinite(rhs_norm) ||
        std::abs(rhs_sum) > 1.0e-12*cells*rhs_norm)
        throw std::runtime_error("M2 incremental diagonal pressure incompatible integrated RHS; no mean repair");

    amrex::MultiFab::Copy(work.physical_rhs, integrated_rhs, 0, 0, 1, 0);
    amrex::MultiFab::Divide(work.physical_rhs, volume, 0, 0, 1, 0);
    pressure.setVal(0);
    work.bc.setVal(0);
    DeferredProjectionResult result;
    if (rhs_norm == 0) return result;

    bool const cartesian = m_mapping_operator.coordinates == CoordinateSystemMode::Cartesian;
    amrex::Array<amrex::LinOpBCType,3> periodic{
        amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic, amrex::LinOpBCType::Periodic};
    std::unique_ptr<amrex::MLCellLinOp> op;
    if (cartesian) {
        amrex::MultiFab::Copy(work.work_rhs, work.physical_rhs, 0, 0, 1, 0);
        work.work_rhs.mult(-1, 0, 1, 0);
        op = std::make_unique<amrex::MLPoisson>(
            amrex::Vector<amrex::Geometry>{m_geom}, amrex::Vector<amrex::BoxArray>{m_ba},
            amrex::Vector<amrex::DistributionMapping>{m_dm});
    } else {
        amrex::MultiFab::Copy(work.work_rhs, integrated_rhs, 0, 0, 1, 0);
        amrex::Array<amrex::MultiFab const*,3> coefficients{};
        auto const h = m_metric_data.logical_grid().spacing;
        for (int d=0; d<3; ++d) {
            amrex::MultiFab::Copy(work.b[d], m_metric_data.face_gradient_metric_fc(d), d, 0, 1, 0);
            amrex::MultiFab::Multiply(work.b[d], m_metric_data.projection_beta_fc(d), 0, 0, 1, 0);
            work.b[d].mult(h[d], 0, 1, 0);
            work.b[d].OverrideSync(m_geom.periodicity());
            coefficients[d] = &work.b[d];
        }
        auto diagonal = std::make_unique<amrex::MLABecLaplacian>(
            amrex::Vector<amrex::Geometry>{m_geom}, amrex::Vector<amrex::BoxArray>{m_ba},
            amrex::Vector<amrex::DistributionMapping>{m_dm});
        diagonal->setScalars(0, 1);
        diagonal->setACoeffs(0, 0);
        diagonal->setBCoeffs(0, coefficients);
        op = std::move(diagonal);
    }
    op->setMaxOrder(2);
    op->setDomainBC(periodic, periodic);
    op->setEnforceSingularSolvable(false);
    op->setLevelBC(0, &work.bc);
    amrex::MLMG mlmg(*op);
    mlmg.setMaxIter(m_mapping_operator.diagonal_max_iterations);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.setBottomVerbose(0);
    mlmg.setThrowException(true);
    // A singular bottom Krylov solve can amplify roundoff in later steps.
    // Use the same robust smoothing policy as the shear solve. Thin diagonal
    // hierarchies also need more than 200 cycles; the configurable work budget
    // changes neither the 1e-12 inner nor the 1e-11 full acceptance tolerance.
    mlmg.setBottomSolver(amrex::MLMG::BottomSolver::smoother);
    mlmg.setBottomSmooth(256);
    try {
        mlmg.solve({&pressure}, {&work.work_rhs}, 1.0e-12, 0);
    } catch (std::exception const& error) {
        throw std::runtime_error(std::string("M2 incremental diagonal solve failed: ") + error.what());
    }
    result.inner_iterations = mlmg.getNumIters();
    result.iterations = 1;

    amrex::MultiFab::Copy(work.weighted, pressure, 0, 0, 1, 0);
    amrex::MultiFab::Multiply(work.weighted, volume, 0, 0, 1, 0);
    auto weighted_sum = work.weighted.sum(0, true);
    auto volume_sum = volume.sum(0, true);
    amrex::ParallelDescriptor::ReduceRealSum(weighted_sum);
    amrex::ParallelDescriptor::ReduceRealSum(volume_sum);
    pressure.plus(-weighted_sum/volume_sum, 0, 1, 0);
    pressure.FillBoundary(m_geom.periodicity());
    // Candidate acceptance remains in advance_incremental_pressure: recheck
    // the actual rounded P_old+Phi with the SAME explicit G_f and physical RHS
    // used for face correction. MLMG's internal residual is not that gate.
    return result;
}

ProjectionDiagnostics AVWiSSolver::project_mapped(amrex::Real dt, amrex::Real alpha)
{
    if (m_mapping_operator.pressure_contract == "incremental_pressure")
        throw std::runtime_error("M2 incremental_pressure requires advance_one_step; standalone projection cannot commit a pressure increment");
    if (m_mapping_operator.projection == ProjectionOperatorMode::OrthogonalMLMG ||
        (m_mapping_operator.mapping_type=="analytic_shear" && m_mapping_operator.shear_epsilon==0.0))
        return project_orthogonal(dt,alpha);
    validate_mapping_operator_config(m_mapping_operator,m_metric_data,m_metric_epoch);
    if (!std::isfinite(dt) || !std::isfinite(alpha) || dt<=0.0 || alpha<=0.0)
        throw std::runtime_error("AVWiS mapped projection requires finite positive dt/alpha");
    if (!m_deferred_workspace)
        throw std::runtime_error("AVWiS deferred projection requires solver-owned workspace");
    fill_ghost_cells();
    amrex::Array<amrex::MultiFab const*,3> fv{&m_ucont[0],&m_ucont[1],&m_ucont[2]};
    compute_metric_divergence(fv,m_metric_data,m_metric_epoch,m_mapping_operator,m_projection_rhs);
    ProjectionDiagnostics report;
    report.singular=true;
    report.max_divergence_before=m_projection_rhs.norm0(0,0,true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_before);
    amrex::Real divergence_scale=1.0;
    auto const h=m_metric_data.logical_grid().spacing;
    bool const reuse_velocity_norms = m_mapping_operator.state_contract == "face_authoritative";
    amrex::Array<amrex::Real,3> velocity_norms{};
    if (reuse_velocity_norms) {
        // Ucat is unchanged throughout this scale calculation. Its three
        // component maxima are independent of logical direction: compute
        // each once, saving six local scans and six global max reductions.
        // Keep this cache local to the call, including retries after failure.
        for (int c=0;c<3;++c) {
            velocity_norms[c]=m_ucat.norm0(c,0,true);
            amrex::ParallelDescriptor::ReduceRealMax(velocity_norms[c]);
        }
    }
    for (int d=0;d<3;++d) for (int c=0;c<3;++c) {
        amrex::Real u=reuse_velocity_norms ? velocity_norms[c] : m_ucat.norm0(c,0,true);
        amrex::Real g=m_metric_data.grad_xi_cc().norm0(3*d+c,0,true);
        if (!reuse_velocity_norms) amrex::ParallelDescriptor::ReduceRealMax(u);
        amrex::ParallelDescriptor::ReduceRealMax(g);
        divergence_scale+=u*g/h[d];
    }
    if (report.max_divergence_before <= 64.0*std::numeric_limits<amrex::Real>::epsilon()*divergence_scale) {
        if (m_mapping_operator.state_contract == "face_authoritative") {
            // A zero multiplier replaces the previous instantaneous value.
            // Reconstruction is prepared before publishing pressure/cell state.
            auto& reconstructed=m_deferred_workspace->reconstructed_cell;
            sync_metric_ucat_from_ucont(fv,m_metric_data,m_metric_epoch,m_mapping_operator,reconstructed);
            if (!reconstructed.is_finite(0,3,0))
                throw std::runtime_error("M2 face-authoritative zero-multiplier reconstruction is nonfinite");
            amrex::MultiFab::Copy(m_ucat,reconstructed,0,0,3,0);
            m_p.setVal(0.0);
            mark_valid_modified();
        }
        m_phi.setVal(0.0);
        if (m_mapping_operator.state_contract == "face_authoritative") fill_ghost_cells();
        report.max_divergence_after=report.max_divergence_before;
        return report;
    }
    if (m_mapping_operator.state_contract == "face_authoritative") {
        // MLMG consumes -alpha/dt times the authoritative net face flux.
        // Form it directly, without rounding net/V and multiplying V back.
        compute_metric_net_flux(fv,m_metric_data,m_metric_epoch,m_mapping_operator,m_projection_rhs);
    } else {
        amrex::MultiFab::Multiply(m_projection_rhs,m_metric_data.cell_volume_cc(),0,0,1,0);
    }
    m_projection_rhs.mult(-alpha/dt,0,1,0);
    auto const result=solve_deferred_metric_poisson(m_phi,m_projection_rhs,m_metric_data,
        m_metric_epoch,m_mapping_operator,m_geom,*m_deferred_workspace);
    report.outer_iterations=result.iterations;
    report.inner_iterations=result.inner_iterations;
    report.full_relative_residual=result.full_relative_residual;
    report.final_residual=result.full_max_residual;
    if (m_mapping_operator.state_contract == "face_authoritative") {
        // U* = U^n + dt I(RHS); DG Phi = D(U*)/dt. Phi is the
        // instantaneous discrete face-constraint multiplier for this step.
        // P=Phi is a semantic choice, not evidence of physical pressure MMS.
        // Prepare projected faces and reconstructed cells before flow commit.
        amrex::Array<amrex::MultiFab const*,3> candidates{};
        for (int d=0;d<3;++d) {
            compute_metric_face_gradient_flux(m_phi,d,m_metric_data,m_metric_epoch,
                m_mapping_operator,m_phi_operator_flux[d]);
            auto& candidate=m_deferred_workspace->face_candidate[d];
            amrex::MultiFab::Copy(candidate,m_ucont[d],0,0,1,0);
            amrex::MultiFab::Saxpy(candidate,-dt/alpha,m_phi_operator_flux[d],0,0,1,0);
            candidate.OverrideSync(m_geom.periodicity());
            candidate.FillBoundary(m_geom.periodicity());
            candidates[d]=&candidate;
        }
        auto& reconstructed=m_deferred_workspace->reconstructed_cell;
        sync_metric_ucat_from_ucont(candidates,m_metric_data,m_metric_epoch,
            m_mapping_operator,reconstructed);
        if (!reconstructed.is_finite(0,3,0) || !m_phi.is_finite(0,1,0))
            throw std::runtime_error("M2 face-authoritative projection candidate is nonfinite");
        compute_metric_divergence(candidates,m_metric_data,m_metric_epoch,
            m_mapping_operator,m_projection_rhs);
        report.max_divergence_after=m_projection_rhs.norm0(0,0,true);
        amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_after);
        if (!std::isfinite(report.max_divergence_after))
            throw std::runtime_error("M2 face-authoritative projection candidate divergence is nonfinite");
        // Runtime/MPI/device faults during copies are fatal. There is no
        // whole-advance rollback: predictor faces/history already changed.
        for (int d=0;d<3;++d)
            amrex::MultiFab::Copy(m_ucont[d],*candidates[d],0,0,1,0);
        amrex::MultiFab::Copy(m_ucat,reconstructed,0,0,3,0);
        amrex::MultiFab::Copy(m_p,m_phi,0,0,1,0);
        mark_valid_modified();fill_ghost_cells();
        amrex::Print()<<"M2 face-authoritative projection: pressure=instantaneous_pressure iterations="
            <<report.outer_iterations<<" full_relative_residual="<<report.full_relative_residual
            <<" max_div_after="<<report.max_divergence_after<<"\n";
        return report;
    }
    // Prepare every correction before writing flow state. Ucont is the mass
    // authority; Ucat is the transported cell velocity with incremental cell
    // gradient correction, NOT an inverse interpolation of projected Ucont.
    // Solver/validation failures preserve Ucat/Ucont/P valid values and the
    // valid epoch. Phi/halos/scratch are not rolled back; an entire advance is
    // still fatal on failure (history/predictor rollback is not implemented).
    auto& gradient=m_deferred_workspace->cell_gradient;
    compute_metric_cell_gradient(m_phi,m_metric_data,m_metric_epoch,m_mapping_operator,gradient);
    for (int d=0;d<3;++d)
        compute_metric_face_gradient_flux(m_phi,d,m_metric_data,m_metric_epoch,
            m_mapping_operator,m_phi_operator_flux[d]);
    // No application-owned MultiFab allocation or operator validation remains.
    // Fatal runtime/MPI/device errors (including communication allocation)
    // are outside the recoverable numerical-failure contract.
    for (int d=0;d<3;++d) {
        amrex::MultiFab::Saxpy(m_ucont[d],-dt/alpha,m_phi_operator_flux[d],0,0,1,0);
        m_ucont[d].OverrideSync(m_geom.periodicity());
        m_ucont[d].FillBoundary(m_geom.periodicity());
    }
    amrex::MultiFab::Add(m_p,m_phi,0,0,1,0);
    amrex::MultiFab::Saxpy(m_ucat,-dt/alpha,gradient,0,0,3,0);
    mark_valid_modified();fill_ghost_cells();
    compute_metric_divergence(fv,m_metric_data,m_metric_epoch,m_mapping_operator,m_projection_rhs);
    report.max_divergence_after=m_projection_rhs.norm0(0,0,true);
    amrex::ParallelDescriptor::ReduceRealMax(report.max_divergence_after);
    amrex::Print()<<"C3 deferred projection: iterations="<<report.outer_iterations
        <<" full_relative_residual="<<report.full_relative_residual
        <<" max_div_after="<<report.max_divergence_after<<"\n";
    return report;
}
