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
        const amrex::Real integrated_scale = correction_scale * m_face_area[dir];
        for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const flux = m_ucont[dir].array(mfi);
            auto const op_flux = m_phi_operator_flux[dir].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                // MLMG getFluxes supplies the operator flux -grad(Phi). The
                // physical correction is -dt/alpha*grad(Phi), so add the
                // operator flux after converting it to integrated flux with A_d.
                flux(i,j,k) += integrated_scale * op_flux(i,j,k);
            });
        }
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
    }

    amrex::MultiFab::Add(m_p, m_phi, 0, 0, 1, 0);
    mark_valid_modified();
    // Do not re-impose P3's pre-projection outlet-flux constraint here. The
    // pressure-Dirichlet outlet is the one boundary whose normal flux the
    // projection is allowed to correct.
    sync_ucat_from_ucont_impl(false);

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
    if (!(dt > 0.0) || !(time_coefficient > 0.0)) {
        throw std::runtime_error("C2.2 orthogonal projection requires positive dt and time_coefficient");
    }
    validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_mapping_operator.projection != ProjectionOperatorMode::OrthogonalMLMG) {
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

    // MLABecLaplacian represents -div_xi(b grad_xi). With b_f=beta Q_mm h_m,
    // its cell action is the negative integrated physical operator. Therefore
    // the matching RHS is -alpha/dt times the net integrated predicted flux.
    auto const& volume = m_metric_data.cell_volume_cc();
    amrex::Real const rhs_scale = -time_coefficient / dt;
    for (amrex::MFIter mfi(m_projection_rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const rhs = m_projection_rhs.array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
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
        for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
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
    mark_valid_modified();
    sync_ucat_from_ucont_impl(false);

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
