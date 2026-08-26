#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

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

void VwisAmrExSolver::validate_projection_boundary_policy() const
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

void VwisAmrExSolver::compute_cartesian_divergence(amrex::MultiFab& divergence) const
{
    if (divergence.boxArray() != m_ba || divergence.nComp() != 1) {
        throw std::runtime_error("P4 divergence destination must be one cell-centred component");
    }
    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const div = divergence.array(mfi);
        auto const fx = m_ucont[0].const_array(mfi);
        auto const fy = m_ucont[1].const_array(mfi);
        auto const fz = m_ucont[2].const_array(mfi);
        const amrex::Real inverse_volume = 1.0 / m_cell_volume;
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Ucont is integrated normal volume flux. The only metric scaling
            // here is net face flux divided by Cartesian cell volume.
            div(i,j,k) = ((fx(i+1,j,k)-fx(i,j,k)) +
                          (fy(i,j+1,k)-fy(i,j,k)) +
                          (fz(i,j,k+1)-fz(i,j,k))) * inverse_volume;
        });
    }
    amrex::Gpu::streamSynchronize();
}

amrex::Real
VwisAmrExSolver::validate_singular_rhs_compatibility(amrex::MultiFab const& rhs) const
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
VwisAmrExSolver::project_cartesian(amrex::Real dt, amrex::Real time_coefficient)
{
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

void VwisAmrExSolver::run_p4_projection_contract_checks(
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
    amrex::Print() << "VWiS AMReX P4 Cartesian projection contract: PASS "
                   << "(integrated-flux RHS/BC+datum/MLMG/correction/Ucat sync)\n";
}
