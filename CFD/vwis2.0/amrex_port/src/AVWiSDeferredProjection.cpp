#include "AVWiSMappedOperators.H"

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <sstream>
#include <stdexcept>

void compute_metric_face_gradient_flux(amrex::MultiFab const& scalar, int direction,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::MultiFab& face_flux, MetricFluxPart part)
{
    validate_mapping_operator_config(config,metric,epoch);
    if (direction<0 || direction>=3) throw std::out_of_range("AVWiS metric face direction");
    auto const& qfield = metric.face_gradient_metric_fc(direction);
    auto const& volume = metric.cell_volume_cc();
    if (scalar.nComp()!=1 || scalar.nGrow()<1 || scalar.boxArray()!=volume.boxArray() ||
        scalar.DistributionMap()!=volume.DistributionMap() || face_flux.nComp()!=1 ||
        face_flux.boxArray()!=qfield.boxArray() || face_flux.DistributionMap()!=qfield.DistributionMap())
        throw std::runtime_error("AVWiS full gradient requires matching scalar/face layouts and corner halos");
    auto const h=metric.logical_grid().spacing;
    amrex::GpuArray<amrex::Real,3> ih{1.0/h[0],1.0/h[1],1.0/h[2]};
    // Preserve the established orthogonal stencil, including tiny geometric
    // off-diagonal roundoff. A zero shear uses the exact diagonal specialization.
    if (part==MetricFluxPart::Full && config.shear_epsilon==0.0)
        part=MetricFluxPart::Diagonal;
    for (amrex::MFIter mfi(face_flux,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const p=scalar.const_array(mfi);
        auto const q=qfield.const_array(mfi);
        auto const beta=metric.projection_beta_fc(direction).const_array(mfi);
        auto const f=face_flux.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            f(i,j,k)=beta(i,j,k)*metric_face_gradient_value(p,q,i,j,k,direction,0,ih,part);
        });
    }
    amrex::Gpu::streamSynchronize();
    amrex::Array<int,3> periods{};
    for (int d=0;d<3;++d) periods[d]=metric.logical_grid().periodic[d]
        ? metric.logical_grid().cell_domain.length(d):0;
    face_flux.OverrideSync(amrex::Periodicity(amrex::IntVect(periods)));
}

DeferredProjectionWorkspace::DeferredProjectionWorkspace(MetricData const& metric)
    : metric_owner(&metric), metric_epoch(metric.epoch())
{
    auto const& v=metric.cell_volume_cc();
    auto const& ba=v.boxArray();
    auto const& dm=v.DistributionMap();
    for (int d=0;d<3;++d) {
        auto const& faces=metric.face_gradient_metric_fc(d).boxArray();
        b[d].define(faces,dm,1,0);
        flux[d].define(faces,dm,1,0);
        face_candidate[d].define(faces,dm,1,1);
    }
    bc.define(ba,dm,1,1);
    work_rhs.define(ba,dm,1,0);action.define(ba,dm,1,0);
    residual.define(ba,dm,1,0);physical_rhs.define(ba,dm,1,0);
    weighted.define(ba,dm,1,0);cell_gradient.define(ba,dm,3,0);
    predictor_rate.define(ba,dm,3,1);reconstructed_cell.define(ba,dm,3,0);
}

DeferredProjectionResult solve_deferred_metric_poisson(
    amrex::MultiFab& phi, amrex::MultiFab const& integrated_rhs,
    MetricData const& metric, std::uint64_t epoch,
    MappingOperatorConfig const& config, amrex::Geometry const& geometry)
{
    DeferredProjectionWorkspace workspace(metric);
    return solve_deferred_metric_poisson(phi,integrated_rhs,metric,epoch,config,geometry,workspace);
}

DeferredProjectionResult solve_deferred_metric_poisson(
    amrex::MultiFab& phi, amrex::MultiFab const& integrated_rhs,
    MetricData const& metric, std::uint64_t epoch,
    MappingOperatorConfig const& config, amrex::Geometry const& geometry,
    DeferredProjectionWorkspace& workspace, DeferredProjectionTrace* trace)
{
    if (trace) trace->clear();
    validate_mapping_operator_config(config,metric,epoch);
    if (workspace.metric_owner!=&metric || workspace.metric_epoch!=epoch ||
        workspace.work_rhs.boxArray()!=metric.cell_volume_cc().boxArray() ||
        workspace.work_rhs.DistributionMap()!=metric.cell_volume_cc().DistributionMap())
        throw std::runtime_error("AVWiS deferred workspace metric owner/epoch/layout mismatch");
    if (config.projection!=ProjectionOperatorMode::DeferredNonorthogonal)
        throw std::runtime_error("AVWiS deferred solve requires deferred_nonorthogonal");
    auto const& volume=metric.cell_volume_cc();
    auto const& ba=volume.boxArray();
    auto const& dm=volume.DistributionMap();
    if (phi.boxArray()!=ba || integrated_rhs.boxArray()!=ba || phi.DistributionMap()!=dm ||
        integrated_rhs.DistributionMap()!=dm || phi.nComp()!=1 || integrated_rhs.nComp()!=1 || phi.nGrow()<1 ||
        geometry.Domain()!=metric.logical_grid().cell_domain)
        throw std::runtime_error("AVWiS deferred pressure layouts/domain differ");
    for (int d=0;d<3;++d)
        if (!geometry.isPeriodic(d)) throw std::runtime_error("AVWiS deferred pressure requires periodic geometry");

    amrex::Real rhs_sum=integrated_rhs.sum(0,true);
    amrex::Real rhs_norm=integrated_rhs.norm0(0,0,true);
    amrex::ParallelDescriptor::ReduceRealSum(rhs_sum);
    amrex::ParallelDescriptor::ReduceRealMax(rhs_norm);
    amrex::Real const cells=static_cast<amrex::Real>(geometry.Domain().numPts());
    if (!std::isfinite(rhs_norm) || !std::isfinite(rhs_sum) ||
        std::abs(rhs_sum)>1.0e-12*cells*rhs_norm)
        throw std::runtime_error("AVWiS deferred pressure incompatible integrated RHS; no mean repair");

    auto& b=workspace.b;
    auto& flux=workspace.flux;
    amrex::Array<amrex::MultiFab const*,3> bview{},fview{};
    auto const h=metric.logical_grid().spacing;
    for (int d=0;d<3;++d) {
        auto const& qfield=metric.face_gradient_metric_fc(d);
        amrex::Real const hd=h[d];
        for (amrex::MFIter mfi(b[d],amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto const out=b[d].array(mfi);
            auto const q=qfield.const_array(mfi);
            auto const beta=metric.projection_beta_fc(d).const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                out(i,j,k)=beta(i,j,k)*q(i,j,k,d)*hd;
            });
        }
        b[d].OverrideSync(geometry.periodicity());
        bview[d]=&b[d]; fview[d]=&flux[d];
    }
    auto& bc=workspace.bc;
    auto& work_rhs=workspace.work_rhs;
    auto& action=workspace.action;
    auto& residual=workspace.residual;
    auto& physical_rhs=workspace.physical_rhs;
    auto& weighted=workspace.weighted;
    // A failed prior attempt leaves scratch dirty. Never use it as an initial
    // guess or correction: coefficients, RHS, Phi and all actions are rebuilt.
    bc.setVal(0.0);phi.setVal(0.0);
    amrex::Array<amrex::LinOpBCType,3> periodic{amrex::LinOpBCType::Periodic,
        amrex::LinOpBCType::Periodic,amrex::LinOpBCType::Periodic};
    amrex::MLABecLaplacian op({geometry},{ba},{dm});
    op.setMaxOrder(2);op.setDomainBC(periodic,periodic);
    op.setScalars(0.0,1.0);op.setACoeffs(0,0.0);op.setBCoeffs(0,bview);op.setLevelBC(0,&bc);
    amrex::MLMG mlmg(op);
    mlmg.setMaxIter(1600);mlmg.setMaxFmgIter(0);mlmg.setVerbose(0);mlmg.setBottomVerbose(0);
    op.setEnforceSingularSolvable(false);
    // A shallow hierarchy (e.g. 64x64x4) needs substantial bottom smoothing.
    // Smoothing avoids Krylov null-mode amplification near roundoff while
    // keeping automatic RHS mean repair disabled. Mixed-mode 64x64x4 later
    // steps were still decaying at 800 iterations (400:8.7e-8, 600:9.8e-11,
    // 800:1.2e-12, absolute residual); allow more work without relaxing either
    // inner tolerance or the full residual gate. Cost remains an M4 gate.
    mlmg.setBottomSolver(amrex::MLMG::BottomSolver::smoother);
    mlmg.setBottomSmooth(256);
    mlmg.setThrowException(true);
    amrex::MultiFab::Copy(physical_rhs,integrated_rhs,0,0,1,0);
    amrex::MultiFab::Divide(physical_rhs,volume,0,0,1,0);
    amrex::Real physical_scale=physical_rhs.norm0(0,0,true);
    amrex::ParallelDescriptor::ReduceRealMax(physical_scale);
    amrex::Real volume_sum=volume.sum(0,true);
    amrex::ParallelDescriptor::ReduceRealSum(volume_sum);
    DeferredProjectionResult result;
    if (physical_scale==0.0) return result;
    for (int iteration=1;iteration<=config.deferred_max_iterations;++iteration) {
        phi.FillBoundary(geometry.periodicity());
        for (int d=0;d<3;++d) compute_metric_face_gradient_flux(phi,d,metric,epoch,config,flux[d],MetricFluxPart::Cross);
        // A_diag=-V*L_diag; A_diag phi = integrated_rhs + V*L_cross(phi_old).
        if (config.state_contract == "face_authoritative") {
            compute_metric_net_flux(fview,metric,epoch,config,action);
        } else {
            compute_metric_divergence(fview,metric,epoch,config,action);
            amrex::MultiFab::Multiply(action,volume,0,0,1,0);
        }
        amrex::MultiFab::Copy(work_rhs,integrated_rhs,0,0,1,0);
        amrex::MultiFab::Add(work_rhs,action,0,0,1,0);
        // The inner solve is a preconditioner step, not the acceptance gate.
        // One decade tighter than the full residual avoids demanding accuracy
        // below the cancellation floor for Phi~1/dt in a coupled predictor.
        try {
            mlmg.solve({&phi},{&work_rhs},1.0e-12,0.0);
        } catch (std::exception const& error) {
            std::ostringstream failure;
            failure << "AVWiS deferred inner solve failed: mapping=" << metric.mapping_id()
                << " epoch=" << epoch << " epsilon=" << config.shear_epsilon
                << " outer=" << iteration << " inner=" << mlmg.getNumIters()
                << " initial_residual=" << mlmg.getInitResidual()
                << " final_residual=" << mlmg.getFinalResidual()
                << " cause=" << error.what();
            auto const& history=mlmg.getResidualHistory();
            if (history.size()>=600)
                failure << " residual_400=" << history[399] << " residual_600=" << history[599];
            throw std::runtime_error(failure.str());
        }
        result.inner_iterations += mlmg.getNumIters();
        amrex::MultiFab::Copy(weighted,phi,0,0,1,0);
        amrex::MultiFab::Multiply(weighted,volume,0,0,1,0);
        amrex::Real mean=weighted.sum(0,true);
        amrex::ParallelDescriptor::ReduceRealSum(mean);
        phi.plus(-mean/volume_sum,0,1,0);
        phi.FillBoundary(geometry.periodicity());
        for (int d=0;d<3;++d) compute_metric_face_gradient_flux(phi,d,metric,epoch,config,flux[d]);
        compute_metric_divergence(fview,metric,epoch,config,residual);
        amrex::MultiFab::Add(residual,physical_rhs,0,0,1,0);
        result.iterations=iteration;
        result.full_max_residual=residual.norm0(0,0,true);
        amrex::ParallelDescriptor::ReduceRealMax(result.full_max_residual);
        result.full_relative_residual=result.full_max_residual/physical_scale;
        if (trace) trace->push_back({iteration,mlmg.getNumIters(),
            mlmg.getInitResidual(),mlmg.getFinalResidual(),mean/volume_sum,
            result.full_max_residual,result.full_relative_residual});
        if (std::isfinite(result.full_relative_residual) && result.full_relative_residual<=1.0e-11)
            return result;
    }
    std::ostringstream failure;
    failure << "AVWiS deferred pressure exhausted max_iterations=" << config.deferred_max_iterations
            << " mapping=" << metric.mapping_id() << " epoch=" << epoch
            << " epsilon=" << config.shear_epsilon
            << " full_relative_residual=" << result.full_relative_residual
            << " full_max_residual=" << result.full_max_residual;
    throw std::runtime_error(failure.str());
}
