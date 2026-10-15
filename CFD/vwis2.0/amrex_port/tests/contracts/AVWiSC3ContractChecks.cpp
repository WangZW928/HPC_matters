#include "AVWiSContractTestAccess.H"
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
amrex::Real global_max(amrex::MultiFab const& f,int c=0) {
    amrex::Real r=f.norm0(c,0,true);amrex::ParallelDescriptor::ReduceRealMax(r);return r;
}
amrex::Real global_sum(amrex::MultiFab const& f,int c=0) {
    amrex::Real r=f.sum(c,true);amrex::ParallelDescriptor::ReduceRealSum(r);return r;
}
void require(bool condition,char const* message) {
    if (!condition) throw std::runtime_error(std::string("C3 contract: ")+message);
}
}

void AVWiSContractTestAccess::run_c3_shear_contract(
    std::string const& path,amrex::Real dt,amrex::Real nu)
{
    require(m_mapping_operator.mapping_type=="analytic_shear" && nu>0.0,"requires shear and positive nu");
    auto const& metric=m_metric_data;
    auto const epoch=m_metric_epoch;
    auto const config=m_mapping_operator;
    auto const h=metric.logical_grid().spacing;
    auto const scale=config.analytic_parameters.scale;
    auto const stretch=config.analytic_parameters.stretch;
    amrex::GpuArray<amrex::Real,3> sc{scale[0],scale[1],scale[2]};
    amrex::GpuArray<amrex::Real,3> st{stretch[0],stretch[1],stretch[2]};
    amrex::Real const eps=config.shear_epsilon;
    constexpr amrex::Real w=6.283185307179586476925286766559;
    constexpr amrex::Real speed=0.35;
    amrex::MultiFab scalar(m_ba,m_dm,1,2),lap(m_ba,m_dm,1,0),cross(m_ba,m_dm,1,0);
    amrex::MultiFab work(m_ba,m_dm,8,0),adv(m_ba,m_dm,3,0),visc(m_ba,m_dm,3,0);
    amrex::MultiFab gradient(m_ba,m_dm,3,0);
    amrex::Array<amrex::MultiFab,3> flux;
    amrex::Array<amrex::MultiFab const*,3> fv{};
    for (int d=0;d<3;++d) {
        flux[d].define(m_ucont[d].boxArray(),m_dm,1,0);fv[d]=&flux[d];
    }
    // Constant-state GCL, transform and triangular normalized six-face LS.
    m_ucat.setVal(0.0);
    m_ucat.setVal(0.3,0,1);m_ucat.setVal(-0.2,1,1);m_ucat.setVal(0.1,2,1);
    mark_valid_modified();sync_ucont_from_ucat();sync_ucat_from_ucont();
    for (int c=0;c<3;++c) {
        amrex::MultiFab::Copy(work,m_ucat,c,0,1,0);
        work.plus(c==0?-0.3:(c==1?0.2:-0.1),0,1,0);
        amrex::Real const reconstruction_error=global_max(work);
        if (!(reconstruction_error<1e-12))
            throw std::runtime_error("C3 constant velocity reconstruction component="+
                std::to_string(c)+" error="+std::to_string(reconstruction_error));
    }
    amrex::Array<amrex::MultiFab const*,3> velocity{&m_ucont[0],&m_ucont[1],&m_ucont[2]};
    compute_metric_divergence(velocity,metric,epoch,config,lap);
    require(global_max(lap)<1e-12,"constant GCL divergence");
    // MMS in logical coordinates, including both mixed derivatives. Fields
    // are periodic under the shear lattice translations, not Cartesian axes.
    for (amrex::MFIter mfi(scalar,false);mfi.isValid();++mfi) {
        auto const p=scalar.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            p(i,j,k)=std::sin(w*((i+0.5)*h[0]+(j+0.5)*h[1]));
        });
    }
    scalar.FillBoundary(m_geom.periodicity());
    compute_metric_cell_gradient(scalar,metric,epoch,config,gradient);
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const u=m_ucat.array(mfi);auto const p=scalar.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            u(i,j,k,0)=speed;u(i,j,k,1)=0.0;u(i,j,k,2)=p(i,j,k);
        });
    }
    mark_valid_modified();sync_ucont_from_ucat();
    compute_advection_rhs(adv);compute_viscous_rhs(visc,nu);
    for (int d=0;d<3;++d) compute_metric_face_gradient_flux(scalar,d,metric,epoch,config,flux[d]);
    compute_metric_divergence(fv,metric,epoch,config,lap);
    for (int d=0;d<3;++d) compute_metric_face_gradient_flux(scalar,d,metric,epoch,config,flux[d],MetricFluxPart::Cross);
    compute_metric_divergence(fv,metric,epoch,config,cross);
    for (amrex::MFIter mfi(work,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const e=work.array(mfi);auto const p=scalar.const_array(mfi);
        auto const l=lap.const_array(mfi);auto const x=cross.const_array(mfi);
        auto const a=adv.const_array(mfi);auto const v=visc.const_array(mfi);
        auto const grad=gradient.const_array(mfi);
        auto const vol=metric.cell_volume_cc().const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            amrex::Real xi=(i+0.5)*h[0],eta=(j+0.5)*h[1];
            amrex::Real ax=sc[0]*(1+st[0]*std::cos(w*xi));
            amrex::Real by=sc[1]*(1+st[1]*std::cos(w*eta));
            amrex::Real axp=-sc[0]*st[0]*w*std::sin(w*xi);
            amrex::Real byp=-sc[1]*st[1]*w*std::sin(w*eta);
            amrex::Real s=p(i,j,k),c=std::cos(w*(xi+eta));
            amrex::Real exact=(1+eps*eps)*(-w*w*s/(ax*ax)-w*c*axp/(ax*ax*ax))
                -w*w*s/(by*by)-w*c*byp/(by*by*by)+2*eps*w*w*s/(ax*by);
            amrex::Real de=l(i,j,k)-exact;
            amrex::Real ae=a(i,j,k,2)+speed*w*c/ax;
            amrex::Real gx=w*c/ax,gy=w*c/by-eps*gx;
            e(i,j,k,6)=((grad(i,j,k,0)-gx)*(grad(i,j,k,0)-gx)+
                (grad(i,j,k,1)-gy)*(grad(i,j,k,1)-gy)+grad(i,j,k,2)*grad(i,j,k,2))*vol(i,j,k);
            e(i,j,k,7)=s*l(i,j,k)*vol(i,j,k);
            e(i,j,k,0)=de*de*vol(i,j,k);
            e(i,j,k,1)=ae*ae*vol(i,j,k);
            e(i,j,k,2)=l(i,j,k)*vol(i,j,k);
            e(i,j,k,3)=a(i,j,k,2)*vol(i,j,k);
            e(i,j,k,4)=std::abs(v(i,j,k,2)/nu-l(i,j,k));
            // Independent assembled affine stencil: 3 diagonal second
            // differences and xy four-corner mixed derivative.
            amrex::Real eigen=-4*(1+eps*eps)*std::pow(std::sin(w*h[0]/2)/(sc[0]*h[0]),2)
                -4*std::pow(std::sin(w*h[1]/2)/(sc[1]*h[1]),2)
                +2*eps*std::sin(w*h[0])*std::sin(w*h[1])/(sc[0]*sc[1]*h[0]*h[1]);
            amrex::Real ce=2*eps*std::sin(w*h[0])*std::sin(w*h[1])/(sc[0]*sc[1]*h[0]*h[1])*s;
            e(i,j,k,5)=std::abs(l(i,j,k)-eigen*s)+std::abs(x(i,j,k)-ce);
        });
    }
    amrex::Gpu::streamSynchronize();
    amrex::Real const volume_sum=global_sum(metric.cell_volume_cc());
    amrex::Real const lap_error=std::sqrt(global_sum(work,0)/volume_sum);
    amrex::Real const adv_error=std::sqrt(global_sum(work,1)/volume_sum);
    amrex::Real const conservation=std::abs(global_sum(work,2));
    amrex::Real const adv_conservation=std::abs(global_sum(work,3));
    amrex::Real const action_error=global_max(work,5);
    amrex::Real const gradient_error=std::sqrt(global_sum(work,6)/volume_sum);
    amrex::Real const energy_rate=global_sum(work,7);
    require(energy_rate<0.0,"diffusion energy production");
    require(global_max(work,4)<1e-11,"viscosity/pressure flux mismatch");
    require(conservation<1e-12 && adv_conservation<1e-12,"periodic integrated conservation");
    if (stretch[0]==0.0 && stretch[1]==0.0 && stretch[2]==0.0)
        require(action_error<2e-10,"assembled affine action/cross sign");
    if (eps!=0.0) require(global_max(cross)>0.1,"cross term must be nontrivial");
    amrex::Real assembled_relative_l2=0.0;
    if (stretch[0]==0.0 && stretch[1]==0.0 && stretch[2]==0.0) {
        // Independent assembled 11-point action for this xy-shear subset.
        // A deterministic irregular periodic vector exercises every coupling,
        // rather than only the single sine mode used by the continuous MMS.
        amrex::MultiFab arbitrary(m_ba,m_dm,1,1),matrix_free(m_ba,m_dm,1,0);
        for (amrex::MFIter mfi(arbitrary,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto const f=arbitrary.array(mfi);
            amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                f(i,j,k)=std::sin(17.0*i+13.0*j+7.0*k)+0.3*std::cos(3.0*i-11.0*j+5.0*k);
            });
        }
        arbitrary.FillBoundary(m_geom.periodicity());
        for (int d=0;d<3;++d) compute_metric_face_gradient_flux(arbitrary,d,metric,epoch,config,flux[d]);
        compute_metric_divergence(fv,metric,epoch,config,matrix_free);
        for (amrex::MFIter mfi(work,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
            auto const f=arbitrary.const_array(mfi);auto const a=matrix_free.const_array(mfi);
            auto const e=work.array(mfi);auto const vol=metric.cell_volume_cc().const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                amrex::Real center=f(i,j,k);
                amrex::Real reference=(1+eps*eps)/(sc[0]*sc[0]*h[0]*h[0])*
                    (f(i+1,j,k)-2*center+f(i-1,j,k))+
                    (f(i,j+1,k)-2*center+f(i,j-1,k))/(sc[1]*sc[1]*h[1]*h[1])+
                    (f(i,j,k+1)-2*center+f(i,j,k-1))/(sc[2]*sc[2]*h[2]*h[2])-
                    eps/(2*sc[0]*sc[1]*h[0]*h[1])*
                    (f(i+1,j+1,k)-f(i+1,j-1,k)-f(i-1,j+1,k)+f(i-1,j-1,k));
                e(i,j,k,0)=(a(i,j,k)-reference)*(a(i,j,k)-reference)*vol(i,j,k);
                e(i,j,k,1)=reference*reference*vol(i,j,k);
            });
        }
        assembled_relative_l2=std::sqrt(global_sum(work,0)/global_sum(work,1));
        require(assembled_relative_l2<=1e-12,"arbitrary-vector assembled action");
    }
    // Reader and writer reject shear before touching any path, on every
    // backend/rank and including epsilon=0 (schema-v3 lacks shear provenance).
    for (bool reading : {false,true}) {
        bool checkpoint_rejected=false;
        try {
            if (reading) m_solver.read_checkpoint("");
            else m_solver.write_checkpoint("");
        } catch (std::runtime_error const& error) {
            checkpoint_rejected=std::string(error.what()).find(
                "analytic_shear checkpoint/restart provenance is not implemented")!=std::string::npos;
        }
        require(checkpoint_rejected,"shear checkpoint/restart rejection reason");
    }
    // Reject incompatible RHS, stale epoch, and solver exhaustion independently.
    amrex::MultiFab bad_rhs(m_ba,m_dm,1,0),test_phi(m_ba,m_dm,1,2);
    // A different metric with identical mapping/epoch/layout is still a
    // different owner. Reject it before changing caller Phi or scratch.
    {
    MetricData other_metric;other_metric.define(m_ba,m_dm,1);
    auto other_mapping=make_coordinate_mapping(config.mapping_type,config.analytic_parameters,eps);
    other_metric.build(*other_mapping,metric.logical_grid(),m_geom);
    DeferredProjectionWorkspace other_workspace(other_metric);
    test_phi.setVal(7.0);bad_rhs.setVal(0.0);
    bool wrong_owner=false;
    try { solve_deferred_metric_poisson(test_phi,bad_rhs,metric,epoch,config,m_geom,other_workspace); }
    catch (std::runtime_error const& error) {
        wrong_owner=std::string(error.what()).find("workspace metric owner/epoch/layout mismatch")!=std::string::npos;
    }
    require(wrong_owner && global_max(test_phi)==7.0,"workspace owner rejection changed Phi");
    }
    bad_rhs.setVal(1.0);
    bool rejected=false;
    try { solve_deferred_metric_poisson(test_phi,bad_rhs,metric,epoch,config,m_geom); }
    catch (std::runtime_error const&) { rejected=true; }
    require(rejected,"incompatible RHS accepted");
    rejected=false;
    try { compute_metric_face_gradient_flux(scalar,0,metric,epoch+1,config,flux[0]); }
    catch (std::runtime_error const&) { rejected=true; }
    require(rejected,"stale metric accepted");
    amrex::MultiFab::Copy(bad_rhs,lap,0,0,1,0);
    amrex::MultiFab::Multiply(bad_rhs,metric.cell_volume_cc(),0,0,1,0);bad_rhs.mult(-1.0);
    if (eps!=0.0) {
        auto limited=config;limited.deferred_max_iterations=1;rejected=false;
        try { solve_deferred_metric_poisson(test_phi,bad_rhs,metric,epoch,limited,m_geom); }
        catch (std::runtime_error const&) { rejected=true; }
        require(rejected,"outer iteration exhaustion accepted");
    }
    // Predictor is dt*G(scalar), so projected face flux must vanish and Phi
    // must recover the discrete scalar with only a volume-weighted datum.
    for (int d=0;d<3;++d) {
        compute_metric_face_gradient_flux(scalar,d,metric,epoch,config,flux[d]);
        amrex::MultiFab::Copy(m_ucont[d],flux[d],0,0,1,0);m_ucont[d].mult(dt);
    }
    mark_valid_modified();
    amrex::Real degeneration=0.0;
    if (eps==0.0) {
        auto orth_config=config;
        orth_config.mapping_type="analytic_orthogonal";
        orth_config.projection=ProjectionOperatorMode::OrthogonalMLMG;
        amrex::Vector<int> nc{m_geom.Domain().length(0),m_geom.Domain().length(1),m_geom.Domain().length(2)};
        amrex::RealBox box(m_geom.ProbLo(),m_geom.ProbHi());
        AVWiSSolver orth_solver(nc,m_geom.Domain().length(0),2,box,{1,1,1},{},orth_config);
        orth_solver.initialize();
        AVWiSContractTestAccess peer(orth_solver);
        // Different decomposition deliberately tests ParallelCopy as well.
        peer.m_ucat.ParallelCopy(m_ucat,0,0,3);
        for (int d=0;d<3;++d) peer.m_ucont[d].ParallelCopy(m_ucont[d],0,0,1);
        peer.mark_valid_modified();
        auto const orth_report=orth_solver.project_orthogonal(dt);
        (void)orth_report;
        m_solver.project_mapped(dt);
        amrex::MultiFab reference(m_ba,m_dm,3,0);
        reference.ParallelCopy(peer.m_ucat,0,0,3);
        amrex::MultiFab::Subtract(reference,m_ucat,0,0,3,0);
        degeneration=std::max({global_max(reference,0),global_max(reference,1),global_max(reference,2)});
        reference.ParallelCopy(peer.m_phi,0,0,1);
        amrex::MultiFab::Subtract(reference,m_phi,0,0,1,0);
        degeneration=std::max(degeneration,global_max(reference));
        require(degeneration<1e-10,"epsilon zero orthogonal projection regression");
        // Restore the manufactured predictor before the primary report solve.
        for (int d=0;d<3;++d) {
            amrex::MultiFab::Copy(m_ucont[d],flux[d],0,0,1,0);m_ucont[d].mult(dt);
        }
        mark_valid_modified();
    }
    if (eps!=0.0) {
        amrex::MultiFab saved_ucat(m_ba,m_dm,3,0),saved_p(m_ba,m_dm,1,0);
        amrex::MultiFab::Copy(saved_ucat,m_ucat,0,0,3,0);
        amrex::MultiFab::Copy(saved_p,m_p,0,0,1,0);
        auto const valid_before=m_valid_epoch;
        m_mapping_operator.deferred_max_iterations=1;rejected=false;
        try { m_solver.project_mapped(dt); }
        catch (std::runtime_error const&) { rejected=true; }
        m_mapping_operator=config;
        require(rejected,"projection exhaustion accepted");
        require(m_valid_epoch==valid_before,"failed projection advanced valid epoch");
        amrex::MultiFab::Subtract(saved_ucat,m_ucat,0,0,3,0);
        amrex::MultiFab::Subtract(saved_p,m_p,0,0,1,0);
        require(global_max(saved_ucat,0)==0.0 && global_max(saved_ucat,1)==0.0 &&
            global_max(saved_ucat,2)==0.0 && global_max(saved_p)==0.0,
            "failed projection changed velocity/pressure");
        for (int d=0;d<3;++d) {
            amrex::MultiFab saved_flux(m_ucont[d].boxArray(),m_dm,1,0);
            amrex::MultiFab::Copy(saved_flux,flux[d],0,0,1,0);saved_flux.mult(dt);
            amrex::MultiFab::Subtract(saved_flux,m_ucont[d],0,0,1,0);
            require(global_max(saved_flux)==0.0,"failed projection changed face flux");
        }
    }
    auto const projection=m_solver.project_mapped(dt);
    require(projection.max_divergence_before>1e-5 && projection.max_divergence_after<1e-10,
        "projection divergence");
    amrex::MultiFab::Copy(work,scalar,0,0,1,0);
    amrex::MultiFab::Multiply(work,metric.cell_volume_cc(),0,0,1,0);
    amrex::Real const datum=global_sum(work)/volume_sum;
    amrex::MultiFab::Copy(work,m_phi,0,0,1,0);
    amrex::MultiFab::Subtract(work,scalar,0,0,1,0);work.plus(datum,0,1,0);
    amrex::Real const phi_error=global_max(work);
    require(phi_error<1e-10,"discrete pressure recovery");
    // Coupled production predictor/project step: this velocity has nonzero
    // predicted divergence, so the step must actually solve pressure rather
    // than take the roundoff/free-stream shortcut.
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const u=m_ucat.array(mfi);auto const p=scalar.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            u(i,j,k,0)=speed+0.1*p(i,j,k);
            u(i,j,k,1)=0.05*p(i,j,k);u(i,j,k,2)=p(i,j,k);
        });
    }
    mark_valid_modified();sync_ucont_from_ucat();m_solver.advance_one_step(dt,nu);
    compute_metric_divergence(velocity,metric,epoch,config,lap);
    amrex::Real const step_div=global_max(lap);
    require(step_div<1e-10,"production step divergence");
    require(global_max(m_phi)>1e-3,"coupled step did not solve nontrivial pressure");
    // Four actual coupled advances with mixed xy/xz modes and nonzero P.
    // Cell momentum state and conservative face state follow distinct updates;
    // halo freshness does not mean they are inverse interpolations.
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const u=m_ucat.array(mfi);auto const p=m_p.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            amrex::Real a=std::sin(w*((i+0.5)*h[0]+(j+0.5)*h[1]));
            amrex::Real b=std::cos(w*((i+0.5)*h[0]-(k+0.5)*h[2]));
            u(i,j,k,0)=speed+0.1*a+0.03*b;
            u(i,j,k,1)=0.05*a;u(i,j,k,2)=a+0.07*b;
            p(i,j,k)=0.2*a;
        });
    }
    mark_valid_modified();sync_ucont_from_ucat();
    amrex::MultiFab previous(m_ba,m_dm,3,m_nghost),older(m_ba,m_dm,3,0);
    amrex::MultiFab expected(m_ba,m_dm,3,m_nghost),difference(m_ba,m_dm,3,0);
    amrex::Array<amrex::MultiFab,3> previous_flux,older_flux,expected_flux;
    amrex::Array<amrex::MultiFab*,3> expected_faces{};
    for (int d=0;d<3;++d) {
        previous_flux[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        older_flux[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        expected_flux[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        expected_faces[d]=&expected_flux[d];
    }
    auto momentum=[&]() {
        amrex::Array<amrex::Real,3> result{};
        amrex::MultiFab::Copy(difference,m_ucat,0,0,3,0);
        for (int c=0;c<3;++c) {
            amrex::MultiFab::Multiply(difference,metric.cell_volume_cc(),0,c,1,0);
            result[c]=global_sum(difference,c);
        }
        return result;
    };
    auto const initial_momentum=momentum();
    auto const start_step=m_step;auto const start_time=m_time;
    auto* const workspace=m_solver.m_deferred_workspace.get();
    std::vector<amrex::Real const*> scratch_addresses;
    if (workspace) for (amrex::MFIter mfi(workspace->weighted,false);mfi.isValid();++mfi)
        scratch_addresses.push_back(workspace->weighted[mfi].dataPtr());
    std::unique_ptr<AVWiSSolver> zero_peer;
    if (eps==0.0) {
        auto orth=config;orth.mapping_type="analytic_orthogonal";
        orth.projection=ProjectionOperatorMode::OrthogonalMLMG;
        zero_peer=std::make_unique<AVWiSSolver>(
            amrex::Vector<int>{m_geom.Domain().length(0),m_geom.Domain().length(1),m_geom.Domain().length(2)},
            m_geom.Domain().length(0),m_nghost,amrex::RealBox(m_geom.ProbLo(),m_geom.ProbHi()),
            amrex::Vector<int>{1,1,1},CartesianBoundaryConfig{},orth);
        zero_peer->initialize();AVWiSContractTestAccess peer(*zero_peer);
        peer.m_ucat.ParallelCopy(m_ucat,0,0,3);peer.m_p.ParallelCopy(m_p,0,0,1);
        peer.mark_valid_modified();peer.sync_ucont_from_ucat();
    }
    std::vector<ProjectionDiagnostics> multi_reports;
    amrex::Real multi_div=0.0,momentum_drift=0.0,state_error=0.0,history_error=0.0,multi_zero=0.0;
    for (int step=0;step<4;++step) {
        amrex::MultiFab::Copy(previous,m_ucat,0,0,3,m_nghost);
        amrex::MultiFab::Copy(older,m_ucat_old,0,0,3,0);
        for (int d=0;d<3;++d) {
            amrex::MultiFab::Copy(previous_flux[d],m_ucont[d],0,0,1,0);
            amrex::MultiFab::Copy(older_flux[d],m_ucont_old[d],0,0,1,0);
        }
        compute_advection_rhs(adv);compute_viscous_rhs(visc,nu);
        amrex::MultiFab::Copy(expected,previous,0,0,3,0);
        amrex::MultiFab::Saxpy(expected,dt,adv,0,0,3,0);
        amrex::MultiFab::Saxpy(expected,dt,visc,0,0,3,0);
        expected.FillBoundary(m_geom.periodicity());
        sync_metric_ucont_from_ucat(expected,expected_faces,metric,epoch,config,m_geom.periodicity());
        auto const valid_before=m_valid_epoch;
        m_solver.advance_one_step(dt,nu);
        auto const report=m_solver.last_step_projection();multi_reports.push_back(report);
        require(m_valid_epoch>valid_before && m_halo_epoch==m_valid_epoch,"advance epoch/halo freshness");
        require_ghosts_fresh("c3-coupled-multistep");
        require(report.max_divergence_before>1e-8 && global_max(m_phi)>1e-6,
            "multistep skipped nontrivial pressure");
        require(report.max_divergence_after<=1e-10 && report.full_relative_residual<=1e-11,
            "multistep pressure/divergence gate");
        multi_div=std::max(multi_div,report.max_divergence_after);
        compute_metric_cell_gradient(m_phi,metric,epoch,config,gradient);
        amrex::MultiFab::Saxpy(expected,-dt,gradient,0,0,3,0);
        amrex::MultiFab::Subtract(expected,m_ucat,0,0,3,0);
        amrex::MultiFab::Subtract(previous,m_ucat_old,0,0,3,0);
        amrex::MultiFab::Subtract(older,m_ucat_older,0,0,3,0);
        for (int c=0;c<3;++c) {
            state_error=std::max(state_error,global_max(expected,c));
            history_error=std::max({history_error,global_max(previous,c),global_max(older,c)});
        }
        for (int d=0;d<3;++d) {
            compute_metric_face_gradient_flux(m_phi,d,metric,epoch,config,flux[d]);
            amrex::MultiFab::Saxpy(expected_flux[d],-dt,flux[d],0,0,1,0);
            amrex::MultiFab::Subtract(expected_flux[d],m_ucont[d],0,0,1,0);
            state_error=std::max(state_error,global_max(expected_flux[d]));
            amrex::MultiFab::Subtract(previous_flux[d],m_ucont_old[d],0,0,1,0);
            amrex::MultiFab::Subtract(older_flux[d],m_ucont_older[d],0,0,1,0);
            history_error=std::max({history_error,global_max(previous_flux[d]),global_max(older_flux[d])});
        }
        auto const now=momentum();
        for (int c=0;c<3;++c) momentum_drift=std::max(momentum_drift,std::abs(now[c]-initial_momentum[c]));
        if (zero_peer) {
            zero_peer->advance_one_step(dt,nu);AVWiSContractTestAccess peer(*zero_peer);
            difference.ParallelCopy(peer.m_ucat,0,0,3);
            amrex::MultiFab::Subtract(difference,m_ucat,0,0,3,0);
            for (int c=0;c<3;++c) multi_zero=std::max(multi_zero,global_max(difference,c));
            difference.ParallelCopy(peer.m_p,0,0,1);
            amrex::MultiFab::Subtract(difference,m_p,0,0,1,0);
            multi_zero=std::max(multi_zero,global_max(difference));
            for (int d=0;d<3;++d) {
                expected_flux[d].ParallelCopy(peer.m_ucont[d],0,0,1);
                amrex::MultiFab::Subtract(expected_flux[d],m_ucont[d],0,0,1,0);
                multi_zero=std::max(multi_zero,global_max(expected_flux[d]));
            }
        }
        require(m_solver.m_deferred_workspace.get()==workspace,"workspace owner replaced");
        if (workspace) {
            int local=0;
            for (amrex::MFIter mfi(workspace->weighted,false);mfi.isValid();++mfi)
                require(workspace->weighted[mfi].dataPtr()==scratch_addresses[local++],"scratch allocation replaced");
        }
    }
    require(state_error<1e-11 && history_error==0.0,"multistep cell/face update or history rotation");
    require(momentum_drift<1e-12,"multistep volume-weighted momentum conservation");
    require(multi_zero<1e-10,"multistep epsilon=0 orthogonal regression");
    require(m_step==start_step+4 && m_history_depth==3 &&
        std::abs(m_time-start_time-4*dt)<1e-14,"multistep time/history metadata");
    for (amrex::MFIter mfi(work,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const out=work.array(mfi);auto const u=m_ucat.const_array(mfi);
        auto const p=m_p.const_array(mfi);auto const v=metric.cell_volume_cc().const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            out(i,j,k,0)=(u(i,j,k,0)*u(i,j,k,0)+u(i,j,k,1)*u(i,j,k,1)+
                u(i,j,k,2)*u(i,j,k,2))*v(i,j,k);
            out(i,j,k,1)=p(i,j,k)*p(i,j,k)*v(i,j,k);
        });
    }
    amrex::Real const multi_velocity_l2=std::sqrt(global_sum(work,0)/volume_sum);
    amrex::Real const multi_pressure_l2=std::sqrt(global_sum(work,1)/volume_sum);
    // One hundred common steps preserve a constant free stream.
    m_ucat.setVal(0.0);m_ucat.setVal(speed,0,1);m_ucat.setVal(0.1,2,1);
    m_p.setVal(0.0);mark_valid_modified();sync_ucont_from_ucat();
    for (int step=0;step<100;++step) m_solver.advance_one_step(dt,nu);
    amrex::MultiFab::Copy(work,m_ucat,0,0,3,0);work.plus(-speed,0,1,0);work.plus(-0.1,2,1,0);
    amrex::Real const free_stream=std::max({global_max(work,0),global_max(work,1),global_max(work,2)});
    require(free_stream<1e-11 && global_max(m_p)<1e-11,"100 step free stream drift");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream out(path);require(bool(out),"cannot write report");
        out<<std::setprecision(17)<<"{\n\"status\":\"PASS\",\n\"epsilon\":"<<eps
           <<",\n\"cells\":"<<m_geom.Domain().length(0)<<",\n\"ranks\":"<<amrex::ParallelDescriptor::NProcs()
           <<",\n\"lap_l2\":"<<lap_error<<",\n\"adv_l2\":"<<adv_error
           <<",\n\"gradient_l2\":"<<gradient_error<<",\n\"energy_rate\":"<<energy_rate
           <<",\n\"conservation\":"<<conservation<<",\n\"adv_conservation\":"<<adv_conservation
           <<",\n\"action_error\":";
        if (stretch[0]==0.0 && stretch[1]==0.0 && stretch[2]==0.0) out << action_error;
        else out << "null";
        out<<",\n\"assembled_relative_l2\":";
        if (stretch[0]==0.0 && stretch[1]==0.0 && stretch[2]==0.0) out << assembled_relative_l2;
        else out << "null";
        out<<",\n\"pressure_error\":"<<phi_error
           <<",\n\"full_relative_residual\":"<<projection.full_relative_residual
           <<",\n\"outer_iterations\":"<<projection.outer_iterations
           <<",\n\"inner_iterations\":"<<projection.inner_iterations
           <<",\n\"projection_divergence\":"<<projection.max_divergence_after
           <<",\n\"step_divergence\":"<<step_div<<",\n\"degeneration\":"<<degeneration
           <<",\n\"multistep_velocity_l2\":"<<multi_velocity_l2
           <<",\n\"multistep_pressure_l2\":"<<multi_pressure_l2
           <<",\n\"multistep_divergence\":"<<multi_div
           <<",\n\"momentum_drift\":"<<momentum_drift
           <<",\n\"state_update_error\":"<<state_error
           <<",\n\"history_error\":"<<history_error
           <<",\n\"multistep_zero_error\":"<<multi_zero
           <<",\n\"coupled_steps\":4,\n\"step_projections\":[";
        for (std::size_t i=0;i<multi_reports.size();++i) {
            if (i) out<<",";
            auto const& r=multi_reports[i];
            out<<"{\"outer\":"<<r.outer_iterations<<",\"inner\":"<<r.inner_iterations
               <<",\"full_relative_residual\":"<<r.full_relative_residual
               <<",\"divergence_before\":"<<r.max_divergence_before
               <<",\"divergence_after\":"<<r.max_divergence_after<<"}";
        }
        out<<"],\n\"free_stream\":"<<free_stream<<"\n}\n";
    }
    amrex::Print()<<"AVWiS C3 shear contract PASS lap_l2="<<lap_error<<" adv_l2="<<adv_error<<"\n";
}
