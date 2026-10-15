#include "AVWiSContractTestAccess.H"
#include "RunMode.H"
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {
void check(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(std::string("pressure reference: ")+message);
}
// Canonical SoA, x fastest, valid cells, both periodic endpoints, overlaps once.
void dump(std::ofstream& out, std::string const& name, amrex::MultiFab const& f,
          amrex::Box const& cells, bool& first) {
    auto const domain=amrex::convert(cells,f.ixType());
    amrex::DistributionMapping io(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
    amrex::MultiFab gathered(amrex::BoxArray(domain),io,f.nComp(),0);
    gathered.ParallelCopy(f,0,0,f.nComp(),0,0); amrex::Gpu::streamSynchronize();
    int ok=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::FArrayBox host(domain,f.nComp(),amrex::The_Pinned_Arena());
        amrex::Gpu::dtoh_memcpy(host.dataPtr(),gathered[0].dataPtr(),host.size()*sizeof(amrex::Real));
        if (!first) out<<',';
        first=false; out<<'"'<<name<<"\":[";
        for (amrex::Long n=0;n<host.size();++n) {
            if (!std::isfinite(host.dataPtr()[n])) ok=0;
            if (n) out<<',';
            out<<host.dataPtr()[n];
        }
        out<<']'; out.flush(); ok=ok && bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(ok); check(ok,"snapshot finite/write gate");
}
void load_rhs(std::string const& path,amrex::MultiFab& f,amrex::Box const& domain) {
    // Every rank reads the same immutable canonical text, then stages its local boxes.
    std::ifstream in(path); std::vector<amrex::Real> values(domain.numPts());
    int ok=bool(in);
    for (auto& x:values) { in>>x; ok=ok && bool(in) && std::isfinite(x); }
    std::string extra; if (in>>extra) ok=0;
    amrex::ParallelDescriptor::ReduceIntMin(ok); check(ok,"frozen RHS length/finite/read gate");
    for (amrex::MFIter mfi(f,false);mfi.isValid();++mfi) {
        auto const b=mfi.validbox(); amrex::FArrayBox host(b,1,amrex::The_Pinned_Arena());
        auto a=host.array();
        for (int k=b.smallEnd(2);k<=b.bigEnd(2);++k)
            for (int j=b.smallEnd(1);j<=b.bigEnd(1);++j)
                for (int i=b.smallEnd(0);i<=b.bigEnd(0);++i)
                    a(i,j,k)=values[((k-domain.smallEnd(2))*domain.length(1)+j-domain.smallEnd(1))*domain.length(0)+i-domain.smallEnd(0)];
        amrex::Gpu::htod_memcpy(f[mfi].dataPtr(),host.dataPtr(),host.size()*sizeof(amrex::Real));
    }
    amrex::Gpu::streamSynchronize();
}
void trace_json(std::ofstream& out, DeferredProjectionTrace const& trace) {
    out<<'[';
    for (std::size_t n=0;n<trace.size();++n) {
        if (n) out<<',';
        auto const& r=trace[n];
        out<<"{\"outer\":"<<r.outer<<",\"inner\":"<<r.inner
           <<",\"inner_initial\":"<<r.inner_initial<<",\"inner_final\":"<<r.inner_final
           <<",\"removed_mean\":"<<r.removed_mean<<",\"full_max\":"<<r.full_max
           <<",\"full_relative\":"<<r.full_relative<<'}';
    }
    out<<']';
}
}

void AVWiSContractTestAccess::run_pressure_reference(
    std::string const& path,amrex::Real dt,amrex::Real nu)
{
    check(m_mapping_operator.mapping_type=="analytic_shear" && m_mapping_operator.shear_epsilon!=0 &&
          std::isfinite(dt) && dt>0 && std::isfinite(nu) && nu>0,"nonzero periodic shear and positive dt/nu required");
    auto const& metric=m_metric_data; auto const& cfg=m_mapping_operator;
    auto const domain=m_geom.Domain(); auto const h=metric.logical_grid().spacing;
    auto const epoch=m_metric_epoch;
    int warmup=0;amrex::ParmParse(avwis_input_namespace()).query("pressure_reference_warmup",warmup);
    check(warmup>=0 && warmup<=16,"warmup must be in 0..16");
    amrex::MultiFab scalar(m_ba,m_dm,1,1),div(m_ba,m_dm,1,0),rhs(m_ba,m_dm,1,0);
    amrex::MultiFab gradient(m_ba,m_dm,3,1),saved(m_ba,m_dm,3,2),predictor(m_ba,m_dm,3,2);
    amrex::MultiFab adv(m_ba,m_dm,3,0),visc(m_ba,m_dm,3,0),initial_phi(m_ba,m_dm,1,1);
    amrex::Array<amrex::MultiFab,3> scratch,saved_faces;
    amrex::Array<amrex::MultiFab*,3> outfaces{};
    amrex::Array<amrex::MultiFab const*,3> views{},actual{};
    for (int d=0;d<3;++d) {
        scratch[d].define(m_ucont[d].boxArray(),m_dm,1,1);
        saved_faces[d].define(m_ucont[d].boxArray(),m_dm,1,2);
        outfaces[d]=&scratch[d]; views[d]=&scratch[d]; actual[d]=&m_ucont[d];
    }
    std::ofstream out; int writable=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {out.open(path);out.precision(17);writable=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"cannot open snapshot");
    int active=m_ucat.local_size()>0; amrex::ParallelDescriptor::ReduceIntSum(active);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out<<"{\"schema\":\"avwis-pressure-reference-v1\",\"status\":\"OBSERVATION\","
           <<"\"amrex_version\":\""<<amrex::Version()<<"\",\"locked_sha\":\""<<AVWIS_LOCKED_GIT_SHA<<"\","
#ifdef AMREX_USE_CUDA
           <<"\"backend\":\"CUDA\","
#else
           <<"\"backend\":\"CPU\","
#endif
           <<"\"ranks\":"<<amrex::ParallelDescriptor::NProcs()<<",\"boxes\":"<<m_ba.size()<<",\"active_ranks\":"<<active
           <<",\"n_cell\":["<<domain.length(0)<<','<<domain.length(1)<<','<<domain.length(2)<<']'
           <<",\"spacing\":["<<h[0]<<','<<h[1]<<','<<h[2]<<"],\"dt\":"<<dt<<",\"nu\":"<<nu
           <<",\"warmup\":"<<warmup<<",\"epsilon\":"<<cfg.shear_epsilon<<",\"scale\":["<<cfg.analytic_parameters.scale[0]<<','<<cfg.analytic_parameters.scale[1]<<','<<cfg.analytic_parameters.scale[2]<<']'
           <<",\"stretch\":["<<cfg.analytic_parameters.stretch[0]<<','<<cfg.analytic_parameters.stretch[1]<<','<<cfg.analytic_parameters.stretch[2]<<']'
           <<",\"old_pressure_in_predictor\":false,\"pressure_update\":\"P += Phi\","
           <<"\"gauge\":\"physical-volume-weighted zero mean per outer iteration\","
           <<"\"reduction\":\"local MultiFab sum/norm0 then ParallelDescriptor SUM/MAX; canonical gather to IO rank\","
#ifdef AMREX_USE_MPI
           <<"\"mpi_compiled\":true,"
#else
           <<"\"mpi_compiled\":false,"
#endif
           <<"\"inner_rtol\":1e-12,\"full_rtol\":1e-11,\"bottom_solver\":\"smoother\",\"bottom_smooth\":256,\"fields\":{";
    }
    bool first=true;
    auto emit=[&](std::string const& name,amrex::MultiFab const& f) {dump(out,name,f,domain,first);};
    auto di=[&](amrex::MultiFab& u,std::string const& name) {
        u.FillBoundary(m_geom.periodicity());
        sync_metric_ucont_from_ucat(u,outfaces,metric,epoch,cfg,m_geom.periodicity());
        compute_metric_divergence(views,metric,epoch,cfg,div); emit(name,div);
    };
    auto actions=[&](amrex::MultiFab& p,std::string const& prefix) {
        p.FillBoundary(m_geom.periodicity());
        for (auto part:{MetricFluxPart::Full,MetricFluxPart::Diagonal,MetricFluxPart::Cross}) {
            for (int d=0;d<3;++d) compute_metric_face_gradient_flux(p,d,metric,epoch,cfg,scratch[d],part);
            compute_metric_divergence(views,metric,epoch,cfg,div);
            emit(prefix+(part==MetricFluxPart::Full ? "_DG" : part==MetricFluxPart::Diagonal ? "_DG_diag" : "_DG_cross"),div);
        }
        compute_metric_cell_gradient(p,metric,epoch,cfg,gradient); di(gradient,prefix+"_DIC");
    };
    emit("cell_volume",metric.cell_volume_cc()); emit("grad_xi",metric.grad_xi_cc());
    for (int d=0;d<3;++d) {
        emit("Q_"+std::to_string(d),metric.face_gradient_metric_fc(d));
        emit("S_"+std::to_string(d),metric.face_area_vector_fc(d));
        emit("beta_"+std::to_string(d),metric.projection_beta_fc(d));
    }
    // Exactly representable probe separates operator action from trig/RHS/solver noise.
    for (amrex::MFIter mfi(scalar,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto a=scalar.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            a(i,j,k)=((17*i+13*j+7*k)%31-15)/32.0;
        });
    }
    emit("probe",scalar); actions(scalar,"probe");
    constexpr amrex::Real w=6.283185307179586476925286766559;
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto u=m_ucat.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            auto a=std::sin(w*((i+0.5)*h[0]+(j+0.5)*h[1]));
            auto b=std::cos(w*((i+0.5)*h[0]-(k+0.5)*h[2]));
            u(i,j,k,0)=0.35+0.1*a+0.03*b;u(i,j,k,1)=0.05*a;u(i,j,k,2)=a+0.07*b;
        });
    }
    mark_valid_modified();sync_ucont_from_ucat();
    compute_metric_divergence(actual,metric,epoch,cfg,div);emit("raw_divergence",div);
    auto const init=m_solver.project_mapped(1,1);
    check(init.max_divergence_after<=1e-10 && init.full_relative_residual<=1e-11,"initial projection gate");
    amrex::MultiFab::Copy(initial_phi,m_phi,0,0,1,1);emit("initial_Phi",initial_phi);actions(initial_phi,"initial_phi");
    emit("initial_Ucat",m_ucat);di(m_ucat,"initial_DI_Ucat");
    compute_metric_divergence(actual,metric,epoch,cfg,div);emit("initial_D_Ucont",div);
    m_p.setVal(0);mark_valid_modified();fill_ghost_cells();
    for (int n=0;n<warmup;++n) m_solver.advance_one_step(dt,nu);
    emit("reference_Ucat",m_ucat);di(m_ucat,"reference_DI_Ucat");
    compute_metric_divergence(actual,metric,epoch,cfg,div);emit("reference_D_Ucont",div);
    amrex::MultiFab::Copy(saved,m_ucat,0,0,3,2);
    for (int d=0;d<3;++d) amrex::MultiFab::Copy(saved_faces[d],m_ucont[d],0,0,1,2);
    // Form the same predictor outside advance to retain its RHS; actual advance below is the authority.
    compute_advection_rhs(adv);compute_viscous_rhs(visc,nu);
    amrex::MultiFab::Copy(predictor,saved,0,0,3,2);
    for (amrex::MFIter mfi(predictor,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto u=predictor.array(mfi);auto a=adv.const_array(mfi);auto v=visc.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),3,[=] AMREX_GPU_DEVICE(int i,int j,int k,int c) noexcept {
            u(i,j,k,c)+=dt*(a(i,j,k,c)+v(i,j,k,c)+0.0);
        });
    }
    emit("predictor_Ucat",predictor);di(predictor,"predictor_divergence");
    amrex::MultiFab::Copy(rhs,div,0,0,1,0);
    amrex::MultiFab::Multiply(rhs,metric.cell_volume_cc(),0,0,1,0);rhs.mult(-1/dt,0,1,0);
    emit("formed_rhs",rhs);
    m_p.setVal(0);mark_valid_modified();fill_ghost_cells();m_solver.advance_one_step(dt,nu);
    auto const step=m_solver.last_step_projection();
    check(step.max_divergence_after<=1e-10 && step.full_relative_residual<=1e-11,"actual step gate");
    emit("step_Ucat",m_ucat);emit("step_Phi",m_phi);emit("step_P",m_p);
    di(m_ucat,"step_DI_Ucat"); compute_metric_divergence(actual,metric,epoch,cfg,div);emit("step_D_Ucont",div);
    for (int d=0;d<3;++d) emit("step_Ucont_"+std::to_string(d),m_ucont[d]);
    // Perturb old pressure by a spatial field, restore the same velocities, and use actual advance.
    amrex::MultiFab::Copy(m_ucat,saved,0,0,3,2);
    for (int d=0;d<3;++d) amrex::MultiFab::Copy(m_ucont[d],saved_faces[d],0,0,1,2);
    amrex::MultiFab::Copy(m_p,scalar,0,0,1,0);emit("old_P_perturbation",m_p);
    mark_valid_modified();fill_ghost_cells();m_solver.advance_one_step(dt,nu);
    emit("perturbed_P_step_Ucat",m_ucat);emit("perturbed_P_step_Phi",m_phi);emit("perturbed_P_step_P",m_p);
    // Re-solve formed or imported RHS with the SAME production solver and opt-in trace.
    std::string frozen;amrex::ParmParse(avwis_input_namespace()).query("pressure_reference_rhs",frozen);
    if (!frozen.empty()) load_rhs(frozen,rhs,domain);
    emit("solve_rhs",rhs);
    DeferredProjectionWorkspace workspace(metric);DeferredProjectionTrace trace;
    try {
        auto const r=solve_deferred_metric_poisson(scalar,rhs,metric,epoch,cfg,m_geom,workspace,&trace);
        check(r.full_relative_residual<=1e-11,"replay full residual gate");
    } catch (...) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            out<<"},\"solve_failed\":true,\"trace\":";trace_json(out,trace);out<<"}\n";out.flush();
        }
        throw;
    }
    emit("solve_Phi",scalar);actions(scalar,"solve_phi");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out<<"},\"imported_rhs\":"<<(!frozen.empty() ? "true" : "false")
           <<",\"initial_projection\":{\"before\":"<<init.max_divergence_before<<",\"after\":"<<init.max_divergence_after
           <<",\"outer\":"<<init.outer_iterations<<",\"inner\":"<<init.inner_iterations<<",\"full_relative\":"<<init.full_relative_residual<<'}'
           <<",\"actual_step\":{\"before\":"<<step.max_divergence_before<<",\"after\":"<<step.max_divergence_after
           <<",\"outer\":"<<step.outer_iterations<<",\"inner\":"<<step.inner_iterations<<",\"full_relative\":"<<step.full_relative_residual<<'}'
           <<",\"trace\":";trace_json(out,trace);out<<"}\n";out.flush();writable=bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"final snapshot write gate");
}

// A separate reference, never selected by a production run. Initial cells are
// authoritative; faces are I(cells). Production retains its dual state after
// projection, so this does not claim that DG equals DIC or freeze runtime M2.
void AVWiSContractTestAccess::run_consistent_reference(
    std::string const& path,amrex::Real dt,amrex::Real nu,int steps)
{
    auto const cfg=m_mapping_operator;
    check(cfg.mapping_type=="analytic_shear" && cfg.shear_epsilon!=0 &&
          cfg.analytic_parameters.stretch==amrex::Array<amrex::Real,3>{0,0,0} &&
          std::isfinite(dt) && dt>0 && std::isfinite(nu) && nu>0 && steps>=1 && steps<=64,
          "consistent reference requires nonzero affine shear, positive dt/nu, 1..64 steps");
    auto const& metric=m_metric_data;auto const epoch=m_metric_epoch;
    auto const domain=m_geom.Domain();auto const h=metric.logical_grid().spacing;
    auto const sc=cfg.analytic_parameters.scale;auto const eps=cfg.shear_epsilon;
    constexpr amrex::Real w=6.283185307179586476925286766559;
    constexpr amrex::Real amplitude=0.04;
    // psi=a sin(2pi xi)sin(2pi eta), logical velocity=(delta_eta psi,
    // -delta_xi psi,0), physical velocity=A*logical. Centered differences
    // commute, so D I A curl(psi)=0 for this affine periodic mapping.
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto u=m_ucat.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            auto x=w*(i+0.5)*h[0],y=w*(j+0.5)*h[1];
            auto a=amplitude*std::sin(x)*std::cos(y)*std::sin(w*h[1])/h[1];
            auto b=-amplitude*std::cos(x)*std::sin(y)*std::sin(w*h[0])/h[0];
            u(i,j,k,0)=sc[0]*a+eps*sc[1]*b;
            u(i,j,k,1)=sc[1]*b;u(i,j,k,2)=0.1*std::sin(x+y);
        });
    }
    m_p.setVal(0);m_phi.setVal(0);mark_valid_modified();sync_ucont_from_ucat();
    for (auto* u:{&m_ucat_old,&m_ucat_older}) amrex::MultiFab::Copy(*u,m_ucat,0,0,3,m_nghost);
    for (int d=0;d<3;++d) for (auto* u:{&m_ucont_old[d],&m_ucont_older[d]})
        amrex::MultiFab::Copy(*u,m_ucont[d],0,0,1,m_nghost);
    m_history_depth=3;
    amrex::MultiFab div(m_ba,m_dm,1,0),rhs(m_ba,m_dm,1,0),pressure(m_ba,m_dm,1,1);
    amrex::MultiFab adv(m_ba,m_dm,3,0),visc(m_ba,m_dm,3,0),rate(m_ba,m_dm,3,1),grad(m_ba,m_dm,3,1);
    amrex::Array<amrex::MultiFab,3> interpolated;
    amrex::Array<amrex::MultiFab*,3> iv{};amrex::Array<amrex::MultiFab const*,3> ic{},fv{};
    for (int d=0;d<3;++d) {
        interpolated[d].define(m_ucont[d].boxArray(),m_dm,1,1);
        iv[d]=&interpolated[d];ic[d]=&interpolated[d];fv[d]=&m_ucont[d];
    }
    std::ofstream out;int writable=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {out.open(path);out.precision(17);writable=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"consistent snapshot open");
    int active=m_ucat.local_size()>0;amrex::ParallelDescriptor::ReduceIntSum(active);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out<<"{\"schema\":\"avwis-consistent-reference-v1\",\"status\":\"OBSERVATION\","
           <<"\"amrex_version\":\""<<amrex::Version()<<"\",\"locked_sha\":\""<<AVWIS_LOCKED_GIT_SHA<<"\","
#ifdef AMREX_USE_CUDA
           <<"\"backend\":\"CUDA\","
#else
           <<"\"backend\":\"CPU\","
#endif
           <<"\"ranks\":"<<amrex::ParallelDescriptor::NProcs()<<",\"boxes\":"<<m_ba.size()<<",\"active_ranks\":"<<active
           <<",\"n_cell\":["<<domain.length(0)<<','<<domain.length(1)<<','<<domain.length(2)<<']'
           <<",\"spacing\":["<<h[0]<<','<<h[1]<<','<<h[2]<<"],\"dt\":"<<dt<<",\"nu\":"<<nu<<",\"steps\":"<<steps
           <<",\"epsilon\":"<<eps<<",\"scale\":["<<sc[0]<<','<<sc[1]<<','<<sc[2]<<"],\"stretch\":[0,0,0],"
           <<"\"initial_authority\":\"cell streamfunction; Ucont=I(Ucat)\","
           <<"\"runtime_authority\":\"experimental dual state; faces conserve mass\","
           <<"\"pressure_datum\":\"initial DG multiplier with physical-volume zero mean; production P += Phi\",\"fields\":{";
    }
    bool first=true;
    auto emit=[&](std::string const& name,amrex::MultiFab const& f) {dump(out,name,f,domain,first);};
    auto maximum=[](amrex::MultiFab const& f,int ng=0) {
        amrex::Real r=0;
        for (int c=0;c<f.nComp();++c) {auto x=f.norm0(c,ng,true);amrex::ParallelDescriptor::ReduceRealMax(x);r=std::max(r,x);}
        check(std::isfinite(r),"nonfinite diagnostic maximum");return r;
    };
    auto constraints=[&](std::string const& prefix) {
        compute_metric_divergence(fv,metric,epoch,cfg,div);emit(prefix+"_D_Ucont",div);
        m_ucat.FillBoundary(m_geom.periodicity());
        sync_metric_ucont_from_ucat(m_ucat,iv,metric,epoch,cfg,m_geom.periodicity());
        compute_metric_divergence(ic,metric,epoch,cfg,div);emit(prefix+"_DI_Ucat",div);
        for (int d=0;d<3;++d) {
            emit(prefix+"_I_Ucat_"+std::to_string(d),interpolated[d]);
            emit(prefix+"_Ucont_"+std::to_string(d),m_ucont[d]);
        }
    };
    auto actions=[&](amrex::MultiFab& p,std::string const& prefix) {
        p.FillBoundary(m_geom.periodicity());
        for (int d=0;d<3;++d) compute_metric_face_gradient_flux(p,d,metric,epoch,cfg,interpolated[d]);
        compute_metric_divergence(ic,metric,epoch,cfg,div);emit(prefix+"_DG",div);
        compute_metric_cell_gradient(p,metric,epoch,cfg,grad);grad.FillBoundary(m_geom.periodicity());
        sync_metric_ucont_from_ucat(grad,iv,metric,epoch,cfg,m_geom.periodicity());
        compute_metric_divergence(ic,metric,epoch,cfg,div);emit(prefix+"_DIC",div);
    };
    emit("cell_volume",metric.cell_volume_cc());emit("grad_xi",metric.grad_xi_cc());
    for (int d=0;d<3;++d) {emit("Q_"+std::to_string(d),metric.face_gradient_metric_fc(d));emit("S_"+std::to_string(d),metric.face_area_vector_fc(d));emit("beta_"+std::to_string(d),metric.projection_beta_fc(d));}
    emit("initial_Ucat",m_ucat);constraints("initial");
    check(maximum(div)<=1e-10,"consistent initial DI constraint");
    compute_advection_rhs(adv);compute_viscous_rhs(visc,nu);
    amrex::MultiFab::Copy(rate,adv,0,0,3,0);amrex::MultiFab::Add(rate,visc,0,0,3,0);
    rate.FillBoundary(m_geom.periodicity());emit("initial_rate",rate);
    sync_metric_ucont_from_ucat(rate,iv,metric,epoch,cfg,m_geom.periodicity());
    compute_metric_divergence(ic,metric,epoch,cfg,div);emit("initial_pressure_source",div);
    check(maximum(div)>1e-3,"nontrivial multiplier source required");
    amrex::MultiFab::Copy(rhs,div,0,0,1,0);
    amrex::MultiFab::Multiply(rhs,metric.cell_volume_cc(),0,0,1,0);rhs.mult(-1);
    auto const reference=solve_deferred_metric_poisson(pressure,rhs,metric,epoch,cfg,m_geom);
    check(reference.full_relative_residual<=1e-11,"initial multiplier DG residual");
    emit("initial_multiplier",pressure);actions(pressure,"initial_multiplier");
    // A physical datum is defined for the reference; no production gauge is changed.
    amrex::MultiFab::Copy(m_p,pressure,0,0,1,0);mark_valid_modified();fill_ghost_cells();
    std::vector<ProjectionDiagnostics> reports;
    for (int n=0;n<steps;++n) {
        m_solver.advance_one_step(dt,nu);auto const r=m_solver.last_step_projection();
        check(r.max_divergence_after<=1e-10 && r.full_relative_residual<=1e-11,"consistent per-step projection gate");
        reports.push_back(r);
        if (n==0) {emit("first_Phi",m_phi);emit("first_Ucat",m_ucat);constraints("first");}
    }
    emit("final_Ucat",m_ucat);emit("final_P",m_p);emit("final_Phi",m_phi);constraints("final");actions(m_phi,"final_phi");
    // Captured state covers every registered field, all three layers, and every
    // allocated halo. Canonical JSON stores valid values; exact full-halo change
    // norms below cover local ghosts that cannot have a unique canonical gather.
    std::vector<std::pair<std::string,amrex::MultiFab*>> fields{
        {"Ucat",&m_ucat},{"Ucat_old",&m_ucat_old},{"Ucat_older",&m_ucat_older},
        {"P",&m_p},{"Phi",&m_phi},{"Nvert",&m_nvert}};
    for (int d=0;d<3;++d) {
        fields.emplace_back("Ucont_"+std::to_string(d),&m_ucont[d]);
        fields.emplace_back("Ucont_old_"+std::to_string(d),&m_ucont_old[d]);
        fields.emplace_back("Ucont_older_"+std::to_string(d),&m_ucont_older[d]);
    }
    auto metadata=[&]() {return std::array<amrex::Real,7>{m_time,amrex::Real(m_step),amrex::Real(m_history_depth),
        amrex::Real(m_valid_epoch),amrex::Real(m_halo_epoch),amrex::Real(m_physical_epoch),m_last_advance_seconds};};
    auto save=[&](std::string const& prefix) {
        std::vector<std::unique_ptr<amrex::MultiFab>> saved;
        for (auto const& f:fields) {
            auto const& src=*f.second;
            saved.push_back(std::make_unique<amrex::MultiFab>(src.boxArray(),m_dm,src.nComp(),src.nGrow()));
            amrex::MultiFab::Copy(*saved.back(),src,0,0,src.nComp(),src.nGrow());
            emit(prefix+"_"+f.first,src);
        }
        return saved;
    };
    auto changed=[&](auto const& saved,std::string const& prefix) {
        std::vector<std::array<amrex::Real,2>> result;
        for (std::size_t n=0;n<fields.size();++n) {
            auto const& src=*fields[n].second;
            amrex::MultiFab delta(src.boxArray(),m_dm,src.nComp(),src.nGrow());
            amrex::MultiFab::Copy(delta,src,0,0,src.nComp(),src.nGrow());
            amrex::MultiFab::Subtract(delta,*saved[n],0,0,src.nComp(),src.nGrow());
            result.push_back({maximum(delta),maximum(delta,src.nGrow())});
            emit(prefix+"_"+fields[n].first,src);
        }
        return result;
    };
    auto direct_saved=save("direct_before");auto const direct_meta=metadata();
    auto const direct=m_solver.project_mapped(dt);
    auto const direct_changes=changed(direct_saved,"direct_after");auto const direct_after_meta=metadata();
    // Reinterpolation deliberately rebuilds faces and exposes the DG/DIC seam.
    sync_ucont_from_ucat();constraints("reinterpolated_before");
    auto const reinterpolated=m_solver.project_mapped(dt);
    emit("reinterpolated_after_Ucat",m_ucat);emit("reinterpolated_after_Phi",m_phi);constraints("reinterpolated_after");
    // Force a real advance to fail in its projection, not an early input guard.
    auto failed_saved=save("failed_before");auto const failed_meta=metadata();
    std::string failure;
    m_mapping_operator.deferred_max_iterations=1;
    try { m_solver.advance_one_step(dt,nu); }
    catch (std::runtime_error const& e) {failure=e.what();}
    m_mapping_operator=cfg;
    check(failure.find("deferred pressure exhausted max_iterations=1")!=std::string::npos,
          "real advance did not reject exact deferred exhaustion");
    auto const failure_changes=changed(failed_saved,"failed_after");auto const failed_after_meta=metadata();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        auto nums=[&](auto const& values) {out<<'[';bool first_value=true;for (auto x:values) {if (!first_value) out<<',';first_value=false;out<<x;}out<<']';};
        auto changes=[&](auto const& values) {out<<'{';for (std::size_t n=0;n<fields.size();++n) {if (n) out<<',';out<<'"'<<fields[n].first<<"\":";nums(values[n]);}out<<'}';};
        out<<"},\"metadata_order\":[\"time\",\"step\",\"history_depth\",\"valid_epoch\",\"halo_epoch\",\"physical_epoch\",\"last_advance_seconds\"],"
           <<"\"direct_metadata_before\":";nums(direct_meta);out<<",\"direct_metadata_after\":";nums(direct_after_meta);
        out<<",\"failed_metadata_before\":";nums(failed_meta);out<<",\"failed_metadata_after\":";nums(failed_after_meta);
        out<<",\"change_norm_order\":[\"valid_max_abs\",\"including_ghost_max_abs\"],\"direct_changes\":";changes(direct_changes);
        out<<",\"failed_changes\":";changes(failure_changes);
        out<<",\"failure_reason\":\""<<failure<<"\",\"failure_contract\":\"fatal; no entire-advance rollback\","
           <<"\"direct_divergence_after\":"<<direct.max_divergence_after
           <<",\"reinterpolated_divergence_before\":"<<reinterpolated.max_divergence_before
           <<",\"reinterpolated_divergence_after\":"<<reinterpolated.max_divergence_after<<",\"projections\":[";
        for (std::size_t n=0;n<reports.size();++n) {if (n) out<<',';auto const& r=reports[n];out<<"{\"before\":"<<r.max_divergence_before<<",\"after\":"<<r.max_divergence_after<<",\"full_relative\":"<<r.full_relative_residual<<",\"outer\":"<<r.outer_iterations<<",\"inner\":"<<r.inner_iterations<<'}';}
        out<<"]}\n";out.flush();writable=bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"consistent final snapshot write");
    // This test exits immediately: the failed production solver is not reused.
}
