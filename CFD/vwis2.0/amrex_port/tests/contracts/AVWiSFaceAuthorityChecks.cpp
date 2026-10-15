#include "AVWiSContractTestAccess.H"
#include "RunMode.H"
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {
void check(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(std::string("face authority audit: ")+message);
}
amrex::Real maximum(amrex::MultiFab const& f, int ghosts=0) {
    amrex::Real result=0;
    for (int c=0;c<f.nComp();++c) {
        auto x=f.norm0(c,ghosts,true);amrex::ParallelDescriptor::ReduceRealMax(x);
        result=std::max(result,x);
    }
    check(std::isfinite(result),"nonfinite norm");return result;
}
amrex::Real total(amrex::MultiFab const& f,int c) {
    auto x=f.sum(c,true);amrex::ParallelDescriptor::ReduceRealSum(x);
    check(std::isfinite(x),"nonfinite integral");return x;
}
// Canonical SoA valid data, x fastest, shared faces gathered once; retain both
// periodic endpoints and every near-zero value. Ghosts have no unique gather.
void dump(std::ofstream& out,std::string const& name,amrex::MultiFab const& f,
          amrex::Box const& cells,bool& first) {
    auto const box=amrex::convert(cells,f.ixType());
    amrex::DistributionMapping io(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
    amrex::MultiFab gathered(amrex::BoxArray(box),io,f.nComp(),0);
    gathered.ParallelCopy(f,0,0,f.nComp(),0,0);amrex::Gpu::streamSynchronize();
    int ok=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::FArrayBox host(box,f.nComp(),amrex::The_Pinned_Arena());
        amrex::Gpu::dtoh_memcpy(host.dataPtr(),gathered[0].dataPtr(),host.size()*sizeof(amrex::Real));
        if (!first) out<<',';
        first=false;out<<'"'<<name<<"\":[";
        for (amrex::Long n=0;n<host.size();++n) {
            if (n) out<<',';
            auto const x=host.dataPtr()[n];ok=ok && std::isfinite(x);out<<x;
        }
        out<<']';out.flush();ok=ok && bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(ok);check(ok,"canonical snapshot write/finite gate");
}
void quoted(std::ofstream& out,std::string const& s) {
    out<<'"';for (char c:s) {
        if (c=='"' || c=='\\') out<<'\\'<<c;
        else if (c=='\n') out<<"\\n";
        else if (c=='\r') out<<"\\r";
        else if (c=='\t') out<<"\\t";
        else out<<c;
    }out<<'"';
}
}

void AVWiSContractTestAccess::run_face_authority_contract(
    std::string const& path,amrex::Real dt,amrex::Real nu,int steps)
{
    auto const cfg=m_mapping_operator;
    check(cfg.state_contract=="face_authoritative" && cfg.pressure_contract=="instantaneous_pressure" &&
        cfg.mapping_type=="analytic_shear" && cfg.shear_epsilon!=0 &&
        cfg.analytic_parameters.stretch==amrex::Array<amrex::Real,3>{0,0,0} &&
        dt>0 && std::isfinite(dt) && nu>0 && std::isfinite(nu) && steps>=1 && steps<=4096,
        "requires face_authoritative/instantaneous_pressure, nonzero affine shear, positive dt/nu and 1..4096 steps");
    for (int d=0;d<3;++d) check(m_geom.isPeriodic(d),"requires periodic geometry");
    auto const& metric=m_metric_data;auto const epoch=m_metric_epoch;
    auto const domain=m_geom.Domain();auto const h=metric.logical_grid().spacing;
    constexpr amrex::Real w=6.283185307179586476925286766559;
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const u=m_ucat.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            auto const a=std::sin(w*((i+0.5)*h[0]+(j+0.5)*h[1]));
            auto const b=std::cos(w*((i+0.5)*h[0]-(k+0.5)*h[2]));
            u(i,j,k,0)=0.35+0.1*a+0.03*b;u(i,j,k,1)=0.05*a;u(i,j,k,2)=a+0.07*b;
        });
    }
    m_p.setVal(0);m_phi.setVal(0);mark_valid_modified();sync_ucont_from_ucat();
    auto const initial_projection=m_solver.project_mapped(1.0);
    check(initial_projection.max_divergence_before>1e-3 && initial_projection.max_divergence_after<=1e-10 &&
        initial_projection.full_relative_residual<=1e-11,"initial projection gate");
    for (auto* u:{&m_ucat_old,&m_ucat_older}) amrex::MultiFab::Copy(*u,m_ucat,0,0,3,m_nghost);
    for (int d=0;d<3;++d) for (auto* u:{&m_ucont_old[d],&m_ucont_older[d]})
        amrex::MultiFab::Copy(*u,m_ucont[d],0,0,1,m_nghost);
    m_history_depth=3;mark_valid_modified();fill_ghost_cells();
    std::vector<std::pair<std::string,amrex::MultiFab*>> fields{
        {"Ucat",&m_ucat},{"Ucat_old",&m_ucat_old},{"Ucat_older",&m_ucat_older},
        {"P",&m_p},{"Phi",&m_phi},{"Nvert",&m_nvert}};
    for (int d=0;d<3;++d) {
        fields.emplace_back("Ucont_"+std::to_string(d),&m_ucont[d]);
        fields.emplace_back("Ucont_old_"+std::to_string(d),&m_ucont_old[d]);
        fields.emplace_back("Ucont_older_"+std::to_string(d),&m_ucont_older[d]);
    }
    check(fields.size()==m_fields.size(),"registered field coverage");
    std::ofstream out;int writable=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {out.open(path);out.precision(17);writable=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"snapshot open");
    int active=m_ucat.local_size()>0;amrex::ParallelDescriptor::ReduceIntSum(active);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out<<"{\"schema\":\"avwis-face-authority-v1\",\"status\":\"OBSERVATION\","
           <<"\"state_contract\":\"face_authoritative\",\"pressure_contract\":\"instantaneous_pressure\","
           <<"\"pressure_interpretation\":\"DG projection multiplier; physical pressure convergence not established\","
           <<"\"amrex_version\":\""<<amrex::Version()<<"\",\"locked_sha\":\""<<AVWIS_LOCKED_GIT_SHA<<"\","
#ifdef AMREX_USE_CUDA
           <<"\"backend\":\"CUDA\","
#else
           <<"\"backend\":\"CPU\","
#endif
           <<"\"ranks\":"<<amrex::ParallelDescriptor::NProcs()<<",\"boxes\":"<<m_ba.size()<<",\"active_ranks\":"<<active
           <<",\"n_cell\":["<<domain.length(0)<<','<<domain.length(1)<<','<<domain.length(2)<<']'
           <<",\"spacing\":["<<h[0]<<','<<h[1]<<','<<h[2]<<"],\"dt\":"<<dt<<",\"nu\":"<<nu<<",\"steps\":"<<steps
           <<",\"epsilon\":"<<cfg.shear_epsilon<<",\"scale\":["<<cfg.analytic_parameters.scale[0]<<','<<cfg.analytic_parameters.scale[1]<<','<<cfg.analytic_parameters.scale[2]
           <<"],\"stretch\":[0,0,0],\"fields\":{";
    }
    bool first=true;
    auto emit=[&](std::string const& name,amrex::MultiFab const& f) {dump(out,name,f,domain,first);};
    auto emit_state=[&](std::string const& prefix) {for (auto const& f:fields) emit(prefix+"_"+f.first,*f.second);};
    auto metadata=[&]() {return std::array<amrex::Real,7>{m_time,amrex::Real(m_step),amrex::Real(m_history_depth),
        amrex::Real(m_valid_epoch),amrex::Real(m_halo_epoch),amrex::Real(m_physical_epoch),m_last_advance_seconds};};
    auto save=[&]() {
        std::vector<std::unique_ptr<amrex::MultiFab>> result;
        for (auto const& f:fields) {
            auto const& src=*f.second;
            result.push_back(std::make_unique<amrex::MultiFab>(src.boxArray(),m_dm,src.nComp(),src.nGrow()));
            amrex::MultiFab::Copy(*result.back(),src,0,0,src.nComp(),src.nGrow());
        }return result;
    };
    auto changes=[&](auto const& saved) {
        std::vector<std::array<amrex::Real,2>> result;
        for (std::size_t n=0;n<fields.size();++n) {
            auto const& src=*fields[n].second;amrex::MultiFab delta(src.boxArray(),m_dm,src.nComp(),src.nGrow());
            amrex::MultiFab::Copy(delta,src,0,0,src.nComp(),src.nGrow());
            amrex::MultiFab::Subtract(delta,*saved[n],0,0,src.nComp(),src.nGrow());
            result.push_back({maximum(delta),maximum(delta,src.nGrow())});
        }return result;
    };
    amrex::MultiFab rate(m_ba,m_dm,3,1),adv(m_ba,m_dm,3,0),visc(m_ba,m_dm,3,0);
    amrex::MultiFab recovered(m_ba,m_dm,3,0),work(m_ba,m_dm,3,0),delta_p(m_ba,m_dm,1,0);
    amrex::Array<amrex::MultiFab,3> predictor,flux;
    amrex::Array<amrex::MultiFab*,3> prediction_out{};
    amrex::Array<amrex::MultiFab const*,3> face_view{};
    for (int d=0;d<3;++d) {
        predictor[d].define(m_ucont[d].boxArray(),m_dm,1,0);flux[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        prediction_out[d]=&predictor[d];face_view[d]=&m_ucont[d];
    }
    auto momentum=[&]() {
        amrex::MultiFab::Copy(work,m_ucat,0,0,3,0);std::array<amrex::Real,3> r{};
        for (int c=0;c<3;++c) {amrex::MultiFab::Multiply(work,metric.cell_volume_cc(),0,c,1,0);r[c]=total(work,c);}return r;
    };
    auto recovery_error=[&]() {
        sync_metric_ucat_from_ucont(face_view,metric,epoch,cfg,recovered);
        amrex::MultiFab::Subtract(recovered,m_ucat,0,0,3,0);return maximum(recovered);
    };
    emit("cell_volume",metric.cell_volume_cc());emit("grad_xi",metric.grad_xi_cc());
    for (int d=0;d<3;++d) {
        emit("Q_"+std::to_string(d),metric.face_gradient_metric_fc(d));
        emit("S_"+std::to_string(d),metric.face_area_vector_fc(d));emit("beta_"+std::to_string(d),metric.projection_beta_fc(d));
    }
    emit_state("initial");auto const initial_metadata=metadata();auto const initial_momentum=momentum();
    amrex::Real drift=0,history_error=0,recovery= recovery_error(),identity_error=0,max_cfl=0,max_diffusion=0;
    check(recovery==0,"initial R identity");
    std::vector<ProjectionDiagnostics> reports;
    auto prepare=[&](std::string const& prefix,amrex::Array<amrex::Real,3> force) {
        compute_advection_rhs(adv);compute_viscous_rhs(visc,nu);
        amrex::MultiFab::Copy(rate,adv,0,0,3,0);amrex::MultiFab::Add(rate,visc,0,0,3,0);
        for (int c=0;c<3;++c) rate.plus(force[c],c,1,0);
        rate.FillBoundary(m_geom.periodicity());emit(prefix+"_rate",rate);
        sync_metric_ucont_from_ucat(rate,prediction_out,metric,epoch,cfg,m_geom.periodicity());
        for (int d=0;d<3;++d) {
            emit(prefix+"_before_Ucont_"+std::to_string(d),m_ucont[d]);
            predictor[d].mult(dt);amrex::MultiFab::Add(predictor[d],m_ucont[d],0,0,1,0);
            emit(prefix+"_predictor_Ucont_"+std::to_string(d),predictor[d]);
        }
    };
    auto update_identity=[&](std::string const& prefix) {
        for (int d=0;d<3;++d) {
            compute_metric_face_gradient_flux(m_phi,d,metric,epoch,cfg,flux[d]);
            amrex::MultiFab::Saxpy(predictor[d],-dt,flux[d],0,0,1,0);
            amrex::MultiFab::Subtract(predictor[d],m_ucont[d],0,0,1,0);
            identity_error=std::max(identity_error,maximum(predictor[d]));
            emit(prefix+"_after_Ucont_"+std::to_string(d),m_ucont[d]);
        }
        emit(prefix+"_Phi",m_phi);emit(prefix+"_P",m_p);emit(prefix+"_Ucat",m_ucat);
        amrex::MultiFab::Copy(delta_p,m_p,0,0,1,0);amrex::MultiFab::Subtract(delta_p,m_phi,0,0,1,0);
        check(maximum(delta_p)==0,"P must equal Phi exactly");recovery=std::max(recovery,recovery_error());
    };
    for (int n=0;n<steps;++n) {
        auto const prefix="step"+std::to_string(n);auto previous=save();
        auto const stability=time_step_diagnostics(dt,nu);
        max_cfl=std::max(max_cfl,stability.advective_cfl);max_diffusion=std::max(max_diffusion,stability.diffusive_number);
        prepare(prefix,{});m_solver.advance_one_step(dt,nu);auto const r=m_solver.last_step_projection();
        reports.push_back(r);check(r.max_divergence_before>1e-8 && maximum(m_phi)>1e-6,"nonzero per-step projection");
        check(r.max_divergence_after<=1e-10 && r.full_relative_residual<=1e-11,"per-step divergence/residual gate");
        require_ghosts_fresh("face-authority-audit");check(m_valid_epoch==m_halo_epoch,"epoch freshness");
        update_identity(prefix);
        for (std::size_t i=0;i<fields.size();++i) {
            auto const& name=fields[i].first;
            if (name=="Ucat_old" || name=="Ucat_older" || name.find("Ucont_old_")==0 || name.find("Ucont_older_")==0) {
                std::string source=name;
                auto p=source.find("_older");if (p!=std::string::npos) source.replace(p,6,"_old");
                else {p=source.find("_old");source.erase(p,4);}
                auto const found=std::find_if(fields.begin(),fields.end(),[&](auto const& f) {return f.first==source;});
                auto const index=std::size_t(found-fields.begin());auto const& current=*fields[i].second;
                amrex::MultiFab difference(current.boxArray(),m_dm,current.nComp(),0);
                amrex::MultiFab::Copy(difference,current,0,0,current.nComp(),0);
                amrex::MultiFab::Subtract(difference,*previous[index],0,0,current.nComp(),0);
                history_error=std::max(history_error,maximum(difference));
            }
        }
        auto const now=momentum();for (int c=0;c<3;++c) drift=std::max(drift,std::abs(now[c]-initial_momentum[c]));
        check(m_step==std::uint64_t(initial_metadata[1])+n+1 && m_history_depth==3 &&
            std::abs(m_time-initial_metadata[0]-(n+1)*dt)<=1e-13,"per-step time/history metadata");
    }
    check(history_error==0 && recovery==0 && identity_error<=1e-12 && drift<=1e-12,"history/R/predictor/momentum gates");
    emit_state("final");auto const final_metadata=metadata();
    // A real additional advance with nonzero body force also witnesses I(rate)
    // and proves that changing an old, spatially varying P changes no result.
    auto baseline=save();auto const baseline_metadata=metadata();auto const last_report=m_solver.m_last_step_projection;
    amrex::Array<amrex::Real,3> const force{0.013,-0.007,0.005};
    prepare("force_probe",force);m_solver.advance_one_step(dt,nu,force);update_identity("force_probe");
    emit_state("perturb_base");auto probe=save();
    for (std::size_t n=0;n<fields.size();++n) {
        auto& f=*fields[n].second;amrex::MultiFab::Copy(f,*baseline[n],0,0,f.nComp(),f.nGrow());
    }
    m_time=baseline_metadata[0];m_step=std::uint64_t(baseline_metadata[1]);m_history_depth=int(baseline_metadata[2]);
    m_valid_epoch=std::uint64_t(baseline_metadata[3]);m_halo_epoch=std::uint64_t(baseline_metadata[4]);
    m_physical_epoch=std::uint64_t(baseline_metadata[5]);m_last_advance_seconds=baseline_metadata[6];m_solver.m_last_step_projection=last_report;
    for (amrex::MFIter mfi(m_p,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const p=m_p.array(mfi);amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            p(i,j,k)+=3.0+0.7*std::sin(w*((i+0.5)*h[0]-(j+0.5)*h[1]))+0.2*std::cos(w*(k+0.5)*h[2]);
        });
    }
    mark_valid_modified();fill_ghost_cells();emit_state("perturb_before");
    m_solver.advance_one_step(dt,nu,force);emit_state("perturb_after");auto const perturb_changes=changes(probe);
    for (std::size_t n=0;n<fields.size();++n) check(perturb_changes[n][1]==0,"old P affected actual advance state");
    emit_state("failed_before");auto failed=save();auto const failed_metadata=metadata();std::string failure;
    m_mapping_operator.deferred_max_iterations=1;
    try {m_solver.advance_one_step(dt,nu);} catch (std::runtime_error const& e) {failure=e.what();}
    m_mapping_operator=cfg;
    check(failure.find("deferred pressure exhausted max_iterations=1")!=std::string::npos,"actual deferred failure witness");
    emit_state("failed_after");auto const failed_changes=changes(failed);auto const failed_after_metadata=metadata();
    check(std::equal(failed_metadata.begin(),failed_metadata.begin()+3,failed_after_metadata.begin()) &&
        failed_metadata[6]==failed_after_metadata[6],"failed counters/timing committed");
    for (std::size_t n=0;n<fields.size();++n) {
        auto const& name=fields[n].first;
        if (name=="P" || name=="Ucat" || name=="Nvert") check(failed_changes[n][1]==0,"failed pressure/R/mask committed");
        if (name.find("Ucont_")==0 && name.find("old")==std::string::npos)
            check(failed_changes[n][0]>0,"failed predictor did not execute");
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        auto nums=[&](auto const& values) {out<<'[';bool f=true;for (auto x:values) {if (!f) out<<',';f=false;out<<x;}out<<']';};
        auto change_json=[&](auto const& values) {out<<'{';for (std::size_t n=0;n<fields.size();++n) {if (n) out<<',';quoted(out,fields[n].first);out<<':';nums(values[n]);}out<<'}';};
        out<<"},\"registered_fields\":[";for (std::size_t n=0;n<fields.size();++n) {if (n) out<<',';quoted(out,fields[n].first);}out<<']'
           <<",\"metadata_order\":[\"time\",\"step\",\"history_depth\",\"valid_epoch\",\"halo_epoch\",\"physical_epoch\",\"last_advance_seconds\"]"
           <<",\"initial_metadata\":";nums(initial_metadata);out<<",\"final_metadata\":";nums(final_metadata);
        out<<",\"failed_metadata_before\":";nums(failed_metadata);out<<",\"failed_metadata_after\":";nums(failed_after_metadata);
        out<<",\"change_norm_order\":[\"valid_max_abs\",\"including_ghost_max_abs\"],\"perturb_changes\":";change_json(perturb_changes);
        out<<",\"failed_changes\":";change_json(failed_changes);out<<",\"failure_reason\":";quoted(out,failure);
        out<<",\"failure_contract\":\"fatal; face predictor and histories may change; no entire-advance rollback\","
           <<"\"momentum_drift\":"<<drift<<",\"history_error\":"<<history_error<<",\"R_error\":"<<recovery
           <<",\"predictor_correction_error\":"<<identity_error<<",\"max_cfl\":"<<max_cfl<<",\"max_diffusion\":"<<max_diffusion
           <<",\"force_probe_body_force\":[0.013,-0.007,0.005],\"initial_projection\":{\"before\":"<<initial_projection.max_divergence_before
           <<",\"after\":"<<initial_projection.max_divergence_after<<",\"residual\":"<<initial_projection.full_relative_residual<<"},\"projections\":[";
        for (std::size_t n=0;n<reports.size();++n) {
            if (n) out<<',';auto const& r=reports[n];
            out<<"{\"before\":"<<r.max_divergence_before<<",\"after\":"<<r.max_divergence_after<<",\"residual\":"<<r.full_relative_residual
               <<",\"outer\":"<<r.outer_iterations<<",\"inner\":"<<r.inner_iterations<<'}';
        }
        out<<"]}\n";out.flush();writable=bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(writable);check(writable,"snapshot final write");
}
