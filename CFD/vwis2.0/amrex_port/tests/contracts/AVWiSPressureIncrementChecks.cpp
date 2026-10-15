#include "AVWiSContractTestAccess.H"
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, char const* why) {
    if (!ok) throw std::runtime_error(std::string("pressure increment audit: ")+why);
}
amrex::Real maximum(amrex::MultiFab const& f, int ghosts=0) {
    amrex::Real r=0;
    for (int c=0;c<f.nComp();++c) {
        auto v=f.norm0(c,ghosts,true);amrex::ParallelDescriptor::ReduceRealMax(v);r=std::max(r,v);
    }
    check(std::isfinite(r),"finite norm");return r;
}
void dump(std::ofstream& out,amrex::MultiFab const& f,amrex::Box const& cells) {
    amrex::DistributionMapping io(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
    amrex::MultiFab gathered(amrex::BoxArray(cells),io,1,0);
    gathered.ParallelCopy(f,0,0,1,0,0);amrex::Gpu::streamSynchronize();
    int ok=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::FArrayBox host(cells,1,amrex::The_Pinned_Arena());
        amrex::Gpu::dtoh_memcpy(host.dataPtr(),gathered[0].dataPtr(),host.size()*sizeof(amrex::Real));
        out<<'[';
        for (amrex::Long n=0;n<host.size();++n) {if(n) out<<',';out<<host.dataPtr()[n];}
        out<<']';ok=bool(out);
    }
    amrex::ParallelDescriptor::ReduceIntMin(ok);check(ok,"snapshot write");
}
}

void AVWiSContractTestAccess::run_pressure_increment_contract(
    std::string const& path,amrex::Real dt,amrex::Real nu,int steps)
{
    check(m_mapping_operator.pressure_contract=="incremental_pressure" &&
          m_solver.m_pressure_increment && dt>0 && nu>0 && steps>=2 && steps<=64,
          "requires incremental_pressure and 2..64 steps");
    auto const domain=m_geom.Domain();auto const h=m_metric_data.logical_grid().spacing;
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
    // Initialization projection is a velocity operation, not a time increment.
    // Restore the requested contract and explicitly start P^0=Phi^0=0.
    ProjectionDiagnostics initial;
    bool const shear=m_mapping_operator.mapping_type=="analytic_shear";
    if (shear) {
        m_mapping_operator.pressure_contract="instantaneous_pressure";
        initial=m_solver.project_mapped(1.0);
        m_mapping_operator.pressure_contract="incremental_pressure";
    } else {
        m_mapping_operator.state_contract="experimental_dual_state";
        m_mapping_operator.pressure_contract="accumulated_correction";
        initial=m_mapping_operator.coordinates==CoordinateSystemMode::Cartesian
            ? m_solver.project_cartesian(1.0) : m_solver.project_mapped(1.0);
        m_mapping_operator.state_contract="face_authoritative";
        m_mapping_operator.pressure_contract="incremental_pressure";
        m_solver.sync_ucat_from_ucont();
    }
    check(initial.max_divergence_after<=1e-10,"consistent initial faces");
    m_p.setVal(0);m_phi.setVal(0);mark_valid_modified();fill_ghost_cells();
    amrex::MultiFab previous_p(m_ba,m_dm,1,0),delta(m_ba,m_dm,1,0),weighted(m_ba,m_dm,3,0);
    auto momentum=[&]() {
        amrex::MultiFab::Copy(weighted,m_ucat,0,0,3,0);
        std::array<amrex::Real,3> r{};
        for (int c=0;c<3;++c) {
            amrex::MultiFab::Multiply(weighted,m_metric_data.cell_volume_cc(),0,c,1,0);
            r[c]=weighted.sum(c,true);amrex::ParallelDescriptor::ReduceRealSum(r[c]);
        }return r;
    };
    auto const initial_momentum=momentum();
    std::ofstream out;int ok=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {out.open(path);out.precision(17);ok=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(ok);check(ok,"report open");
    if (amrex::ParallelDescriptor::IOProcessor())
        out<<"{\"schema\":\"avwis-pressure-increment-v1\",\"physical_pressure_gate\":\"NOT_TESTED\","
           <<"\"dt\":"<<dt<<",\"steps\":"<<steps<<",\"series\":[";
    amrex::Real drift=0,max_div=0,max_residual=0,update_error=0;
    for (int n=0;n<steps;++n) {
        amrex::MultiFab::Copy(previous_p,m_p,0,0,1,0);
        m_solver.advance_one_step(dt,nu);
        amrex::MultiFab::Copy(delta,previous_p,0,0,1,0);
        amrex::MultiFab::Add(delta,m_phi,0,0,1,0);
        amrex::MultiFab::Subtract(delta,m_p,0,0,1,0);
        update_error=std::max(update_error,maximum(delta));check(update_error==0,"P_old+Phi exactly once");
        if (n==0) check(maximum(previous_p)==0 && maximum(m_phi)>1e-6,"P^0=0 and nontrivial first increment");
        auto const& r=m_solver.last_step_projection();
        max_div=std::max(max_div,r.max_divergence_after);max_residual=std::max(max_residual,r.full_relative_residual);
        auto const m=momentum();for(int c=0;c<3;++c) drift=std::max(drift,std::abs(m[c]-initial_momentum[c]));
        if (amrex::ParallelDescriptor::IOProcessor()) {if(n)out<<',';out<<"{\"P\":";}
        dump(out,m_p,domain);
        if (amrex::ParallelDescriptor::IOProcessor()) out<<",\"Phi\":";
        dump(out,m_phi,domain);
        if (amrex::ParallelDescriptor::IOProcessor()) out<<'}';
    }
    amrex::Print()<<"Pressure increment validation: momentum_drift="<<drift
        <<" max_divergence="<<max_div<<" full_relative_residual="<<max_residual<<'\n';
    check(drift<=1e-12 && max_div<=1e-10 && max_residual<=1e-11,"momentum/divergence/full residual");
    // A repeated standalone projection cannot replay the last accepted Phi.
    // Failure is injected into a REAL advance after nonzero P has been committed.
    std::vector<amrex::MultiFab*> fields{&m_p,&m_phi,&m_ucat,&m_ucat_old,&m_ucat_older,&m_nvert};
    for(int d=0;d<3;++d) for(auto* f:{&m_ucont[d],&m_ucont_old[d],&m_ucont_older[d]}) fields.push_back(f);
    auto save=[&]() {
        std::vector<std::unique_ptr<amrex::MultiFab>> v;
        for(auto* f:fields) {
            v.push_back(std::make_unique<amrex::MultiFab>(f->boxArray(),m_dm,f->nComp(),f->nGrow()));
            amrex::MultiFab::Copy(*v.back(),*f,0,0,f->nComp(),f->nGrow());
        }return v;
    };
    auto const saved=save();
    auto unchanged=[&]() {
        for(std::size_t n=0;n<fields.size();++n) {
            auto& f=*fields[n];amrex::MultiFab v(f.boxArray(),m_dm,f.nComp(),f.nGrow());
            amrex::MultiFab::Copy(v,f,0,0,f.nComp(),f.nGrow());
            amrex::MultiFab::Subtract(v,*saved[n],0,0,f.nComp(),f.nGrow());
            check(maximum(v,f.nGrow())==0,"failed/replayed pressure or velocity/history published");
        }
    };
    auto const time=m_time;auto const step=m_step;auto const depth=m_history_depth;
    auto const valid=m_valid_epoch;auto const halo=m_halo_epoch;auto const physical=m_physical_epoch;
    auto const timing=m_last_advance_seconds;auto const last=m_solver.last_step_projection();
    // Invalid spatial loads must be rejected before predictor/history writes.
    // Reuse the existing full-state witness rather than adding another mode.
    amrex::MultiFab bad_components(m_ba,m_dm,2,0);
    bad_components.setVal(0);
    amrex::MultiFab nonfinite_force(m_ba,m_dm,3,0);
    nonfinite_force.setVal(std::numeric_limits<amrex::Real>::quiet_NaN());
    for (auto const* force : {&bad_components, &nonfinite_force}) {
        std::string reason;
        try {m_solver.advance_one_step(dt,nu,{},force);}
        catch(std::exception const& e) {reason=e.what();}
        check(reason.find("spatial body force must be finite cell acceleration")!=std::string::npos,
              "spatial load rejected before publication");
        unchanged();
    }
    for(int repeat=0;repeat<2;++repeat) {
        std::string reason;try {m_solver.project_mapped(dt);}catch(std::exception const& e){reason=e.what();}
        check(reason.find("standalone projection cannot commit")!=std::string::npos,"standalone replay rejection");unchanged();
    }
    auto& iteration_budget=shear ? m_mapping_operator.deferred_max_iterations
                                 : m_mapping_operator.diagonal_max_iterations;
    auto const budget=iteration_budget;
    iteration_budget=1;
    std::string failure;try {m_solver.advance_one_step(dt,nu);}catch(std::exception const& e){failure=e.what();}
    iteration_budget=budget;
    check(failure.find(shear ? "deferred pressure exhausted max_iterations=1"
                             : "incremental diagonal solve failed")!=std::string::npos,"actual solver failure witness");
    unchanged();
    check(m_time==time && m_step==step && m_history_depth==depth && m_valid_epoch==valid &&
        m_halo_epoch==halo && m_physical_epoch==physical && m_last_advance_seconds==timing &&
        m_solver.last_step_projection().full_relative_residual==last.full_relative_residual &&
        m_solver.m_pressure_increment->phase==AVWiSSolver::PressureIncrementPhase::Ready,
        "failed metadata/phase commit");
    // Dirty scratch is rebuilt; a valid subsequent advance commits exactly once.
    m_solver.advance_one_step(dt,nu);check(m_step==step+1 && m_time==time+dt,"retry accepted exactly once");
    amrex::MultiFab::Copy(delta,*saved[0],0,0,1,0);amrex::MultiFab::Add(delta,m_phi,0,0,1,0);
    amrex::MultiFab::Subtract(delta,m_p,0,0,1,0);check(maximum(delta)==0,"retry pressure update");
    // With B=0, a nonzero previous P is cancelled by Phi=-P. A second
    // zero step must keep P=Phi=0, never reapply the previous increment.
    for(auto* f:{&m_ucat,&m_ucat_old,&m_ucat_older}) f->setVal(0);
    for(int d=0;d<3;++d) for(auto* f:{&m_ucont[d],&m_ucont_old[d],&m_ucont_older[d]}) f->setVal(0);
    amrex::MultiFab::Copy(previous_p,m_p,0,0,1,0);mark_valid_modified();fill_ghost_cells();
    m_solver.advance_one_step(dt,nu);
    amrex::MultiFab::Copy(delta,previous_p,0,0,1,0);amrex::MultiFab::Add(delta,m_phi,0,0,1,0);
    check(maximum(delta)==0 && maximum(m_p)==0,"zero RHS cancels old predictor pressure");
    m_solver.advance_one_step(dt,nu);check(maximum(m_p)==0 && maximum(m_phi)==0,"zero update idempotence");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out<<"],\"update_error\":"<<update_error<<",\"momentum_drift\":"<<drift
           <<",\"max_divergence\":"<<max_div<<",\"max_full_relative_residual\":"<<max_residual
           <<",\"first_step_P_equals_Phi\":true,\"standalone_replay_rejected\":true,"
           <<"\"failed_advance_all_state_exact\":true,\"retry_commits_once\":true,"
           <<"\"zero_update_idempotent\":true,\"status\":\"PASS\"}\n";
    }
}
