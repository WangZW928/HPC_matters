#include "AVWiSContractTestAccess.H"
#include "RunMode.H"

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(std::string("C3 temporal audit: ")+message);
}
amrex::Real maximum(amrex::MultiFab const& f, int c=0) {
    auto r=f.norm0(c,0,true);amrex::ParallelDescriptor::ReduceRealMax(r);
    require(std::isfinite(r),"nonfinite norm");return r;
}
amrex::Real sum(amrex::MultiFab const& f, int c=0) {
    auto r=f.sum(c,true);amrex::ParallelDescriptor::ReduceRealSum(r);
    require(std::isfinite(r),"nonfinite integral");return r;
}
// Test-only canonical valid snapshot. Both periodic face endpoints are retained,
// but Box overlaps are gathered once. Host staging is outside the advance loop.
void field(std::ofstream& out,char const* name,amrex::MultiFab const& f,
           amrex::Box const& cells,bool& first) {
    auto const domain=amrex::convert(cells,f.ixType());
    amrex::DistributionMapping io(amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
    amrex::MultiFab gathered(amrex::BoxArray(domain),io,f.nComp(),0);
    gathered.ParallelCopy(f,0,0,f.nComp(),0,0);amrex::Gpu::streamSynchronize();
    amrex::FArrayBox host(domain,f.nComp(),amrex::The_Pinned_Arena());
    int finite=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Gpu::dtoh_memcpy(host.dataPtr(),gathered[0].dataPtr(),host.size()*sizeof(amrex::Real));
        for (amrex::Long n=0;n<host.size();++n) if (!std::isfinite(host.dataPtr()[n])) finite=0;
    }
    amrex::ParallelDescriptor::ReduceIntMin(finite);
    require(finite==1,"nonfinite snapshot");
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    if (!first) out<<',';
    first=false;out<<'"'<<name<<"\":[";
    for (amrex::Long n=0;n<host.size();++n) {
        if (n) out<<',';
        out<<host.dataPtr()[n];
    }
    out<<']';
}
}

void AVWiSContractTestAccess::run_c3_temporal_contract(
    std::string const& path,amrex::Real dt,amrex::Real nu,int steps)
{
    bool consistent_reference=false;
    amrex::ParmParse(avwis_input_namespace()).query("consistent_reference",consistent_reference);
    bool pressure_reference=false;
    amrex::ParmParse(avwis_input_namespace()).query("pressure_reference",pressure_reference);
    bool face_authority_reference=false;
    amrex::ParmParse(avwis_input_namespace()).query("face_authority_reference",face_authority_reference);
    bool pressure_increment_reference=false;
    amrex::ParmParse(avwis_input_namespace()).query("pressure_increment_reference",pressure_increment_reference);
    require(!(pressure_increment_reference && (consistent_reference || pressure_reference || face_authority_reference)),
            "reference selectors are mutually exclusive");
    if (pressure_increment_reference) { run_pressure_increment_contract(path,dt,nu,steps); return; }
    require(!(face_authority_reference && (consistent_reference || pressure_reference)),
            "reference selectors are mutually exclusive");
    require(!(consistent_reference && pressure_reference),"reference selectors are mutually exclusive");
    if (face_authority_reference) { run_face_authority_contract(path,dt,nu,steps); return; }
    require(m_mapping_operator.state_contract=="experimental_dual_state",
            "legacy temporal/reference requires experimental_dual_state; use face_authority_reference=1");
    if (consistent_reference) { run_consistent_reference(path,dt,nu,steps); return; }
    if (pressure_reference) { run_pressure_reference(path,dt,nu); return; }
    require(m_mapping_operator.mapping_type=="analytic_shear" &&
        std::isfinite(dt) && dt>0 && std::isfinite(nu) && nu>0 && steps>=16 && steps<=4096,
        "requires shear, positive dt/nu and 16..4096 real steps");
    auto const& metric=m_metric_data;auto const config=m_mapping_operator;
    auto const epoch=m_metric_epoch;auto const domain=m_geom.Domain();
    auto const h=metric.logical_grid().spacing;
    constexpr amrex::Real w=6.283185307179586476925286766559;
    // Same mixed xy/xz state as the four-step contract. One common dt=1
    // projection prepares every run identically, including pressure gauge.
    for (amrex::MFIter mfi(m_ucat,amrex::TilingIfNotGPU());mfi.isValid();++mfi) {
        auto const u=m_ucat.array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            auto const a=std::sin(w*((i+0.5)*h[0]+(j+0.5)*h[1]));
            auto const b=std::cos(w*((i+0.5)*h[0]-(k+0.5)*h[2]));
            u(i,j,k,0)=0.35+0.1*a+0.03*b;
            u(i,j,k,1)=0.05*a;u(i,j,k,2)=a+0.07*b;
        });
    }
    mark_valid_modified();sync_ucont_from_ucat();
    auto const initial_projection=m_solver.project_mapped(1.0,1.0);
    require(initial_projection.max_divergence_before>1e-3 &&
        initial_projection.max_divergence_after<=1e-10 &&
        initial_projection.full_relative_residual<=1e-11,"initial projection gate");
    m_p.setVal(0.0);mark_valid_modified();fill_ghost_cells();
    amrex::MultiFab initial(m_ba,m_dm,3,0),previous(m_ba,m_dm,3,0),older(m_ba,m_dm,3,0);
    amrex::MultiFab work(m_ba,m_dm,3,0),divergence(m_ba,m_dm,1,0),alias_div(m_ba,m_dm,1,0);
    amrex::MultiFab::Copy(initial,m_ucat,0,0,3,0);
    amrex::Array<amrex::MultiFab,3> previous_faces,older_faces,interpolated;
    amrex::Array<amrex::MultiFab*,3> alias_out{};
    amrex::Array<amrex::MultiFab const*,3> alias_view{},faces{};
    for (int d=0;d<3;++d) {
        previous_faces[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        older_faces[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        interpolated[d].define(m_ucont[d].boxArray(),m_dm,1,0);
        alias_out[d]=&interpolated[d];alias_view[d]=&interpolated[d];faces[d]=&m_ucont[d];
    }
    auto momentum=[&]() {
        amrex::Array<amrex::Real,3> r{};
        amrex::MultiFab::Copy(work,m_ucat,0,0,3,0);
        for (int c=0;c<3;++c) {
            amrex::MultiFab::Multiply(work,metric.cell_volume_cc(),0,c,1,0);r[c]=sum(work,c);
        }
        return r;
    };
    auto const initial_momentum=momentum();
    sync_metric_ucont_from_ucat(m_ucat,alias_out,metric,epoch,config,m_geom.periodicity());
    compute_metric_divergence(alias_view,metric,epoch,config,alias_div);
    auto const initial_alias_max=maximum(alias_div);
    auto const start_step=m_step;auto const start_time=m_time;
    std::vector<ProjectionDiagnostics> reports;
    std::vector<amrex::Real> phi_magnitudes;
    amrex::Real drift=0,history_error=0,mass_error=0,max_cfl=0,max_diffusion=0;
    amrex::Real elapsed=amrex::second();
    for (int step=0;step<steps;++step) {
        amrex::MultiFab::Copy(previous,m_ucat,0,0,3,0);
        amrex::MultiFab::Copy(older,m_ucat_old,0,0,3,0);
        for (int d=0;d<3;++d) {
            amrex::MultiFab::Copy(previous_faces[d],m_ucont[d],0,0,1,0);
            amrex::MultiFab::Copy(older_faces[d],m_ucont_old[d],0,0,1,0);
        }
        auto const stability=time_step_diagnostics(dt,nu);
        max_cfl=std::max(max_cfl,stability.advective_cfl);
        max_diffusion=std::max(max_diffusion,stability.diffusive_number);
        m_solver.advance_one_step(dt,nu);
        auto const r=m_solver.last_step_projection();reports.push_back(r);
        auto const phi_max=maximum(m_phi);phi_magnitudes.push_back(phi_max);
        require(r.max_divergence_before>1e-8 && phi_max>1e-6,"pressure shortcut in coupled advance");
        require(r.max_divergence_after<=1e-10 && r.full_relative_residual<=1e-11,"per-step residual/divergence gate");
        require_ghosts_fresh("c3-temporal-advance");
        require(m_valid_epoch==m_halo_epoch,"halo epoch");
        amrex::MultiFab::Subtract(previous,m_ucat_old,0,0,3,0);
        amrex::MultiFab::Subtract(older,m_ucat_older,0,0,3,0);
        for (int c=0;c<3;++c) history_error=std::max({history_error,maximum(previous,c),maximum(older,c)});
        for (int d=0;d<3;++d) {
            amrex::MultiFab::Subtract(previous_faces[d],m_ucont_old[d],0,0,1,0);
            amrex::MultiFab::Subtract(older_faces[d],m_ucont_older[d],0,0,1,0);
            history_error=std::max({history_error,maximum(previous_faces[d]),maximum(older_faces[d])});
        }
        compute_metric_divergence(faces,metric,epoch,config,divergence);
        amrex::MultiFab::Multiply(divergence,metric.cell_volume_cc(),0,0,1,0);
        mass_error=std::max(mass_error,std::abs(sum(divergence)));
        auto const now=momentum();
        for (int c=0;c<3;++c) drift=std::max(drift,std::abs(now[c]-initial_momentum[c]));
    }
    elapsed=amrex::second()-elapsed;amrex::ParallelDescriptor::ReduceRealMax(elapsed);
    require(drift<=1e-12 && mass_error<=1e-12 && history_error==0,"momentum/mass/history gate");
    require(m_step==start_step+steps && m_history_depth==3 &&
        std::abs(m_time-start_time-steps*dt)<=1e-13,"time metadata");
    // Observe I(Ucat) without modifying projected Ucont or epochs. This
    // inconsistency is evidence for M2, never an instruction to overwrite it.
    sync_metric_ucont_from_ucat(m_ucat,alias_out,metric,epoch,config,m_geom.periodicity());
    compute_metric_divergence(alias_view,metric,epoch,config,alias_div);
    auto const alias_max=maximum(alias_div);
    compute_metric_divergence(faces,metric,epoch,config,divergence);
    int active_ranks=m_ucat.local_size()>0 ? 1 : 0;
    amrex::ParallelDescriptor::ReduceIntSum(active_ranks);
    std::ofstream out;
    int writable=1;
    if (amrex::ParallelDescriptor::IOProcessor()) {out.open(path);writable=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(writable);
    require(writable==1,"cannot open snapshot");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        out.precision(17);
        out<<"{\"schema\":\"avwis-c3-temporal-v1\",\"invariant_status\":\"PASS\","
           <<"\"amrex_version\":\""<<amrex::Version()<<"\",\"locked_sha\":\""<<AVWIS_LOCKED_GIT_SHA<<"\","
#ifdef AMREX_USE_CUDA
           <<"\"backend\":\"CUDA\","
#else
           <<"\"backend\":\"CPU\","
#endif
           <<"\"ranks\":"<<amrex::ParallelDescriptor::NProcs()<<",\"boxes\":"<<m_ba.size()
           <<",\"active_ranks\":"<<active_ranks<<",\"step\":"<<m_step
           <<",\"scale\":["<<config.analytic_parameters.scale[0]<<','<<config.analytic_parameters.scale[1]<<','<<config.analytic_parameters.scale[2]<<']'
           <<",\"stretch\":["<<config.analytic_parameters.stretch[0]<<','<<config.analytic_parameters.stretch[1]<<','<<config.analytic_parameters.stretch[2]<<']'
           <<",\"n_cell\":["<<domain.length(0)<<','<<domain.length(1)<<','<<domain.length(2)<<']'
           <<",\"epsilon\":"<<config.shear_epsilon<<",\"dt\":"<<dt<<",\"steps\":"<<steps
           <<",\"time\":"<<m_time<<",\"history_depth\":"<<m_history_depth
           <<",\"momentum_drift\":"<<drift<<",\"mass_error\":"<<mass_error
           <<",\"initial_mass_divergence\":"<<initial_projection.max_divergence_after
           <<",\"initial_interpolated_divergence\":"<<initial_alias_max
           <<",\"history_error\":"<<history_error<<",\"max_cfl\":"<<max_cfl
           <<",\"max_diffusion\":"<<max_diffusion<<",\"interpolated_divergence\":"<<alias_max
           <<",\"audited_advance_seconds_max\":"<<elapsed<<",\"projections\":[";
        for (std::size_t n=0;n<reports.size();++n) {
            if (n) out<<',';
            auto const& r=reports[n];
            out<<"{\"outer\":"<<r.outer_iterations<<",\"inner\":"<<r.inner_iterations
               <<",\"phi_max\":"<<phi_magnitudes[n]<<",\"residual\":"<<r.full_relative_residual
               <<",\"before\":"<<r.max_divergence_before<<",\"after\":"<<r.max_divergence_after<<'}';
        }
        out<<"],\"fields\":{";
    }
    bool first=true;
    field(out,"initial_Ucat",initial,domain,first);field(out,"Ucat",m_ucat,domain,first);
    field(out,"P",m_p,domain,first);field(out,"Phi",m_phi,domain,first);
    field(out,"divergence",divergence,domain,first);field(out,"interpolated_divergence",alias_div,domain,first);
    field(out,"cell_volume",metric.cell_volume_cc(),domain,first);
    for (int d=0;d<3;++d) {
        auto const name="Ucont_"+std::to_string(d);field(out,name.c_str(),m_ucont[d],domain,first);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {out<<"}}\n";out.flush();writable=bool(out);}
    amrex::ParallelDescriptor::ReduceIntMin(writable);
    require(writable==1,"snapshot write failed");
}
