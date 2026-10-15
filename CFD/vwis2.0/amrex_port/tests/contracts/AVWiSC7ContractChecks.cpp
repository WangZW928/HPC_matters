#include "AVWiSContractTestAccess.H"

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
// A test-only snapshot: gather valid values into one canonical domain Box on
// the IO rank. This is deliberately not a production MPI/GPU I/O interface.
// Nodal face domains include both periodic endpoints but no shared Box faces.
void write_field(std::ofstream& output, char const* name,
                 amrex::MultiFab const& source, amrex::Box const& cell_domain,
                 bool& first)
{
    const amrex::Box domain = amrex::convert(cell_domain, source.ixType());
    amrex::DistributionMapping io_mapping(
        amrex::Vector<int>{amrex::ParallelDescriptor::IOProcessorNumber()});
    amrex::MultiFab gathered(amrex::BoxArray(domain), io_mapping, source.nComp(), 0);
    gathered.ParallelCopy(source, 0, 0, source.nComp(), 0, 0);
    amrex::Gpu::streamSynchronize();
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    amrex::FArrayBox host(domain, source.nComp(), amrex::The_Pinned_Arena());
    amrex::Gpu::dtoh_memcpy(host.dataPtr(), gathered[0].dataPtr(),
                          host.size() * sizeof(amrex::Real));
    if (!first) output << ",\n";
    first = false;
    output << "    \"" << name << "\": [";
    // AMReX FAB storage order is i,j,k,component; this order is independent of
    // rank count, tiling, and Box decomposition and preserves every value.
    for (amrex::Long index = 0; index < host.size(); ++index) {
        const amrex::Real value = host.dataPtr()[index];
        if (!std::isfinite(value)) throw std::runtime_error("C7 snapshot contains nonfinite values");
        if (index) output << ',';
        output << value;
    }
    output << ']';
}
}

void AVWiSContractTestAccess::run_c7_backend_contract(
    std::string const& report_path, amrex::Real dt, amrex::Real viscosity)
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Mapped ||
        m_metric_data.mapping_id() != "analytic_orthogonal" ||
        !std::isfinite(dt) || dt <= 0.0 || !std::isfinite(viscosity) || viscosity <= 0.0 ||
        report_path.empty()) {
        throw std::runtime_error("C7 requires analytic orthogonal mapping, positive dt/viscosity and report path");
    }
    const auto domain = m_geom.Domain();
    constexpr amrex::Real two_pi = 6.283185307179586476925286766559;
    const int nx = domain.length(0), ny = domain.length(1), nz = domain.length(2);
    const int ilo = domain.smallEnd(0), jlo = domain.smallEnd(1), klo = domain.smallEnd(2);
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto u = m_ucat.array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = two_pi * (i-ilo+0.5)/nx;
            const amrex::Real y = two_pi * (j-jlo+0.5)/ny;
            const amrex::Real z = two_pi * (k-klo+0.5)/nz;
            u(i,j,k,0) = 0.35 + 0.1*std::sin(x)*std::cos(y);
            u(i,j,k,1) = -0.1*std::cos(x)*std::sin(y);
            u(i,j,k,2) = 0.05*std::sin(z)*std::cos(x);
        });
    }
    amrex::Gpu::streamSynchronize();
    mark_valid_modified();
    sync_ucont_from_ucat();
    const auto initial_projection = project_orthogonal(dt, 1.0);
    if (!(initial_projection.max_divergence_before > 1.0e-3) ||
        !(initial_projection.max_divergence_after < 1.0e-10)) {
        throw std::runtime_error("C7 initial nontrivial projection gate failed");
    }

    amrex::MultiFab advection(m_ba, m_dm, 3, 0), viscous(m_ba, m_dm, 3, 0);
    compute_advection_rhs(advection);
    compute_viscous_rhs(viscous, viscosity);
    amrex::Real elapsed = amrex::second();
    for (int step = 0; step < 3; ++step) advance_one_step(dt, viscosity);
    elapsed = amrex::second()-elapsed;
    amrex::ParallelDescriptor::ReduceRealMax(elapsed);
    const auto diagnostic = p3_diagnostics("C7 snapshot", true);
    if (!(diagnostic.max_abs_divergence < 1.0e-10) ||
        std::abs(diagnostic.integrated_divergence-diagnostic.outward_boundary_flux) > 1.0e-12 ||
        m_step != 3 || m_history_depth != 3) {
        throw std::runtime_error("C7 advance/mass/history gate failed");
    }
    const auto stability = time_step_diagnostics(dt, viscosity);
    amrex::MultiFab divergence(m_ba, m_dm, 1, 0);
    compute_cartesian_divergence(divergence);
    int active_ranks = m_ucat.local_size() > 0 ? 1 : 0;
    amrex::ParallelDescriptor::ReduceIntSum(active_ranks);

    std::ofstream output;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output.open(report_path);
        if (!output) throw std::runtime_error("cannot write C7 snapshot: " + report_path);
        output.precision(std::numeric_limits<amrex::Real>::max_digits10);
        output << "{\n  \"schema\": \"avwis-c7-backend-v1\",\n"
               << "  \"amrex_version\": \"" << amrex::Version() << "\",\n"
               << "  \"locked_sha\": \"" << AVWIS_LOCKED_GIT_SHA << "\",\n"
#ifdef AMREX_USE_CUDA
               << "  \"backend\": \"CUDA\",\n"
#else
               << "  \"backend\": \"CPU\",\n"
#endif
               << "  \"nprocs\": " << amrex::ParallelDescriptor::NProcs() << ",\n"
               << "  \"active_ranks\": " << active_ranks << ",\n"
               << "  \"boxes\": " << m_ba.size() << ",\n"
               << "  \"n_cell\": [" << nx << ',' << ny << ',' << nz << "],\n"
               << "  \"step\": " << m_step << ", \"history_depth\": " << m_history_depth << ",\n"
               << "  \"time\": " << m_time << ",\n"
               << "  \"advance_seconds_max\": " << elapsed << ",\n"
               << "  \"reductions\": [" << initial_projection.max_divergence_before << ','
               << initial_projection.max_divergence_after << ','
               << diagnostic.max_abs_divergence << ',' << diagnostic.integrated_divergence << ','
               << diagnostic.outward_boundary_flux << ',' << stability.advective_cfl << ','
               << stability.diffusive_number << "],\n  \"fields\": {\n";
    }
    bool first = true;
    write_field(output, "Ucat", m_ucat, domain, first);
    write_field(output, "Ucat_old", m_ucat_old, domain, first);
    write_field(output, "Ucat_older", m_ucat_older, domain, first);
    write_field(output, "P", m_p, domain, first);
    write_field(output, "Phi", m_phi, domain, first);
    write_field(output, "advection_initial", advection, domain, first);
    write_field(output, "viscous_initial", viscous, domain, first);
    write_field(output, "divergence", divergence, domain, first);
    write_field(output, "cell_volume", m_metric_data.cell_volume_cc(), domain, first);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const std::string suffix = std::to_string(dir);
        write_field(output, ("Ucont_"+suffix).c_str(), m_ucont[dir], domain, first);
        write_field(output, ("Ucont_old_"+suffix).c_str(), m_ucont_old[dir], domain, first);
        write_field(output, ("Ucont_older_"+suffix).c_str(), m_ucont_older[dir], domain, first);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        output << "\n  }\n}\n";
        output.close();
        if (!output) throw std::runtime_error("C7 snapshot write failed");
    }
    amrex::Print() << "AVWiS C7 backend snapshot: PASS ranks="
                   << amrex::ParallelDescriptor::NProcs() << " report=" << report_path << '\n';
}
