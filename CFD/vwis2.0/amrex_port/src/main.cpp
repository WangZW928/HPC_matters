#include "VwisAmrExSolver.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <exception>
#include <string>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        amrex::ParmParse pp("vwis");
        amrex::Vector<int> n_cell(AMREX_SPACEDIM, 16);
        amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
        pp.queryarr("n_cell", n_cell, 0, AMREX_SPACEDIM);
        pp.queryarr("is_periodic", is_periodic, 0, AMREX_SPACEDIM);

        int max_grid_size = 32;
        int nghost = 2;
        amrex::Real dt = 1.0e-3;
        bool run_contract_checks = false;
        bool run_p2_transform_checks = false;
        bool run_p3_boundary_checks = false;
        std::string metadata_file;
        pp.query("max_grid_size", max_grid_size);
        pp.query("nghost", nghost);
        pp.query("dt", dt);
        pp.query("run_contract_checks", run_contract_checks);
        pp.query("run_p2_transform_checks", run_p2_transform_checks);
        pp.query("run_p3_boundary_checks", run_p3_boundary_checks);
        pp.query("metadata_file", metadata_file);

        amrex::RealBox physical_domain(
            {AMREX_D_DECL(0.0, 0.0, 0.0)},
            {AMREX_D_DECL(1.0, 1.0, 1.0)});
        auto boundary = read_cartesian_boundary_config(is_periodic);
        VwisAmrExSolver solver(n_cell, max_grid_size, nghost,
                                physical_domain, is_periodic, boundary);
        solver.initialize();
        if (run_contract_checks) {
            solver.run_runtime_contract_checks();
        }
        if (run_p2_transform_checks) {
            solver.run_p2_transform_layout_checks();
        }
        if (run_p3_boundary_checks) {
            solver.run_p3_boundary_contract_checks();
        }
        // Explicit P3 no-op: no physical state is advanced.
        solver.advance_one_step(dt);
        if (!metadata_file.empty()) solver.write_metadata_manifest(metadata_file);
        solver.diagnostics();
    } catch (std::exception const& error) {
        amrex::Print() << "VWiS AMReX P3 error: " << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
