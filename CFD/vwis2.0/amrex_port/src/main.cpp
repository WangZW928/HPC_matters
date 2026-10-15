#include "RunMode.H"
#include "AVWiSSolver.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <exception>
#include <stdexcept>
#include <string>

namespace {
MappingOperatorConfig read_mapping_operator_config(amrex::ParmParse& pp)
{
    MappingOperatorConfig config;
    std::string coordinates = "cartesian";
    std::string projection = "cartesian_mlmg";
    pp.query("coordinates", coordinates);
    config.coordinates = parse_coordinate_system_mode(coordinates);

    std::string const root = avwis_input_namespace();
    amrex::ParmParse mapping_pp(root + ".mapping");
    amrex::ParmParse projection_pp(root + ".projection");
    mapping_pp.query("type", config.mapping_type);
    projection_pp.query("operator", projection);
    mapping_pp.query("epsilon", config.shear_epsilon);
    projection_pp.query("max_iterations", config.deferred_max_iterations);
    projection_pp.query("diagonal_max_iterations", config.diagonal_max_iterations);
    config.projection = parse_projection_operator_mode(projection);

    // An explicit opt-in chooses face-authority evolution. Pressure selection
    // names the discrete multiplier's time semantics, not physical validation.
    pp.query("state_contract", config.state_contract);
    pp.query("pressure_contract", config.pressure_contract);
    pp.query("failure_contract", config.failure_contract);
    if (config.mapping_type == "analytic_shear") {
        if (config.state_contract != "experimental_dual_state" &&
            config.state_contract != "face_authoritative")
            throw std::runtime_error(
                "M2 analytic_shear state_contract supports experimental_dual_state or face_authoritative; "
                "time-converged physical pressure remain open");
    }

    amrex::Vector<amrex::Real> scale(AMREX_SPACEDIM, 1.0);
    amrex::Vector<amrex::Real> stretch(AMREX_SPACEDIM, 0.0);
    mapping_pp.queryarr("scale", scale, 0, AMREX_SPACEDIM);
    mapping_pp.queryarr("stretch", stretch, 0, AMREX_SPACEDIM);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        config.analytic_parameters.scale[dir] = scale[dir];
        config.analytic_parameters.stretch[dir] = stretch[dir];
    }
    return config;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        amrex::ParmParse pp(avwis_input_namespace());
        const RunConfiguration run = read_run_configuration(pp);

        amrex::Vector<int> n_cell(AMREX_SPACEDIM, 16);
        amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
        pp.queryarr("n_cell", n_cell, 0, AMREX_SPACEDIM);
        pp.queryarr("is_periodic", is_periodic, 0, AMREX_SPACEDIM);

        int max_grid_size = 32;
        int nghost = 2;
        pp.query("max_grid_size", max_grid_size);
        pp.query("nghost", nghost);

        amrex::RealBox physical_domain(
            {AMREX_D_DECL(0.0, 0.0, 0.0)},
            {AMREX_D_DECL(1.0, 1.0, 1.0)});
        auto boundary = read_cartesian_boundary_config(is_periodic);
        auto mapping_operator = read_mapping_operator_config(pp);
        if (mapping_operator.state_contract == "face_authoritative" &&
            run.mode != RunMode::Solve && run.mode != RunMode::ContractC3Temporal &&
            run.mode != RunMode::CaseMappedPressureWave)
            throw std::runtime_error("M2 face_authoritative supports solve or contract_c3_temporal, or case_mapped_pressure_wave");
        AVWiSSolver solver(n_cell, max_grid_size, nghost,
                           physical_domain, is_periodic, boundary, mapping_operator);
        if (run.restart_file.empty()) solver.initialize();
        else solver.read_checkpoint(run.restart_file);

        dispatch_run_mode(solver, run);
        solver.diagnostics();
    } catch (std::exception const& error) {
        amrex::Print() << "AVWiS port error: " << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
