#include "RunMode.H"

#include "CartesianBenchmarkCase.H"
#include "LidDrivenCavityCase.H"
#include "MappedPressureWaveCase.H"
#include "PhysicalChannelCase.H"
#include "MappedChannelReferenceCase.H"
#include "TaylorGreenMMSCase.H"
#include "AVWiSSolver.H"
#include "AVWiSContractRunner.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
bool use_legacy_input_namespace()
{
    amrex::ParmParse current("avwis");
    if (current.contains("mode")) return false;
    amrex::ParmParse legacy("vwis");
    return legacy.contains("mode");
}

RunMode parse_run_mode(amrex::ParmParse& pp)
{
    std::string name;
    if (!pp.query("mode", name)) {
        throw std::runtime_error(
            "avwis.mode is required; expected solve, contract_base, contract_base_p2, "
            "contract_p2, contract_p3, contract_p4, contract_p5_advection, contract_p5_viscous, "
            "contract_p5_time, contract_p5_orthogonal, contract_p5_mapped_boundary, "
            "contract_p5_orthogonal_time, contract_p5_mapped_boundary_c52, "
            "contract_p8_restart, contract_p8_sampling, contract_c7_backend, contract_c3_shear, contract_c3_temporal, "
            "benchmark_cartesian, benchmark_cartesian_mms, benchmark_physical_channel, "
            "benchmark_mapped_channel_reference, case_lid_cavity, or case_mapped_pressure_wave");
    }

    static const std::unordered_map<std::string, RunMode> modes = {
        {"solve", RunMode::Solve},
        {"contract_base", RunMode::ContractBase},
        {"contract_base_p2", RunMode::ContractBaseP2},
        {"contract_p2", RunMode::ContractP2},
        {"contract_p3", RunMode::ContractP3},
        {"contract_p4", RunMode::ContractP4},
        {"contract_p5_advection", RunMode::ContractP5Advection},
        {"contract_p5_viscous", RunMode::ContractP5Viscous},
        {"contract_p5_time", RunMode::ContractP5Time},
        {"contract_p5_orthogonal", RunMode::ContractP5Orthogonal},
        {"contract_p5_orthogonal_time", RunMode::ContractP5OrthogonalTime},
        {"contract_p5_mapped_boundary", RunMode::ContractP5MappedBoundary},
        {"contract_p5_mapped_boundary_c52", RunMode::ContractP5MappedBoundaryC52},
        {"contract_p8_restart", RunMode::ContractP8Restart},
        {"contract_p8_sampling", RunMode::ContractP8Sampling},
        {"contract_c7_backend", RunMode::ContractC7Backend},
        {"contract_c3_shear", RunMode::ContractC3Shear},
        {"contract_c3_temporal", RunMode::ContractC3Temporal},
        {"benchmark_cartesian", RunMode::BenchmarkCartesian},
        {"benchmark_cartesian_mms", RunMode::BenchmarkCartesianMMS},
        {"benchmark_physical_channel", RunMode::BenchmarkPhysicalChannel},
        {"benchmark_mapped_channel_reference", RunMode::BenchmarkMappedChannelReference},
        {"case_lid_cavity", RunMode::CaseLidCavity},
        {"case_mapped_pressure_wave", RunMode::CaseMappedPressureWave},
    };
    const auto found = modes.find(name);
    if (found == modes.end()) {
        throw std::runtime_error("unknown avwis.mode '" + name + "'");
    }
    return found->second;
}

void require_file(std::string const& value, char const* message)
{
    if (value.empty()) throw std::runtime_error(message);
}

// The formal solve lifecycle is intentionally separate from mode dispatch:
// cases, benchmarks, and contracts own their complete stepping/output flows.
void run_solve_loop(AVWiSSolver& solver, RunConfiguration const& config)
{
    for (int step = 0; step < config.run_steps; ++step) {
        solver.advance_one_step(config.dt, config.viscosity, config.body_force);
    }
    if (!config.checkpoint_file.empty()) {
        solver.write_checkpoint(config.checkpoint_file);
    }
    if (!config.metadata_file.empty()) {
        solver.write_metadata_manifest(config.metadata_file);
    }
}
} // namespace

char const* avwis_input_namespace()
{
    static const bool legacy = use_legacy_input_namespace();
    static const bool warned = [] {
        if (use_legacy_input_namespace()) {
            amrex::Print() << "WARNING: legacy vwis.* input namespace is deprecated; use avwis.*\n";
        }
        return true;
    }();
    (void)warned;
    return legacy ? "vwis" : "avwis";
}

RunConfiguration read_run_configuration(amrex::ParmParse& pp)
{
    RunConfiguration config{parse_run_mode(pp)};
    std::string integrator = "explicit_euler";
    std::string viscous_operator = "component_laplacian";
    pp.query("time_integrator", integrator);
    pp.query("viscous_operator", viscous_operator);
    if (integrator != "explicit_euler")
        throw std::runtime_error("P5-005 time_integrator supports only explicit_euler; "
                                 "SNES/semi-implicit/BDF2 residual contracts remain open");
    if (viscous_operator != "component_laplacian")
        throw std::runtime_error("M6 viscous_operator supports only component_laplacian; "
                                 "full stress divergence remains open");
    pp.query("dt", config.dt);
    pp.query("run_steps", config.run_steps);
    pp.query("restart_total_steps", config.restart_total_steps);
    pp.query("restart_checkpoint_step", config.restart_checkpoint_step);
    pp.query("viscosity", config.viscosity);
    pp.query("final_time", config.final_time);
    pp.query("projection_time_coefficient", config.projection_time_coefficient);
    // Only these standalone projection contracts consume a configurable alpha.
    // Every advance uses explicit Euler's alpha=1; accepting another value here
    // would silently ignore a requested time scheme (e.g. BDF2's 3/2).
    bool const configurable_projection =
        config.mode == RunMode::ContractP4 ||
        config.mode == RunMode::ContractP5Orthogonal ||
        config.mode == RunMode::ContractP5MappedBoundary ||
        config.mode == RunMode::ContractP5MappedBoundaryC52;
    if (!configurable_projection && config.projection_time_coefficient != 1.0)
        throw std::runtime_error(
            "P5-005 projection_time_coefficient must be 1 outside standalone projection contracts; "
            "explicit_euler fixes alpha=1, SNES/semi-implicit/BDF2 remain open");
    amrex::Vector<amrex::Real> body_force(
        config.body_force.begin(), config.body_force.end());
    pp.queryarr("body_force", body_force, 0, AMREX_SPACEDIM);
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        config.body_force[comp] = body_force[comp];
    }
    pp.query("metadata_file", config.metadata_file);
    pp.query("checkpoint_file", config.checkpoint_file);
    pp.query("restart_file", config.restart_file);
    pp.query("plane_file", config.plane_file);
    pp.query("field_file", config.field_file);
    pp.query("centerline_file", config.centerline_file);
    pp.query("history_file", config.history_file);
    return config;
}

void dispatch_run_mode(AVWiSSolver& solver, RunConfiguration const& config)
{
    switch (config.mode) {
    case RunMode::CaseMappedPressureWave:
        require_file(config.metadata_file, "mapped pressure wave requires avwis.metadata_file");
        if (!config.restart_file.empty() || !config.checkpoint_file.empty())
            throw std::runtime_error("mapped pressure wave initializes its analytic state; checkpoint/restart is not supported");
        for (auto value : config.body_force)
            if (value != 0)
                throw std::runtime_error("mapped pressure wave supplies analytic forcing; uniform body_force must be zero");
        run_mapped_pressure_wave_case(solver, config.dt, config.final_time,
                                     config.viscosity, config.metadata_file, config.field_file);
        return;
    case RunMode::ContractC3Temporal:
        require_file(config.metadata_file,"C3 temporal audit requires metadata_file");
        AVWiSContractRunner(solver).run_c3_temporal_contract(
            config.metadata_file,config.dt,config.viscosity,config.run_steps);
        return;
    case RunMode::ContractC3Shear:
        {
        require_file(config.metadata_file,"C3 shear contract requires metadata_file");
        AVWiSContractRunner(solver).run_c3_shear_contract(config.metadata_file,config.dt,config.viscosity);
        return;
        }
    case RunMode::ContractC7Backend:
        {
        require_file(config.metadata_file,
                     "avwis.metadata_file is required for the C7 backend contract");
        AVWiSContractRunner contracts(solver);
        contracts.run_c7_backend_contract(config.metadata_file, config.dt, config.viscosity);
        return;
        }
    case RunMode::Solve:
        run_solve_loop(solver, config);
        return;
    case RunMode::ContractBase:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_runtime_contract_checks();
        return;
        }
    case RunMode::ContractBaseP2:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_runtime_contract_checks();
        contracts.run_p2_transform_layout_checks();
        return;
        }
    case RunMode::ContractP2:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p2_transform_layout_checks();
        return;
        }
    case RunMode::ContractP3:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p3_boundary_contract_checks();
        return;
        }
    case RunMode::ContractP4:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p4_projection_contract_checks(config.dt, config.projection_time_coefficient);
        return;
        }
    case RunMode::ContractP5Advection:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_advection_contract_checks();
        return;
        }
    case RunMode::ContractP5Viscous:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_viscous_contract_checks(config.viscosity);
        return;
        }
    case RunMode::ContractP5Time:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_time_contract_checks(config.dt, config.final_time, config.viscosity);
        return;
        }
    case RunMode::ContractP5Orthogonal:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_orthogonal_projection_contract_checks(
            config.dt, config.projection_time_coefficient);
        return;
        }
    case RunMode::ContractP5OrthogonalTime:
        {
        require_file(config.metadata_file,
                     "avwis.metadata_file is required for the C2.3 mapped time contract");
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_orthogonal_time_contract_checks(
            config.metadata_file, config.dt, config.viscosity);
        return;
        }
    case RunMode::ContractP5MappedBoundary:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_mapped_boundary_contract_checks(
            config.dt, config.projection_time_coefficient);
        return;
        }
    case RunMode::ContractP5MappedBoundaryC52:
        {
        AVWiSContractRunner contracts(solver);
        contracts.run_p5_mapped_boundary_c52_contract_checks(
            config.dt, config.projection_time_coefficient);
        return;
        }
    case RunMode::ContractP8Restart:
        {
        require_file(config.checkpoint_file,
                     "avwis.checkpoint_file is required for the P8 restart contract");
        AVWiSContractRunner contracts(solver);
        contracts.run_p8_restart_contract_checks(
            config.checkpoint_file, config.dt, config.restart_total_steps,
            config.restart_checkpoint_step, config.viscosity);
        // Preserve the old optional post-contract manifest without invoking
        // the solve loop or duplicating the P8 contract's checkpoint writes.
        if (!config.metadata_file.empty()) {
            solver.write_metadata_manifest(config.metadata_file);
        }
        return;
        }
    case RunMode::ContractP8Sampling:
        {
        if (config.metadata_file.empty() || config.plane_file.empty()) {
            throw std::runtime_error(
                "avwis.metadata_file and avwis.plane_file are required for the P8 sampling contract");
        }
        AVWiSContractRunner contracts(solver);
        contracts.run_p8_sampling_statistics_contract(config.metadata_file, config.plane_file);
        return;
        }
    case RunMode::BenchmarkCartesian:
        require_file(config.metadata_file,
                     "avwis.metadata_file is required for the Cartesian benchmark");
        run_cartesian_benchmark(solver, config.dt, config.run_steps, config.viscosity,
                                config.metadata_file);
        return;
    case RunMode::BenchmarkCartesianMMS:
        require_file(config.metadata_file,
                     "avwis.metadata_file is required for the Cartesian MMS benchmark");
        run_taylor_green_mms(solver, config.dt, config.final_time,
                             config.viscosity, config.metadata_file);
        return;
    case RunMode::BenchmarkPhysicalChannel:
        require_file(config.metadata_file,
                     "avwis.metadata_file is required for the physical benchmark");
        run_physical_channel_benchmark(solver, config.dt, config.run_steps,
                                       config.viscosity, config.metadata_file);
        return;
    case RunMode::BenchmarkMappedChannelReference:
        if (config.metadata_file.empty() || config.plane_file.empty()) {
            throw std::runtime_error(
                "mapped channel reference requires avwis.metadata_file and plane_file");
        }
        run_mapped_channel_reference(
            solver, config.dt, config.final_time, config.viscosity,
            config.body_force, config.metadata_file, config.plane_file);
        return;
    case RunMode::CaseLidCavity:
        if (config.metadata_file.empty() || config.field_file.empty() ||
            config.centerline_file.empty() || config.history_file.empty()) {
            throw std::runtime_error(
                "lid cavity requires avwis.metadata_file, field_file, centerline_file, and history_file");
        }
        run_lid_driven_cavity_case(
            solver, config.dt, config.run_steps, config.viscosity, config.metadata_file,
            config.field_file, config.centerline_file, config.history_file);
        return;
    }

    throw std::runtime_error("unhandled vwis run mode");
}
