#ifndef VWIS_RUN_MODE_H_
#define VWIS_RUN_MODE_H_

#include <AMReX_REAL.H>

#include <string>

namespace amrex {
class ParmParse;
}

class AVWiSSolver;

enum class RunMode {
    Solve,
    ContractBase,
    ContractBaseP2,
    ContractP2,
    ContractP3,
    ContractP4,
    ContractP5Advection,
    ContractP5Viscous,
    ContractP5Time,
    ContractP5Orthogonal,
    ContractP5MappedBoundary,
    ContractP5MappedBoundaryC52,
    ContractP8Restart,
    ContractP8Sampling,
    BenchmarkCartesian,
    BenchmarkPhysicalChannel,
    CaseLidCavity
};

struct RunConfiguration {
    RunMode mode;
    amrex::Real dt = 1.0e-3;
    int run_steps = 0;
    int restart_total_steps = 8;
    int restart_checkpoint_step = 3;
    amrex::Real viscosity = 0.1;
    amrex::Real final_time = 8.0e-3;
    amrex::Real projection_time_coefficient = 1.0;
    std::string metadata_file;
    std::string checkpoint_file;
    std::string restart_file;
    std::string plane_file;
    std::string field_file;
    std::string centerline_file;
    std::string history_file;
};

RunConfiguration read_run_configuration(amrex::ParmParse& pp);
/** Select the new product namespace, falling back to legacy `vwis.*` inputs. */
char const* avwis_input_namespace();
void dispatch_run_mode(AVWiSSolver& solver, RunConfiguration const& config);

#endif
