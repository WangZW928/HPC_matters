#ifndef VWIS_CASE_RUNNER_ACCESS_H_
#define VWIS_CASE_RUNNER_ACCESS_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

/** Internal bridge for case setup/export that must coordinate solver-owned fields. */
class AVWiSCaseRunnerAccess {
public:
    static void run_cartesian_benchmark_impl(
        AVWiSSolver& solver, amrex::Real dt, int steps, amrex::Real viscosity,
        std::string const& report_path);
    static void run_physical_channel_benchmark_impl(
        AVWiSSolver& solver, amrex::Real dt, int steps, amrex::Real viscosity,
        std::string const& report_path);
    static void run_lid_driven_cavity_case_impl(
        AVWiSSolver& solver, amrex::Real dt, int steps, amrex::Real viscosity,
        std::string const& report_path, std::string const& field_path,
        std::string const& centerline_path, std::string const& history_path);
};

#endif
