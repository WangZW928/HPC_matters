#ifndef VWIS_CONTRACT_RUNNER_H_
#define VWIS_CONTRACT_RUNNER_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

/** Test-only facade composed with a solver; a contract runner is not a solver subtype. */
class AVWiSContractRunner {
public:
    explicit AVWiSContractRunner(AVWiSSolver& solver) : m_solver(solver) {}

    void run_runtime_contract_checks();
    void run_p2_transform_layout_checks();
    void run_p3_boundary_contract_checks();
    void run_p4_projection_contract_checks(amrex::Real dt,
                                           amrex::Real time_coefficient = 1.0);
    void run_p5_advection_contract_checks();
    void run_p5_viscous_contract_checks(amrex::Real viscosity);
    void run_p5_time_contract_checks(amrex::Real dt, amrex::Real final_time,
                                     amrex::Real viscosity);
    void run_p5_orthogonal_projection_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p5_mapped_boundary_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p5_mapped_boundary_c52_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p8_restart_contract_checks(std::string const& path, amrex::Real dt,
                                        int total_steps, int checkpoint_step,
                                        amrex::Real viscosity);
    void run_p8_sampling_statistics_contract(std::string const& report_path,
                                             std::string const& plane_path);

private:
    AVWiSSolver& m_solver;
};

#endif
