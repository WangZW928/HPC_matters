#include "AVWiSContractRunner.H"

#include "AVWiSContractTestAccess.H"

void AVWiSContractRunner::run_runtime_contract_checks()
{
    AVWiSContractTestAccess(m_solver).run_runtime_contract_checks();
}

void AVWiSContractRunner::run_p2_transform_layout_checks()
{
    AVWiSContractTestAccess(m_solver).run_p2_transform_layout_checks();
}

void AVWiSContractRunner::run_p3_boundary_contract_checks()
{
    AVWiSContractTestAccess(m_solver).run_p3_boundary_contract_checks();
}

void AVWiSContractRunner::run_p4_projection_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    AVWiSContractTestAccess(m_solver).run_p4_projection_contract_checks(dt, time_coefficient);
}

void AVWiSContractRunner::run_p5_advection_contract_checks()
{
    AVWiSContractTestAccess(m_solver).run_p5_advection_contract_checks();
}

void AVWiSContractRunner::run_p5_viscous_contract_checks(amrex::Real viscosity)
{
    AVWiSContractTestAccess(m_solver).run_p5_viscous_contract_checks(viscosity);
}

void AVWiSContractRunner::run_p5_time_contract_checks(
    amrex::Real dt, amrex::Real final_time, amrex::Real viscosity)
{
    AVWiSContractTestAccess(m_solver).run_p5_time_contract_checks(dt, final_time, viscosity);
}

void AVWiSContractRunner::run_p5_orthogonal_projection_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    AVWiSContractTestAccess(m_solver).run_p5_orthogonal_projection_contract_checks(
        dt, time_coefficient);
}

void AVWiSContractRunner::run_p5_mapped_boundary_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    AVWiSContractTestAccess(m_solver).run_p5_mapped_boundary_contract_checks(
        dt, time_coefficient);
}

void AVWiSContractRunner::run_p5_mapped_boundary_c52_contract_checks(
    amrex::Real dt, amrex::Real time_coefficient)
{
    AVWiSContractTestAccess(m_solver).run_p5_mapped_boundary_c52_contract_checks(
        dt, time_coefficient);
}

void AVWiSContractRunner::run_p8_restart_contract_checks(
    std::string const& path, amrex::Real dt, int total_steps,
    int checkpoint_step, amrex::Real viscosity)
{
    AVWiSContractTestAccess(m_solver).run_p8_restart_contract_checks(
        path, dt, total_steps, checkpoint_step, viscosity);
}

void AVWiSContractRunner::run_p8_sampling_statistics_contract(
    std::string const& report_path, std::string const& plane_path)
{
    AVWiSContractTestAccess(m_solver).run_p8_sampling_statistics_contract(
        report_path, plane_path);
}
