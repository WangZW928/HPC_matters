#ifndef VWIS_LID_DRIVEN_CAVITY_CASE_H_
#define VWIS_LID_DRIVEN_CAVITY_CASE_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

void run_lid_driven_cavity_case(
    AVWiSSolver& solver, amrex::Real dt, int steps, amrex::Real viscosity,
    std::string const& report_path, std::string const& field_path,
    std::string const& centerline_path, std::string const& history_path);

#endif
