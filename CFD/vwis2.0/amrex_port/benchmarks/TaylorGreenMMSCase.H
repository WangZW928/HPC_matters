#ifndef AVWIS_TAYLOR_GREEN_MMS_CASE_H_
#define AVWIS_TAYLOR_GREEN_MMS_CASE_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

void run_taylor_green_mms(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path);

#endif
