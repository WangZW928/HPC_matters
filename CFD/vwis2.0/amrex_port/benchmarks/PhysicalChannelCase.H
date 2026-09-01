#ifndef VWIS_PHYSICAL_CHANNEL_CASE_H_
#define VWIS_PHYSICAL_CHANNEL_CASE_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

void run_physical_channel_benchmark(AVWiSSolver& solver, amrex::Real dt,
                                    int steps, amrex::Real viscosity,
                                    std::string const& report_path);

#endif
