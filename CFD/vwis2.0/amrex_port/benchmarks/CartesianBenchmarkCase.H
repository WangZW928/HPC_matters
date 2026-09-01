#ifndef VWIS_CARTESIAN_BENCHMARK_CASE_H_
#define VWIS_CARTESIAN_BENCHMARK_CASE_H_

#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

void run_cartesian_benchmark(AVWiSSolver& solver, amrex::Real dt, int steps,
                             amrex::Real viscosity, std::string const& report_path);

#endif
