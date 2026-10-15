#ifndef AVWIS_MAPPED_PRESSURE_WAVE_CASE_H_
#define AVWIS_MAPPED_PRESSURE_WAVE_CASE_H_

#include <AMReX_REAL.H>
#include <string>

class AVWiSSolver;

/** Evolving solenoidal wave with nonzero analytic pressure and spatial forcing
 * on the periodic affine-shear domain. Uses the production Euler advance.
 */
void run_mapped_pressure_wave_case(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity, std::string const& report_path,
    std::string const& field_path);

#endif
