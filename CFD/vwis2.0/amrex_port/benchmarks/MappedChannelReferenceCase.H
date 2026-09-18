#ifndef AVWIS_MAPPED_CHANNEL_REFERENCE_CASE_H_
#define AVWIS_MAPPED_CHANNEL_REFERENCE_CASE_H_

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <string>

class AVWiSSolver;

void run_mapped_channel_reference(
    AVWiSSolver& solver, amrex::Real dt, amrex::Real final_time,
    amrex::Real viscosity,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& body_force,
    std::string const& report_path, std::string const& profile_path);

#endif
