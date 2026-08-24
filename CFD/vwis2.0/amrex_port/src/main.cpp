#include "VwisAmrExSolver.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <exception>
#include <stdexcept>
#include <string>

namespace {
CartesianBC named_bc(std::string const& name)
{
    if (name == "noslip") return CartesianBC::NoSlipWall;
    if (name == "slip") return CartesianBC::SlipWall;
    if (name == "symmetry") return CartesianBC::Symmetry;
    if (name == "inflow") return CartesianBC::Inflow;
    if (name == "outflow") return CartesianBC::Outflow;
    throw std::runtime_error("unsupported vwisbcs named Cartesian BC '" + name +
                             "' (supported: noslip, slip, symmetry, inflow, outflow)");
}

CartesianBC legacy_bc(int code)
{
    switch (code) {
    case 1: return CartesianBC::NoSlipWall;
    case 3: return CartesianBC::Symmetry;
    case 4: return CartesianBC::Outflow;
    case 5: return CartesianBC::Inflow;
    default:
        throw std::runtime_error("unsupported legacy BC integer " + std::to_string(code) +
                                 "; P3 supports only 1/3/4/5. Use a named Cartesian BC for slip; "
                                 "legacy case-specific 0/2/6/8/10/11/12/13/14/-1/-2 are rejected");
    }
}

CartesianBoundaryConfig read_boundary_config(amrex::Vector<int> const& periodic)
{
    amrex::ParmParse pp("vwisbcs");
    CartesianBoundaryConfig config;
    pp.query("enabled", config.enabled);
    if (!config.enabled) return config;

    amrex::Vector<std::string> lo_name(AMREX_SPACEDIM, "");
    amrex::Vector<std::string> hi_name(AMREX_SPACEDIM, "");
    amrex::Vector<int> legacy(2 * AMREX_SPACEDIM, -999);
    pp.queryarr("lo", lo_name, 0, AMREX_SPACEDIM);
    pp.queryarr("hi", hi_name, 0, AMREX_SPACEDIM);
    pp.queryarr("legacy_codes", legacy, 0, 2 * AMREX_SPACEDIM);
    pp.query("inlet_profile", config.inlet_profile);
    pp.query("inlet_target_flux", config.inlet_target_flux);
    pp.query("profile_offset", config.profile_offset);
    pp.query("profile_slope_0", config.profile_slope_0);
    pp.query("profile_slope_1", config.profile_slope_1);
    pp.query("constrain_outlet_flux", config.constrain_outlet_flux);

    amrex::Vector<amrex::Real> pressure(2 * AMREX_SPACEDIM, 0.0);
    pp.queryarr("pressure", pressure, 0, 2 * AMREX_SPACEDIM);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side = 0; side < 2; ++side) {
            const int slot = 2 * dir + side;
            auto& spec = config.sides[slot];
            spec.pressure = pressure[slot];
            if (periodic[dir]) {
                if (!(side == 0 ? lo_name[dir] : hi_name[dir]).empty() || legacy[slot] != -999) {
                    throw std::runtime_error("periodic direction must not also specify a physical BC");
                }
                spec.velocity = CartesianBC::Periodic;
            } else {
                auto const& name = side == 0 ? lo_name[dir] : hi_name[dir];
                if (!name.empty() && legacy[slot] != -999) {
                    throw std::runtime_error("choose either named vwisbcs.lo/hi or legacy_codes, not both");
                }
                if (!name.empty()) spec.velocity = named_bc(name);
                else if (legacy[slot] != -999) {
                    spec.velocity = legacy_bc(legacy[slot]);
                    spec.legacy_code = legacy[slot];
                } else {
                    throw std::runtime_error("every non-periodic side needs an explicit vwisbcs.lo/hi or legacy_codes entry");
                }
            }
        }
    }
    return config;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        amrex::ParmParse pp("vwis");
        amrex::Vector<int> n_cell(AMREX_SPACEDIM, 16);
        amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0);
        pp.queryarr("n_cell", n_cell, 0, AMREX_SPACEDIM);
        pp.queryarr("is_periodic", is_periodic, 0, AMREX_SPACEDIM);

        int max_grid_size = 32;
        int nghost = 2;
        amrex::Real dt = 1.0e-3;
        bool run_contract_checks = false;
        bool run_p2_transform_checks = false;
        bool run_p3_boundary_checks = false;
        std::string metadata_file;
        pp.query("max_grid_size", max_grid_size);
        pp.query("nghost", nghost);
        pp.query("dt", dt);
        pp.query("run_contract_checks", run_contract_checks);
        pp.query("run_p2_transform_checks", run_p2_transform_checks);
        pp.query("run_p3_boundary_checks", run_p3_boundary_checks);
        pp.query("metadata_file", metadata_file);

        amrex::RealBox physical_domain(
            {AMREX_D_DECL(0.0, 0.0, 0.0)},
            {AMREX_D_DECL(1.0, 1.0, 1.0)});
        auto boundary = read_boundary_config(is_periodic);
        VwisAmrExSolver solver(n_cell, max_grid_size, nghost,
                                physical_domain, is_periodic, boundary);
        solver.initialize();
        if (run_contract_checks) {
            solver.run_runtime_contract_checks();
        }
        if (run_p2_transform_checks) {
            solver.run_p2_transform_layout_checks();
        }
        if (run_p3_boundary_checks) {
            solver.run_p3_boundary_contract_checks();
        }
        // Explicit P3 no-op: no physical state is advanced.
        solver.advance_one_step(dt);
        if (!metadata_file.empty()) solver.write_metadata_manifest(metadata_file);
        solver.diagnostics();
    } catch (std::exception const& error) {
        amrex::Print() << "VWiS AMReX P3 error: " << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
