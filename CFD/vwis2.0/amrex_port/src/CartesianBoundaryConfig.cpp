#include "CartesianBoundaryConfig.H"

#include <AMReX_ParmParse.H>

#include <stdexcept>

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
} // namespace

CartesianBoundaryConfig read_cartesian_boundary_config(
    amrex::Vector<int> const& is_periodic)
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
            if (is_periodic[dir]) {
                if (!(side == 0 ? lo_name[dir] : hi_name[dir]).empty() || legacy[slot] != -999) {
                    throw std::runtime_error("periodic direction must not also specify a physical BC");
                }
                spec.velocity = CartesianBC::Periodic;
                continue;
            }

            auto const& name = side == 0 ? lo_name[dir] : hi_name[dir];
            if (!name.empty() && legacy[slot] != -999) {
                throw std::runtime_error("choose either named vwisbcs.lo/hi or legacy_codes, not both");
            }
            if (!name.empty()) {
                spec.velocity = named_bc(name);
            } else if (legacy[slot] != -999) {
                spec.velocity = legacy_bc(legacy[slot]);
                spec.legacy_code = legacy[slot];
            } else {
                throw std::runtime_error(
                    "every non-periodic side needs an explicit vwisbcs.lo/hi or legacy_codes entry");
            }
        }
    }
    return config;
}

char const* cartesian_bc_name(CartesianBC bc) noexcept
{
    switch (bc) {
    case CartesianBC::Periodic: return "periodic";
    case CartesianBC::NoSlipWall: return "noslip";
    case CartesianBC::SlipWall: return "slip";
    case CartesianBC::Symmetry: return "symmetry";
    case CartesianBC::Inflow: return "inflow";
    case CartesianBC::Outflow: return "outflow";
    }
    return "unknown";
}
