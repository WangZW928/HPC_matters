#ifndef VWIS_CARTESIAN_BOUNDARY_CONFIG_H_
#define VWIS_CARTESIAN_BOUNDARY_CONFIG_H_

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

/** P3 single-level Cartesian physical-boundary semantics. */
enum class CartesianBC { Periodic, NoSlipWall, SlipWall, Symmetry, Inflow, Outflow };

struct BoundarySideSpec {
    CartesianBC velocity = CartesianBC::Periodic;
    int legacy_code = -999; // -999 means no legacy integer code was supplied.
    amrex::Real pressure = 0.0; // Fixed only for Outflow; otherwise homogeneous Neumann.
};

struct CartesianBoundaryConfig {
    amrex::Array<BoundarySideSpec, 2 * AMREX_SPACEDIM> sides{}; // lo, hi per direction.
    std::string inlet_profile = "uniform"; // uniform or linear_plane
    amrex::Real inlet_target_flux = 1.0;    // positive volume entering the domain
    amrex::Real profile_offset = 1.0;
    amrex::Real profile_slope_0 = 0.0;
    amrex::Real profile_slope_1 = 0.0;
    bool constrain_outlet_flux = true;
    bool enabled = false;
};

/** Parse the stable `vwisbcs` input namespace and map supported legacy codes. */
CartesianBoundaryConfig read_cartesian_boundary_config(
    amrex::Vector<int> const& is_periodic);

/** Stable diagnostic/input spelling for a Cartesian boundary kind. */
char const* cartesian_bc_name(CartesianBC bc) noexcept;

#endif
