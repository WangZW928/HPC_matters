#ifndef AVWIS_COORDINATE_MAPPING_H_
#define AVWIS_COORDINATE_MAPPING_H_

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>

#include <cmath>
#include <memory>
#include <string>

/** Logical-coordinate metadata; spacing is never inferred from a physical metric. */
struct LogicalGrid {
    amrex::Box cell_domain;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> lower{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> spacing{};
    amrex::Array<int, AMREX_SPACEDIM> periodic{};

    static LogicalGrid from_cartesian_geometry(amrex::Geometry const& geometry);
};

/**
 * Production boundary for a fixed structured coordinate mapping. Providers
 * write physical coordinates at logical nodes; they do not own metric fields.
 */
class CoordinateMapping {
public:
    virtual void fill_nodes(amrex::MultiFab& node_coordinates_nd,
                            LogicalGrid const& logical) const = 0;
    virtual std::string id() const = 0;
    virtual ~CoordinateMapping() = default;
};

/** x(xi)=xi using the logical origin and spacing supplied by LogicalGrid. */
class IdentityCoordinateMapping final : public CoordinateMapping {
public:
    void fill_nodes(amrex::MultiFab& node_coordinates_nd,
                    LogicalGrid const& logical) const override;
    std::string id() const override { return "identity"; }
};

/**
 * Parameters for a globally monotone separable mapping.  In direction m,
 *
 *   x_m = xi_lo_m + scale_m L_m
 *         [t_m + stretch_m sin(2 pi t_m)/(2 pi)],
 *   t_m = (xi_m-xi_lo_m)/L_m.
 *
 * scale=1 and stretch=0 is exactly the identity mapping.  The strict
 * |stretch|<1 bound makes dx_m/dxi_m positive throughout the logical domain
 * and throughout its analytic ghost extension.
 */
struct AnalyticOrthogonalMappingParameters {
    amrex::Array<amrex::Real, AMREX_SPACEDIM> scale{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> stretch{AMREX_D_DECL(0.0, 0.0, 0.0)};
};

/** Smooth separable x=f(xi), y=g(eta), z=h(zeta) coordinate provider. */
class AnalyticOrthogonalCoordinateMapping final : public CoordinateMapping {
public:
    explicit AnalyticOrthogonalCoordinateMapping(
        AnalyticOrthogonalMappingParameters parameters = {});

    void fill_nodes(amrex::MultiFab& node_coordinates_nd,
                    LogicalGrid const& logical) const override;
    std::string id() const override { return "analytic_orthogonal"; }
    AnalyticOrthogonalMappingParameters const& parameters() const noexcept { return m_parameters; }

    /** GPU-safe one-dimensional map used by the provider and analytic audits. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real map_coordinate(amrex::Real logical_coordinate,
                                      amrex::Real logical_lower,
                                      amrex::Real logical_length,
                                      amrex::Real scale,
                                      amrex::Real stretch) noexcept
    {
        constexpr amrex::Real two_pi = 6.283185307179586476925286766559005768;
        amrex::Real const t = (logical_coordinate - logical_lower) / logical_length;
        return logical_lower + scale * logical_length
             * (t + stretch * std::sin(two_pi * t) / two_pi);
    }

    /** Exact positive diagonal derivative dx_m/dxi_m for contract/reference use. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real coordinate_derivative(amrex::Real logical_coordinate,
                                             amrex::Real logical_lower,
                                             amrex::Real logical_length,
                                             amrex::Real scale,
                                             amrex::Real stretch) noexcept
    {
        constexpr amrex::Real two_pi = 6.283185307179586476925286766559005768;
        amrex::Real const t = (logical_coordinate - logical_lower) / logical_length;
        return scale * (1.0 + stretch * std::cos(two_pi * t));
    }

private:
    AnalyticOrthogonalMappingParameters m_parameters;
};

/** Strict factory boundary for future input parsing; unknown types fail. */
std::unique_ptr<CoordinateMapping> make_coordinate_mapping(
    std::string const& type,
    AnalyticOrthogonalMappingParameters const& analytic_parameters = {});

#endif
