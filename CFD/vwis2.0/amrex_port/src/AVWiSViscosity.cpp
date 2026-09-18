#include "AVWiSSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real quadratic_dirichlet_wall_derivative(
    amrex::Real value_near, amrex::Real value_next, amrex::Real wall_value,
    amrex::Real distance_near, amrex::Real distance_next) noexcept
{
    const amrex::Real shifted_near = value_near-wall_value;
    const amrex::Real shifted_next = value_next-wall_value;
    return (distance_next*shifted_near/distance_near -
            distance_near*shifted_next/distance_next) /
           (distance_next-distance_near);
}
} // namespace

void AVWiSSolver::compute_cartesian_viscous_rhs(
    amrex::MultiFab& rhs, amrex::Real viscosity)
{
    if (m_mapping_operator.coordinates != CoordinateSystemMode::Cartesian) {
        throw std::runtime_error(
            "P5 Cartesian viscosity entry point requires Cartesian coordinates");
    }
    compute_viscous_rhs(rhs, viscosity);
}

void AVWiSSolver::compute_viscous_rhs(
    amrex::MultiFab& rhs, amrex::Real viscosity)
{
    if (!rhs.boxArray().ixType().cellCentered() || rhs.boxArray() != m_ba ||
        rhs.DistributionMap() != m_dm || rhs.nComp() != AMREX_SPACEDIM) {
        throw std::runtime_error(
            "P5 viscous RHS must match the solver cell layout and DistributionMapping");
    }
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
        throw std::runtime_error("P5 viscosity must be finite and non-negative");
    }
    if (m_nghost < 1) {
        throw std::runtime_error("P5 viscosity requires avwis.nghost>=1");
    }
    if (m_boundary.enabled) {
        apply_boundary_pipeline("pre-p5-viscosity");
    } else {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P5 viscosity requires explicit avwisbcs on non-periodic directions");
            }
        }
        fill_ghost_cells();
    }
    require_ghosts_fresh("P5 physical viscosity");

    if (m_mapping_operator.coordinates == CoordinateSystemMode::Mapped) {
        validate_mapping_operator_config(m_mapping_operator, m_metric_data, m_metric_epoch);
        auto const spacing = m_metric_data.logical_grid().spacing;
        const amrex::Box domain = m_geom.Domain();
        const auto lo = domain.smallEnd();
        const auto hi = domain.bigEnd();
        amrex::GpuArray<int, 2*AMREX_SPACEDIM> kinds{};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> moving_wall{};
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            moving_wall[comp] = m_boundary.moving_wall_velocity[comp];
        }
        for (int slot = 0; slot < 2*AMREX_SPACEDIM; ++slot) {
            kinds[slot] = m_boundary.enabled
                ? static_cast<int>(m_boundary.sides[slot].velocity)
                : static_cast<int>(CartesianBC::Periodic);
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (domain.length(dir) < 2 &&
                (m_boundary.sides[2*dir].velocity == CartesianBC::NoSlipWall ||
                 m_boundary.sides[2*dir].velocity == CartesianBC::MovingWall ||
                 m_boundary.sides[2*dir+1].velocity == CartesianBC::NoSlipWall ||
                 m_boundary.sides[2*dir+1].velocity == CartesianBC::MovingWall)) {
                throw std::runtime_error(
                    "mapped second-order wall viscosity requires at least two cells normal to the wall");
            }
        }
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const visc = rhs.array(mfi);
            auto const u = m_ucat.const_array(mfi);
            auto const volume = m_metric_data.cell_volume_cc().const_array(mfi);
            auto const coordinates =
                m_metric_data.cell_center_coordinates_cc().const_array(mfi);
            auto const nodes = m_metric_data.node_coordinates_nd().const_array(mfi);
            auto const qx = m_metric_data.face_gradient_metric_fc(0).const_array(mfi);
            auto const qy = m_metric_data.face_gradient_metric_fc(1).const_array(mfi);
            auto const qz = m_metric_data.face_gradient_metric_fc(2).const_array(mfi);
            auto const sx = m_metric_data.face_area_vector_fc(0).const_array(mfi);
            auto const sy = m_metric_data.face_area_vector_fc(1).const_array(mfi);
            auto const sz = m_metric_data.face_area_vector_fc(2).const_array(mfi);
            const amrex::Real ih0 = 1.0 / spacing[0];
            const amrex::Real ih1 = 1.0 / spacing[1];
            const amrex::Real ih2 = 1.0 / spacing[2];
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    amrex::Real flux_x_hi = qx(i+1,j,k,0) *
                        (u(i+1,j,k,comp)-u(i,j,k,comp)) * ih0;
                    amrex::Real flux_x_lo = qx(i,j,k,0) *
                        (u(i,j,k,comp)-u(i-1,j,k,comp)) * ih0;
                    amrex::Real flux_y_hi = qy(i,j+1,k,1) *
                        (u(i,j+1,k,comp)-u(i,j,k,comp)) * ih1;
                    amrex::Real flux_y_lo = qy(i,j,k,1) *
                        (u(i,j,k,comp)-u(i,j-1,k,comp)) * ih1;
                    amrex::Real flux_z_hi = qz(i,j,k+1,2) *
                        (u(i,j,k+1,comp)-u(i,j,k,comp)) * ih2;
                    amrex::Real flux_z_lo = qz(i,j,k,2) *
                        (u(i,j,k,comp)-u(i,j,k-1,comp)) * ih2;

                    // Mirror ghosts impose the wall value but yield only a
                    // first-order face derivative for curved profiles. On a
                    // separable orthogonal wall, fit the wall value and the
                    // first two physical cell centres instead. This preserves
                    // the existing ghost contract while making the viscous
                    // boundary flux second-order on smooth stretched grids.
                    const int no_slip = static_cast<int>(CartesianBC::NoSlipWall);
                    const int moving = static_cast<int>(CartesianBC::MovingWall);
                    if (i == lo[0] && (kinds[0] == no_slip || kinds[0] == moving)) {
                        const amrex::Real wall = kinds[0] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real xwall = nodes(i,j,k,0);
                        const amrex::Real derivative = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i+1,j,k,comp), wall,
                            coordinates(i,j,k,0)-xwall,
                            coordinates(i+1,j,k,0)-xwall);
                        flux_x_lo = sx(i,j,k,0)*derivative;
                    }
                    if (i == hi[0] && (kinds[1] == no_slip || kinds[1] == moving)) {
                        const amrex::Real wall = kinds[1] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real xwall = nodes(i+1,j,k,0);
                        const amrex::Real inward = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i-1,j,k,comp), wall,
                            xwall-coordinates(i,j,k,0),
                            xwall-coordinates(i-1,j,k,0));
                        flux_x_hi = -sx(i+1,j,k,0)*inward;
                    }
                    if (j == lo[1] && (kinds[2] == no_slip || kinds[2] == moving)) {
                        const amrex::Real wall = kinds[2] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real ywall = nodes(i,j,k,1);
                        const amrex::Real derivative = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i,j+1,k,comp), wall,
                            coordinates(i,j,k,1)-ywall,
                            coordinates(i,j+1,k,1)-ywall);
                        flux_y_lo = sy(i,j,k,1)*derivative;
                    }
                    if (j == hi[1] && (kinds[3] == no_slip || kinds[3] == moving)) {
                        const amrex::Real wall = kinds[3] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real ywall = nodes(i,j+1,k,1);
                        const amrex::Real inward = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i,j-1,k,comp), wall,
                            ywall-coordinates(i,j,k,1),
                            ywall-coordinates(i,j-1,k,1));
                        flux_y_hi = -sy(i,j+1,k,1)*inward;
                    }
                    if (k == lo[2] && (kinds[4] == no_slip || kinds[4] == moving)) {
                        const amrex::Real wall = kinds[4] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real zwall = nodes(i,j,k,2);
                        const amrex::Real derivative = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i,j,k+1,comp), wall,
                            coordinates(i,j,k,2)-zwall,
                            coordinates(i,j,k+1,2)-zwall);
                        flux_z_lo = sz(i,j,k,2)*derivative;
                    }
                    if (k == hi[2] && (kinds[5] == no_slip || kinds[5] == moving)) {
                        const amrex::Real wall = kinds[5] == moving ? moving_wall[comp] : 0.0;
                        const amrex::Real zwall = nodes(i,j,k+1,2);
                        const amrex::Real inward = quadratic_dirichlet_wall_derivative(
                            u(i,j,k,comp), u(i,j,k-1,comp), wall,
                            zwall-coordinates(i,j,k,2),
                            zwall-coordinates(i,j,k-1,2));
                        flux_z_hi = -sz(i,j,k+1,2)*inward;
                    }
                    visc(i,j,k,comp) = viscosity *
                        ((flux_x_hi-flux_x_lo) + (flux_y_hi-flux_y_lo) +
                         (flux_z_hi-flux_z_lo)) / volume(i,j,k);
                });
        }
        amrex::Gpu::streamSynchronize();
        return;
    }

    const amrex::Real idx2 = 1.0 / (m_dx[0] * m_dx[0]);
    const amrex::Real idy2 = 1.0 / (m_dx[1] * m_dx[1]);
    const amrex::Real idz2 = 1.0 / (m_dx[2] * m_dx[2]);
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const visc = rhs.array(mfi);
        auto const u = m_ucat.const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                visc(i,j,k,comp) = viscosity * (
                    (u(i+1,j,k,comp) - 2.0*u(i,j,k,comp) + u(i-1,j,k,comp)) * idx2 +
                    (u(i,j+1,k,comp) - 2.0*u(i,j,k,comp) + u(i,j-1,k,comp)) * idy2 +
                    (u(i,j,k+1,comp) - 2.0*u(i,j,k,comp) + u(i,j,k-1,comp)) * idz2);
            });
    }
    amrex::Gpu::streamSynchronize();
}
