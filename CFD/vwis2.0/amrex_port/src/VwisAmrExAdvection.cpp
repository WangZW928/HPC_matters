#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <stdexcept>

void VwisAmrExSolver::compute_cartesian_advection_rhs(amrex::MultiFab& rhs)
{
    if (rhs.boxArray() != m_ba || rhs.nComp() != AMREX_SPACEDIM) {
        throw std::runtime_error("P5 Cartesian advection RHS must be a three-component cell MultiFab");
    }
    if (m_nghost < 1) {
        throw std::runtime_error("P5 Cartesian advection requires vwis.nghost>=1");
    }
    if (m_boundary.enabled) {
        apply_boundary_pipeline("pre-p5-advection");
    } else {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P5 Cartesian advection requires explicit vwisbcs on non-periodic directions");
            }
        }
        fill_ghost_cells();
    }
    require_ghosts_fresh("P5 Cartesian advection");

    const amrex::Real inverse_volume = 1.0 / m_cell_volume;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = rhs.array(mfi);
        auto const u = m_ucat.const_array(mfi);
        auto const fx = m_ucont[0].const_array(mfi);
        auto const fy = m_ucont[1].const_array(mfi);
        auto const fz = m_ucont[2].const_array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real flux_x_hi = fx(i+1,j,k) * 0.5 * (u(i,j,k,comp) + u(i+1,j,k,comp));
                const amrex::Real flux_x_lo = fx(i,j,k)   * 0.5 * (u(i-1,j,k,comp) + u(i,j,k,comp));
                const amrex::Real flux_y_hi = fy(i,j+1,k) * 0.5 * (u(i,j,k,comp) + u(i,j+1,k,comp));
                const amrex::Real flux_y_lo = fy(i,j,k)   * 0.5 * (u(i,j-1,k,comp) + u(i,j,k,comp));
                const amrex::Real flux_z_hi = fz(i,j,k+1) * 0.5 * (u(i,j,k,comp) + u(i,j,k+1,comp));
                const amrex::Real flux_z_lo = fz(i,j,k)   * 0.5 * (u(i,j,k-1,comp) + u(i,j,k,comp));
                adv(i,j,k,comp) = -((flux_x_hi-flux_x_lo) +
                                     (flux_y_hi-flux_y_lo) +
                                     (flux_z_hi-flux_z_lo)) * inverse_volume;
            });
    }
    amrex::Gpu::streamSynchronize();
}

void VwisAmrExSolver::run_p5_advection_contract_checks()
{
    amrex::MultiFab rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    const amrex::Real roundoff = 4096.0 * std::numeric_limits<amrex::Real>::epsilon();

    if (m_boundary.enabled) {
        int inflow_dir = -1;
        int inflow_side = 0;
        int outflow_dir = -1;
        int outflow_side = 0;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            for (int side = 0; side < 2; ++side) {
                const auto kind = m_boundary.sides[2 * dir + side].velocity;
                if (kind == CartesianBC::Inflow) { inflow_dir = dir; inflow_side = side; }
                if (kind == CartesianBC::Outflow) { outflow_dir = dir; outflow_side = side; }
            }
        }
        if (inflow_dir < 0 || outflow_dir != inflow_dir || outflow_side == inflow_side ||
            !m_boundary.constrain_outlet_flux) {
            throw std::runtime_error(
                "P5 boundary contract requires opposite inflow/constrained-outflow faces");
        }
        amrex::Real cross_section = 1.0;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (dir != inflow_dir) cross_section *= m_geom.ProbLength(dir);
        }
        const amrex::Real speed = (inflow_side == 0 ? 1.0 : -1.0) *
                                  m_boundary.inlet_target_flux / cross_section;
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    u(i,j,k,comp) = comp == inflow_dir ? speed : 0.0;
                });
        }
        mark_valid_modified();
        sync_ucont_from_ucat();
        compute_cartesian_advection_rhs(rhs);
        amrex::Real max_error = 0.0;
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            max_error = amrex::max(max_error, rhs.norm0(comp, 0, true));
        }
        amrex::ParallelDescriptor::ReduceRealMax(max_error);
        if (max_error > roundoff * amrex::max(1.0, speed * speed / m_dx[inflow_dir])) {
            throw std::runtime_error("P5 constant boundary advection RHS is not zero");
        }
        amrex::Print() << "VWiS AMReX P5-001 boundary/multi-Box advection: PASS max_error="
                       << max_error << "\n";
        return;
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5 manufactured advection contract requires a fully periodic domain");
        }
    }

    constexpr amrex::Real advecting_speed = 0.75;
    const amrex::Real xlo = m_geom.ProbLo(0);
    const amrex::Real length = m_geom.ProbLength(0);
    const amrex::Real dx = m_dx[0];
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
            u(i,j,k,0) = advecting_speed;
            u(i,j,k,1) = amrex::Math::sinpi(2.0 * (x-xlo) / length);
            u(i,j,k,2) = 0.0;
        });
    }
    mark_valid_modified();
    sync_ucont_from_ucat();
    compute_cartesian_advection_rhs(rhs);

    amrex::MultiFab discrete_error(m_ba, m_dm, 1, 0);
    amrex::MultiFab continuous_error(m_ba, m_dm, 1, 0);
    const amrex::Real pi = 3.141592653589793238462643383279502884;
    const amrex::Real discrete_wavenumber = std::sin(2.0 * pi * dx / length) / dx;
    const amrex::Real exact_wavenumber = 2.0 * pi / length;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const adv = rhs.const_array(mfi);
        auto const disc = discrete_error.array(mfi);
        auto const cont = continuous_error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
            const amrex::Real cosine = amrex::Math::cospi(2.0 * (x-xlo) / length);
            disc(i,j,k) = adv(i,j,k,1) + advecting_speed * discrete_wavenumber * cosine;
            cont(i,j,k) = adv(i,j,k,1) + advecting_speed * exact_wavenumber * cosine;
        });
    }
    amrex::Real stencil_error = discrete_error.norm0(0, 0, true);
    amrex::Real continuous_linf = continuous_error.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
    amrex::ParallelDescriptor::ReduceRealMax(continuous_linf);
    const amrex::Real scale = std::abs(advecting_speed * exact_wavenumber);
    const amrex::Real second_order_bound = std::abs(advecting_speed) *
        exact_wavenumber * exact_wavenumber * exact_wavenumber * dx * dx / 6.0;
    if (stencil_error > roundoff * amrex::max(1.0, scale) ||
        continuous_linf > second_order_bound) {
        throw std::runtime_error("P5 periodic manufactured advection stencil/accuracy failed");
    }
    for (int comp : {0, 2}) {
        amrex::Real norm = rhs.norm0(comp, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(norm);
        if (norm > roundoff) throw std::runtime_error("P5 constant advected component RHS is not zero");
    }
    amrex::Print() << "VWiS AMReX P5-001 periodic manufactured advection: PASS dx=" << dx
                   << " stencil_error=" << stencil_error
                   << " continuous_Linf=" << continuous_linf
                   << " error_over_dx2=" << continuous_linf / (dx * dx) << "\n";
}
