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

void VwisAmrExSolver::compute_cartesian_viscous_rhs(
    amrex::MultiFab& rhs, amrex::Real viscosity)
{
    if (rhs.boxArray() != m_ba || rhs.nComp() != AMREX_SPACEDIM) {
        throw std::runtime_error("P5 Cartesian viscous RHS must be a three-component cell MultiFab");
    }
    if (!std::isfinite(viscosity) || viscosity < 0.0) {
        throw std::runtime_error("P5 Cartesian viscosity must be finite and non-negative");
    }
    if (m_nghost < 1) {
        throw std::runtime_error("P5 Cartesian viscosity requires vwis.nghost>=1");
    }
    if (m_boundary.enabled) {
        apply_boundary_pipeline("pre-p5-viscosity");
    } else {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!m_geom.isPeriodic(dir)) {
                throw std::runtime_error(
                    "P5 Cartesian viscosity requires explicit vwisbcs on non-periodic directions");
            }
        }
        fill_ghost_cells();
    }
    require_ghosts_fresh("P5 Cartesian viscosity");

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

void VwisAmrExSolver::run_p5_viscous_contract_checks(amrex::Real viscosity)
{
    if (!std::isfinite(viscosity) || viscosity <= 0.0) {
        throw std::runtime_error("P5 viscous contract requires finite positive vwis.viscosity");
    }
    amrex::MultiFab rhs(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab error(m_ba, m_dm, AMREX_SPACEDIM, 0);
    amrex::MultiFab work(m_ba, m_dm, AMREX_SPACEDIM, 0);
    const amrex::Real roundoff = 8192.0 * std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real amplitudes[AMREX_SPACEDIM] = {AMREX_D_DECL(1.0, -0.5, 0.25)};

    if (m_boundary.enabled) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            for (int side = 0; side < 2; ++side) {
                if (m_boundary.sides[2 * dir + side].velocity != CartesianBC::NoSlipWall) {
                    throw std::runtime_error(
                        "P5 physical viscous contract requires all non-periodic faces to be noslip");
                }
            }
        }
        const auto problo = m_geom.ProbLoArray();
        const auto dx = m_geom.CellSizeArray();
        const amrex::Real lx = m_geom.ProbLength(0);
        const amrex::Real ly = m_geom.ProbLength(1);
        const amrex::Real lz = m_geom.ProbLength(2);
        for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    const amrex::Real x = problo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
                    const amrex::Real y = problo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
                    const amrex::Real z = problo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
                    u(i,j,k,comp) = amplitudes[comp] *
                        amrex::Math::sinpi((x-problo[0]) / lx) *
                        amrex::Math::sinpi((y-problo[1]) / ly) *
                        amrex::Math::sinpi((z-problo[2]) / lz);
                });
        }
        mark_valid_modified();
        compute_cartesian_viscous_rhs(rhs, viscosity);

        const amrex::Real eigenvalue = -4.0 * viscosity * (
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[0] / lx) / m_dx[0], 2) +
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[1] / ly) / m_dx[1], 2) +
            std::pow(std::sin(0.5 * 3.14159265358979323846 * m_dx[2] / lz) / m_dx[2], 2));
        const auto lo = m_geom.Domain().smallEnd();
        const auto hi = m_geom.Domain().bigEnd();
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = m_ucat.const_array(mfi);
            auto const visc = rhs.const_array(mfi);
            auto const err = error.array(mfi);
            auto const wall = work.array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), AMREX_SPACEDIM,
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                    err(i,j,k,comp) = visc(i,j,k,comp) - eigenvalue * u(i,j,k,comp);
                    amrex::Real flux = 0.0;
                    if (i == lo[0] || i == hi[0]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[1]*dx[2] / dx[0];
                    if (j == lo[1] || j == hi[1]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[0]*dx[2] / dx[1];
                    if (k == lo[2] || k == hi[2]) flux -= 2.0 * viscosity * u(i,j,k,comp) * dx[0]*dx[1] / dx[2];
                    wall(i,j,k,comp) = flux;
                });
        }
        amrex::Real max_error = 0.0;
        amrex::Real max_balance_error = 0.0;
        for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
            amrex::Real stencil_error = error.norm0(comp, 0, true);
            amrex::Real volume_rhs = rhs.sum(comp, true) * m_cell_volume;
            amrex::Real wall_flux = work.sum(comp, true);
            amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
            amrex::ParallelDescriptor::ReduceRealSum(volume_rhs);
            amrex::ParallelDescriptor::ReduceRealSum(wall_flux);
            max_error = amrex::max(max_error, stencil_error);
            max_balance_error = amrex::max(max_balance_error, std::abs(volume_rhs-wall_flux));
        }
        const amrex::Real scale = std::abs(eigenvalue);
        if (max_error > roundoff * amrex::max(1.0, scale) ||
            max_balance_error > roundoff * amrex::max(1.0, scale)) {
            throw std::runtime_error("P5 no-slip viscous stencil/boundary-flux balance failed");
        }
        amrex::Print() << "VWiS AMReX P5-002 boundary/multi-Box viscosity: PASS max_error="
                       << max_error << " boundary_balance_error=" << max_balance_error << "\n";
        return;
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!m_geom.isPeriodic(dir)) {
            throw std::runtime_error("P5 manufactured viscous contract requires a fully periodic domain");
        }
    }
    const amrex::Real xlo = m_geom.ProbLo(0);
    const amrex::Real length = m_geom.ProbLength(0);
    const amrex::Real dx = m_dx[0];
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                const amrex::Real x = xlo + (static_cast<amrex::Real>(i) + 0.5) * dx;
                u(i,j,k,comp) = amplitudes[comp] * amrex::Math::sinpi(2.0 * (x-xlo) / length);
            });
    }
    mark_valid_modified();
    compute_cartesian_viscous_rhs(rhs, viscosity);

    const amrex::Real pi = 3.141592653589793238462643383279502884;
    const amrex::Real exact_wavenumber = 2.0 * pi / length;
    const amrex::Real discrete_eigenvalue = -4.0 * viscosity *
        std::pow(std::sin(pi * dx / length) / dx, 2);
    const amrex::Real exact_eigenvalue = -viscosity * exact_wavenumber * exact_wavenumber;
    for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const visc = rhs.const_array(mfi);
        auto const err = error.array(mfi);
        auto const energy = work.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                err(i,j,k,comp) = visc(i,j,k,comp) - discrete_eigenvalue * u(i,j,k,comp);
                energy(i,j,k,comp) = u(i,j,k,comp) * visc(i,j,k,comp);
            });
    }
    amrex::Real max_error = 0.0;
    amrex::Real max_momentum = 0.0;
    amrex::Real energy = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        amrex::Real stencil_error = error.norm0(comp, 0, true);
        amrex::Real momentum = rhs.sum(comp, true) * m_cell_volume;
        amrex::Real component_energy = work.sum(comp, true) * m_cell_volume;
        amrex::ParallelDescriptor::ReduceRealMax(stencil_error);
        amrex::ParallelDescriptor::ReduceRealSum(momentum);
        amrex::ParallelDescriptor::ReduceRealSum(component_energy);
        max_error = amrex::max(max_error, stencil_error);
        max_momentum = amrex::max(max_momentum, std::abs(momentum));
        energy += component_energy;
    }
    amrex::MultiFab::Copy(error, rhs, 0, 0, AMREX_SPACEDIM, 0);
    for (amrex::MFIter mfi(error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = m_ucat.const_array(mfi);
        auto const cont = error.array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), AMREX_SPACEDIM,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                cont(i,j,k,comp) -= exact_eigenvalue * u(i,j,k,comp);
            });
    }
    amrex::Real continuous_linf = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        amrex::Real norm = error.norm0(comp, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(norm);
        continuous_linf = amrex::max(continuous_linf, norm);
    }
    const amrex::Real accuracy_bound = viscosity * std::pow(exact_wavenumber, 4) * dx * dx / 12.0;
    const amrex::Real scale = std::abs(exact_eigenvalue);
    if (max_error > roundoff * amrex::max(1.0, scale) ||
        max_momentum > roundoff * amrex::max(1.0, scale) ||
        !(energy < 0.0) || continuous_linf > accuracy_bound) {
        throw std::runtime_error("P5 periodic viscous stencil/conservation/dissipation failed");
    }
    amrex::Print() << "VWiS AMReX P5-002 periodic manufactured viscosity: PASS dx=" << dx
                   << " stencil_error=" << max_error
                   << " continuous_Linf=" << continuous_linf
                   << " momentum_error=" << max_momentum
                   << " energy_rate=" << energy << "\n";
}
