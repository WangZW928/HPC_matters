#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <AMReX_BC_TYPES.H>

#include <stdexcept>

namespace {
// Keep extended device lambdas in free functions: nvcc rejects lambdas whose
// enclosing class member is private, even though the lambda captures no `this`.
amrex::Real inlet_profile_integral_impl(
    amrex::MultiFab const& face_flux, amrex::DistributionMapping const& dm,
    amrex::Geometry const& geom, CartesianBoundaryConfig const& boundary,
    amrex::Real face_area, int dir, bool high)
{
    amrex::MultiFab profile(face_flux.boxArray(), dm, 1, 0);
    profile.setVal(0.0);
    const amrex::Box face_domain =
        amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(dir));
    const int boundary_index = high ? face_domain.bigEnd(dir) : face_domain.smallEnd(dir);
    const int linear = boundary.inlet_profile == "linear_plane";
    const amrex::Real offset = boundary.profile_offset;
    const amrex::Real slope0 = boundary.profile_slope_0;
    const amrex::Real slope1 = boundary.profile_slope_1;
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const int td0 = (dir + 1) % AMREX_SPACEDIM;
    const int td1 = (dir + 2) % AMREX_SPACEDIM;
    for (amrex::MFIter mfi(profile, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const value = profile.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
            if (index[dir] == boundary_index) {
                const amrex::Real x0 = problo[td0] +
                    (static_cast<amrex::Real>(index[td0]) + 0.5) * dx[td0];
                const amrex::Real x1 = problo[td1] +
                    (static_cast<amrex::Real>(index[td1]) + 0.5) * dx[td1];
                value(i,j,k) = face_area * (offset + linear * (slope0 * x0 + slope1 * x1));
            }
        });
    }
    amrex::Real local_sum = profile.sum_unique(0, true, geom.periodicity());
    amrex::ParallelDescriptor::ReduceRealSum(local_sum);
    return local_sum;
}

amrex::Real boundary_flux_impl(
    amrex::MultiFab const& face_flux, amrex::DistributionMapping const& dm,
    amrex::Geometry const& geom, int dir, bool high)
{
    amrex::MultiFab plane(face_flux.boxArray(), dm, 1, 0);
    plane.setVal(0.0);
    const amrex::Box face_domain =
        amrex::convert(geom.Domain(), amrex::IntVect::TheDimensionVector(dir));
    const int boundary_index = high ? face_domain.bigEnd(dir) : face_domain.smallEnd(dir);
    for (amrex::MFIter mfi(plane, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const dst = plane.array(mfi);
        auto const src = face_flux.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int index = dir == 0 ? i : (dir == 1 ? j : k);
            if (index == boundary_index) dst(i,j,k) = src(i,j,k);
        });
    }
    amrex::Real coordinate_flux = plane.sum_unique(0, true, geom.periodicity());
    amrex::ParallelDescriptor::ReduceRealSum(coordinate_flux);
    return high ? coordinate_flux : -coordinate_flux;
}
} // namespace

void VwisAmrExSolver::validate_boundary_config() const
{
    if (!m_boundary.enabled) return;
    if (m_nghost < 1) throw std::runtime_error("P3 physical BC requires vwisbcs.enabled=1 and vwis.nghost>=1");
    if (m_boundary.inlet_profile != "uniform" && m_boundary.inlet_profile != "linear_plane") {
        throw std::runtime_error("vwisbcs.inlet_profile must be uniform or linear_plane");
    }
    int inlet_count = 0;
    int outlet_count = 0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (int side = 0; side < 2; ++side) {
            auto const kind = m_boundary.sides[2 * dir + side].velocity;
            if (m_geom.isPeriodic(dir)) {
                if (kind != CartesianBC::Periodic) throw std::runtime_error("periodic direction has a physical BC");
            } else if (kind == CartesianBC::Periodic) {
                throw std::runtime_error("non-periodic direction is missing an explicit physical BC");
            }
            inlet_count += kind == CartesianBC::Inflow;
            outlet_count += kind == CartesianBC::Outflow;
        }
    }
    if ((inlet_count == 1 || outlet_count == 1) &&
        (!(m_boundary.inlet_target_flux > 0.0))) {
        throw std::runtime_error("vwisbcs.inlet_target_flux must be positive (magnitude entering domain)");
    }
    if (!((inlet_count == 1 && outlet_count == 1) ||
          (inlet_count == 0 && outlet_count == 0))) {
        throw std::runtime_error("Cartesian boundary path requires one inflow and one outflow, or a closed no-penetration domain");
    }
}

void VwisAmrExSolver::define_boundary_metadata()
{
    // FillBoundary is deliberately only inter-box/periodic communication.
    // ext_dir faces are owned by fill_physical_ghost_cells(); periodic faces
    // are int_dir and never passed to the physical fill.
    m_cell_bcs.assign(AMREX_SPACEDIM, amrex::BCRec{});
    for (auto& bc : m_cell_bcs) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int type = m_geom.isPeriodic(dir) ? amrex::BCType::int_dir : amrex::BCType::ext_dir;
            bc.setLo(dir, type);
            bc.setHi(dir, type);
        }
    }
}

amrex::Real VwisAmrExSolver::inlet_profile_integral(int dir, bool high) const
{
    return inlet_profile_integral_impl(
        m_ucont[dir], m_dm, m_geom, m_boundary, m_face_area[dir], dir, high);
}

amrex::Real VwisAmrExSolver::boundary_flux(int dir, bool high) const
{
    return boundary_flux_impl(m_ucont[dir], m_dm, m_geom, dir, high);
}

void VwisAmrExSolver::fill_physical_ghost_cells()
{
    fill_physical_ghost_cells_impl(true);
}

void VwisAmrExSolver::fill_physical_ghost_cells_impl(bool impose_boundary_flux)
{
    if (!m_boundary.enabled) {
        throw std::runtime_error("physical ghost fill requested without vwisbcs.enabled=1");
    }
    if (m_halo_epoch != m_valid_epoch) {
        throw std::runtime_error("physical ghost fill requires FillBoundary halo at the current valid epoch");
    }

    const amrex::Box domain = m_geom.Domain();
    amrex::GpuArray<int, 2 * AMREX_SPACEDIM> kinds{};
    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> pressures{};
    amrex::GpuArray<amrex::Real, 2 * AMREX_SPACEDIM> inlet_scales{};
    for (int slot = 0; slot < 2 * AMREX_SPACEDIM; ++slot) {
        kinds[slot] = static_cast<int>(m_boundary.sides[slot].velocity);
        pressures[slot] = m_boundary.sides[slot].pressure;
        if (m_boundary.sides[slot].velocity == CartesianBC::Inflow) {
            const amrex::Real raw = inlet_profile_integral(slot / 2, (slot % 2) != 0);
            if (!(raw > 0.0)) throw std::runtime_error("inlet plane profile has non-positive integrated weight");
            inlet_scales[slot] = m_boundary.inlet_target_flux / raw;
        }
    }
    const auto lo = domain.smallEnd();
    const auto hi = domain.bigEnd();
    const auto cell_problo = m_geom.ProbLoArray();
    const auto cell_dx = m_geom.CellSizeArray();
    const int linear_profile = m_boundary.inlet_profile == "linear_plane";
    const amrex::Real profile_offset = m_boundary.profile_offset;
    const amrex::Real profile_slope0 = m_boundary.profile_slope_0;
    const amrex::Real profile_slope1 = m_boundary.profile_slope_1;

    // Cell ghosts use mirror Dirichlet for velocity inlet/walls, even
    // extrapolation for outlet/slip tangential velocity, homogeneous pressure
    // Neumann except fixed outlet pressure. Edge/corner precedence is x,y,z.
    auto fill_cell = [&](amrex::MultiFab& mf, int role) {
        const int ncomp = mf.nComp();
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const value = mf.array(mfi);
            const amrex::Box grown = mfi.fabbox();
            amrex::ParallelFor(grown, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int comp) noexcept {
                int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                if (index[0] >= lo[0] && index[0] <= hi[0] && index[1] >= lo[1] && index[1] <= hi[1]
                    && index[2] >= lo[2] && index[2] <= hi[2]) return;
                int boundary_dir = -1;
                int side = 0;
                int source[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (source[d] < lo[d]) { if (boundary_dir < 0) { boundary_dir = d; side = 0; } source[d] = lo[d]; }
                    if (source[d] > hi[d]) { if (boundary_dir < 0) { boundary_dir = d; side = 1; } source[d] = hi[d]; }
                }
                const int kind = kinds[2 * boundary_dir + side];
                const amrex::Real interior = value(source[0], source[1], source[2], comp);
                if (role == 2) { value(i,j,k,comp) = interior; return; } // Nvert classifier: zero-gradient only.
                if (role == 1) { // accumulated pressure
                    value(i,j,k,comp) = kind == static_cast<int>(CartesianBC::Outflow)
                        ? 2.0 * pressures[2 * boundary_dir + side] - interior : interior;
                    return;
                }
                if (role == 3) { // pressure correction: homogeneous Dirichlet at pressure outlet
                    value(i,j,k,comp) = kind == static_cast<int>(CartesianBC::Outflow)
                        ? -interior : interior;
                    return;
                }
                // Ucat: inlet profile is imposed consistently from boundary Ucont below;
                // only the normal component is nonzero in this minimal interface.
                if (kind == static_cast<int>(CartesianBC::NoSlipWall)) value(i,j,k,comp) = -interior;
                else if (kind == static_cast<int>(CartesianBC::SlipWall) || kind == static_cast<int>(CartesianBC::Symmetry))
                    value(i,j,k,comp) = comp == boundary_dir ? -interior : interior;
                else if (kind == static_cast<int>(CartesianBC::Inflow)) {
                    if (comp == boundary_dir) {
                        const int td0 = (boundary_dir + 1) % AMREX_SPACEDIM;
                        const int td1 = (boundary_dir + 2) % AMREX_SPACEDIM;
                        const amrex::Real x0 = cell_problo[td0] + (static_cast<amrex::Real>(source[td0]) + 0.5) * cell_dx[td0];
                        const amrex::Real x1 = cell_problo[td1] + (static_cast<amrex::Real>(source[td1]) + 0.5) * cell_dx[td1];
                        const amrex::Real weight = profile_offset + linear_profile * (profile_slope0 * x0 + profile_slope1 * x1);
                        const amrex::Real boundary_velocity = (side ? -1.0 : 1.0) * inlet_scales[2 * boundary_dir + side] * weight;
                        value(i,j,k,comp) = 2.0 * boundary_velocity - interior;
                    } else value(i,j,k,comp) = -interior;
                }
                else value(i,j,k,comp) = interior;
            });
        }
    };
    fill_cell(m_p, 1); fill_cell(m_phi, 3); fill_cell(m_nvert, 2);
    fill_cell(m_ucat, 0); fill_cell(m_ucat_old, 0);

    // First make every physical normal boundary face authoritative. Inflow is
    // a globally normalized plane profile; constrained outflow has equal and
    // opposite outward flux. Wall/symmetry normal flux is exactly zero.
    if (impose_boundary_flux) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const amrex::Box face_domain =
                amrex::convert(domain, amrex::IntVect::TheDimensionVector(dir));
            for (int side = 0; side < 2; ++side) {
                const bool high = side != 0;
                const auto kind = m_boundary.sides[2 * dir + side].velocity;
                if (kind == CartesianBC::Periodic) continue;
                amrex::Real profile_scale = 0.0;
                if (kind == CartesianBC::Inflow) {
                    profile_scale = inlet_scales[2 * dir + side];
                }
                const int boundary_index = high ? face_domain.bigEnd(dir) : face_domain.smallEnd(dir);
                const int linear = m_boundary.inlet_profile == "linear_plane";
                const amrex::Real offset = m_boundary.profile_offset;
                const amrex::Real slope0 = m_boundary.profile_slope_0;
                const amrex::Real slope1 = m_boundary.profile_slope_1;
                const amrex::Real area = m_face_area[dir];
                const auto problo = m_geom.ProbLoArray();
                const auto dx = m_geom.CellSizeArray();
                const int td0 = (dir + 1) % AMREX_SPACEDIM;
                const int td1 = (dir + 2) % AMREX_SPACEDIM;
                const amrex::Real outflow_density = m_boundary.inlet_target_flux /
                    (static_cast<amrex::Real>(domain.length(td0) * domain.length(td1)));
                const bool constrain_outlet = m_boundary.constrain_outlet_flux;
                for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    auto const flux = m_ucont[dir].array(mfi);
                    amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                        if (index[dir] != boundary_index) return;
                        if (kind == CartesianBC::Inflow) {
                            const amrex::Real x0 = problo[td0] + (static_cast<amrex::Real>(index[td0]) + 0.5) * dx[td0];
                            const amrex::Real x1 = problo[td1] + (static_cast<amrex::Real>(index[td1]) + 0.5) * dx[td1];
                            const amrex::Real weight = offset + linear * (slope0 * x0 + slope1 * x1);
                            flux(i,j,k) = (high ? -1.0 : 1.0) * profile_scale * weight * area;
                        } else if (kind == CartesianBC::Outflow && constrain_outlet) {
                            flux(i,j,k) = (high ? 1.0 : -1.0) * outflow_density;
                        } else if (kind != CartesianBC::Outflow) {
                            flux(i,j,k) = 0.0;
                        }
                    });
                }
            }
        }
    }

    auto fill_face = [&](amrex::MultiFab& mf, int face_dir) {
        const amrex::Box face_domain = amrex::convert(domain, amrex::IntVect::TheDimensionVector(face_dir));
        const auto flo = face_domain.smallEnd();
        const auto fhi = face_domain.bigEnd();
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const value = mf.array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                if (index[0] >= flo[0] && index[0] <= fhi[0] && index[1] >= flo[1] && index[1] <= fhi[1]
                    && index[2] >= flo[2] && index[2] <= fhi[2]) return;
                int boundary_dir = -1; int side = 0;
                int source[AMREX_SPACEDIM] = {AMREX_D_DECL(i,j,k)};
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    if (source[d] < flo[d]) { if (boundary_dir < 0) { boundary_dir=d; side=0; } source[d]=flo[d]; }
                    if (source[d] > fhi[d]) { if (boundary_dir < 0) { boundary_dir=d; side=1; } source[d]=fhi[d]; }
                }
                const int kind = kinds[2 * boundary_dir + side];
                const amrex::Real interior = value(source[0],source[1],source[2]);
                bool odd = kind == static_cast<int>(CartesianBC::NoSlipWall) ||
                           ((kind == static_cast<int>(CartesianBC::SlipWall) || kind == static_cast<int>(CartesianBC::Symmetry)) && face_dir == boundary_dir) ||
                           (kind == static_cast<int>(CartesianBC::Inflow) && face_dir != boundary_dir);
                value(i,j,k) = odd ? -interior : interior;
            });
        }
    };
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        fill_face(m_ucont[dir], dir);
        fill_face(m_ucont_old[dir], dir);
        fill_face(m_ucont_older[dir], dir);
    }
    amrex::Gpu::streamSynchronize();
    m_physical_epoch = m_valid_epoch;
}

void VwisAmrExSolver::apply_boundary_pipeline(char const* stage)
{
    fill_ghost_cells();
    fill_physical_ghost_cells();
    (void)p3_diagnostics(stage, true);
}
