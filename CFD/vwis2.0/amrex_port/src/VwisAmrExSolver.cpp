#include "VwisAmrExSolver.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <stdexcept>

#include <iomanip>
#include <sstream>

namespace {
FieldLocation face_location(int dir)
{
    return dir == 0 ? FieldLocation::XFace :
           (dir == 1 ? FieldLocation::YFace : FieldLocation::ZFace);
}
} // namespace

VwisAmrExSolver::VwisAmrExSolver(
    amrex::Vector<int> const& n_cell, int max_grid_size, int nghost,
    amrex::RealBox const& physical_domain,
    amrex::Vector<int> const& is_periodic,
    CartesianBoundaryConfig const& boundary)
    : m_boundary(boundary), m_nghost(nghost)
{
    if (n_cell.size() != AMREX_SPACEDIM || is_periodic.size() != AMREX_SPACEDIM ||
        nghost < 0 || max_grid_size <= 0) {
        throw std::runtime_error("vwis: invalid n_cell/is_periodic dimensionality, nghost, or max_grid_size");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (n_cell[dir] <= 0 || (is_periodic[dir] != 0 && is_periodic[dir] != 1)) {
            throw std::runtime_error("vwis: n_cell must be positive and is_periodic must be 0 or 1");
        }
    }

    amrex::IntVect small_end(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect big_end(AMREX_D_DECL(n_cell[0] - 1, n_cell[1] - 1, n_cell[2] - 1));
    amrex::Box domain(small_end, big_end);
    amrex::Array<int, AMREX_SPACEDIM> periodicity{};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) periodicity[dir] = is_periodic[dir];
    m_geom.define(domain, &physical_domain, 0, periodicity.data());
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) m_dx[dir] = m_geom.CellSize()[dir];
    m_cell_volume = m_dx[0] * m_dx[1] * m_dx[2];
    m_face_area[0] = m_dx[1] * m_dx[2];
    m_face_area[1] = m_dx[0] * m_dx[2];
    m_face_area[2] = m_dx[0] * m_dx[1];
    m_ba.define(domain);
    m_ba.maxSize(max_grid_size);
    m_dm = amrex::DistributionMapping(m_ba);

    m_p.define(m_ba, m_dm, 1, m_nghost);
    m_phi.define(m_ba, m_dm, 1, m_nghost);
    m_nvert.define(m_ba, m_dm, 1, m_nghost);
    m_ucat.define(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    m_ucat_old.define(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    m_ucat_older.define(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    m_projection_rhs.define(m_ba, m_dm, 1, 0);
    m_projection_bc.define(m_ba, m_dm, 1, 1);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(dir));
        m_ucont[dir].define(face_ba, m_dm, 1, m_nghost);
        m_ucont_old[dir].define(face_ba, m_dm, 1, m_nghost);
        m_ucont_older[dir].define(face_ba, m_dm, 1, m_nghost);
        m_phi_operator_flux[dir].define(face_ba, m_dm, 1, 0);
    }
    register_fields();
    register_metric_metadata();
    validate_boundary_config();
    define_boundary_metadata();
}

void VwisAmrExSolver::register_fields()
{
    m_fields = {
        {"P", FieldLocation::Cell, 1, m_nghost, "legacy nondimensional pressure; physical conversion and datum unresolved", "n", "P", "cell-owned"},
        {"Phi", FieldLocation::Cell, 1, m_nghost, "legacy nondimensional pressure correction; same scale as P", "workspace", "Phi", "cell-owned"},
        {"Nvert", FieldLocation::Cell, 1, m_nghost, "legacy IBM classification code; not EB volume fraction", "n", "Nvert", "cell-owned"},
        {"Ucat", FieldLocation::Cell, AMREX_SPACEDIM, m_nghost, "legacy nondimensional Cartesian velocity", "n", "Ux,Uy,Uz", "cell-owned"},
        {"Ucat_old", FieldLocation::Cell, AMREX_SPACEDIM, m_nghost, "legacy nondimensional Cartesian velocity", "n-1", "Ux,Uy,Uz", "cell-owned"},
        {"Ucat_older", FieldLocation::Cell, AMREX_SPACEDIM, m_nghost, "legacy nondimensional Cartesian velocity", "n-2", "Ux,Uy,Uz", "cell-owned"},
    };
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        std::string axis(1, static_cast<char>('x' + dir));
        m_fields.push_back({"Ucont_" + axis, face_location(dir), 1, m_nghost,
                            "legacy nondimensional Cartesian volume flux (normal velocity times face area)", "n", axis + "-normal volume flux", "overlapping face boxes; lowest global box owns shared faces"});
        m_fields.push_back({"Ucont_" + axis + "_old", face_location(dir), 1, m_nghost,
                            "legacy nondimensional Cartesian volume flux (normal velocity times face area)", "n-1", axis + "-normal volume flux", "overlapping face boxes; lowest global box owns shared faces"});
        m_fields.push_back({"Ucont_" + axis + "_older", face_location(dir), 1, m_nghost,
                            "legacy nondimensional Cartesian volume flux (normal velocity times face area)", "n-2", axis + "-normal volume flux", "overlapping face boxes; lowest global box owns shared faces"});
    }
}

void VwisAmrExSolver::register_metric_metadata()
{
    auto real_text = [](amrex::Real value) {
        std::ostringstream stream;
        stream << std::setprecision(17) << value;
        return stream.str();
    };
    m_metrics = {
        {"coordinate_system", "level", "Cartesian", "orthogonal uniform Cartesian coordinates"},
        {"dx", "level", real_text(m_dx[0]) + "," + real_text(m_dx[1]) + "," + real_text(m_dx[2]),
         "cell widths (dx,dy,dz) from Geometry"},
        {"cell_volume", "cell", real_text(m_cell_volume), "dx*dy*dz; no Jacobian factor"},
        {"face_area_x", "x-face", real_text(m_face_area[0]), "dy*dz; multiplies normal velocity to form Ucont_x"},
        {"face_area_y", "y-face", real_text(m_face_area[1]), "dx*dz; multiplies normal velocity to form Ucont_y"},
        {"face_area_z", "z-face", real_text(m_face_area[2]), "dx*dy; multiplies normal velocity to form Ucont_z"},
        {"legacy_Aj_equivalent", "not allocated", real_text(1.0 / m_cell_volume),
         "inverse Cartesian cell volume for unit-index computational spacing; curvilinear Aj is unsupported"}
    };
}

void VwisAmrExSolver::initialize()
{
    const amrex::Real start = amrex::second();
    // Array4 handles are captured by value, so the GPU lambda captures no host state.
    for (amrex::MFIter mfi(m_p, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const p = m_p.array(mfi);
        auto const phi = m_phi.array(mfi);
        auto const nvert = m_nvert.array(mfi);
        auto const ucat = m_ucat.array(mfi);
        auto const ucat_old = m_ucat_old.array(mfi);
        auto const ucat_older = m_ucat_older.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = 0.0;
            phi(i, j, k) = 0.0;
            nvert(i, j, k) = 0.0;
            for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                ucat(i, j, k, comp) = 0.0;
                ucat_old(i, j, k, comp) = 0.0;
                ucat_older(i, j, k, comp) = 0.0;
            }
        });
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].setVal(0.0);
        m_ucont_old[dir].setVal(0.0);
        m_ucont_older[dir].setVal(0.0);
        m_phi_operator_flux[dir].setVal(0.0);
    }
    m_projection_rhs.setVal(0.0);
    m_projection_bc.setVal(0.0);
    m_time = 0.0;
    m_step = 0;
    m_history_depth = 1;
    mark_valid_modified();
    if (m_boundary.enabled) apply_boundary_pipeline("initialize");
    else fill_ghost_cells();
    m_initialize_seconds = amrex::second() - start;
}

void VwisAmrExSolver::mark_valid_modified()
{
    ++m_valid_epoch;
}

void VwisAmrExSolver::require_ghosts_fresh(char const* consumer) const
{
    if (m_halo_epoch != m_valid_epoch || (m_boundary.enabled && m_physical_epoch != m_valid_epoch)) {
        throw std::runtime_error(std::string("stale ghost read before ") + consumer +
                                 ": valid/halo/physical epochs=" + std::to_string(m_valid_epoch) + "/" +
                                 std::to_string(m_halo_epoch) + "/" + std::to_string(m_physical_epoch));
    }
}

void VwisAmrExSolver::fill_ghost_cells()
{
    m_p.FillBoundary(m_geom.periodicity());
    m_phi.FillBoundary(m_geom.periodicity());
    m_nvert.FillBoundary(m_geom.periodicity());
    m_ucat.FillBoundary(m_geom.periodicity());
    m_ucat_old.FillBoundary(m_geom.periodicity());
    m_ucat_older.FillBoundary(m_geom.periodicity());
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont_old[dir].OverrideSync(m_geom.periodicity());
        m_ucont_older[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
        m_ucont_old[dir].FillBoundary(m_geom.periodicity());
        m_ucont_older[dir].FillBoundary(m_geom.periodicity());
    }
    m_halo_epoch = m_valid_epoch;
}

void VwisAmrExSolver::sync_ucat_from_ucont()
{
    sync_ucat_from_ucont_impl(true);
}

void VwisAmrExSolver::sync_ucat_from_ucont_impl(bool reapply_boundary_flux)
{
    // Face valid regions overlap at Box boundaries.  OverrideSync first makes
    // the AMReX owner authoritative, then FillBoundary supplies stencil ghosts.
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
    }
    for (amrex::MFIter mfi(m_ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const ucat = m_ucat.array(mfi);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            auto const ucont = m_ucont[dir].const_array(mfi);
            const amrex::Real inverse_face_area = 1.0 / m_face_area[dir];
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int ip = i + (dir == 0);
                const int jp = j + (dir == 1);
                const int kp = k + (dir == 2);
                ucat(i, j, k, dir) = 0.5 * (ucont(i, j, k) + ucont(ip, jp, kp)) * inverse_face_area;
            });
        }
    }
    mark_valid_modified();
    if (m_boundary.enabled) {
        fill_ghost_cells();
        fill_physical_ghost_cells_impl(reapply_boundary_flux);
        (void)p3_diagnostics("sync-ucat-from-ucont", true);
    } else {
        fill_ghost_cells();
    }
}

void VwisAmrExSolver::sync_ucont_from_ucat()
{
    if (m_boundary.enabled) apply_boundary_pipeline("pre-sync-ucont-from-ucat");
    else m_ucat.FillBoundary(m_geom.periodicity());
    const amrex::Box& domain = m_geom.Domain();
    const int lo[AMREX_SPACEDIM] = {AMREX_D_DECL(domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2))};
    const int hi[AMREX_SPACEDIM] = {AMREX_D_DECL(domain.bigEnd(0), domain.bigEnd(1), domain.bigEnd(2))};
    const int periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(m_geom.isPeriodic(0), m_geom.isPeriodic(1), m_geom.isPeriodic(2))};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const amrex::Real face_area = m_face_area[dir];
        for (amrex::MFIter mfi(m_ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const ucont = m_ucont[dir].array(mfi);
            auto const ucat = m_ucat.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int face = dir == 0 ? i : (dir == 1 ? j : k);
                int il = i - (dir == 0); int jl = j - (dir == 1); int kl = k - (dir == 2);
                int ir = i;               int jr = j;               int kr = k;
                if (!periodic[dir] && face == lo[dir]) {
                    ucont(i, j, k) = ucat(ir, jr, kr, dir) * face_area;
                } else if (!periodic[dir] && face == hi[dir] + 1) {
                    ucont(i, j, k) = ucat(il, jl, kl, dir) * face_area;
                } else {
                    ucont(i, j, k) = 0.5 * (ucat(il, jl, kl, dir) + ucat(ir, jr, kr, dir)) * face_area;
                }
            });
        }
        // This is required even though the manufactured values agree: it is the
        // canonical shared-face ownership rule for future independently written boxes.
        m_ucont[dir].OverrideSync(m_geom.periodicity());
        m_ucont[dir].FillBoundary(m_geom.periodicity());
    }
    mark_valid_modified();
    if (m_boundary.enabled) apply_boundary_pipeline("sync-ucont-from-ucat");
}
