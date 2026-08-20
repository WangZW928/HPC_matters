#include "VwisAmrExSolver.H"

#include <AMReX_BC_TYPES.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <stdexcept>

namespace {
FieldLocation face_location(int dir)
{
    return dir == 0 ? FieldLocation::XFace :
           (dir == 1 ? FieldLocation::YFace : FieldLocation::ZFace);
}

char const* location_name(FieldLocation location)
{
    switch (location) {
    case FieldLocation::Cell: return "cell";
    case FieldLocation::XFace: return "x-face";
    case FieldLocation::YFace: return "y-face";
    case FieldLocation::ZFace: return "z-face";
    }
    return "unknown";
}
} // namespace

VwisAmrExSolver::VwisAmrExSolver(
    amrex::Vector<int> const& n_cell, int max_grid_size, int nghost,
    amrex::RealBox const& physical_domain,
    amrex::Vector<int> const& is_periodic)
    : m_nghost(nghost)
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
    m_ba.define(domain);
    m_ba.maxSize(max_grid_size);
    m_dm = amrex::DistributionMapping(m_ba);

    m_p.define(m_ba, m_dm, 1, m_nghost);
    m_phi.define(m_ba, m_dm, 1, m_nghost);
    m_nvert.define(m_ba, m_dm, 1, m_nghost);
    m_ucat.define(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    m_ucat_old.define(m_ba, m_dm, AMREX_SPACEDIM, m_nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::BoxArray face_ba = amrex::convert(m_ba, amrex::IntVect::TheDimensionVector(dir));
        m_ucont[dir].define(face_ba, m_dm, 1, m_nghost);
        m_ucont_old[dir].define(face_ba, m_dm, 1, m_nghost);
        m_ucont_older[dir].define(face_ba, m_dm, 1, m_nghost);
    }
    register_fields();
    define_boundary_metadata();
}

void VwisAmrExSolver::register_fields()
{
    m_fields = {
        {"pressure", FieldLocation::Cell, 1, m_nghost, "legacy nondimensional pressure (conversion unresolved)", "n"},
        {"phi", FieldLocation::Cell, 1, m_nghost, "pressure-correction units unresolved", "workspace"},
        {"nvert", FieldLocation::Cell, 1, m_nghost, "legacy IBM classification code", "n"},
        {"ucat", FieldLocation::Cell, AMREX_SPACEDIM, m_nghost, "legacy nondimensional Cartesian velocity", "n"},
        {"ucat_old", FieldLocation::Cell, AMREX_SPACEDIM, m_nghost, "legacy nondimensional Cartesian velocity", "n-1"},
    };
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        std::string axis(1, static_cast<char>('x' + dir));
        m_fields.push_back({"ucont_" + axis, face_location(dir), 1, m_nghost,
                            "contravariant normal flux/velocity; normalization unresolved", "n"});
        m_fields.push_back({"ucont_" + axis + "_old", face_location(dir), 1, m_nghost,
                            "contravariant normal flux/velocity; normalization unresolved", "n-1"});
        m_fields.push_back({"ucont_" + axis + "_older", face_location(dir), 1, m_nghost,
                            "contravariant normal flux/velocity; normalization unresolved", "n-2"});
    }
}

void VwisAmrExSolver::define_boundary_metadata()
{
    // ext_dir assigns future physical-ghost ownership to a BC functor.
    // FillBoundary below is deliberately only inter-box/periodic communication.
    m_cell_bcs.assign(AMREX_SPACEDIM, amrex::BCRec{});
    for (auto& bc : m_cell_bcs) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bc.setLo(dir, amrex::BCType::ext_dir);
            bc.setHi(dir, amrex::BCType::ext_dir);
        }
    }
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
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = 0.0;
            phi(i, j, k) = 0.0;
            nvert(i, j, k) = 0.0;
            for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
                ucat(i, j, k, comp) = 0.0;
                ucat_old(i, j, k, comp) = 0.0;
            }
        });
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].setVal(0.0);
        m_ucont_old[dir].setVal(0.0);
        m_ucont_older[dir].setVal(0.0);
    }
    fill_ghost_cells();
    m_initialize_seconds = amrex::second() - start;
}

void VwisAmrExSolver::fill_ghost_cells()
{
    m_p.FillBoundary(m_geom.periodicity());
    m_phi.FillBoundary(m_geom.periodicity());
    m_nvert.FillBoundary(m_geom.periodicity());
    m_ucat.FillBoundary(m_geom.periodicity());
    m_ucat_old.FillBoundary(m_geom.periodicity());
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_ucont[dir].FillBoundary(m_geom.periodicity());
        m_ucont_old[dir].FillBoundary(m_geom.periodicity());
        m_ucont_older[dir].FillBoundary(m_geom.periodicity());
    }
}

void VwisAmrExSolver::advance_one_step(amrex::Real dt)
{
    if (dt <= 0.0) throw std::runtime_error("vwis.dt must be positive even for the P1 no-op");
    const amrex::Real start = amrex::second();
    fill_ghost_cells();
    m_last_noop_seconds = amrex::second() - start;
}

void VwisAmrExSolver::diagnostics() const
{
    amrex::Print() << "VWiS AMReX P1: boxes=" << m_ba.size()
                   << ", ranks=" << amrex::ParallelDescriptor::NProcs()
                   << ", ghosts=" << m_nghost
                   << ", max(|P|)=" << m_p.norm0(0, 0, true)
                   << ", init_s=" << m_initialize_seconds
                   << ", noop_s=" << m_last_noop_seconds << "\n";
    for (auto const& field : m_fields) {
        amrex::Print() << "  field " << field.name << " location=" << location_name(field.location)
                       << " nComp=" << field.components << " nGrow=" << field.ghost_cells
                       << " layer=" << field.time_layer << " units=" << field.units << "\n";
    }
}

void VwisAmrExSolver::write_metadata_manifest(std::string const& path) const
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write P1 metadata manifest: " + path);
    output << "{\n  \"schema\": \"vwis-amrex-p1-metadata-v1\",\n"
           << "  \"payload_written\": false,\n"
           << "  \"note\": \"Not a plotfile or checkpoint; no restart payload exists in P1.\",\n"
           << "  \"fields\": [\n";
    for (std::size_t i = 0; i < m_fields.size(); ++i) {
        auto const& field = m_fields[i];
        output << "    {\"name\": \"" << field.name << "\", \"location\": \""
               << location_name(field.location) << "\", \"components\": " << field.components
               << ", \"ngrow\": " << field.ghost_cells << ", \"units\": \"" << field.units
               << "\", \"time_layer\": \"" << field.time_layer << "\"}"
               << (i + 1 == m_fields.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}
