#include "AVWiSCoordinateMapping.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void require_nodal_coordinates(amrex::MultiFab const& node_coordinates_nd,
                               char const* mapping_name)
{
    if (node_coordinates_nd.nComp() != AMREX_SPACEDIM ||
        !node_coordinates_nd.boxArray().ixType().nodeCentered()) {
        throw std::runtime_error(std::string("AVWiS ") + mapping_name
                                 + " mapping requires a three-component nodal MultiFab");
    }
}

void require_positive_logical_spacing(LogicalGrid const& logical)
{
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!std::isfinite(logical.lower[dir]) ||
            !std::isfinite(logical.spacing[dir]) || !(logical.spacing[dir] > 0.0)) {
            throw std::runtime_error("AVWiS logical mapping origin must be finite and spacing positive");
        }
    }
}
} // namespace

LogicalGrid LogicalGrid::from_cartesian_geometry(amrex::Geometry const& geometry)
{
    static_assert(AMREX_SPACEDIM == 3, "AVWiS fixed curvilinear metrics require a 3-D AMReX build");
    LogicalGrid logical;
    logical.cell_domain = geometry.Domain();
    auto const* prob_lo = geometry.ProbLo();
    auto const* dx = geometry.CellSize();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        logical.lower[dir] = prob_lo[dir];
        logical.spacing[dir] = dx[dir];
        logical.periodic[dir] = geometry.isPeriodic(dir);
    }
    return logical;
}

void IdentityCoordinateMapping::fill_nodes(
    amrex::MultiFab& node_coordinates_nd, LogicalGrid const& logical) const
{
    require_nodal_coordinates(node_coordinates_nd, "identity");
    require_positive_logical_spacing(logical);

    const int ilo = logical.cell_domain.smallEnd(0);
    const int jlo = logical.cell_domain.smallEnd(1);
    const int klo = logical.cell_domain.smallEnd(2);
    const amrex::Real xlo = logical.lower[0];
    const amrex::Real ylo = logical.lower[1];
    const amrex::Real zlo = logical.lower[2];
    const amrex::Real dxi = logical.spacing[0];
    const amrex::Real deta = logical.spacing[1];
    const amrex::Real dzet = logical.spacing[2];
    for (amrex::MFIter mfi(node_coordinates_nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = node_coordinates_nd.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            xyz(i, j, k, 0) = xlo + static_cast<amrex::Real>(i - ilo) * dxi;
            xyz(i, j, k, 1) = ylo + static_cast<amrex::Real>(j - jlo) * deta;
            xyz(i, j, k, 2) = zlo + static_cast<amrex::Real>(k - klo) * dzet;
        });
    }
    amrex::Gpu::streamSynchronize();

    // Periodic coordinate ghosts carry a physical translation and therefore
    // must not be overwritten by periodic image values. Interior shared nodes
    // still use the canonical AMReX owner and non-periodic halo exchange.
    node_coordinates_nd.OverrideSync(amrex::Periodicity::NonPeriodic());
    node_coordinates_nd.FillBoundary(amrex::Periodicity::NonPeriodic());
}

AnalyticOrthogonalCoordinateMapping::AnalyticOrthogonalCoordinateMapping(
    AnalyticOrthogonalMappingParameters parameters)
    : m_parameters(parameters)
{
    // A finite strict margin also prevents a nominally positive derivative
    // from collapsing to zero after roundoff at cos(2*pi*t) = +/-1.
    amrex::Real const stretch_limit = 1.0 - 64.0 * std::numeric_limits<amrex::Real>::epsilon();
    amrex::Real minimum_jacobian = 1.0;
    amrex::Real maximum_jacobian = 1.0;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!std::isfinite(m_parameters.scale[dir]) || !(m_parameters.scale[dir] > 0.0)) {
            throw std::invalid_argument("AVWiS analytic orthogonal mapping scale must be finite and positive");
        }
        if (!std::isfinite(m_parameters.stretch[dir]) ||
            !(std::abs(m_parameters.stretch[dir]) < stretch_limit)) {
            throw std::invalid_argument(
                "AVWiS analytic orthogonal mapping requires finite abs(stretch) < 1 with a roundoff margin");
        }
        minimum_jacobian *= m_parameters.scale[dir]
                          * (1.0 - std::abs(m_parameters.stretch[dir]));
        maximum_jacobian *= m_parameters.scale[dir]
                          * (1.0 + std::abs(m_parameters.stretch[dir]));
        if (!std::isfinite(maximum_jacobian) ||
            !(minimum_jacobian > std::numeric_limits<amrex::Real>::min())) {
            throw std::invalid_argument(
                "AVWiS analytic orthogonal mapping parameters risk a non-representable Jacobian");
        }
    }
}

void AnalyticOrthogonalCoordinateMapping::fill_nodes(
    amrex::MultiFab& node_coordinates_nd, LogicalGrid const& logical) const
{
    require_nodal_coordinates(node_coordinates_nd, "analytic orthogonal");
    require_positive_logical_spacing(logical);

    const int ilo = logical.cell_domain.smallEnd(0);
    const int jlo = logical.cell_domain.smallEnd(1);
    const int klo = logical.cell_domain.smallEnd(2);
    const amrex::Real xlo = logical.lower[0];
    const amrex::Real ylo = logical.lower[1];
    const amrex::Real zlo = logical.lower[2];
    const amrex::Real dxi = logical.spacing[0];
    const amrex::Real deta = logical.spacing[1];
    const amrex::Real dzet = logical.spacing[2];
    const amrex::Real lx = dxi * static_cast<amrex::Real>(logical.cell_domain.length(0));
    const amrex::Real ly = deta * static_cast<amrex::Real>(logical.cell_domain.length(1));
    const amrex::Real lz = dzet * static_cast<amrex::Real>(logical.cell_domain.length(2));
    const amrex::Real sx = m_parameters.scale[0];
    const amrex::Real sy = m_parameters.scale[1];
    const amrex::Real sz = m_parameters.scale[2];
    const amrex::Real ax = m_parameters.stretch[0];
    const amrex::Real ay = m_parameters.stretch[1];
    const amrex::Real az = m_parameters.stretch[2];
    for (amrex::MFIter mfi(node_coordinates_nd, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = node_coordinates_nd.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const xi = xlo + static_cast<amrex::Real>(i - ilo) * dxi;
            amrex::Real const eta = ylo + static_cast<amrex::Real>(j - jlo) * deta;
            amrex::Real const zeta = zlo + static_cast<amrex::Real>(k - klo) * dzet;
            xyz(i, j, k, 0) = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                xi, xlo, lx, sx, ax);
            xyz(i, j, k, 1) = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                eta, ylo, ly, sy, ay);
            xyz(i, j, k, 2) = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                zeta, zlo, lz, sz, az);
        });
    }
    amrex::Gpu::streamSynchronize();

    // The formula extends analytically into non-periodic ghosts.  For a
    // periodic direction, t->t+1 adds exactly scale*L while all derivatives
    // repeat, so coordinate ghosts retain the required physical translation.
    node_coordinates_nd.OverrideSync(amrex::Periodicity::NonPeriodic());
    node_coordinates_nd.FillBoundary(amrex::Periodicity::NonPeriodic());
}

std::unique_ptr<CoordinateMapping> make_coordinate_mapping(
    std::string const& type, AnalyticOrthogonalMappingParameters const& analytic_parameters)
{
    if (type == "identity") {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (analytic_parameters.scale[dir] != 1.0 ||
                analytic_parameters.stretch[dir] != 0.0) {
                throw std::invalid_argument(
                    "AVWiS identity mapping does not accept analytic orthogonal parameters");
            }
        }
        return std::make_unique<IdentityCoordinateMapping>();
    }
    if (type == "analytic_orthogonal") {
        return std::make_unique<AnalyticOrthogonalCoordinateMapping>(analytic_parameters);
    }
    throw std::invalid_argument(
        "AVWiS unknown coordinate mapping type '" + type +
        "' (supported: identity, analytic_orthogonal; general non-orthogonal/curved "
        "mappings require C3/C4 and are not implemented)");
}
