#include "AVWiSMetricData.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
struct Vec3 {
    amrex::Real x;
    amrex::Real y;
    amrex::Real z;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 add(Vec3 a, Vec3 b) noexcept
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 subtract(Vec3 a, Vec3 b) noexcept
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 scale(Vec3 a, amrex::Real value) noexcept
{
    return {value * a.x, value * a.y, value * a.z};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real dot(Vec3 a, Vec3 b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 cross(Vec3 a, Vec3 b) noexcept
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real magnitude(Vec3 a) noexcept
{
    return std::sqrt(dot(a, a));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 node(amrex::Array4<amrex::Real const> const& xyz, int i, int j, int k) noexcept
{
    return {xyz(i, j, k, 0), xyz(i, j, k, 1), xyz(i, j, k, 2)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 average2(Vec3 a, Vec3 b) noexcept
{
    return scale(add(a, b), 0.5);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 cell_center(amrex::Array4<amrex::Real const> const& xyz, int i, int j, int k) noexcept
{
    Vec3 result{0.0, 0.0, 0.0};
    for (int dk = 0; dk <= 1; ++dk) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) result = add(result, node(xyz, i + di, j + dj, k + dk));
        }
    }
    return scale(result, 0.125);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void face_vertices(amrex::Array4<amrex::Real const> const& xyz, int dir,
                   int i, int j, int k, Vec3& p0, Vec3& p1, Vec3& p2, Vec3& p3) noexcept
{
    if (dir == 0) {
        p0 = node(xyz, i, j, k);
        p1 = node(xyz, i, j + 1, k);
        p2 = node(xyz, i, j + 1, k + 1);
        p3 = node(xyz, i, j, k + 1);
    } else if (dir == 1) {
        p0 = node(xyz, i, j, k);
        p1 = node(xyz, i, j, k + 1);
        p2 = node(xyz, i + 1, j, k + 1);
        p3 = node(xyz, i + 1, j, k);
    } else {
        p0 = node(xyz, i, j, k);
        p1 = node(xyz, i + 1, j, k);
        p2 = node(xyz, i + 1, j + 1, k);
        p3 = node(xyz, i, j + 1, k);
    }
}

/** Fixed p0--p2 diagonal; both triangle vectors point toward increasing xi^dir. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void face_geometry(amrex::Array4<amrex::Real const> const& xyz, int dir,
                   int i, int j, int k, Vec3& area, amrex::Real& first_moment) noexcept
{
    Vec3 p0{}, p1{}, p2{}, p3{};
    face_vertices(xyz, dir, i, j, k, p0, p1, p2, p3);
    Vec3 const triangle0 = scale(cross(subtract(p1, p0), subtract(p2, p0)), 0.5);
    Vec3 const triangle1 = scale(cross(subtract(p2, p0), subtract(p3, p0)), 0.5);
    area = add(triangle0, triangle1);
    Vec3 const centroid0 = scale(add(add(p0, p1), p2), 1.0 / 3.0);
    Vec3 const centroid1 = scale(add(add(p0, p2), p3), 1.0 / 3.0);
    first_moment = dot(centroid0, triangle0) + dot(centroid1, triangle1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Vec3 face_center(amrex::Array4<amrex::Real const> const& xyz, int dir,
                 int i, int j, int k) noexcept
{
    Vec3 p0{}, p1{}, p2{}, p3{};
    face_vertices(xyz, dir, i, j, k, p0, p1, p2, p3);
    return scale(add(add(p0, p1), add(p2, p3)), 0.25);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void cell_basis(amrex::Array4<amrex::Real const> const& xyz, int i, int j, int k,
                amrex::Real h0, amrex::Real h1, amrex::Real h2, Vec3 basis[3]) noexcept
{
    basis[0] = scale(subtract(face_center(xyz, 0, i + 1, j, k),
                              face_center(xyz, 0, i, j, k)), 1.0 / h0);
    basis[1] = scale(subtract(face_center(xyz, 1, i, j + 1, k),
                              face_center(xyz, 1, i, j, k)), 1.0 / h1);
    basis[2] = scale(subtract(face_center(xyz, 2, i, j, k + 1),
                              face_center(xyz, 2, i, j, k)), 1.0 / h2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void face_basis(amrex::Array4<amrex::Real const> const& xyz,
                amrex::Array4<amrex::Real const> const& centers,
                int dir, int i, int j, int k,
                amrex::Real h0, amrex::Real h1, amrex::Real h2, Vec3 basis[3]) noexcept
{
    int il = i - (dir == 0); int jl = j - (dir == 1); int kl = k - (dir == 2);
    int ir = i;              int jr = j;              int kr = k;
    Vec3 const left{centers(il, jl, kl, 0), centers(il, jl, kl, 1), centers(il, jl, kl, 2)};
    Vec3 const right{centers(ir, jr, kr, 0), centers(ir, jr, kr, 1), centers(ir, jr, kr, 2)};
    const amrex::Real h[3] = {h0, h1, h2};
    basis[dir] = scale(subtract(right, left), 1.0 / h[dir]);

    if (dir == 0) {
        basis[1] = scale(subtract(average2(node(xyz, i, j + 1, k), node(xyz, i, j + 1, k + 1)),
                                  average2(node(xyz, i, j, k), node(xyz, i, j, k + 1))), 1.0 / h1);
        basis[2] = scale(subtract(average2(node(xyz, i, j, k + 1), node(xyz, i, j + 1, k + 1)),
                                  average2(node(xyz, i, j, k), node(xyz, i, j + 1, k))), 1.0 / h2);
    } else if (dir == 1) {
        basis[0] = scale(subtract(average2(node(xyz, i + 1, j, k), node(xyz, i + 1, j, k + 1)),
                                  average2(node(xyz, i, j, k), node(xyz, i, j, k + 1))), 1.0 / h0);
        basis[2] = scale(subtract(average2(node(xyz, i, j, k + 1), node(xyz, i + 1, j, k + 1)),
                                  average2(node(xyz, i, j, k), node(xyz, i + 1, j, k))), 1.0 / h2);
    } else {
        basis[0] = scale(subtract(average2(node(xyz, i + 1, j, k), node(xyz, i + 1, j + 1, k)),
                                  average2(node(xyz, i, j, k), node(xyz, i, j + 1, k))), 1.0 / h0);
        basis[1] = scale(subtract(average2(node(xyz, i, j + 1, k), node(xyz, i + 1, j + 1, k)),
                                  average2(node(xyz, i, j, k), node(xyz, i + 1, j, k))), 1.0 / h1);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real inverse_basis(Vec3 const basis[3], Vec3 gradients[3]) noexcept
{
    Vec3 const cofactor0 = cross(basis[1], basis[2]);
    Vec3 const cofactor1 = cross(basis[2], basis[0]);
    Vec3 const cofactor2 = cross(basis[0], basis[1]);
    amrex::Real const determinant = dot(basis[0], cofactor0);
    if (determinant > 0.0) {
        gradients[0] = scale(cofactor0, 1.0 / determinant);
        gradients[1] = scale(cofactor1, 1.0 / determinant);
        gradients[2] = scale(cofactor2, 1.0 / determinant);
    } else {
        gradients[0] = gradients[1] = gradients[2] = {0.0, 0.0, 0.0};
    }
    return determinant;
}

void reduce_min(amrex::Real& value)
{
    amrex::ParallelDescriptor::ReduceRealMin(value);
}

void reduce_max(amrex::Real& value)
{
    amrex::ParallelDescriptor::ReduceRealMax(value);
}
} // namespace

void MetricData::define(amrex::BoxArray const& cell_boxes,
                        amrex::DistributionMapping const& distribution, int nghost)
{
    static_assert(AMREX_SPACEDIM == 3, "AVWiS fixed curvilinear metrics require a 3-D AMReX build");
    if (!cell_boxes.ixType().cellCentered() || nghost < 1) {
        throw std::runtime_error("AVWiS MetricData requires cell boxes and at least one ghost cell");
    }
    m_cell_boxes = cell_boxes;
    m_distribution = distribution;
    m_nghost = nghost;

    m_node_coordinates_nd.define(amrex::convert(cell_boxes, amrex::IntVect::TheNodeVector()),
                                 distribution, AMREX_SPACEDIM, nghost + 1);
    m_cell_center_coordinates_cc.define(cell_boxes, distribution, AMREX_SPACEDIM, nghost + 1);
    m_mapping_jacobian_cc.define(cell_boxes, distribution, 1, nghost);
    m_inverse_mapping_jacobian_cc.define(cell_boxes, distribution, 1, nghost);
    m_grad_xi_cc.define(cell_boxes, distribution, AMREX_SPACEDIM * AMREX_SPACEDIM, nghost);
    m_area_cofactor_cc.define(cell_boxes, distribution, AMREX_SPACEDIM * AMREX_SPACEDIM, nghost);
    m_cell_volume_cc.define(cell_boxes, distribution, 1, nghost);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        auto const face_boxes = amrex::convert(cell_boxes, amrex::IntVect::TheDimensionVector(dir));
        m_face_area_vector_fc[dir].define(face_boxes, distribution, AMREX_SPACEDIM, nghost);
        m_face_gradient_metric_fc[dir].define(face_boxes, distribution, AMREX_SPACEDIM, nghost);
        m_projection_beta_fc[dir].define(face_boxes, distribution, 1, 0);
    }
    m_mapping_id.clear();
    m_epoch = 0;
    m_defined = true;
    m_built = false;
}

void MetricData::build(CoordinateMapping const& mapping, LogicalGrid const& logical,
                       amrex::Geometry const& geometry, MetricTolerance const& tolerance)
{
    if (m_epoch != 0) throw std::runtime_error("AVWiS MetricData::build called twice; use rebuild explicitly");
    rebuild(mapping, logical, geometry, tolerance);
}

void MetricData::rebuild(CoordinateMapping const& mapping, LogicalGrid const& logical,
                         amrex::Geometry const& geometry, MetricTolerance const& tolerance)
{
    if (!m_defined) throw std::runtime_error("AVWiS MetricData must be defined before build/rebuild");
    if (logical.cell_domain != geometry.Domain() || m_cell_boxes.minimalBox() != geometry.Domain()) {
        throw std::runtime_error("AVWiS MetricData logical, Geometry, and BoxArray domains differ");
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (!(logical.spacing[dir] > 0.0) || logical.periodic[dir] != geometry.isPeriodic(dir)) {
            throw std::runtime_error("AVWiS MetricData logical spacing/periodicity is inconsistent");
        }
    }

    m_logical = logical;
    m_periodicity = geometry.periodicity();
    std::string const next_mapping_id = mapping.id();
    if (next_mapping_id.empty()) throw std::runtime_error("AVWiS coordinate mapping id must not be empty");
    // A failed rebuild must never leave partially replaced fields observable as
    // a valid epoch. Access remains unavailable until validation succeeds.
    m_built = false;
    m_mapping_id = next_mapping_id;
    mapping.fill_nodes(m_node_coordinates_nd, logical);

    const amrex::Real h0 = logical.spacing[0];
    const amrex::Real h1 = logical.spacing[1];
    const amrex::Real h2 = logical.spacing[2];
    for (amrex::MFIter mfi(m_cell_center_coordinates_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = m_node_coordinates_nd.const_array(mfi);
        auto const centers = m_cell_center_coordinates_cc.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Vec3 const center = cell_center(xyz, i, j, k);
            centers(i, j, k, 0) = center.x;
            centers(i, j, k, 1) = center.y;
            centers(i, j, k, 2) = center.z;
        });
    }
    for (amrex::MFIter mfi(m_cell_volume_cc, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = m_node_coordinates_nd.const_array(mfi);
        auto const mapping_jacobian = m_mapping_jacobian_cc.array(mfi);
        auto const inverse_mapping_jacobian = m_inverse_mapping_jacobian_cc.array(mfi);
        auto const grad_xi = m_grad_xi_cc.array(mfi);
        auto const cofactor = m_area_cofactor_cc.array(mfi);
        auto const volume = m_cell_volume_cc.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Vec3 basis[3]{};
            Vec3 gradients[3]{};
            cell_basis(xyz, i, j, k, h0, h1, h2, basis);
            amrex::Real const determinant = inverse_basis(basis, gradients);
            mapping_jacobian(i, j, k) = determinant;
            inverse_mapping_jacobian(i, j, k) = determinant > 0.0 ? 1.0 / determinant : 0.0;
            for (int m = 0; m < 3; ++m) {
                grad_xi(i, j, k, 3 * m) = gradients[m].x;
                grad_xi(i, j, k, 3 * m + 1) = gradients[m].y;
                grad_xi(i, j, k, 3 * m + 2) = gradients[m].z;
                Vec3 const area_cofactor = scale(gradients[m], determinant);
                cofactor(i, j, k, 3 * m) = area_cofactor.x;
                cofactor(i, j, k, 3 * m + 1) = area_cofactor.y;
                cofactor(i, j, k, 3 * m + 2) = area_cofactor.z;
            }

            amrex::Real volume_sum = 0.0;
            for (int dir = 0; dir < 3; ++dir) {
                Vec3 unused{};
                amrex::Real low_moment = 0.0;
                amrex::Real high_moment = 0.0;
                face_geometry(xyz, dir, i, j, k, unused, low_moment);
                face_geometry(xyz, dir, i + (dir == 0), j + (dir == 1), k + (dir == 2),
                              unused, high_moment);
                volume_sum += high_moment - low_moment;
            }
            volume(i, j, k) = volume_sum / 3.0;
        });
    }
    amrex::Gpu::streamSynchronize();

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        for (amrex::MFIter mfi(m_face_area_vector_fc[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const xyz = m_node_coordinates_nd.const_array(mfi);
            auto const centers = m_cell_center_coordinates_cc.const_array(mfi);
            auto const area = m_face_area_vector_fc[dir].array(mfi);
            auto const q = m_face_gradient_metric_fc[dir].array(mfi);
            auto const beta = m_projection_beta_fc[dir].array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Vec3 face_area{};
                amrex::Real unused = 0.0;
                face_geometry(xyz, dir, i, j, k, face_area, unused);
                area(i, j, k, 0) = face_area.x;
                area(i, j, k, 1) = face_area.y;
                area(i, j, k, 2) = face_area.z;
                Vec3 basis[3]{};
                Vec3 gradients[3]{};
                face_basis(xyz, centers, dir, i, j, k, h0, h1, h2, basis);
                inverse_basis(basis, gradients);
                for (int m = 0; m < 3; ++m) q(i, j, k, m) = dot(face_area, gradients[m]);
            });
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                beta(i, j, k) = 1.0;
            });
        }
        m_face_area_vector_fc[dir].OverrideSync(m_periodicity);
        m_face_gradient_metric_fc[dir].OverrideSync(m_periodicity);
        m_projection_beta_fc[dir].OverrideSync(m_periodicity);
        m_face_area_vector_fc[dir].FillBoundary(m_periodicity);
        m_face_gradient_metric_fc[dir].FillBoundary(m_periodicity);
    }
    m_cell_center_coordinates_cc.FillBoundary(m_periodicity);
    m_mapping_jacobian_cc.FillBoundary(m_periodicity);
    m_inverse_mapping_jacobian_cc.FillBoundary(m_periodicity);
    m_grad_xi_cc.FillBoundary(m_periodicity);
    m_area_cofactor_cc.FillBoundary(m_periodicity);
    m_cell_volume_cc.FillBoundary(m_periodicity);
    amrex::Gpu::streamSynchronize();

    m_built = true;
    MetricDiagnostics const diagnostics = validate(tolerance);
    if (!diagnostics.passed) {
        m_built = false;
        std::ostringstream message;
        message << "AVWiS metric validation failed for mapping '" << m_mapping_id
                << "': min(Jx)=" << diagnostics.minimum_mapping_jacobian
                << ", min(V)=" << diagnostics.minimum_cell_volume
                << ", min(|S|)=" << diagnostics.minimum_face_area
                << ", reciprocal=" << diagnostics.maximum_reciprocal_relative_error
                << ", GCL=" << diagnostics.maximum_gcl_relative_error
                << ", reciprocity=" << diagnostics.maximum_reciprocity_relative_error;
        throw std::runtime_error(message.str());
    }
    ++m_epoch;
}

MetricDiagnostics MetricData::validate(MetricTolerance const& tolerance) const
{
    require_ready("validate");
    amrex::MultiFab checks(m_cell_boxes, m_distribution, 5, 0);
    const amrex::Real h0 = m_logical.spacing[0];
    const amrex::Real h1 = m_logical.spacing[1];
    const amrex::Real h2 = m_logical.spacing[2];
    const amrex::Real tiny = std::numeric_limits<amrex::Real>::min();
    for (amrex::MFIter mfi(checks, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const xyz = m_node_coordinates_nd.const_array(mfi);
        auto const jacobian = m_mapping_jacobian_cc.const_array(mfi);
        auto const inverse_jacobian = m_inverse_mapping_jacobian_cc.const_array(mfi);
        auto const cofactor = m_area_cofactor_cc.const_array(mfi);
        auto const volume = m_cell_volume_cc.const_array(mfi);
        auto const sx = m_face_area_vector_fc[0].const_array(mfi);
        auto const sy = m_face_area_vector_fc[1].const_array(mfi);
        auto const sz = m_face_area_vector_fc[2].const_array(mfi);
        auto const out = checks.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const jx = jacobian(i, j, k);
            out(i, j, k, 0) = jx;
            out(i, j, k, 1) = volume(i, j, k);
            out(i, j, k, 2) = std::abs(jx * inverse_jacobian(i, j, k) - 1.0)
                            / amrex::max(amrex::Real(1.0), std::abs(jx * inverse_jacobian(i, j, k)));

            Vec3 const xlo{sx(i, j, k, 0), sx(i, j, k, 1), sx(i, j, k, 2)};
            Vec3 const xhi{sx(i + 1, j, k, 0), sx(i + 1, j, k, 1), sx(i + 1, j, k, 2)};
            Vec3 const ylo{sy(i, j, k, 0), sy(i, j, k, 1), sy(i, j, k, 2)};
            Vec3 const yhi{sy(i, j + 1, k, 0), sy(i, j + 1, k, 1), sy(i, j + 1, k, 2)};
            Vec3 const zlo{sz(i, j, k, 0), sz(i, j, k, 1), sz(i, j, k, 2)};
            Vec3 const zhi{sz(i, j, k + 1, 0), sz(i, j, k + 1, 1), sz(i, j, k + 1, 2)};
            Vec3 const closure = add(add(subtract(xhi, xlo), subtract(yhi, ylo)), subtract(zhi, zlo));
            amrex::Real const area_sum = magnitude(xlo) + magnitude(xhi) + magnitude(ylo)
                                       + magnitude(yhi) + magnitude(zlo) + magnitude(zhi);
            out(i, j, k, 3) = magnitude(closure) / amrex::max(area_sum, tiny);

            Vec3 basis[3]{};
            cell_basis(xyz, i, j, k, h0, h1, h2, basis);
            amrex::Real reciprocity = 0.0;
            for (int m = 0; m < 3; ++m) {
                Vec3 const a{cofactor(i, j, k, 3 * m), cofactor(i, j, k, 3 * m + 1),
                             cofactor(i, j, k, 3 * m + 2)};
                for (int direction = 0; direction < 3; ++direction) {
                    amrex::Real const expected = m == direction ? jx : 0.0;
                    reciprocity = amrex::max(reciprocity,
                        std::abs(dot(a, basis[direction]) - expected) / amrex::max(std::abs(jx), tiny));
                }
            }
            out(i, j, k, 4) = reciprocity;
        });
    }
    amrex::Gpu::streamSynchronize();

    MetricDiagnostics diagnostics;
    diagnostics.minimum_mapping_jacobian = checks.min(0, 0, true);
    diagnostics.minimum_cell_volume = checks.min(1, 0, true);
    diagnostics.maximum_reciprocal_relative_error = checks.norm0(2, 0, true);
    diagnostics.maximum_gcl_relative_error = checks.norm0(3, 0, true);
    diagnostics.maximum_reciprocity_relative_error = checks.norm0(4, 0, true);
    reduce_min(diagnostics.minimum_mapping_jacobian);
    reduce_min(diagnostics.minimum_cell_volume);
    reduce_max(diagnostics.maximum_reciprocal_relative_error);
    reduce_max(diagnostics.maximum_gcl_relative_error);
    reduce_max(diagnostics.maximum_reciprocity_relative_error);

    diagnostics.minimum_face_area = std::numeric_limits<amrex::Real>::max();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab area_norm(m_face_area_vector_fc[dir].boxArray(), m_distribution, 1, 0);
        for (amrex::MFIter mfi(area_norm, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const area = m_face_area_vector_fc[dir].const_array(mfi);
            auto const norm = area_norm.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                norm(i, j, k) = std::sqrt(area(i, j, k, 0) * area(i, j, k, 0)
                                          + area(i, j, k, 1) * area(i, j, k, 1)
                                          + area(i, j, k, 2) * area(i, j, k, 2));
            });
        }
        amrex::Gpu::streamSynchronize();
        diagnostics.minimum_face_area = amrex::min(diagnostics.minimum_face_area,
                                                    area_norm.min(0, 0, true));
    }
    reduce_min(diagnostics.minimum_face_area);
    diagnostics.passed = diagnostics.minimum_mapping_jacobian > tolerance.minimum_positive
                      && diagnostics.minimum_cell_volume > tolerance.minimum_positive
                      && diagnostics.minimum_face_area > tolerance.minimum_positive
                      && diagnostics.maximum_reciprocal_relative_error <= tolerance.reciprocal_relative
                      && diagnostics.maximum_gcl_relative_error <= tolerance.gcl_relative
                      && diagnostics.maximum_reciprocity_relative_error <= tolerance.reciprocity_relative;
    return diagnostics;
}

amrex::MultiFab const& MetricData::face_area_vector_fc(int dir) const
{
    require_ready("face_area_vector_fc");
    if (dir < 0 || dir >= AMREX_SPACEDIM) throw std::out_of_range("AVWiS face-area direction");
    return m_face_area_vector_fc[dir];
}

amrex::MultiFab const& MetricData::face_gradient_metric_fc(int dir) const
{
    require_ready("face_gradient_metric_fc");
    if (dir < 0 || dir >= AMREX_SPACEDIM) throw std::out_of_range("AVWiS face-gradient direction");
    return m_face_gradient_metric_fc[dir];
}

amrex::MultiFab const& MetricData::projection_beta_fc(int dir) const
{
    require_ready("projection_beta_fc");
    if (dir < 0 || dir >= AMREX_SPACEDIM) throw std::out_of_range("AVWiS projection-beta direction");
    return m_projection_beta_fc[dir];
}

void MetricData::require_ready(char const* operation) const
{
    if (!m_defined || !m_built) {
        throw std::runtime_error(std::string("AVWiS MetricData is not built before ") + operation);
    }
}
