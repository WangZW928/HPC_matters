#include "AVWiSCoordinateMapping.H"
#include "AVWiSMetricData.H"

#include <AMReX.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
struct CaseResult {
    amrex::Real jacobian_error = 0.0;
    amrex::Real maximum_geometric_error = 0.0;
};

AnalyticOrthogonalMappingParameters strong_parameters()
{
    AnalyticOrthogonalMappingParameters parameters;
    parameters.scale = {AMREX_D_DECL(1.4, 0.75, 1.2)};
    parameters.stretch = {AMREX_D_DECL(0.94, -0.82, 0.67)};
    return parameters;
}

amrex::Real maximum_difference(amrex::MultiFab const& lhs, amrex::MultiFab const& rhs,
                               int nghost)
{
    if (lhs.boxArray() != rhs.boxArray() || lhs.DistributionMap() != rhs.DistributionMap() ||
        lhs.nComp() != rhs.nComp() || lhs.nGrow() < nghost || rhs.nGrow() < nghost) {
        throw std::runtime_error("P5-003 C2.1 comparison layout mismatch");
    }
    amrex::MultiFab difference(lhs.boxArray(), lhs.DistributionMap(), lhs.nComp(), nghost);
    amrex::MultiFab::Copy(difference, lhs, 0, 0, lhs.nComp(), nghost);
    amrex::MultiFab::Subtract(difference, rhs, 0, 0, lhs.nComp(), nghost);
    amrex::Real result = 0.0;
    for (int comp = 0; comp < lhs.nComp(); ++comp) {
        result = amrex::max(result, difference.norm0(comp, nghost, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}

void check_identity_limit(amrex::BoxArray const& boxes,
                          amrex::DistributionMapping const& distribution,
                          LogicalGrid const& logical, amrex::Geometry const& geometry)
{
    MetricData identity;
    identity.define(boxes, distribution, 1);
    IdentityCoordinateMapping identity_mapping;
    identity.build(identity_mapping, logical, geometry);

    MetricData analytic;
    analytic.define(boxes, distribution, 1);
    AnalyticOrthogonalCoordinateMapping analytic_mapping;
    analytic.build(analytic_mapping, logical, geometry);

    amrex::Real difference = 0.0;
    difference = amrex::max(difference, maximum_difference(
        identity.node_coordinates_nd(), analytic.node_coordinates_nd(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.cell_center_coordinates_cc(), analytic.cell_center_coordinates_cc(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.mapping_jacobian_cc(), analytic.mapping_jacobian_cc(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.inverse_mapping_jacobian_cc(), analytic.inverse_mapping_jacobian_cc(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.grad_xi_cc(), analytic.grad_xi_cc(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.area_cofactor_cc(), analytic.area_cofactor_cc(), 1));
    difference = amrex::max(difference, maximum_difference(
        identity.cell_volume_cc(), analytic.cell_volume_cc(), 1));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        difference = amrex::max(difference, maximum_difference(
            identity.face_area_vector_fc(dir), analytic.face_area_vector_fc(dir), 1));
        difference = amrex::max(difference, maximum_difference(
            identity.face_gradient_metric_fc(dir), analytic.face_gradient_metric_fc(dir), 1));
        difference = amrex::max(difference, maximum_difference(
            identity.projection_beta_fc(dir), analytic.projection_beta_fc(dir), 0));
    }
    amrex::Real const tolerance = 512.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (difference > tolerance || analytic.mapping_id() != "analytic_orthogonal") {
        throw std::runtime_error("P5-003 C2.1 analytic identity limit differs from C0 identity");
    }
}

CaseResult run_strong_case(int n, int max_grid_size, bool audit_contracts)
{
    amrex::IntVect const lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect const hi(AMREX_D_DECL(n - 1, n - 1, n - 1));
    amrex::Box const domain(lo, hi);
    amrex::RealBox const logical_box({AMREX_D_DECL(-0.4, 0.2, 1.1)},
                                     {AMREX_D_DECL(0.6, 1.2, 2.1)});
    amrex::Array<int, AMREX_SPACEDIM> const periodic{AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geometry(domain, &logical_box, 0, periodic.data());
    amrex::BoxArray boxes(domain);
    boxes.maxSize(max_grid_size);
    amrex::DistributionMapping distribution(boxes);
    LogicalGrid const logical = LogicalGrid::from_cartesian_geometry(geometry);
    AnalyticOrthogonalMappingParameters const parameters = strong_parameters();
    AnalyticOrthogonalCoordinateMapping mapping(parameters);

    MetricData metric;
    metric.define(boxes, distribution, 1);
    metric.build(mapping, logical, geometry);
    MetricDiagnostics const diagnostics = metric.validate();
    if (!diagnostics.passed || diagnostics.minimum_mapping_jacobian <= 0.0 ||
        diagnostics.minimum_cell_volume <= 0.0 || diagnostics.minimum_face_area <= 0.0) {
        throw std::runtime_error("P5-003 C2.1 strong stretch positivity/metric validation failed");
    }

    amrex::MultiFab cell_checks(boxes, distribution, 4, 0);
    int const ilo = domain.smallEnd(0);
    int const jlo = domain.smallEnd(1);
    int const klo = domain.smallEnd(2);
    amrex::Real const lower0 = logical.lower[0];
    amrex::Real const lower1 = logical.lower[1];
    amrex::Real const lower2 = logical.lower[2];
    amrex::Real const h0 = logical.spacing[0];
    amrex::Real const h1 = logical.spacing[1];
    amrex::Real const h2 = logical.spacing[2];
    amrex::Real const length0 = h0 * static_cast<amrex::Real>(domain.length(0));
    amrex::Real const length1 = h1 * static_cast<amrex::Real>(domain.length(1));
    amrex::Real const length2 = h2 * static_cast<amrex::Real>(domain.length(2));
    amrex::Real const scale0 = parameters.scale[0];
    amrex::Real const scale1 = parameters.scale[1];
    amrex::Real const scale2 = parameters.scale[2];
    amrex::Real const stretch0 = parameters.stretch[0];
    amrex::Real const stretch1 = parameters.stretch[1];
    amrex::Real const stretch2 = parameters.stretch[2];

    for (amrex::MFIter mfi(cell_checks, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const check = cell_checks.array(mfi);
        auto const jacobian = metric.mapping_jacobian_cc().const_array(mfi);
        auto const volume = metric.cell_volume_cc().const_array(mfi);
        auto const grad = metric.grad_xi_cc().const_array(mfi);
        auto const cofactor = metric.area_cofactor_cc().const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const xi = lower0 + (static_cast<amrex::Real>(i - ilo) + 0.5) * h0;
            amrex::Real const eta = lower1 + (static_cast<amrex::Real>(j - jlo) + 0.5) * h1;
            amrex::Real const zeta = lower2 + (static_cast<amrex::Real>(k - klo) + 0.5) * h2;
            amrex::Real const exact_jx =
                AnalyticOrthogonalCoordinateMapping::coordinate_derivative(
                    xi, lower0, length0, scale0, stretch0)
              * AnalyticOrthogonalCoordinateMapping::coordinate_derivative(
                    eta, lower1, length1, scale1, stretch1)
              * AnalyticOrthogonalCoordinateMapping::coordinate_derivative(
                    zeta, lower2, length2, scale2, stretch2);
            check(i, j, k, 0) = std::abs(jacobian(i, j, k) - exact_jx);

            amrex::Real off_diagonal = 0.0;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    if (row != column) {
                        off_diagonal = amrex::max(off_diagonal,
                            std::abs(grad(i, j, k, 3 * row + column)));
                        off_diagonal = amrex::max(off_diagonal,
                            std::abs(cofactor(i, j, k, 3 * row + column)));
                    }
                }
            }
            check(i, j, k, 1) = off_diagonal;

            amrex::Real const x0 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower0 + static_cast<amrex::Real>(i - ilo) * h0,
                lower0, length0, scale0, stretch0);
            amrex::Real const x1 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower0 + static_cast<amrex::Real>(i - ilo + 1) * h0,
                lower0, length0, scale0, stretch0);
            amrex::Real const y0 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower1 + static_cast<amrex::Real>(j - jlo) * h1,
                lower1, length1, scale1, stretch1);
            amrex::Real const y1 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower1 + static_cast<amrex::Real>(j - jlo + 1) * h1,
                lower1, length1, scale1, stretch1);
            amrex::Real const z0 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower2 + static_cast<amrex::Real>(k - klo) * h2,
                lower2, length2, scale2, stretch2);
            amrex::Real const z1 = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                lower2 + static_cast<amrex::Real>(k - klo + 1) * h2,
                lower2, length2, scale2, stretch2);
            check(i, j, k, 2) = std::abs(volume(i, j, k)
                                               - (x1 - x0) * (y1 - y0) * (z1 - z0));
            check(i, j, k, 3) = amrex::min(x1 - x0, amrex::min(y1 - y0, z1 - z0));
        });
    }
    amrex::Gpu::streamSynchronize();

    CaseResult result;
    result.jacobian_error = cell_checks.norm0(0, 0, true);
    amrex::Real const off_diagonal = cell_checks.norm0(1, 0, true);
    amrex::Real const volume_error = cell_checks.norm0(2, 0, true);
    amrex::Real const minimum_edge = cell_checks.min(3, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(result.jacobian_error);
    amrex::Real reduced_off_diagonal = off_diagonal;
    amrex::Real reduced_volume_error = volume_error;
    amrex::Real reduced_minimum_edge = minimum_edge;
    amrex::ParallelDescriptor::ReduceRealMax(reduced_off_diagonal);
    amrex::ParallelDescriptor::ReduceRealMax(reduced_volume_error);
    amrex::ParallelDescriptor::ReduceRealMin(reduced_minimum_edge);
    result.maximum_geometric_error = amrex::max(reduced_off_diagonal, reduced_volume_error);

    amrex::Real const roundoff = 4096.0 * std::numeric_limits<amrex::Real>::epsilon();
    if (!(reduced_minimum_edge > 0.0) || result.maximum_geometric_error > roundoff) {
        throw std::runtime_error("P5-003 C2.1 diagonal metric/exact volume contract failed");
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::MultiFab face_checks(metric.face_area_vector_fc(dir).boxArray(), distribution, 2, 0);
        for (amrex::MFIter mfi(face_checks, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const check = face_checks.array(mfi);
            auto const area = metric.face_area_vector_fc(dir).const_array(mfi);
            auto const q = metric.face_gradient_metric_fc(dir).const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                amrex::Real off_diagonal = 0.0;
                for (int comp = 0; comp < 3; ++comp) {
                    if (comp != dir) {
                        off_diagonal = amrex::max(off_diagonal, std::abs(area(i, j, k, comp)));
                        off_diagonal = amrex::max(off_diagonal, std::abs(q(i, j, k, comp)));
                    }
                }
                check(i, j, k, 0) = off_diagonal;
                check(i, j, k, 1) = area(i, j, k, dir);
            });
        }
        amrex::Gpu::streamSynchronize();
        amrex::Real face_off_diagonal = face_checks.norm0(0, 0, true);
        amrex::Real minimum_area = face_checks.min(1, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(face_off_diagonal);
        amrex::ParallelDescriptor::ReduceRealMin(minimum_area);
        result.maximum_geometric_error = amrex::max(result.maximum_geometric_error,
                                                     face_off_diagonal);
        if (face_off_diagonal > roundoff || !(minimum_area > 0.0)) {
            throw std::runtime_error("P5-003 C2.1 face orthogonality/positive area contract failed");
        }

        if (audit_contracts) {
            amrex::Long expected_unique_faces = 1;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                expected_unique_faces *= domain.length(d) + (d == dir ? 1 : 0);
            }
            auto owner = metric.face_area_vector_fc(dir).OwnerMask(geometry.periodicity());
            if (owner->sum(0, 0, false) != expected_unique_faces) {
                throw std::runtime_error("P5-003 C2.1 shared-face owner count failed");
            }
        }
    }

    if (audit_contracts) {
        amrex::Gpu::DeviceScalar<int> device_errors(0);
        int* const errors = device_errors.dataPtr();
        for (amrex::MFIter mfi(metric.node_coordinates_nd(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const xyz = metric.node_coordinates_nd().const_array(mfi);
            amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                amrex::Real const logical_coordinate[3] = {
                    lower0 + static_cast<amrex::Real>(i - ilo) * h0,
                    lower1 + static_cast<amrex::Real>(j - jlo) * h1,
                    lower2 + static_cast<amrex::Real>(k - klo) * h2};
                amrex::Real const lower[3] = {lower0, lower1, lower2};
                amrex::Real const length[3] = {length0, length1, length2};
                amrex::Real const scale[3] = {scale0, scale1, scale2};
                amrex::Real const stretch[3] = {stretch0, stretch1, stretch2};
                for (int comp = 0; comp < 3; ++comp) {
                    amrex::Real const expected = AnalyticOrthogonalCoordinateMapping::map_coordinate(
                        logical_coordinate[comp], lower[comp], length[comp], scale[comp], stretch[comp]);
                    amrex::Real const tolerance = 512.0 * std::numeric_limits<amrex::Real>::epsilon()
                                                * amrex::max(amrex::Real(1.0), std::abs(expected));
                    if (std::abs(xyz(i, j, k, comp) - expected) > tolerance) {
                        amrex::Gpu::Atomic::Add(errors, 1);
                    }
                }
            });
        }
        amrex::Gpu::streamSynchronize();
        int error_count = device_errors.dataValue();
        amrex::ParallelDescriptor::ReduceIntSum(error_count);
        if (error_count != 0) {
            throw std::runtime_error("P5-003 C2.1 analytic node ghost/owner values failed");
        }

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
        amrex::Real const velocity[3] = {0.8, -1.1, 2.3};
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            flux[dir].define(metric.face_area_vector_fc(dir).boxArray(), distribution, 1, 1);
            for (amrex::MFIter mfi(flux[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const area = metric.face_area_vector_fc(dir).const_array(mfi);
                auto const ucont = flux[dir].array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    ucont(i, j, k) = velocity[0] * area(i, j, k, 0)
                                   + velocity[1] * area(i, j, k, 1)
                                   + velocity[2] * area(i, j, k, 2);
                });
            }
            flux[dir].OverrideSync(geometry.periodicity());
            flux[dir].FillBoundary(geometry.periodicity());
        }
        amrex::MultiFab divergence(boxes, distribution, 1, 0);
        for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const div = divergence.array(mfi);
            auto const volume = metric.cell_volume_cc().const_array(mfi);
            auto const fx = flux[0].const_array(mfi);
            auto const fy = flux[1].const_array(mfi);
            auto const fz = flux[2].const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                div(i, j, k) = ((fx(i + 1, j, k) - fx(i, j, k))
                              + (fy(i, j + 1, k) - fy(i, j, k))
                              + (fz(i, j, k + 1) - fz(i, j, k))) / volume(i, j, k);
            });
        }
        amrex::Gpu::streamSynchronize();
        amrex::Real max_divergence = divergence.norm0(0, 0, true);
        amrex::ParallelDescriptor::ReduceRealMax(max_divergence);
        if (max_divergence > 2.0e-11) {
            throw std::runtime_error("P5-003 C2.1 Ucont=u dot S/free-stream geometric divergence failed");
        }

        check_identity_limit(boxes, distribution, logical, geometry);
    }
    return result;
}

void check_invalid_parameters()
{
    auto must_reject = [](AnalyticOrthogonalMappingParameters const& parameters) {
        try {
            AnalyticOrthogonalCoordinateMapping mapping(parameters);
            (void)mapping;
        } catch (std::invalid_argument const&) {
            return;
        }
        throw std::runtime_error("P5-003 C2.1 accepted invalid mapping parameters");
    };

    auto parameters = strong_parameters();
    parameters.scale[0] = 0.0;
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.scale[1] = -1.0;
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.scale[2] = std::numeric_limits<amrex::Real>::infinity();
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.stretch[0] = 1.0;
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.stretch[1] = -1.0;
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.stretch[2] = std::numeric_limits<amrex::Real>::quiet_NaN();
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.scale = {AMREX_D_DECL(std::numeric_limits<amrex::Real>::min(), 1.0, 1.0)};
    must_reject(parameters);
    parameters = strong_parameters();
    parameters.scale = {AMREX_D_DECL(std::numeric_limits<amrex::Real>::max() / 4.0,
                                     std::numeric_limits<amrex::Real>::max() / 4.0,
                                     1.0)};
    must_reject(parameters);

    bool unused_rejected = false;
    try {
        parameters = strong_parameters();
        auto mapping = make_coordinate_mapping("identity", parameters);
        (void)mapping;
    } catch (std::invalid_argument const&) {
        unused_rejected = true;
    }
    if (!unused_rejected) {
        throw std::runtime_error("P5-003 C2.1 identity mapping accepted unknown analytic parameters");
    }

    bool unknown_rejected = false;
    try {
        auto mapping = make_coordinate_mapping("general_nonorthogonal");
        (void)mapping;
    } catch (std::invalid_argument const&) {
        unknown_rejected = true;
    }
    if (!unknown_rejected) {
        throw std::runtime_error("P5-003 C2.1 accepted an unknown mapping type");
    }
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        check_invalid_parameters();
        CaseResult const coarse = run_strong_case(12, 12, true);  // one Box
        CaseResult const multibox = run_strong_case(12, 4, true); // many Boxes/shared owners
        CaseResult const medium = run_strong_case(24, 24, false);
        CaseResult const fine = run_strong_case(48, 48, false);

        amrex::Real const coarse_multibox_tolerance =
            4096.0 * std::numeric_limits<amrex::Real>::epsilon();
        if (std::abs(coarse.jacobian_error - multibox.jacobian_error) >
                coarse_multibox_tolerance * amrex::max(amrex::Real(1.0), coarse.jacobian_error) ||
            coarse.maximum_geometric_error > coarse_multibox_tolerance ||
            multibox.maximum_geometric_error > coarse_multibox_tolerance) {
            throw std::runtime_error("P5-003 C2.1 one-Box/multi-Box geometry differs");
        }
        amrex::Real const rate0 = std::log(coarse.jacobian_error / medium.jacobian_error)
                                / std::log(amrex::Real(2.0));
        amrex::Real const rate1 = std::log(medium.jacobian_error / fine.jacobian_error)
                                / std::log(amrex::Real(2.0));
        if (!(rate0 >= 1.8) || !(rate1 >= 1.8)) {
            throw std::runtime_error("P5-003 C2.1 analytic Jacobian refinement rate is below second order");
        }

        amrex::Print() << "AVWiS P5-003 C2.1 analytic orthogonal metric contract: PASS "
                       << "Jx_Linf=" << coarse.jacobian_error << ","
                       << medium.jacobian_error << "," << fine.jacobian_error
                       << " rates=" << rate0 << "," << rate1 << "\n";
    } catch (std::exception const& error) {
        amrex::Print() << "AVWiS P5-003 C2.1 analytic orthogonal contract error: "
                       << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
