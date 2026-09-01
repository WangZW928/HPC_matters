#include "AVWiSMappedOperators.H"
#include "AVWiSMetricAdapter.H"

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
AnalyticOrthogonalMappingParameters parameters()
{
    AnalyticOrthogonalMappingParameters value;
    value.scale = {AMREX_D_DECL(1.2, 0.85, 1.1)};
    value.stretch = {AMREX_D_DECL(0.35, -0.25, 0.2)};
    return value;
}

MappingOperatorConfig mapped_config(std::string mapping = "analytic_orthogonal")
{
    MappingOperatorConfig config;
    config.coordinates = CoordinateSystemMode::Mapped;
    config.mapping_type = std::move(mapping);
    config.projection = ProjectionOperatorMode::OrthogonalMLMG;
    config.analytic_parameters = parameters();
    if (config.mapping_type == "identity") config.analytic_parameters = {};
    return config;
}

struct GridCase {
    amrex::Box domain;
    amrex::RealBox real_box;
    amrex::Array<int, AMREX_SPACEDIM> periodic{};
    amrex::Geometry geometry;
    amrex::BoxArray boxes;
    amrex::DistributionMapping distribution;
    LogicalGrid logical;
    AnalyticOrthogonalCoordinateMapping mapping;
    MetricData metric;

    GridCase(int n, int max_grid)
        : domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                 amrex::IntVect(AMREX_D_DECL(n-1, n-1, n-1))),
          real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)}),
          geometry(domain, &real_box, 0, periodic.data()), boxes(domain),
          mapping(parameters())
    {
        boxes.maxSize(max_grid);
        distribution = amrex::DistributionMapping(boxes);
        logical = LogicalGrid::from_cartesian_geometry(geometry);
        metric.define(boxes, distribution, 1);
        metric.build(mapping, logical, geometry);
    }
};

amrex::Real gradient_mms_error(int n, int max_grid)
{
    GridCase grid(n, max_grid);
    MappingOperatorConfig const config = mapped_config();
    amrex::MultiFab scalar(grid.boxes, grid.distribution, 1, 1);
    amrex::MultiFab gradient(grid.boxes, grid.distribution, AMREX_SPACEDIM, 0);
    amrex::MultiFab error(grid.boxes, grid.distribution, AMREX_SPACEDIM, 0);
    constexpr amrex::Real two_pi = 6.283185307179586476925286766559005768;
    for (amrex::MFIter mfi(scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.array(mfi);
        auto const xyz = grid.metric.cell_center_coordinates_cc().const_array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            phi(i,j,k) = std::sin(two_pi*xyz(i,j,k,0)) +
                         0.5*std::cos(two_pi*xyz(i,j,k,1)) +
                         0.25*std::sin(two_pi*xyz(i,j,k,2));
        });
    }
    scalar.FillBoundary(grid.geometry.periodicity());
    compute_metric_cell_gradient(scalar, grid.metric, grid.metric.epoch(), config, gradient);
    for (amrex::MFIter mfi(error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const grad = gradient.const_array(mfi);
        auto const xyz = grid.metric.cell_center_coordinates_cc().const_array(mfi);
        auto const delta = error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            delta(i,j,k,0) = grad(i,j,k,0) - two_pi*std::cos(two_pi*xyz(i,j,k,0));
            delta(i,j,k,1) = grad(i,j,k,1) + 0.5*two_pi*std::sin(two_pi*xyz(i,j,k,1));
            delta(i,j,k,2) = grad(i,j,k,2) - 0.25*two_pi*std::cos(two_pi*xyz(i,j,k,2));
        });
    }
    amrex::Real result = 0.0;
    for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
        result = amrex::max(result, error.norm0(comp, 0, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(result);
    return result;
}

void check_constant_linear_transform_and_divergence()
{
    GridCase grid(16, 4); // many Boxes; scalar/velocity stencils cross box ghosts
    MappingOperatorConfig const config = mapped_config();
    amrex::MultiFab scalar(grid.boxes, grid.distribution, 1, 1);
    amrex::MultiFab gradient(grid.boxes, grid.distribution, AMREX_SPACEDIM, 0);
    scalar.setVal(4.25);
    compute_metric_cell_gradient(scalar, grid.metric, grid.metric.epoch(), config, gradient);
    if (gradient.norm0(0, 0, true) != 0.0 || gradient.norm0(1, 0, true) != 0.0 ||
        gradient.norm0(2, 0, true) != 0.0) {
        throw std::runtime_error("C2.2 constant scalar gradient is nonzero");
    }

    amrex::Real const slope[3] = {0.7, -1.1, 0.4};
    int const ilo = grid.domain.smallEnd(0);
    int const jlo = grid.domain.smallEnd(1);
    int const klo = grid.domain.smallEnd(2);
    auto const h = grid.logical.spacing;
    for (amrex::MFIter mfi(scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const xi = (static_cast<amrex::Real>(i-ilo)+0.5)*h[0];
            amrex::Real const eta = (static_cast<amrex::Real>(j-jlo)+0.5)*h[1];
            amrex::Real const zeta = (static_cast<amrex::Real>(k-klo)+0.5)*h[2];
            phi(i,j,k) = slope[0]*xi + slope[1]*eta + slope[2]*zeta;
        });
    }
    scalar.FillBoundary(grid.geometry.periodicity());
    compute_metric_cell_gradient(scalar, grid.metric, grid.metric.epoch(), config, gradient);
    amrex::MultiFab linear_error(grid.boxes, grid.distribution, AMREX_SPACEDIM, 0);
    for (amrex::MFIter mfi(linear_error, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const grad = gradient.const_array(mfi);
        auto const grad_xi = grid.metric.grad_xi_cc().const_array(mfi);
        auto const error = linear_error.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            for (int physical = 0; physical < 3; ++physical) {
                amrex::Real const exact = slope[0]*grad_xi(i,j,k,physical) +
                    slope[1]*grad_xi(i,j,k,3+physical) +
                    slope[2]*grad_xi(i,j,k,6+physical);
                error(i,j,k,physical) = grad(i,j,k,physical)-exact;
            }
        });
    }
    amrex::Real linear_norm = 0.0;
    for (int comp = 0; comp < 3; ++comp) {
        linear_norm = amrex::max(linear_norm, linear_error.norm0(comp, 0, true));
    }
    amrex::ParallelDescriptor::ReduceRealMax(linear_norm);
    if (linear_norm > 4096.0*std::numeric_limits<amrex::Real>::epsilon()) {
        throw std::runtime_error("C2.2 logical-linear scalar gradient contract failed");
    }

    amrex::MultiFab ucat(grid.boxes, grid.distribution, 3, 1);
    amrex::MultiFab reconstructed(grid.boxes, grid.distribution, 3, 0);
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> ucont;
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> ucont_view{};
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const_ucont_view{};
    amrex::Real const velocity[3] = {0.8, -0.6, 1.4};
    ucat.setVal(0.0);
    for (int comp = 0; comp < 3; ++comp) ucat.setVal(velocity[comp], comp, 1, 1);
    for (int dir = 0; dir < 3; ++dir) {
        ucont[dir].define(grid.metric.face_area_vector_fc(dir).boxArray(),
                          grid.distribution, 1, 1);
        ucont_view[dir] = &ucont[dir];
        const_ucont_view[dir] = &ucont[dir];
    }
    sync_orthogonal_ucont_from_ucat(ucat, ucont_view, grid.metric, grid.metric.epoch(),
                                   config, grid.geometry.periodicity());
    sync_orthogonal_ucat_from_ucont(const_ucont_view, grid.metric, grid.metric.epoch(),
                                   config, reconstructed);
    for (int comp = 0; comp < 3; ++comp) reconstructed.plus(-velocity[comp], comp, 1, 0);
    amrex::Real transform_error = 0.0;
    for (int comp = 0; comp < 3; ++comp) {
        transform_error = amrex::max(transform_error, reconstructed.norm0(comp, 0, true));
    }

    amrex::MultiFab divergence(grid.boxes, grid.distribution, 1, 0);
    compute_metric_divergence(const_ucont_view, grid.metric, grid.metric.epoch(), config, divergence);
    amrex::Real free_stream_divergence = divergence.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(transform_error);
    amrex::ParallelDescriptor::ReduceRealMax(free_stream_divergence);
    if (transform_error > 2.0e-13 || free_stream_divergence > 2.0e-11) {
        throw std::runtime_error("C2.2 orthogonal constant transform/free-stream contract failed");
    }

    // u=(x,y,z) gives div(u)=3. Face coordinates are constant over each
    // separable face, so this also audits volume scaling exactly once.
    auto const& xyz = grid.metric.node_coordinates_nd();
    for (int dir = 0; dir < 3; ++dir) {
        for (amrex::MFIter mfi(ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const flux = ucont[dir].array(mfi);
            auto const area = grid.metric.face_area_vector_fc(dir).const_array(mfi);
            auto const node = xyz.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                flux(i,j,k) = node(i,j,k,dir) * area(i,j,k,dir);
            });
        }
    }
    compute_metric_divergence(const_ucont_view, grid.metric, grid.metric.epoch(), config, divergence);
    divergence.plus(-3.0, 0, 1, 0);
    amrex::Real affine_error = divergence.norm0(0, 0, true);
    amrex::ParallelDescriptor::ReduceRealMax(affine_error);
    if (affine_error > 2.0e-11) {
        throw std::runtime_error("C2.2 mapped affine divergence/volume contract failed");
    }
}

void check_identity_and_rejection_contracts()
{
    int const n = 8;
    amrex::Box domain(amrex::IntVect(AMREX_D_DECL(0,0,0)),
                      amrex::IntVect(AMREX_D_DECL(n-1,n-1,n-1)));
    amrex::RealBox real_box({AMREX_D_DECL(0.0,0.0,0.0)}, {AMREX_D_DECL(1.0,1.0,1.0)});
    amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(1,1,1)};
    amrex::Geometry geometry(domain, &real_box, 0, periodic.data());
    amrex::BoxArray boxes(domain);
    boxes.maxSize(3);
    amrex::DistributionMapping distribution(boxes);
    LogicalGrid logical = LogicalGrid::from_cartesian_geometry(geometry);
    MetricData metric;
    metric.define(boxes, distribution, 1);
    IdentityCoordinateMapping identity;
    metric.build(identity, logical, geometry);
    auto const config = mapped_config("identity");

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> flux;
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> view{};
    for (int dir = 0; dir < 3; ++dir) {
        flux[dir].define(metric.face_area_vector_fc(dir).boxArray(), distribution, 1, 1);
        for (amrex::MFIter mfi(flux[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const value = flux[dir].array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                int const index = dir == 0 ? i : (dir == 1 ? j : k);
                value(i,j,k) = static_cast<amrex::Real>(dir+1)*index;
            });
        }
        flux[dir].OverrideSync(geometry.periodicity());
        flux[dir].FillBoundary(geometry.periodicity());
        view[dir] = &flux[dir];
    }
    amrex::MultiFab generic(boxes, distribution, 1, 0);
    amrex::MultiFab legacy(boxes, distribution, 1, 0);
    compute_metric_divergence(view, metric, metric.epoch(), config, generic);
    compute_identity_metric_divergence(view, metric, metric.epoch(), legacy);
    amrex::MultiFab::Subtract(generic, legacy, 0, 0, 1, 0);
    if (generic.norm0(0, 0, true) > 512.0*std::numeric_limits<amrex::Real>::epsilon()) {
        throw std::runtime_error("C2.2 identity-limit divergence is not tightly equivalent");
    }

    amrex::MultiFab scalar(boxes, distribution, 1, 1);
    amrex::MultiFab mapped_face(metric.face_area_vector_fc(0).boxArray(), distribution, 1, 0);
    amrex::MultiFab cartesian_face(metric.face_area_vector_fc(0).boxArray(), distribution, 1, 0);
    auto const h = logical.spacing;
    for (amrex::MFIter mfi(scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.array(mfi);
        amrex::ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            phi(i,j,k) = 0.75*(static_cast<amrex::Real>(i)+0.5)*h[0]
                       - 0.2*(static_cast<amrex::Real>(j)+0.5)*h[1]
                       + 0.4*(static_cast<amrex::Real>(k)+0.5)*h[2];
        });
    }
    scalar.FillBoundary(geometry.periodicity());
    compute_orthogonal_face_gradient_flux(
        scalar, 0, metric, metric.epoch(), config, mapped_face);
    for (amrex::MFIter mfi(cartesian_face, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.const_array(mfi);
        auto const reference = cartesian_face.array(mfi);
        amrex::Real const area = h[1]*h[2];
        amrex::Real const inv_h = 1.0/h[0];
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            reference(i,j,k) = area*(phi(i,j,k)-phi(i-1,j,k))*inv_h;
        });
    }
    amrex::MultiFab::Subtract(mapped_face, cartesian_face, 0, 0, 1, 0);
    if (mapped_face.norm0(0, 0, true) >
        512.0*std::numeric_limits<amrex::Real>::epsilon()) {
        throw std::runtime_error("C2.2 identity-limit face gradient is not tightly equivalent");
    }

    bool mismatch_rejected = false;
    try {
        auto bad = config;
        bad.mapping_type = "analytic_orthogonal";
        validate_mapping_operator_config(bad, metric, metric.epoch());
    } catch (std::runtime_error const&) { mismatch_rejected = true; }
    if (!mismatch_rejected) throw std::runtime_error("C2.2 accepted incompatible mapping data");

    bool mode_rejected = false;
    try {
        MappingOperatorConfig bad;
        bad.projection = ProjectionOperatorMode::OrthogonalMLMG;
        validate_mapping_operator_config(bad, metric, metric.epoch());
    } catch (std::runtime_error const&) { mode_rejected = true; }
    if (!mode_rejected) throw std::runtime_error("C2.2 accepted incompatible operator mode");

    bool layout_rejected = false;
    try {
        amrex::MultiFab wrong_layout(boxes, distribution, 2, 0);
        compute_metric_divergence(view, metric, metric.epoch(), config, wrong_layout);
    } catch (std::runtime_error const&) { layout_rejected = true; }
    if (!layout_rejected) throw std::runtime_error("C2.2 accepted an incompatible cell layout");

    std::uint64_t const stale = metric.epoch();
    metric.rebuild(identity, logical, geometry);
    bool stale_rejected = false;
    try { compute_metric_divergence(view, metric, stale, config, legacy); }
    catch (std::runtime_error const&) { stale_rejected = true; }
    if (!stale_rejected) throw std::runtime_error("C2.2 accepted a stale metric epoch");

    bool unknown_rejected = false;
    try { (void)parse_projection_operator_mode("general_19_point"); }
    catch (std::invalid_argument const&) { unknown_rejected = true; }
    if (!unknown_rejected) throw std::runtime_error("C2.2 accepted an unknown projection mode");
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    try {
        check_constant_linear_transform_and_divergence();
        check_identity_and_rejection_contracts();
        amrex::Real const e12 = gradient_mms_error(12, 4);
        amrex::Real const e24 = gradient_mms_error(24, 6);
        amrex::Real const e48 = gradient_mms_error(48, 8);
        amrex::Real const p0 = std::log(e12/e24)/std::log(amrex::Real(2.0));
        amrex::Real const p1 = std::log(e24/e48)/std::log(amrex::Real(2.0));
        if (p0 < 1.6 || p1 < 1.6) {
            throw std::runtime_error("C2.2 mapped scalar-gradient MMS rate is below tolerance");
        }
        amrex::Print() << "AVWiS P5-003 C2.2 orthogonal operator contract: PASS "
                       << "gradient_Linf=" << e12 << "," << e24 << "," << e48
                       << " rates=" << p0 << "," << p1 << "\n";
    } catch (std::exception const& error) {
        amrex::Print() << "AVWiS P5-003 C2.2 orthogonal operator contract error: "
                       << error.what() << "\n";
        status = 1;
    }
    amrex::Finalize();
    return status;
}
