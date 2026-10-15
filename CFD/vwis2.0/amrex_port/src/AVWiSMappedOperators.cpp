#include "AVWiSMappedOperators.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MFIter.H>
#include <AMReX_MFParallelFor.H>

#include <cmath>
#include <stdexcept>

namespace {
bool is_supported_orthogonal_mapping(std::string const& id)
{
    return id == "identity" || id == "analytic_orthogonal";
}

void require_cell_layout(amrex::MultiFab const& field, int components,
                         amrex::MultiFab const& volume, char const* operation)
{
    if (!field.boxArray().ixType().cellCentered() || field.nComp() != components ||
        field.boxArray() != volume.boxArray() ||
        field.DistributionMap() != volume.DistributionMap()) {
        throw std::runtime_error(std::string("AVWiS ") + operation +
                                 " cell/metric layouts differ");
    }
}

void require_face_layout(amrex::MultiFab const& field, int direction,
                         MetricData const& metric, char const* operation)
{
    auto const& area = metric.face_area_vector_fc(direction);
    if (field.nComp() != 1 || field.boxArray() != area.boxArray() ||
        field.DistributionMap() != area.DistributionMap()) {
        throw std::runtime_error(std::string("AVWiS ") + operation +
                                 " face/metric layouts differ");
    }
}
} // namespace

CoordinateSystemMode parse_coordinate_system_mode(std::string const& name)
{
    if (name == "cartesian") return CoordinateSystemMode::Cartesian;
    if (name == "mapped") return CoordinateSystemMode::Mapped;
    throw std::invalid_argument("AVWiS unknown coordinate-system mode '" + name + "'");
}

ProjectionOperatorMode parse_projection_operator_mode(std::string const& name)
{
    if (name == "cartesian_mlmg") return ProjectionOperatorMode::CartesianMLMG;
    if (name == "orthogonal_mlmg") return ProjectionOperatorMode::OrthogonalMLMG;
    if (name == "deferred_nonorthogonal") return ProjectionOperatorMode::DeferredNonorthogonal;
    throw std::invalid_argument("AVWiS unknown projection operator mode '" + name + "'");
}

char const* coordinate_system_mode_name(CoordinateSystemMode mode) noexcept
{
    return mode == CoordinateSystemMode::Cartesian ? "cartesian" : "mapped";
}

char const* projection_operator_mode_name(ProjectionOperatorMode mode) noexcept
{
    if (mode == ProjectionOperatorMode::DeferredNonorthogonal) return "deferred_nonorthogonal";
    return mode == ProjectionOperatorMode::CartesianMLMG ? "cartesian_mlmg" : "orthogonal_mlmg";
}

void validate_mapping_operator_config(MappingOperatorConfig const& config,
                                      MetricData const& metric,
                                      std::uint64_t expected_metric_epoch)
{
    if (config.failure_contract != "fatal")
        throw std::runtime_error("M2 failure_contract supports only fatal; entire-advance rollback is not implemented");
    if (config.state_contract == "face_authoritative") {
        if (config.pressure_contract != "instantaneous_pressure" &&
            config.pressure_contract != "incremental_pressure")
            throw std::runtime_error("M2 face_authoritative requires pressure_contract=instantaneous_pressure or incremental_pressure; time-converged physical pressure remain open");
        if (config.mapping_type == "analytic_shear" &&
            (config.shear_epsilon == 0.0 ||
             config.analytic_parameters.stretch != amrex::Array<amrex::Real,3>{0,0,0}))
            throw std::runtime_error("M2 face_authoritative currently requires nonzero affine analytic_shear (stretch=0 0 0)");
        if (config.mapping_type != "analytic_shear" && config.pressure_contract != "incremental_pressure")
            throw std::runtime_error("M2 Cartesian/orthogonal face_authoritative requires incremental_pressure");
        for (int d=0;d<3;++d)
            if (!metric.logical_grid().periodic[d])
                throw std::runtime_error("M2 face_authoritative incremental advancement requires all-periodic geometry; physical BCs remain open");
    } else if (config.state_contract != "experimental_dual_state") {
        throw std::runtime_error("M2 state_contract supports experimental_dual_state or face_authoritative; time-converged physical pressure remain open");
    } else if (config.pressure_contract != "accumulated_correction") {
        throw std::runtime_error("M2 experimental_dual_state requires pressure_contract=accumulated_correction");
    }
    bool const diagonal_increment = config.pressure_contract == "incremental_pressure" &&
                                   config.mapping_type != "analytic_shear";
    if (config.diagonal_max_iterations < 1 || config.diagonal_max_iterations > 1600 ||
        (!diagonal_increment && config.diagonal_max_iterations != 1600))
        throw std::runtime_error("projection.diagonal_max_iterations requires a Cartesian/orthogonal incremental solve and a budget in [1,1600]");
    if (expected_metric_epoch == 0 || metric.epoch() != expected_metric_epoch) {
        throw std::runtime_error("AVWiS mapped operator detected a stale metric epoch");
    }
    if (metric.mapping_id() != config.mapping_type) {
        throw std::runtime_error("AVWiS mapped operator configuration and MetricData mapping differ");
    }
    if (config.mapping_type == "analytic_shear") {
        if (config.coordinates != CoordinateSystemMode::Mapped ||
            config.projection != ProjectionOperatorMode::DeferredNonorthogonal ||
            !std::isfinite(config.shear_epsilon) || std::abs(config.shear_epsilon)>0.25 ||
            config.deferred_max_iterations < 1 || config.deferred_max_iterations > 100) {
            throw std::runtime_error("AVWiS analytic_shear requires mapped/deferred_nonorthogonal, "
                                     "abs(epsilon)<=0.25 and max_iterations in [1,100]");
        }
        for (int d=0;d<3;++d) {
            if (!metric.logical_grid().periodic[d])
                throw std::runtime_error("AVWiS analytic_shear currently requires all-periodic geometry");
        }
        return;
    }
    if (config.shear_epsilon != 0.0)
        throw std::runtime_error("AVWiS epsilon requires analytic_shear mapping");
    if (config.coordinates == CoordinateSystemMode::Cartesian) {
        if (config.mapping_type != "identity" ||
            config.projection != ProjectionOperatorMode::CartesianMLMG) {
            throw std::runtime_error(
                "AVWiS Cartesian coordinates require identity mapping and cartesian_mlmg projection");
        }
        return;
    }
    if (!is_supported_orthogonal_mapping(config.mapping_type) ||
        config.projection != ProjectionOperatorMode::OrthogonalMLMG) {
        throw std::runtime_error(
            "AVWiS mapped coordinates currently require identity or analytic_orthogonal mapping "
            "with orthogonal_mlmg projection");
    }
}

void compute_metric_divergence(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& integrated_flux,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& divergence)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    auto const& volume = metric.cell_volume_cc();
    require_cell_layout(divergence, 1, volume, "divergence");
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (integrated_flux[dir] == nullptr) {
            throw std::runtime_error("AVWiS divergence received a null face flux");
        }
        require_face_layout(*integrated_flux[dir], dir, metric, "divergence");
    }

    for (amrex::MFIter mfi(divergence, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const div = divergence.array(mfi);
        auto const cell_volume = volume.const_array(mfi);
        auto const fx = integrated_flux[0]->const_array(mfi);
        auto const fy = integrated_flux[1]->const_array(mfi);
        auto const fz = integrated_flux[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // The inputs already contain u dot S; volume is applied once here.
            div(i,j,k) = ((fx(i+1,j,k)-fx(i,j,k)) +
                          (fy(i,j+1,k)-fy(i,j,k)) +
                          (fz(i,j,k+1)-fz(i,j,k))) / cell_volume(i,j,k);
        });
    }
    amrex::Gpu::streamSynchronize();
}

void compute_metric_net_flux(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& integrated_flux,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& net_flux)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    require_cell_layout(net_flux, 1, metric.cell_volume_cc(), "net flux");
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        if (integrated_flux[d] == nullptr)
            throw std::runtime_error("AVWiS net flux received a null face flux");
        require_face_layout(*integrated_flux[d], d, metric, "net flux");
    }
    for (amrex::MFIter mfi(net_flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const out=net_flux.array(mfi);
        auto const fx=integrated_flux[0]->const_array(mfi);
        auto const fy=integrated_flux[1]->const_array(mfi);
        auto const fz=integrated_flux[2]->const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            out(i,j,k)=((fx(i+1,j,k)-fx(i,j,k)) +
                         (fy(i,j+1,k)-fy(i,j,k))) +
                         (fz(i,j,k+1)-fz(i,j,k));
        });
    }
    amrex::Gpu::streamSynchronize();
}

void compute_metric_cell_gradient(amrex::MultiFab const& scalar,
                                  MetricData const& metric,
                                  std::uint64_t expected_metric_epoch,
                                  MappingOperatorConfig const& config,
                                  amrex::MultiFab& gradient)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    auto const& volume = metric.cell_volume_cc();
    require_cell_layout(scalar, 1, volume, "cell gradient input");
    require_cell_layout(gradient, AMREX_SPACEDIM, volume, "cell gradient output");
    if (scalar.nGrow() < 1) {
        throw std::runtime_error("AVWiS metric cell gradient requires one valid scalar ghost layer");
    }

    auto const spacing = metric.logical_grid().spacing;
    for (amrex::MFIter mfi(gradient, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.const_array(mfi);
        auto const grad_xi = metric.grad_xi_cc().const_array(mfi);
        auto const grad = gradient.array(mfi);
        amrex::Real const h0 = spacing[0];
        amrex::Real const h1 = spacing[1];
        amrex::Real const h2 = spacing[2];
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real const dphi[3] = {
                (phi(i+1,j,k)-phi(i-1,j,k))/(2.0*h0),
                (phi(i,j+1,k)-phi(i,j-1,k))/(2.0*h1),
                (phi(i,j,k+1)-phi(i,j,k-1))/(2.0*h2)};
            for (int physical = 0; physical < 3; ++physical) {
                grad(i,j,k,physical) =
                    grad_xi(i,j,k,physical) * dphi[0] +
                    grad_xi(i,j,k,3+physical) * dphi[1] +
                    grad_xi(i,j,k,6+physical) * dphi[2];
            }
        });
    }
    amrex::Gpu::streamSynchronize();
}

void compute_orthogonal_face_gradient_flux(amrex::MultiFab const& scalar,
                                           int direction,
                                           MetricData const& metric,
                                           std::uint64_t expected_metric_epoch,
                                           MappingOperatorConfig const& config,
                                           amrex::MultiFab& face_flux)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    if (config.projection != ProjectionOperatorMode::OrthogonalMLMG &&
        !(config.mapping_type == "analytic_shear" && config.shear_epsilon == 0.0)) {
        throw std::runtime_error("AVWiS orthogonal face gradient requires orthogonal_mlmg mode");
    }
    if (direction < 0 || direction >= AMREX_SPACEDIM) {
        throw std::out_of_range("AVWiS orthogonal face-gradient direction");
    }
    require_cell_layout(scalar, 1, metric.cell_volume_cc(), "face gradient input");
    require_face_layout(face_flux, direction, metric, "face gradient output");
    if (scalar.nGrow() < 1) {
        throw std::runtime_error("AVWiS orthogonal face gradient requires one scalar ghost layer");
    }

    amrex::Real const inverse_h = 1.0 / metric.logical_grid().spacing[direction];
    for (amrex::MFIter mfi(face_flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const phi = scalar.const_array(mfi);
        auto const q = metric.face_gradient_metric_fc(direction).const_array(mfi);
        auto const beta = metric.projection_beta_fc(direction).const_array(mfi);
        auto const flux = face_flux.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            int const il = i - (direction == 0);
            int const jl = j - (direction == 1);
            int const kl = k - (direction == 2);
            flux(i,j,k) = beta(i,j,k) * q(i,j,k,direction) *
                          (phi(i,j,k)-phi(il,jl,kl)) * inverse_h;
        });
    }
    amrex::Gpu::streamSynchronize();
}

void sync_metric_ucont_from_ucat(
    amrex::MultiFab const& ucat,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::Periodicity const& periodicity)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    if (config.coordinates != CoordinateSystemMode::Mapped &&
        config.pressure_contract != "incremental_pressure") {
        throw std::runtime_error("AVWiS orthogonal Ucat-to-Ucont requires mapped coordinates");
    }
    require_cell_layout(ucat, AMREX_SPACEDIM, metric.cell_volume_cc(), "Ucat-to-Ucont input");
    if (ucat.nGrow() < 1) {
        throw std::runtime_error("AVWiS orthogonal Ucat-to-Ucont requires one velocity ghost layer");
    }
    bool const conservative_interpolation = config.pressure_contract == "incremental_pressure" &&
        config.mapping_type == "analytic_orthogonal" &&
        config.analytic_parameters.stretch != amrex::Array<amrex::Real,3>{0,0,0};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (ucont[dir] == nullptr) throw std::runtime_error("AVWiS null Ucont destination");
        require_face_layout(*ucont[dir], dir, metric, "Ucat-to-Ucont output");
        for (amrex::MFIter mfi(*ucont[dir], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const u = ucat.const_array(mfi);
            auto const area = metric.face_area_vector_fc(dir).const_array(mfi);
            auto const flux = ucont[dir]->array(mfi);
            if (conservative_interpolation) {
                auto const volume = metric.cell_volume_cc().const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
                    int const il=i-(dir==0), jl=j-(dir==1), kl=k-(dir==2);
                    auto const vl=volume(il,jl,kl), vr=volume(i,j,k);
                    amrex::Real value=0;
                    for (int c=0;c<3;++c) {
                        // R uses a pair average. Its global volume weights on
                        // a face are (V_left+V_right)/2. This matching I ensures
                        // sum(V R(I(a)))=sum(V a) on separable orthogonal grids.
                        auto const average=(vl*u(il,jl,kl,c)+vr*u(i,j,k,c))/(vl+vr);
                        value+=average*area(i,j,k,c);
                    }
                    flux(i,j,k)=value;
                });
            } else {
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    int const il = i - (dir == 0);
                    int const jl = j - (dir == 1);
                    int const kl = k - (dir == 2);
                    amrex::Real value = 0.0;
                    for (int comp = 0; comp < 3; ++comp) {
                        value += 0.5 * (u(il,jl,kl,comp) + u(i,j,k,comp)) * area(i,j,k,comp);
                    }
                    flux(i,j,k) = value;
                });
            }
        }
        ucont[dir]->OverrideSync(periodicity);
        ucont[dir]->FillBoundary(periodicity);
    }
    amrex::Gpu::streamSynchronize();
}

void sync_metric_ucat_from_ucont(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t expected_metric_epoch,
    MappingOperatorConfig const& config, amrex::MultiFab& ucat)
{
    validate_mapping_operator_config(config, metric, expected_metric_epoch);
    if (config.coordinates != CoordinateSystemMode::Mapped &&
        config.pressure_contract != "incremental_pressure") {
        throw std::runtime_error("AVWiS orthogonal Ucont-to-Ucat requires mapped coordinates");
    }
    require_cell_layout(ucat, AMREX_SPACEDIM, metric.cell_volume_cc(), "Ucont-to-Ucat output");
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (ucont[dir] == nullptr) throw std::runtime_error("AVWiS null Ucont input");
        require_face_layout(*ucont[dir], dir, metric, "Ucont-to-Ucat input");
    }

    amrex::Real const epsilon = config.shear_epsilon;
    for (amrex::MFIter mfi(ucat, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const u = ucat.array(mfi);
        auto const fx = ucont[0]->const_array(mfi);
        auto const fy = ucont[1]->const_array(mfi);
        auto const fz = ucont[2]->const_array(mfi);
        auto const sx = metric.face_area_vector_fc(0).const_array(mfi);
        auto const sy = metric.face_area_vector_fc(1).const_array(mfi);
        auto const sz = metric.face_area_vector_fc(2).const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Pair-normalized flux averages; the fixed-shear inverse below
            // is the triangular specialization of normalized six-face LS.
            u(i,j,k,0) = 0.5 * (fx(i,j,k)/sx(i,j,k,0) + fx(i+1,j,k)/sx(i+1,j,k,0));
            u(i,j,k,1) = 0.5 * (fy(i,j,k)/sy(i,j,k,1) + fy(i,j+1,k)/sy(i,j+1,k,1));
            u(i,j,k,2) = 0.5 * (fz(i,j,k)/sz(i,j,k,2) + fz(i,j,k+1)/sz(i,j,k+1,2));
            // Fixed shear normals: Sx/Ax=(1,-epsilon,0), Sy/Ay=(0,1,0).
            // Normalized six-face LS reduces exactly to this triangular solve.
            u(i,j,k,0) += epsilon*u(i,j,k,1);
        });
    }
    amrex::Gpu::streamSynchronize();
}

void sync_orthogonal_ucont_from_ucat(amrex::MultiFab const& ucat,
    amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::Periodicity const& periodicity)
{
    if (config.shear_epsilon != 0.0)
        throw std::runtime_error("AVWiS orthogonal transform rejected nonzero shear");
    sync_metric_ucont_from_ucat(ucat,ucont,metric,epoch,config,periodicity);
}

void sync_orthogonal_ucat_from_ucont(
    amrex::Array<amrex::MultiFab const*, AMREX_SPACEDIM> const& ucont,
    MetricData const& metric, std::uint64_t epoch, MappingOperatorConfig const& config,
    amrex::MultiFab& ucat)
{
    if (config.shear_epsilon != 0.0)
        throw std::runtime_error("AVWiS orthogonal transform rejected nonzero shear");
    sync_metric_ucat_from_ucont(ucont,metric,epoch,config,ucat);
}
