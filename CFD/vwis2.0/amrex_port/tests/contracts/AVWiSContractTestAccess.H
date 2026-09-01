#ifndef VWIS_CONTRACT_TEST_ACCESS_H_
#define VWIS_CONTRACT_TEST_ACCESS_H_

#include "AVWiSSolver.H"

class AVWiSContractRunner;

/**
 * Narrow friend bridge used only by AVWiSContractRunner implementations.
 * It mirrors the private names touched by the established assertions so their
 * numerical bodies stay unchanged; it does not publish solver storage.
 */
class AVWiSContractTestAccess {
    friend class AVWiSContractRunner;

    explicit AVWiSContractTestAccess(AVWiSSolver& solver)
        : m_solver(solver),
          m_geom(solver.m_geom), m_ba(solver.m_ba), m_dm(solver.m_dm),
          m_metric_data(solver.m_metric_data), m_metric_epoch(solver.m_metric_epoch),
          m_mapping_operator(solver.m_mapping_operator),
          m_p(solver.m_p), m_phi(solver.m_phi), m_nvert(solver.m_nvert),
          m_ucat(solver.m_ucat),
          m_ucat_old(solver.m_ucat_old), m_ucat_older(solver.m_ucat_older),
          m_projection_rhs(solver.m_projection_rhs), m_ucont(solver.m_ucont),
          m_ucont_old(solver.m_ucont_old), m_ucont_older(solver.m_ucont_older),
          m_fields(solver.m_fields), m_metrics(solver.m_metrics),
          m_boundary(solver.m_boundary), m_dx(solver.m_dx),
          m_face_area(solver.m_face_area), m_cell_volume(solver.m_cell_volume),
          m_nghost(solver.m_nghost), m_initialize_seconds(solver.m_initialize_seconds),
          m_last_advance_seconds(solver.m_last_advance_seconds), m_time(solver.m_time),
          m_step(solver.m_step), m_history_depth(solver.m_history_depth),
          m_valid_epoch(solver.m_valid_epoch), m_halo_epoch(solver.m_halo_epoch),
          m_physical_epoch(solver.m_physical_epoch)
    {}

    void run_runtime_contract_checks();
    void run_p2_transform_layout_checks();
    void run_p3_boundary_contract_checks();
    void run_p4_projection_contract_checks(amrex::Real dt, amrex::Real time_coefficient);
    void run_p5_advection_contract_checks();
    void run_p5_viscous_contract_checks(amrex::Real viscosity);
    void run_p5_time_contract_checks(amrex::Real dt, amrex::Real final_time,
                                     amrex::Real viscosity);
    void run_p5_orthogonal_projection_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p5_mapped_boundary_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p5_mapped_boundary_c52_contract_checks(
        amrex::Real dt, amrex::Real time_coefficient);
    void run_p8_restart_contract_checks(std::string const& path, amrex::Real dt,
                                        int total_steps, int checkpoint_step,
                                        amrex::Real viscosity);
    void run_p8_sampling_statistics_contract(std::string const& report_path,
                                             std::string const& plane_path);

    void fill_ghost_cells() { m_solver.fill_ghost_cells(); }
    void fill_physical_ghost_cells() { m_solver.fill_physical_ghost_cells(); }
    void apply_boundary_pipeline(char const* stage) { m_solver.apply_boundary_pipeline(stage); }
    void require_ghosts_fresh(char const* consumer) const {
        m_solver.require_ghosts_fresh(consumer);
    }
    amrex::Real boundary_flux(int dir, bool high) const {
        return m_solver.boundary_flux(dir, high);
    }
    void mark_valid_modified() { m_solver.mark_valid_modified(); }
    void sync_ucat_from_ucont() { m_solver.sync_ucat_from_ucont(); }
    void sync_ucont_from_ucat() { m_solver.sync_ucont_from_ucat(); }
    P3Diagnostics p3_diagnostics(char const* stage, bool require_fresh) const {
        return m_solver.p3_diagnostics(stage, require_fresh);
    }
    void validate_projection_boundary_policy() const {
        m_solver.validate_projection_boundary_policy();
    }
    void validate_boundary_config() const { m_solver.validate_boundary_config(); }
    ProjectionDiagnostics project_cartesian(amrex::Real dt, amrex::Real coefficient) {
        return m_solver.project_cartesian(dt, coefficient);
    }
    ProjectionDiagnostics project_orthogonal(amrex::Real dt, amrex::Real coefficient) {
        return m_solver.project_orthogonal(dt, coefficient);
    }
    void compute_cartesian_divergence(amrex::MultiFab& divergence) const {
        m_solver.compute_cartesian_divergence(divergence);
    }
    amrex::Real validate_singular_rhs_compatibility(amrex::MultiFab const& rhs) const {
        return m_solver.validate_singular_rhs_compatibility(rhs);
    }
    void compute_cartesian_advection_rhs(amrex::MultiFab& rhs) {
        m_solver.compute_cartesian_advection_rhs(rhs);
    }
    void compute_cartesian_viscous_rhs(amrex::MultiFab& rhs, amrex::Real viscosity) {
        m_solver.compute_cartesian_viscous_rhs(rhs, viscosity);
    }
    TimeStepDiagnostics time_step_diagnostics(amrex::Real dt, amrex::Real viscosity) const {
        return m_solver.time_step_diagnostics(dt, viscosity);
    }
    void advance_one_step(amrex::Real dt, amrex::Real viscosity) {
        m_solver.advance_one_step(dt, viscosity);
    }
    void write_checkpoint(std::string const& path) const {
        m_solver.write_checkpoint(path);
    }
    void read_checkpoint(std::string const& path) { m_solver.read_checkpoint(path); }
    UniformPointSample sample_uniform_point(
        amrex::Array<amrex::Real, AMREX_SPACEDIM> const& position) const {
        return m_solver.sample_uniform_point(position);
    }
    UniformPlaneStatistics uniform_plane_statistics(int direction, int cell_index) const {
        return m_solver.uniform_plane_statistics(direction, cell_index);
    }
    UniformFlowDiagnostics uniform_flow_diagnostics() const {
        return m_solver.uniform_flow_diagnostics();
    }
    void write_uniform_plane_csv(
        std::string const& path, int direction, int cell_index) const {
        m_solver.write_uniform_plane_csv(path, direction, cell_index);
    }

    AVWiSSolver& m_solver;
    amrex::Geometry& m_geom;
    amrex::BoxArray& m_ba;
    amrex::DistributionMapping& m_dm;
    MetricData& m_metric_data;
    std::uint64_t& m_metric_epoch;
    MappingOperatorConfig& m_mapping_operator;
    amrex::MultiFab& m_p;
    amrex::MultiFab& m_phi;
    amrex::MultiFab& m_nvert;
    amrex::MultiFab& m_ucat;
    amrex::MultiFab& m_ucat_old;
    amrex::MultiFab& m_ucat_older;
    amrex::MultiFab& m_projection_rhs;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& m_ucont;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& m_ucont_old;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& m_ucont_older;
    std::vector<FieldSpec>& m_fields;
    std::vector<MetricSpec>& m_metrics;
    CartesianBoundaryConfig& m_boundary;
    amrex::Array<amrex::Real, AMREX_SPACEDIM>& m_dx;
    amrex::Array<amrex::Real, AMREX_SPACEDIM>& m_face_area;
    amrex::Real& m_cell_volume;
    int& m_nghost;
    amrex::Real& m_initialize_seconds;
    amrex::Real& m_last_advance_seconds;
    amrex::Real& m_time;
    std::uint64_t& m_step;
    int& m_history_depth;
    std::uint64_t& m_valid_epoch;
    std::uint64_t& m_halo_epoch;
    std::uint64_t& m_physical_epoch;
};

#endif
