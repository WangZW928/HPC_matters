#ifndef AVWIS_METRIC_DATA_H_
#define AVWIS_METRIC_DATA_H_

#include "AVWiSCoordinateMapping.H"

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <cstdint>
#include <string>

struct MetricTolerance {
    amrex::Real minimum_positive = 0.0;
    amrex::Real reciprocal_relative = 5.0e-13;
    amrex::Real gcl_relative = 1.0e-12;
    amrex::Real reciprocity_relative = 1.0e-11;
};

struct MetricDiagnostics {
    amrex::Real minimum_mapping_jacobian = 0.0;
    amrex::Real minimum_cell_volume = 0.0;
    amrex::Real minimum_face_area = 0.0;
    amrex::Real maximum_reciprocal_relative_error = 0.0;
    amrex::Real maximum_gcl_relative_error = 0.0;
    amrex::Real maximum_reciprocity_relative_error = 0.0;
    bool passed = false;
};

/**
 * Immutable-after-build metric owner for one fixed, single-level structured
 * mapping. Public field access is const-only; rebuild() is the sole mutation
 * path after define(). No flow operator is implied by this data model.
 */
class MetricData {
public:
    static constexpr char const* discretization_version = "avwis-shared-triangle-metric-v1";

    void define(amrex::BoxArray const& cell_boxes,
                amrex::DistributionMapping const& distribution, int nghost = 1);
    void build(CoordinateMapping const& mapping, LogicalGrid const& logical,
               amrex::Geometry const& geometry,
               MetricTolerance const& tolerance = {});
    void rebuild(CoordinateMapping const& mapping, LogicalGrid const& logical,
                 amrex::Geometry const& geometry,
                 MetricTolerance const& tolerance = {});
    MetricDiagnostics validate(MetricTolerance const& tolerance = {}) const;

    bool is_defined() const noexcept { return m_defined; }
    std::uint64_t epoch() const noexcept { return m_epoch; }
    std::string const& mapping_id() const noexcept { return m_mapping_id; }
    int nghost() const noexcept { return m_nghost; }
    LogicalGrid const& logical_grid() const { require_ready("logical_grid"); return m_logical; }

    amrex::MultiFab const& node_coordinates_nd() const { require_ready("node_coordinates_nd"); return m_node_coordinates_nd; }
    amrex::MultiFab const& cell_center_coordinates_cc() const { require_ready("cell_center_coordinates_cc"); return m_cell_center_coordinates_cc; }
    amrex::MultiFab const& mapping_jacobian_cc() const { require_ready("mapping_jacobian_cc"); return m_mapping_jacobian_cc; }
    amrex::MultiFab const& inverse_mapping_jacobian_cc() const { require_ready("inverse_mapping_jacobian_cc"); return m_inverse_mapping_jacobian_cc; }
    amrex::MultiFab const& grad_xi_cc() const { require_ready("grad_xi_cc"); return m_grad_xi_cc; }
    amrex::MultiFab const& area_cofactor_cc() const { require_ready("area_cofactor_cc"); return m_area_cofactor_cc; }
    amrex::MultiFab const& cell_volume_cc() const { require_ready("cell_volume_cc"); return m_cell_volume_cc; }
    amrex::MultiFab const& face_area_vector_fc(int dir) const;
    amrex::MultiFab const& face_gradient_metric_fc(int dir) const;
    amrex::MultiFab const& projection_beta_fc(int dir) const;

private:
    void require_ready(char const* operation) const;

    amrex::BoxArray m_cell_boxes;
    amrex::DistributionMapping m_distribution;
    amrex::MultiFab m_node_coordinates_nd;
    amrex::MultiFab m_cell_center_coordinates_cc;
    amrex::MultiFab m_mapping_jacobian_cc;
    amrex::MultiFab m_inverse_mapping_jacobian_cc;
    amrex::MultiFab m_grad_xi_cc;
    amrex::MultiFab m_area_cofactor_cc;
    amrex::MultiFab m_cell_volume_cc;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_face_area_vector_fc;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_face_gradient_metric_fc;
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_projection_beta_fc;
    LogicalGrid m_logical;
    amrex::Periodicity m_periodicity;
    std::string m_mapping_id;
    std::uint64_t m_epoch = 0;
    int m_nghost = 0;
    bool m_defined = false;
    bool m_built = false;
};

#endif
