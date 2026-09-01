#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
port="$root/amrex_port"
required=("$port/CMakeLists.txt" "$port/CMakePresets.json" "$port/amrex_version.lock" "$port/inputs/p1_smoke.in" "$port/inputs/p1_multibox.in" "$port/inputs/p1_contract.in" "$port/inputs/p2_contract.in" "$port/inputs/p2_boundary_face.in" "$port/inputs/p3_cartesian_boundary.in" "$port/inputs/p3_legacy_supported.in" "$port/inputs/p3_legacy_rejected.in" "$port/inputs/p4_periodic_projection.in" "$port/inputs/p4_closed_neumann_projection.in" "$port/inputs/p4_inflow_outflow_projection.in" "$port/inputs/p5_periodic_advection_16.in" "$port/inputs/p5_periodic_advection_32.in" "$port/inputs/p5_boundary_multibox_advection.in" "$port/inputs/p5_periodic_viscous_16.in" "$port/inputs/p5_periodic_viscous_32.in" "$port/inputs/p5_boundary_multibox_viscous.in" "$port/inputs/p5_explicit_time.in" "$port/inputs/p0_cartesian_benchmark.in" "$port/inputs/p5_explicit_cfl_rejected.in" "$port/inputs/p8_sampling_statistics.in" "$port/src/CartesianBoundaryConfig.H" "$port/src/CartesianBoundaryConfig.cpp" "$port/src/AVWiSSolver.H" "$port/src/AVWiSSolver.cpp" "$port/src/AVWiSBoundary.cpp" "$port/src/AVWiSProjection.cpp" "$port/src/AVWiSAdvection.cpp" "$port/src/AVWiSViscosity.cpp" "$port/src/AVWiSTime.cpp" "$port/src/AVWiSDiagnostics.cpp" "$port/tests/contracts/AVWiSContractChecks.cpp" "$port/src/AVWiSCaseRunnerAccess.H" "$port/benchmarks/CartesianBenchmarkCase.H" "$port/benchmarks/CartesianBenchmarkCase.cpp" "$port/benchmarks/PhysicalChannelCase.H" "$port/benchmarks/PhysicalChannelCase.cpp" "$port/tests/cartesian_benchmark.cmake" "$port/tests/p8_sampling_statistics.cmake")
required+=("$port/inputs/p10_lid_cavity_sanity.in" "$port/inputs/unknown_mode.in" "$port/src/RunMode.H" "$port/src/RunMode.cpp" "$port/cases/LidDrivenCavityCase.H" "$port/cases/LidDrivenCavityCase.cpp" "$port/tests/lid_driven_cavity.cmake" "$port/tests/contracts/AVWiSContractRunner.H" "$port/tests/contracts/AVWiSContractRunner.cpp" "$port/tests/contracts/AVWiSContractTestAccess.H" "$port/tests/contracts/AVWiSP8ContractChecks.cpp")
required+=("$port/src/AVWiSCoordinateMapping.H" "$port/src/AVWiSCoordinateMapping.cpp" "$port/src/AVWiSMetricData.H" "$port/src/AVWiSMetricData.cpp" "$port/src/AVWiSMetricAdapter.H" "$port/src/AVWiSMetricAdapter.cpp" "$port/tests/curvilinear/MetricIdentityContract.cpp" "$port/tests/curvilinear/MetricIdentityAdapterContract.cpp")
required+=("$port/tests/curvilinear/MetricAnalyticOrthogonalContract.cpp")
required+=("$port/src/AVWiSMappedOperators.H" "$port/src/AVWiSMappedOperators.cpp" "$port/tests/curvilinear/MetricOrthogonalOperatorContract.cpp" "$port/tests/contracts/AVWiSC2ContractChecks.cpp" "$port/inputs/p5_analytic_orthogonal_projection.in" "$port/inputs/p5_mapped_cartesian_rejected.in")
required+=("$port/tests/contracts/AVWiSC5ContractChecks.cpp" "$port/inputs/p5_c5_mapped_wall_boundary.in" "$port/inputs/p5_c5_mapped_periodic_boundary.in")
required+=("$port/inputs/p5_c52_mapped_inflow_outflow.in" "$port/inputs/p5_c52_mapped_wall_modes.in" "$port/inputs/p5_c52_mapped_legacy_rejected.in" "$port/inputs/p5_c52_mapped_moving_normal_rejected.in" "$port/inputs/p5_c52_nonorthogonal_rejected.in")
required+=("$root/_Docs/AMReX_P5-003_G0_identity_metric_20260831.md" "$root/_Docs/AMReX_P5-003_C1_identity_adapter_20260831.md" "$root/_Docs/AMReX_P5-003_C2_analytic_orthogonal_20260831.md")
required+=("$root/_Docs/AMReX_P5-003_C2.2_orthogonal_operators_20260901.md")
required+=("$root/_Docs/AMReX_P5-003_C5.1_mapped_boundary_20260901.md")
required+=("$root/_Docs/AMReX_P5-003_C5.2_mapped_boundary_modes_20260901.md")
for file in "${required[@]}"; do test -f "$file"; done
old_contract_source="$port/src/AVWiSContract"'s.cpp'
test ! -e "$old_contract_source"
rg -q 'tests/contracts/AVWiSContractChecks.cpp' "$port/CMakeLists.txt"
rg -q 'tests/contracts/AVWiSP8ContractChecks.cpp' "$port/CMakeLists.txt"
rg -q 'set\(AVWIS_PRODUCTION_SOURCES' "$port/CMakeLists.txt"
rg -q 'set\(AVWIS_APPLICATION_SOURCES' "$port/CMakeLists.txt"
rg -q 'set\(AVWIS_TEST_MODE_SOURCES' "$port/CMakeLists.txt"
rg -q 'add_library\(avwis_metric STATIC' "$port/CMakeLists.txt"
rg -q 'add_executable\(avwis_metric_identity_contract' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_metric_identity_contract' "$port/CMakeLists.txt"
rg -q 'add_executable\(avwis_metric_identity_adapter_contract' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_metric_identity_adapter_contract' "$port/CMakeLists.txt"
rg -q 'add_executable\(avwis_metric_analytic_orthogonal_contract' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_metric_analytic_orthogonal_contract' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_metric_orthogonal_operator_contract' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_orthogonal_projection' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c5_mapped_wall_boundary' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c5_mapped_periodic_boundary' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c52_mapped_inflow_outflow' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c52_mapped_wall_modes' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c52_mapped_legacy_rejected' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c52_mapped_moving_normal_rejected' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_p5_c52_nonorthogonal_rejected' "$port/CMakeLists.txt"
rg -q 'class CoordinateMapping' "$port/src/AVWiSCoordinateMapping.H"
rg -q 'class IdentityCoordinateMapping' "$port/src/AVWiSCoordinateMapping.H"
rg -q 'class AnalyticOrthogonalCoordinateMapping' "$port/src/AVWiSCoordinateMapping.H"
rg -q 'abs\(stretch\) < 1' "$port/src/AVWiSCoordinateMapping.cpp"
rg -q 'unknown coordinate mapping type' "$port/src/AVWiSCoordinateMapping.cpp"
rg -q 'class MetricData' "$port/src/AVWiSMetricData.H"
rg -q 'mapping_jacobian_cc' "$port/src/AVWiSMetricData.H" "$port/src/AVWiSMetricData.cpp"
rg -q 'inverse_mapping_jacobian_cc' "$port/src/AVWiSMetricData.H" "$port/src/AVWiSMetricData.cpp"
rg -q 'cell_volume_cc' "$port/src/AVWiSMetricData.H" "$port/src/AVWiSMetricData.cpp"
rg -q 'face_area_vector_fc' "$port/src/AVWiSMetricData.H" "$port/src/AVWiSMetricData.cpp"
rg -q 'OverrideSync' "$port/src/AVWiSMetricData.cpp"
rg -q 'MetricData const& metric_data\(\) const' "$port/src/AVWiSSolver.H"
rg -q 'std::uint64_t metric_epoch\(\) const' "$port/src/AVWiSSolver.H"
rg -q 'm_metric_data.define\(m_ba, m_dm, 1\)' "$port/src/AVWiSSolver.cpp"
rg -q 'compute_identity_metric_divergence' "$port/src/AVWiSProjection.cpp" "$port/src/AVWiSMetricAdapter.cpp"
rg -q 'accepts identity MetricData only' "$port/src/AVWiSMetricAdapter.cpp"
rg -q 'stale metric epoch' "$port/src/AVWiSMetricAdapter.cpp"
rg -q 'enum class CoordinateSystemMode' "$port/src/AVWiSMappedOperators.H"
rg -q 'enum class ProjectionOperatorMode' "$port/src/AVWiSMappedOperators.H"
rg -q 'validate_mapping_operator_config' "$port/src/AVWiSMappedOperators.cpp" "$port/src/AVWiSProjection.cpp"
rg -q 'compute_metric_cell_gradient' "$port/src/AVWiSMappedOperators.cpp"
rg -q 'compute_metric_divergence' "$port/src/AVWiSMappedOperators.cpp"
rg -q 'compute_orthogonal_face_gradient_flux' "$port/src/AVWiSMappedOperators.cpp" "$port/src/AVWiSProjection.cpp"
rg -q 'C2.2 orthogonal projection' "$port/src/AVWiSProjection.cpp"
rg -q 'avwis.coordinates = mapped' "$port/inputs/p5_analytic_orthogonal_projection.in"
rg -q 'avwis.projection.operator = orthogonal_mlmg' "$port/inputs/p5_analytic_orthogonal_projection.in"
rg -q 'enum class BoundaryGeometryMode' "$port/src/CartesianBoundaryConfig.H"
rg -q 'avwisbcs.geometry = mapped_orthogonal' "$port/inputs/p5_c5_mapped_wall_boundary.in"
rg -q 'C5.2 mapped physical boundaries require analytic_orthogonal mapping' "$port/src/AVWiSBoundary.cpp"
rg -q 'face_area_vector_fc' "$port/src/AVWiSBoundary.cpp" "$port/tests/contracts/AVWiSC5ContractChecks.cpp"
rg -q 'pre-c5.2-orthogonal-projection' "$port/src/AVWiSProjection.cpp"
rg -q 'P5-003 C5.1 mapped boundary contract: PASS' "$port/tests/contracts/AVWiSC5ContractChecks.cpp"
rg -q 'P5-003 C5.2 mapped boundary contract: PASS' "$port/tests/contracts/AVWiSC5ContractChecks.cpp"
rg -q 'general non-orthogonal/curved' "$port/src/AVWiSCoordinateMapping.cpp"
rg -q 'physical_area \* weight' "$port/src/AVWiSBoundary.cpp"
rg -q 'report.singular = !has_pressure_dirichlet' "$port/src/AVWiSProjection.cpp"
if rg -n 'amrex::MultiFab& (node_coordinates_nd|cell_center_coordinates_cc|mapping_jacobian_cc|inverse_mapping_jacobian_cc|grad_xi_cc|area_cofactor_cc|cell_volume_cc|face_area_vector_fc|face_gradient_metric_fc|projection_beta_fc)' "$port/src/AVWiSMetricData.H"; then
    echo 'MetricData exposes a writable MultiFab accessor' >&2
    exit 1
fi
if rg -n '(contract runner|contract check|manufactured|MMS)' "$port/src/AVWiSMetricData."* "$port/src/AVWiSCoordinateMapping."* "$port/src/AVWiSMetricAdapter."*; then
    echo 'test-only metric contract logic entered production src' >&2
    exit 1
fi
rg -q 'P3Diagnostics AVWiSSolver::p3_diagnostics' "$port/src/AVWiSDiagnostics.cpp"
rg -q 'void AVWiSSolver::diagnostics' "$port/src/AVWiSDiagnostics.cpp"
rg -q 'void AVWiSSolver::write_metadata_manifest' "$port/src/AVWiSDiagnostics.cpp"

# Naming migration guard: the port must expose only AVWiS names.  Historical
# wording is allowed in documentation outside this executable/source tree,
# but no stale product/build identifier may remain in the port itself.
legacy_core='Vwis'"AmrEx"
legacy_exe='vwis_amrex'"_skeleton"
legacy_ctest='vwis_amrex'"_"
if rg -n "$legacy_core|$legacy_exe|$legacy_ctest" "$port" \
        --glob '!tests/static_contract_check.sh'; then
    echo 'stale legacy product/build identifier remains in amrex_port' >&2
    exit 1
fi
rg -q 'add_executable\(avwis ' "$port/CMakeLists.txt"
rg -q 'add_test\(NAME avwis_' "$port/CMakeLists.txt"
rg -q 'class AVWiSSolver' "$port/src/AVWiSSolver.H"
rg -q 'class AVWiSCaseRunnerAccess' "$port/src/AVWiSCaseRunnerAccess.H"
rg -q 'class AVWiSContractRunner' "$port/tests/contracts/AVWiSContractRunner.H"
rg -q 'class AVWiSContractTestAccess' "$port/tests/contracts/AVWiSContractTestAccess.H"
rg -q 'avwis\.mode' "$port/inputs/p1_smoke.in"
rg -q 'legacy vwis\.\* input namespace' "$port/src/RunMode.cpp"
rg -q 'avwis-uniform-diagnostics-v1' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'find_package\(AMReX CONFIG QUIET\)' "$port/CMakeLists.txt"
rg -q 'AMReXConfig.cmake was not found' "$port/CMakeLists.txt"
rg -q 'enum class RunMode' "$port/src/RunMode.H"
rg -q 'unknown avwis.mode' "$port/src/RunMode.cpp"
rg -q 'void run_solve_loop\(' "$port/src/RunMode.cpp"
rg -q 'run_solve_loop\(solver, config\)' "$port/src/RunMode.cpp"
if rg -q 'break;' "$port/src/RunMode.cpp"; then
    echo 'run-mode dispatch must not rely on switch fall-through' >&2
    exit 1
fi
rg -q 'dispatch_run_mode\(solver, run\)' "$port/src/main.cpp"
rg -q 'run_cartesian_benchmark\(solver,' "$port/src/RunMode.cpp"
rg -q 'run_physical_channel_benchmark\(solver,' "$port/src/RunMode.cpp"
rg -q 'run_lid_driven_cavity_case\(' "$port/src/RunMode.cpp"
rg -q 'AVWiSContractRunner contracts\(solver\)' "$port/src/RunMode.cpp"
rg -q 'contracts.run_runtime_contract_checks\(\)' "$port/src/RunMode.cpp"
rg -q 'contracts.run_p2_transform_layout_checks\(\)' "$port/src/RunMode.cpp"
rg -q 'friend class AVWiSContractTestAccess' "$port/src/AVWiSSolver.H"
rg -q 'class AVWiSContractRunner' "$port/tests/contracts/AVWiSContractRunner.H"
if rg -q 'class AVWiSContractRunner[[:space:]]*:' "$port/tests/contracts/AVWiSContractRunner.H"; then
    echo 'contract runner must use composition, not solver inheritance' >&2
    exit 1
fi
if rg -q 'run_(runtime_contract_checks|p2_transform_layout_checks|p3_boundary_contract_checks|p4_projection_contract_checks|p5_advection_contract_checks|p5_viscous_contract_checks|p5_time_contract_checks|p8_restart_contract_checks|p8_sampling_statistics_contract)' "$port/src/AVWiSSolver.H"; then
    echo 'test contract-runner method declaration entered AVWiSSolver.H' >&2
    exit 1
fi
if rg -n '(AVWiSContractTestAccess\.H|run_p8_restart_contract_checks|run_p8_sampling_statistics_contract|manufactured contract test|sampling/statistics manufactured values)' \
        "$port/src" --glob '!RunMode.*'; then
    echo 'test-only implementation or access bridge entered production src' >&2
    exit 1
fi
if rg -q '(CartesianBenchmarkCase|PhysicalChannelCase|LidDrivenCavityCase|run_cartesian_benchmark|run_physical_channel_benchmark|run_lid_driven_cavity_case)' "$port/src/AVWiSSolver.H"; then
    echo 'case-specific runner method declaration entered AVWiSSolver.H' >&2
    exit 1
fi
rg -q 'friend class AVWiSCaseRunnerAccess' "$port/src/AVWiSSolver.H"
if rg -q 'run_(contract_checks|p2_transform_checks|p3_boundary_checks|p4_projection_checks|p5_advection_checks|p5_viscous_checks|p5_time_checks|p8_restart_checks|p8_sampling_checks|cartesian_benchmark|physical_benchmark|lid_driven_cavity)' "$port/src/main.cpp"; then
    echo 'legacy independent run_* dispatch remains in main.cpp' >&2
    exit 1
fi
rg -q 'FieldLocation' "$port/src/AVWiSSolver.H"
rg -q 'component_names' "$port/src/AVWiSSolver.H"
rg -q 'm_cell_volume' "$port/src/AVWiSSolver.H"
rg -q 'm_face_area' "$port/src/AVWiSSolver.H"
rg -q 'm_ucont_older' "$port/src/AVWiSSolver.H"
rg -q 'amrex::BCRec' "$port/src/AVWiSSolver.H"
rg -q 'AMREX_GPU_DEVICE' "$port/src/AVWiSSolver.cpp" "$port/src/AVWiSBoundary.cpp" "$port/src/AVWiSProjection.cpp" "$port/src/AVWiSAdvection.cpp" "$port/src/AVWiSViscosity.cpp" "$port/src/AVWiSTime.cpp" "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'payload_written' "$port/src/AVWiSDiagnostics.cpp"
rg -q 'AVWiSContractTestAccess::run_runtime_contract_checks' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'sync_ucat_from_ucont' "$port/src/AVWiSSolver.cpp"
rg -q 'sync_ucont_from_ucat' "$port/src/AVWiSSolver.cpp"
rg -q 'OverrideSync' "$port/src/AVWiSSolver.cpp"
rg -q 'OwnerMask' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'sum_unique' "$port/src/AVWiSBoundary.cpp" "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'normal velocity times face area' "$port/src/AVWiSSolver.cpp"
rg -q 'derived divergence' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'constant Ucat/Ucont volume-flux contract failed' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'base runtime contract: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'P2-003/004/005: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'enum class CartesianBC' "$port/src/CartesianBoundaryConfig.H"
rg -q 'fill_physical_ghost_cells' "$port/src/AVWiSBoundary.cpp"
rg -q 'stale ghost read' "$port/src/AVWiSSolver.cpp"
rg -q 'ReduceRealSum' "$port/src/AVWiSBoundary.cpp" "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'ReduceRealMax' "$port/src/AVWiSDiagnostics.cpp"
rg -q 'P3-001/002/003/004: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'unsupported legacy BC integer' "$port/src/CartesianBoundaryConfig.cpp"
rg -q 'AMReX_MLPoisson.H' "$port/src/AVWiSProjection.cpp"
rg -q 'Ucont already contains u dot S' "$port/src/AVWiSMetricAdapter.cpp"
rg -q 'm_face_area\[dir\]' "$port/src/AVWiSProjection.cpp"
rg -q 'no automatic mean subtraction is permitted' "$port/src/AVWiSProjection.cpp"
rg -q 'LinOpBCType::Dirichlet' "$port/src/AVWiSProjection.cpp"
rg -q 'LinOpBCType::Neumann' "$port/src/AVWiSProjection.cpp"
rg -Fq 'sync_ucat_from_ucont_impl(false)' "$port/src/AVWiSProjection.cpp"
rg -q 'P4 Cartesian projection contract: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -Fq 'fx(i+1,j,k) * 0.5 * (u(i,j,k,comp) + u(i+1,j,k,comp))' "$port/src/AVWiSAdvection.cpp"
rg -q 'inverse_volume' "$port/src/AVWiSAdvection.cpp"
rg -q 'P5-001 periodic manufactured advection: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'P5-001 boundary/multi-Box advection: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'P5-002 periodic manufactured viscosity: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'm_ucat_older' "$port/src/AVWiSSolver.H" "$port/src/AVWiSTime.cpp"
rg -q 'projection_time_coefficient = 1.0' "$port/src/AVWiSSolver.H"
rg -q 'P5-004 explicit Euler time contract: PASS' "$port/tests/contracts/AVWiSContractChecks.cpp"
rg -q 'AVWiSContractTestAccess::run_p8_restart_contract_checks' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'AVWiSContractTestAccess::run_p8_sampling_statistics_contract' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'uninterrupted and checkpoint/restart trajectories differ' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'sampling/statistics manufactured values failed' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'explicit step rejected' "$port/src/AVWiSTime.cpp"
rg -q 'not legacy SNES' "$port/src/AVWiSDiagnostics.cpp"
rg -q 'sample_uniform_point' "$port/src/AVWiSSolver.H" "$port/src/AVWiSDiagnostics.cpp"
rg -q 'uniform_plane_statistics' "$port/src/AVWiSSolver.H" "$port/src/AVWiSDiagnostics.cpp"
rg -q 'avwis-uniform-diagnostics-v1' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'plotfile_compatible.*false' "$port/tests/contracts/AVWiSP8ContractChecks.cpp"
rg -q 'MovingWall' "$port/src/CartesianBoundaryConfig.H" "$port/src/CartesianBoundaryConfig.cpp"
rg -q 'target = moving_wall_velocity\[comp\]' "$port/src/AVWiSBoundary.cpp"
rg -q 'moving_wall_reconstruction_max_error' "$port/cases/LidDrivenCavityCase.cpp" "$port/tests/lid_driven_cavity.cmake"
rg -q 'demonstration / engineering result; not CFD validation' "$port/cases/LidDrivenCavityCase.cpp"
echo 'static P0-P8-003 plus cavity AMReX contract check: PASS'
