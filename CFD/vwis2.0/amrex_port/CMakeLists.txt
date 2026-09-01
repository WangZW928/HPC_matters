cmake_minimum_required(VERSION 3.20)
# AMReX's MPI package exports both C and C++ MPI dependencies, so C must be
# enabled when consuming an AMReX install configured with AMReX_MPI=ON.
project(avwis_port LANGUAGES C CXX)

# A different AMReX can change CMake targets, GPU defaults, and solver APIs.
# An override is allowed only for an explicitly labelled exploratory build.
set(AVWIS_EXPECTED_VERSION "26.04" CACHE STRING
    "AMReX version recorded by amrex_version.lock")
option(AVWIS_ALLOW_VERSION_MISMATCH
       "Permit an AMReX version other than AVWIS_EXPECTED_VERSION for exploratory builds" OFF)

find_package(AMReX CONFIG QUIET)
if (NOT AMReX_FOUND)
    message(FATAL_ERROR
        "AMReXConfig.cmake was not found. Set -DAMReX_DIR=/path/to/lib/cmake/AMReX or "
        "add its prefix to CMAKE_PREFIX_PATH. No fallback solver is built.")
endif()
if (DEFINED AMReX_VERSION AND NOT AMReX_VERSION VERSION_EQUAL AVWIS_EXPECTED_VERSION)
    if (AVWIS_ALLOW_VERSION_MISMATCH)
        message(WARNING "AMReX ${AMReX_VERSION} differs from locked ${AVWIS_EXPECTED_VERSION}; exploratory build only.")
    else()
        message(FATAL_ERROR "Found AMReX ${AMReX_VERSION}, but P0-P4 locks ${AVWIS_EXPECTED_VERSION}. Use the locked package or set AVWIS_ALLOW_VERSION_MISMATCH=ON for an unvalidated exploratory build.")
    endif()
elseif (NOT DEFINED AMReX_VERSION)
    message(WARNING "AMReX did not export AMReX_VERSION; record its revision before treating a build as evidence.")
endif()

# A CUDA AMReX package exports CUDA device annotations in its public headers.
# Compile this port's kernel-bearing translation units with nvcc, rather than
# letting a host C++ compiler mistake a GPU package for a CPU-only package.
if (AMReX_CUDA)
    enable_language(CUDA)
endif()

set(AVWIS_METRIC_SOURCES
    src/AVWiSCoordinateMapping.cpp
    src/AVWiSMetricData.cpp
    src/AVWiSMetricAdapter.cpp
    src/AVWiSMappedOperators.cpp)
add_library(avwis_metric STATIC ${AVWIS_METRIC_SOURCES})
target_include_directories(avwis_metric PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(avwis_metric PUBLIC AMReX::amrex)
target_compile_features(avwis_metric PUBLIC cxx_std_17)

set(AVWIS_PRODUCTION_SOURCES
    src/CartesianBoundaryConfig.cpp
    src/AVWiSSolver.cpp
    src/AVWiSBoundary.cpp
    src/AVWiSProjection.cpp
    src/AVWiSAdvection.cpp
    src/AVWiSViscosity.cpp
    src/AVWiSTime.cpp
    src/AVWiSCheckpoint.cpp
    src/AVWiSDiagnostics.cpp)
set(AVWIS_APPLICATION_SOURCES
    src/main.cpp
    src/RunMode.cpp)
set(AVWIS_TEST_MODE_SOURCES
    tests/contracts/AVWiSContractChecks.cpp
    tests/contracts/AVWiSP8ContractChecks.cpp
    tests/contracts/AVWiSC2ContractChecks.cpp
    tests/contracts/AVWiSC5ContractChecks.cpp
    tests/contracts/AVWiSContractRunner.cpp)
set(AVWIS_CASE_SOURCES
    benchmarks/CartesianBenchmarkCase.cpp
    benchmarks/PhysicalChannelCase.cpp
    cases/LidDrivenCavityCase.cpp)
set(AVWIS_SOURCES
    ${AVWIS_PRODUCTION_SOURCES}
    ${AVWIS_APPLICATION_SOURCES}
    ${AVWIS_CASE_SOURCES}
    # The public executable intentionally retains established contract modes.
    # Their implementations are isolated in tests/ and linked only for those
    # explicit modes; no manufactured/regression body lives under src/.
    ${AVWIS_TEST_MODE_SOURCES})
add_executable(avwis ${AVWIS_SOURCES})
target_include_directories(avwis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/cases
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/contracts)
target_link_libraries(avwis PRIVATE avwis_metric AMReX::amrex)
target_compile_features(avwis PRIVATE cxx_std_17)
target_compile_definitions(avwis PRIVATE
    AVWIS_LOCKED_VERSION="${AVWIS_EXPECTED_VERSION}"
    AVWIS_LOCKED_GIT_SHA="9219ba416b7ba2073dd1b12bf19fdce27391f17b")
if (AMReX_CUDA)
    set_source_files_properties(${AVWIS_SOURCES} ${AVWIS_METRIC_SOURCES} PROPERTIES LANGUAGE CUDA)
    # AMReX 26.04 CUDA archives are built with relocatable device code; the
    # application must perform the final device link or registration symbols
    # from the AMReX archive remain unresolved.
    set_target_properties(avwis PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(avwis_metric PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

include(CTest)
if (BUILD_TESTING)
    add_executable(avwis_metric_identity_contract
        tests/curvilinear/MetricIdentityContract.cpp)
    target_link_libraries(avwis_metric_identity_contract PRIVATE avwis_metric AMReX::amrex)
    target_compile_features(avwis_metric_identity_contract PRIVATE cxx_std_17)
    if (AMReX_CUDA)
        set_source_files_properties(tests/curvilinear/MetricIdentityContract.cpp PROPERTIES LANGUAGE CUDA)
        set_target_properties(avwis_metric_identity_contract PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    add_test(NAME avwis_p5_metric_identity_contract COMMAND avwis_metric_identity_contract)
    add_executable(avwis_metric_identity_adapter_contract
        tests/curvilinear/MetricIdentityAdapterContract.cpp)
    target_link_libraries(avwis_metric_identity_adapter_contract PRIVATE avwis_metric AMReX::amrex)
    target_compile_features(avwis_metric_identity_adapter_contract PRIVATE cxx_std_17)
    if (AMReX_CUDA)
        set_source_files_properties(tests/curvilinear/MetricIdentityAdapterContract.cpp PROPERTIES LANGUAGE CUDA)
        set_target_properties(avwis_metric_identity_adapter_contract PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    add_test(NAME avwis_p5_metric_identity_adapter_contract
             COMMAND avwis_metric_identity_adapter_contract)
    add_executable(avwis_metric_analytic_orthogonal_contract
        tests/curvilinear/MetricAnalyticOrthogonalContract.cpp)
    target_link_libraries(avwis_metric_analytic_orthogonal_contract PRIVATE avwis_metric AMReX::amrex)
    target_compile_features(avwis_metric_analytic_orthogonal_contract PRIVATE cxx_std_17)
    if (AMReX_CUDA)
        set_source_files_properties(tests/curvilinear/MetricAnalyticOrthogonalContract.cpp
                                    PROPERTIES LANGUAGE CUDA)
        set_target_properties(avwis_metric_analytic_orthogonal_contract PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    add_test(NAME avwis_p5_metric_analytic_orthogonal_contract
             COMMAND avwis_metric_analytic_orthogonal_contract)
    add_executable(avwis_metric_orthogonal_operator_contract
        tests/curvilinear/MetricOrthogonalOperatorContract.cpp)
    target_link_libraries(avwis_metric_orthogonal_operator_contract PRIVATE avwis_metric AMReX::amrex)
    target_compile_features(avwis_metric_orthogonal_operator_contract PRIVATE cxx_std_17)
    if (AMReX_CUDA)
        set_source_files_properties(tests/curvilinear/MetricOrthogonalOperatorContract.cpp
                                    PROPERTIES LANGUAGE CUDA)
        set_target_properties(avwis_metric_orthogonal_operator_contract PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    add_test(NAME avwis_p5_metric_orthogonal_operator_contract
             COMMAND avwis_metric_orthogonal_operator_contract)
    add_test(NAME avwis_single_box COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_smoke.in)
    add_test(NAME avwis_multi_box COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_multibox.in)
    add_test(NAME avwis_base_runtime_contract
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_contract.in)
    add_test(NAME avwis_runtime_contract
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p2_contract.in)
    add_test(NAME avwis_boundary_face_contract
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p2_boundary_face.in)
    add_test(NAME avwis_p3_cartesian_boundary
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_cartesian_boundary.in)
    add_test(NAME avwis_p3_legacy_supported
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_legacy_supported.in)
    add_test(NAME avwis_p3_legacy_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_legacy_rejected.in)
    add_test(NAME avwis_p4_periodic_projection
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p4_periodic_projection.in)
    add_test(NAME avwis_p4_closed_neumann_projection
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p4_closed_neumann_projection.in)
    add_test(NAME avwis_p4_inflow_outflow_projection
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p4_inflow_outflow_projection.in)
    add_test(NAME avwis_p5_periodic_advection_coarse
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_periodic_advection_16.in)
    add_test(NAME avwis_p5_periodic_advection_fine
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_periodic_advection_32.in)
    add_test(NAME avwis_p5_boundary_multibox_advection
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_boundary_multibox_advection.in)
    add_test(NAME avwis_p5_periodic_viscous_coarse
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_periodic_viscous_16.in)
    add_test(NAME avwis_p5_periodic_viscous_fine
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_periodic_viscous_32.in)
    add_test(NAME avwis_p5_boundary_multibox_viscous
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_boundary_multibox_viscous.in)
    add_test(NAME avwis_p5_explicit_time
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_explicit_time.in)
    add_test(NAME avwis_p5_orthogonal_projection
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_analytic_orthogonal_projection.in)
    add_test(NAME avwis_p5_c5_mapped_wall_boundary
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c5_mapped_wall_boundary.in)
    add_test(NAME avwis_p5_c5_mapped_periodic_boundary
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c5_mapped_periodic_boundary.in)
    add_test(NAME avwis_p5_c52_mapped_inflow_outflow
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_inflow_outflow.in)
    add_test(NAME avwis_p5_c52_mapped_wall_modes
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_wall_modes.in)
    add_test(NAME avwis_p5_c52_mapped_legacy_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_legacy_rejected.in)
    add_test(NAME avwis_p5_c52_mapped_moving_normal_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_moving_normal_rejected.in)
    add_test(NAME avwis_p5_c52_nonorthogonal_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_nonorthogonal_rejected.in)
    add_test(NAME avwis_p5_mapped_cartesian_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_mapped_cartesian_rejected.in)
    add_test(NAME avwis_p5_explicit_cfl_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_explicit_cfl_rejected.in)
    add_test(NAME avwis_p8_restart_consistency
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p8_restart.in
                -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p8_restart_consistency
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p8_restart_consistency.cmake)
    add_test(NAME avwis_p8_strict_rejection
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p8_checkpoint_write.in
                -Drestart_input=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p8_checkpoint_restart.in
                -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p8_strict_rejection
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p8_strict_rejection.cmake)
    add_test(NAME avwis_p8_sampling_statistics
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p8_sampling_statistics.in
                -Dreport=${CMAKE_CURRENT_BINARY_DIR}/p8_sampling_statistics.json
                -Dplane=${CMAKE_CURRENT_BINARY_DIR}/p8_plane_x1.csv
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p8_sampling_statistics.cmake)
    add_test(NAME avwis_cartesian_benchmark
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p0_cartesian_benchmark.in
                -Dreport=${CMAKE_CURRENT_BINARY_DIR}/cartesian_benchmark.json
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cartesian_benchmark.cmake)
    add_test(NAME avwis_physical_channel
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/physical_channel.in
                -Dreport=${CMAKE_CURRENT_BINARY_DIR}/physical_channel.json
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/physical_channel.cmake)
    add_test(NAME avwis_lid_driven_cavity_sanity
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p10_lid_cavity_sanity.in
                -Dwork=${CMAKE_CURRENT_BINARY_DIR}/lid_driven_cavity_sanity
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/lid_driven_cavity.cmake)
    add_test(NAME avwis_unknown_mode_rejected
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/unknown_mode.in)
    # The source error text is also pinned by static_contract_check.sh.  CTest's
    # WILL_FAIL inverts the nonzero process status; combining it with
    # PASS_REGULAR_EXPRESSION has version-dependent inversion semantics.
    set_tests_properties(avwis_p3_legacy_rejected avwis_p5_explicit_cfl_rejected
                         avwis_p5_mapped_cartesian_rejected
                         avwis_p5_c52_mapped_legacy_rejected
                         avwis_p5_c52_mapped_moving_normal_rejected
                         avwis_p5_c52_nonorthogonal_rejected
                         avwis_unknown_mode_rejected
                         PROPERTIES WILL_FAIL TRUE)
endif()
