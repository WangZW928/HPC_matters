cmake_minimum_required(VERSION 3.20)
# AMReX's MPI package exports both C and C++ MPI dependencies, so C must be
# enabled when consuming an AMReX install configured with AMReX_MPI=ON.
project(avwis_port LANGUAGES C CXX)

# Keep the build contract and runtime provenance sourced from one lock file.
# A different AMReX can change CMake targets, GPU defaults, and solver APIs;
# an override is allowed only for an explicitly labelled exploratory build.
set(AVWIS_VERSION_LOCK "${CMAKE_CURRENT_SOURCE_DIR}/amrex_version.lock")
file(STRINGS "${AVWIS_VERSION_LOCK}" AVWIS_LOCKED_VERSION_LINE
     REGEX "^AMReX_VERSION=")
file(STRINGS "${AVWIS_VERSION_LOCK}" AVWIS_LOCKED_GIT_SHA_LINE
     REGEX "^AMReX_GIT_SHA=")
if (NOT AVWIS_LOCKED_VERSION_LINE OR NOT AVWIS_LOCKED_GIT_SHA_LINE)
    message(FATAL_ERROR "amrex_version.lock must define AMReX_VERSION and AMReX_GIT_SHA")
endif()
string(REGEX REPLACE "^AMReX_VERSION=" "" AVWIS_LOCKED_VERSION
       "${AVWIS_LOCKED_VERSION_LINE}")
string(REGEX REPLACE "^AMReX_GIT_SHA=" "" AVWIS_LOCKED_GIT_SHA
       "${AVWIS_LOCKED_GIT_SHA_LINE}")
set(AVWIS_EXPECTED_VERSION "${AVWIS_LOCKED_VERSION}")
message(STATUS
    "AVWiS dependency lock: AMReX ${AVWIS_EXPECTED_VERSION} (${AVWIS_LOCKED_GIT_SHA})")
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
    src/AVWiSMappedOperators.cpp
    src/AVWiSDeferredProjection.cpp)
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
    tests/contracts/AVWiSC7ContractChecks.cpp
    tests/contracts/AVWiSC3ContractChecks.cpp
    tests/contracts/AVWiSC3TemporalChecks.cpp
    tests/contracts/AVWiSPressureReference.cpp
    tests/contracts/AVWiSFaceAuthorityChecks.cpp
    tests/contracts/AVWiSPressureIncrementChecks.cpp
    tests/contracts/AVWiSContractRunner.cpp)
set(AVWIS_CASE_SOURCES
    benchmarks/CartesianBenchmarkCase.cpp
    benchmarks/TaylorGreenMMSCase.cpp
    benchmarks/PhysicalChannelCase.cpp
    benchmarks/MappedChannelReferenceCase.cpp
    cases/MappedPressureWaveCase.cpp
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
    AVWIS_LOCKED_GIT_SHA="${AVWIS_LOCKED_GIT_SHA}")
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
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME avwis_m2_pressure_increment
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/pressure_increment_audit.py
                     --exe $<TARGET_FILE:avwis> --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/m2_pressure_increment.in
                     --work ${CMAKE_CURRENT_BINARY_DIR}/m2_pressure_increment)
    set_tests_properties(avwis_m2_pressure_increment PROPERTIES TIMEOUT 360)
    if (AMReX_CUDA)
        set(AVWIS_C7_BACKEND CUDA)
        # Keep the same real 32/64/128 refinement and 256-step long trajectory
        # on CUDA. Small MLMG solves require a larger wall-clock budget; this
        # changes neither numerical gates nor production solver work limits.
        set(AVWIS_C3_AUDIT_FLAGS --max-grid 16 --timeout 480)
    else()
        set(AVWIS_C7_BACKEND CPU)
        set(AVWIS_C3_AUDIT_FLAGS)
    endif()
    add_test(NAME avwis_p5_time_coefficient_contract
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/p5_time_coefficient_contract.py
                     --exe $<TARGET_FILE:avwis> --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                     --work ${CMAKE_CURRENT_BINARY_DIR}/p5_time_coefficient --backend ${AVWIS_C7_BACKEND})
    set_tests_properties(avwis_p5_time_coefficient_contract PROPERTIES TIMEOUT 360)
    add_test(NAME avwis_c3_shear_regression
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/c3_shear_regression.py
                     --exe $<TARGET_FILE:avwis> --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/c3_shear_periodic.in
                     --work ${CMAKE_CURRENT_BINARY_DIR}/c3_shear)
    set_tests_properties(avwis_c3_shear_regression PROPERTIES TIMEOUT 600)
    add_test(NAME avwis_m2_consistent_reference_contract
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_consistent_state_reference.py)
    add_test(NAME avwis_m2_consistent_reference
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/consistent_state_reference.py
                     --exe $<TARGET_FILE:avwis> --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/c3_temporal.in
                     --work ${CMAKE_CURRENT_BINARY_DIR}/m2_consistent_reference)
    set_tests_properties(avwis_m2_consistent_reference PROPERTIES TIMEOUT 360)
    add_test(NAME avwis_m2_face_authority
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/face_authority_audit.py
                     --exe $<TARGET_FILE:avwis> --backend ${AVWIS_C7_BACKEND}
                     --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/m2_face_authority.in
                     --work ${CMAKE_CURRENT_BINARY_DIR}/m2_face_authority)
    set_tests_properties(avwis_m2_face_authority PROPERTIES TIMEOUT 360)
    add_test(NAME avwis_c3_temporal_comparator_contract
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_c3_temporal_audit.py)
    add_test(NAME avwis_c3_temporal_audit
             COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/c3_temporal_audit.py
                     --exe $<TARGET_FILE:avwis> --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/c3_temporal.in
                     --work ${CMAKE_CURRENT_BINARY_DIR}/c3_temporal --backend ${AVWIS_C7_BACKEND}
                     ${AVWIS_C3_AUDIT_FLAGS})
    set_tests_properties(avwis_c3_temporal_audit PROPERTIES TIMEOUT 1800)
    add_test(NAME avwis_c7_comparator_contract
             COMMAND ${Python3_EXECUTABLE} -B
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_c7_backend_compare.py)
    add_test(NAME avwis_c7_layout_comparison
             COMMAND ${Python3_EXECUTABLE} -B
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/c7_backend_compare.py
                     --exe $<TARGET_FILE:avwis> --backend ${AVWIS_C7_BACKEND}
                     --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                     --work ${CMAKE_CURRENT_BINARY_DIR}/c7_layout)
    set_tests_properties(avwis_c7_layout_comparison PROPERTIES TIMEOUT 180)
    if (AMReX_MPI)
        find_package(MPI REQUIRED COMPONENTS C CXX)
        set(AVWIS_C7_MPI_FLAGS)
        foreach(flag IN LISTS MPIEXEC_PREFLAGS)
            list(APPEND AVWIS_C7_MPI_FLAGS "--mpi-preflag=${flag}")
        endforeach()
        foreach(flag IN LISTS MPIEXEC_POSTFLAGS)
            list(APPEND AVWIS_C7_MPI_FLAGS "--mpi-postflag=${flag}")
        endforeach()
        add_test(NAME avwis_p5_time_coefficient_mpi_contract
                 COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/p5_time_coefficient_contract.py
                         --exe $<TARGET_FILE:avwis> --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                         --work ${CMAKE_CURRENT_BINARY_DIR}/p5_time_coefficient_mpi
                         --backend ${AVWIS_C7_BACKEND} --mpi-exec ${MPIEXEC_EXECUTABLE} --ranks 4
                         --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG} ${AVWIS_C7_MPI_FLAGS})
        set_tests_properties(avwis_p5_time_coefficient_mpi_contract PROPERTIES TIMEOUT 360 PROCESSORS 4)
        add_test(NAME avwis_c7_mpi_comparison
                 COMMAND ${Python3_EXECUTABLE} -B
                         ${CMAKE_CURRENT_SOURCE_DIR}/tests/c7_backend_compare.py
                         --exe $<TARGET_FILE:avwis> --backend ${AVWIS_C7_BACKEND}
                         --inputs ${CMAKE_CURRENT_SOURCE_DIR}/inputs
                         --work ${CMAKE_CURRENT_BINARY_DIR}/c7_mpi
                         --mpi-exec ${MPIEXEC_EXECUTABLE}
                         --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG} ${AVWIS_C7_MPI_FLAGS})
        set_tests_properties(avwis_c7_mpi_comparison PROPERTIES TIMEOUT 300 PROCESSORS 4)
        add_test(NAME avwis_c3_temporal_mpi_audit
                 COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/tests/c3_temporal_audit.py
                         --exe $<TARGET_FILE:avwis> --input ${CMAKE_CURRENT_SOURCE_DIR}/inputs/c3_temporal.in
                         --work ${CMAKE_CURRENT_BINARY_DIR}/c3_temporal_mpi
                         --mpi-exec ${MPIEXEC_EXECUTABLE} --ranks 4 --max-grid 16 --tile "4 4 2"
                         --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG} ${AVWIS_C7_MPI_FLAGS}
                         --reference ${CMAKE_CURRENT_BINARY_DIR}/c3_temporal/audit.json --observe-peer)
        set_tests_properties(avwis_c3_temporal_mpi_audit PROPERTIES
                             TIMEOUT 600 PROCESSORS 4 DEPENDS avwis_c3_temporal_audit)
    endif()
    function(add_avwis_rejection_test test_name input_file expected_text)
        add_test(NAME ${test_name}
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/${input_file}
                    -Dexpected=${expected_text}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_failure.cmake)
    endfunction()

    add_avwis_rejection_test(avwis_c3_state_contract_rejected c3_reject_state_contract.in
        "M2 analytic_shear state_contract supports experimental_dual_state or face_authoritative")
    add_avwis_rejection_test(avwis_c3_pressure_contract_rejected c3_reject_pressure_contract.in
        "time-converged physical pressure remain open")
    add_avwis_rejection_test(avwis_c3_operator_rejected c3_reject_operator.in
        "analytic_shear requires mapped/deferred_nonorthogonal")
    add_avwis_rejection_test(avwis_c3_epsilon_rejected c3_reject_epsilon.in
        "finite abs(epsilon) <= 0.25")
    add_avwis_rejection_test(avwis_c3_boundary_rejected c3_reject_boundary.in
        "requires avwisbcs.enabled=0")
    add_avwis_rejection_test(avwis_c3_integrator_rejected c3_reject_integrator.in
        "time_integrator supports only explicit_euler")
    add_avwis_rejection_test(avwis_c3_stress_rejected c3_reject_stress.in
        "viscous_operator supports only component_laplacian")
    add_avwis_rejection_test(avwis_c3_checkpoint_rejected c3_reject_checkpoint.in
        "analytic_shear checkpoint/restart provenance is not implemented")
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
    add_avwis_rejection_test(
        avwis_p3_legacy_rejected p3_legacy_rejected.in
        "unsupported legacy BC integer 8")
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
    add_test(NAME avwis_p5_c23_mapped_rhs_time
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinputs=${CMAKE_CURRENT_SOURCE_DIR}/inputs
                -Dwork=${CMAKE_CURRENT_BINARY_DIR}/c23_mapped_convergence
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c23_mapped_convergence.cmake)
    add_test(NAME avwis_p5_c5_mapped_wall_boundary
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c5_mapped_wall_boundary.in)
    add_test(NAME avwis_p5_c5_mapped_periodic_boundary
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c5_mapped_periodic_boundary.in)
    add_test(NAME avwis_p5_c52_mapped_inflow_outflow
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_inflow_outflow.in)
    add_test(NAME avwis_p5_c52_mapped_wall_modes
             COMMAND avwis ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p5_c52_mapped_wall_modes.in)
    add_avwis_rejection_test(
        avwis_p5_c52_mapped_legacy_rejected p5_c52_mapped_legacy_rejected.in
        "mapped physical boundaries require named BCs")
    add_avwis_rejection_test(
        avwis_p5_c52_mapped_moving_normal_rejected p5_c52_mapped_moving_normal_rejected.in
        "moving_wall_velocity must have zero normal component")
    add_avwis_rejection_test(
        avwis_p5_c52_nonorthogonal_rejected p5_c52_nonorthogonal_rejected.in
        "unknown coordinate mapping type 'general_nonorthogonal'")
    add_avwis_rejection_test(
        avwis_p5_mapped_cartesian_rejected p5_mapped_cartesian_rejected.in
        "mapped coordinates currently require identity or analytic_orthogonal mapping")
    add_avwis_rejection_test(
        avwis_p5_explicit_cfl_rejected p5_explicit_cfl_rejected.in
        "explicit step rejected: advective CFL and diffusive number must each be <= 1")
    # Checkpoint/restart and host-side sampling are deliberately CPU-only.
    # Do not register unsupported cases in a CUDA CTest suite: an explicit
    # runtime rejection remains the production contract, not a GPU test pass.
    if (NOT AMReX_CUDA)
        add_test(NAME avwis_p8_restart_consistency
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p8_restart.in
                    -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p8_restart_consistency
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p8_restart_consistency.cmake)
        add_test(NAME avwis_p6_mapped_restart_consistency
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p6_mapped_restart.in
                    -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p6_mapped_restart_consistency
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p8_restart_consistency.cmake)
        add_test(NAME avwis_p6_mapped_provenance_rejected
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p6_mapped_checkpoint_write.in
                    -Dmismatch_input=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p6_mapped_checkpoint_mismatch.in
                    -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p6_mapped_provenance_rejection
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c6_mapped_provenance_rejection.cmake)
        add_test(NAME avwis_p6_mapped_cross_layout_restart
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dproducer_input=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p6_mapped_checkpoint_write.in
                    -Dconsumer_input=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p6_mapped_checkpoint_cross_layout.in
                    -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p6_mapped_cross_layout
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c6_mapped_cross_layout.cmake)
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
    endif()
    add_test(NAME avwis_cartesian_benchmark
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/inputs/p0_cartesian_benchmark.in
                -Dreport=${CMAKE_CURRENT_BINARY_DIR}/cartesian_benchmark.json
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cartesian_benchmark.cmake)
    add_test(NAME avwis_p10_001_taylor_green_mms
             COMMAND ${CMAKE_COMMAND}
                -Dexe=$<TARGET_FILE:avwis>
                -Dinputs=${CMAKE_CURRENT_SOURCE_DIR}/inputs
                -Dwork=${CMAKE_CURRENT_BINARY_DIR}/p10_taylor_green_mms
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/p10_taylor_green_convergence.cmake)
    if (NOT AMReX_CUDA)
        add_test(NAME avwis_c8_g1_mapped_channel_reference
                 COMMAND ${CMAKE_COMMAND}
                    -Dexe=$<TARGET_FILE:avwis>
                    -Dinputs=${CMAKE_CURRENT_SOURCE_DIR}/inputs
                    -Dwork=${CMAKE_CURRENT_BINARY_DIR}/c8_g1_mapped_channel
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c8_mapped_channel_convergence.cmake)
    endif()
    if (NOT AMReX_CUDA)
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
    endif()
    add_avwis_rejection_test(
        avwis_unknown_mode_rejected unknown_mode.in
        "unknown avwis.mode 'definitely_not_a_mode'")
endif()
