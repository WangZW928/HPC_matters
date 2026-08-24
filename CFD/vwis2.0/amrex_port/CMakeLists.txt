cmake_minimum_required(VERSION 3.20)
# AMReX's MPI package exports both C and C++ MPI dependencies, so C must be
# enabled when consuming an AMReX install configured with AMReX_MPI=ON.
project(vwis_amrex_port LANGUAGES C CXX)

# A different AMReX can change CMake targets, GPU defaults, and solver APIs.
# An override is allowed only for an explicitly labelled exploratory build.
set(VWIS_AMREX_EXPECTED_VERSION "26.04" CACHE STRING
    "AMReX version recorded by amrex_version.lock")
option(VWIS_AMREX_ALLOW_VERSION_MISMATCH
       "Permit an AMReX version other than VWIS_AMREX_EXPECTED_VERSION for exploratory builds" OFF)

find_package(AMReX CONFIG QUIET)
if (NOT AMReX_FOUND)
    message(FATAL_ERROR
        "AMReXConfig.cmake was not found. Set -DAMReX_DIR=/path/to/lib/cmake/AMReX or "
        "add its prefix to CMAKE_PREFIX_PATH. No fallback solver is built.")
endif()
if (DEFINED AMReX_VERSION AND NOT AMReX_VERSION VERSION_EQUAL VWIS_AMREX_EXPECTED_VERSION)
    if (VWIS_AMREX_ALLOW_VERSION_MISMATCH)
        message(WARNING "AMReX ${AMReX_VERSION} differs from locked ${VWIS_AMREX_EXPECTED_VERSION}; exploratory build only.")
    else()
        message(FATAL_ERROR "Found AMReX ${AMReX_VERSION}, but P0-P2 locks ${VWIS_AMREX_EXPECTED_VERSION}. Use the locked package or set VWIS_AMREX_ALLOW_VERSION_MISMATCH=ON for an unvalidated exploratory build.")
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

set(VWIS_AMREX_SOURCES
    src/main.cpp
    src/CartesianBoundaryConfig.cpp
    src/VwisAmrExSolver.cpp
    src/VwisAmrExBoundary.cpp
    src/VwisAmrExContracts.cpp)
add_executable(vwis_amrex_skeleton ${VWIS_AMREX_SOURCES})
target_link_libraries(vwis_amrex_skeleton PRIVATE AMReX::amrex)
target_compile_features(vwis_amrex_skeleton PRIVATE cxx_std_17)
if (AMReX_CUDA)
    set_source_files_properties(${VWIS_AMREX_SOURCES} PROPERTIES LANGUAGE CUDA)
    # AMReX 26.04 CUDA archives are built with relocatable device code; the
    # application must perform the final device link or registration symbols
    # from the AMReX archive remain unresolved.
    set_target_properties(vwis_amrex_skeleton PROPERTIES
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

include(CTest)
if (BUILD_TESTING)
    add_test(NAME vwis_amrex_single_box COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_smoke.in)
    add_test(NAME vwis_amrex_multi_box COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_multibox.in)
    add_test(NAME vwis_amrex_runtime_contract
             COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p2_contract.in)
    add_test(NAME vwis_amrex_boundary_face_contract
             COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p2_boundary_face.in)
    add_test(NAME vwis_amrex_p3_cartesian_boundary
             COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_cartesian_boundary.in)
    add_test(NAME vwis_amrex_p3_legacy_supported
             COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_legacy_supported.in)
    add_test(NAME vwis_amrex_p3_legacy_rejected
             COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p3_legacy_rejected.in)
    # The source error text is also pinned by static_contract_check.sh.  CTest's
    # WILL_FAIL inverts the nonzero process status; combining it with
    # PASS_REGULAR_EXPRESSION has version-dependent inversion semantics.
    set_tests_properties(vwis_amrex_p3_legacy_rejected PROPERTIES WILL_FAIL TRUE)
endif()
