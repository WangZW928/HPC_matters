cmake_minimum_required(VERSION 3.20)
project(vwis_amrex_port LANGUAGES CXX)

# A different AMReX can change CMake targets, GPU defaults, and solver APIs.
# An override is allowed only for an explicitly labelled exploratory build.
set(VWIS_AMREX_EXPECTED_VERSION "25.02" CACHE STRING
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
        message(FATAL_ERROR "Found AMReX ${AMReX_VERSION}, but P0/P1 locks ${VWIS_AMREX_EXPECTED_VERSION}. Use the locked package or set VWIS_AMREX_ALLOW_VERSION_MISMATCH=ON for an unvalidated exploratory build.")
    endif()
elseif (NOT DEFINED AMReX_VERSION)
    message(WARNING "AMReX did not export AMReX_VERSION; record its revision before treating a build as evidence.")
endif()

add_executable(vwis_amrex_skeleton src/main.cpp src/VwisAmrExSolver.cpp)
target_link_libraries(vwis_amrex_skeleton PRIVATE AMReX::amrex)
target_compile_features(vwis_amrex_skeleton PRIVATE cxx_std_17)

include(CTest)
if (BUILD_TESTING)
    add_test(NAME vwis_amrex_single_box COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_smoke.in)
    add_test(NAME vwis_amrex_multi_box COMMAND vwis_amrex_skeleton ${CMAKE_CURRENT_SOURCE_DIR}/inputs/p1_multibox.in)
endif()
