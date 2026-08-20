cmake_minimum_required(VERSION 3.20)
project(vwis_amrex_skeleton LANGUAGES CXX)

find_package(AMReX CONFIG REQUIRED)

add_executable(vwis_amrex_skeleton
    src/main.cpp
    src/VwisAmrExSolver.cpp)

target_link_libraries(vwis_amrex_skeleton PRIVATE AMReX::amrex)
target_compile_features(vwis_amrex_skeleton PRIVATE cxx_std_17)
