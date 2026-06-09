cmake_minimum_required(VERSION 3.24)

# 在 project() 探测 CUDA 编译器前指定 RTX 4060 的 Ada 架构，
# 避免无显示设备的构建节点无法自动探测默认架构。
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
project(earth_moon_sph LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(earth_moon_sph main.cpp sph_kernels.cu)
target_include_directories(earth_moon_sph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(earth_moon_sph PRIVATE CUDA::cudart)
set_target_properties(earth_moon_sph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(earth_moon_sph PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)
