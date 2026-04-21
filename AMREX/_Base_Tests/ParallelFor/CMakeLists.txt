cmake_minimum_required(VERSION 3.20)

project(BaseTestsParallelFor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(parallelfor2d_cpu
    main.cpp
)

target_include_directories(parallelfor2d_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (OpenMP_CXX_FOUND)
    target_link_libraries(parallelfor2d_cpu PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(parallelfor2d_cpu PRIVATE BASE_TESTS_USE_OPENMP=1)
endif()

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/main_cuda.cu
        COPYONLY
    )

    add_executable(parallelfor2d_gpu
        ${CMAKE_CURRENT_BINARY_DIR}/main_cuda.cu
    )

    target_include_directories(parallelfor2d_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(parallelfor2d_gpu PRIVATE cxx_std_17)
    target_compile_options(parallelfor2d_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_compile_definitions(parallelfor2d_gpu PRIVATE BASE_TESTS_COMPILE_CUDA=1)
    set_target_properties(parallelfor2d_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
