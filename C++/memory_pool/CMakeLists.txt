cmake_minimum_required(VERSION 3.16)
project(memory_pool_notes LANGUAGES CXX)

option(MEMORY_POOL_BUILD_CUDA "Build CUDA memory pool demo when CUDA is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(memory_pool_demo src/main.cpp)

target_include_directories(memory_pool_demo PRIVATE include)

if(MEMORY_POOL_BUILD_CUDA)
    include(CheckLanguage)

    if(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    endif()

    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        add_executable(memory_pool_cuda_demo src/cuda_demo.cu)
        target_include_directories(memory_pool_cuda_demo PRIVATE include)
        target_compile_features(memory_pool_cuda_demo PRIVATE cuda_std_17)
        target_compile_definitions(memory_pool_cuda_demo PRIVATE MEMORY_POOL_ENABLE_CUDA)

        set_target_properties(memory_pool_cuda_demo PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 89)
    else()
        message(STATUS "CUDA compiler not found; skipping memory_pool_cuda_demo.")
    endif()
endif()
