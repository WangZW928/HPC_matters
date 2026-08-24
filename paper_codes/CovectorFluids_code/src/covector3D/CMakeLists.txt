find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
#find_package(Half REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)

# OpenVDB 10 (shipped by Ubuntu 24.04) requires C++14 or newer.
set(CMAKE_CXX_STANDARD 14)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61; -gencode arch=compute_75,code=sm_75;
        -gencode arch=compute_75,code=compute_75; -lineinfo)
set(CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_FILES
        main.cpp
        GPU_kernel.cu
        Mapping.cpp
        CovectorSolver.cpp
        )

cuda_add_executable(Covector3D ${SOURCE_FILES})
target_link_libraries(Covector3D  ${TBB_LIBRARIES})
target_link_libraries(Covector3D  ${OPENVDB_LIBRARIES})
#target_link_libraries(Covector3D  Half)
target_link_libraries(Covector3D ${Boost_LIBRARIES})
#target_link_libraries(Covector3D rt ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib)
