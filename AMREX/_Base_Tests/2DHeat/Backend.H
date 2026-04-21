#ifndef BASE_TESTS_2DHEAT_BACKEND_H_
#define BASE_TESTS_2DHEAT_BACKEND_H_

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define BASE_GPU_DEVICE __device__
#define BASE_GPU_HOST_DEVICE __host__ __device__
#else
#define BASE_GPU_DEVICE
#define BASE_GPU_HOST_DEVICE
#endif

inline constexpr char const* backendLabel () noexcept
{
#if defined(__CUDACC__)
    return "GPU";
#else
    return "CPU";
#endif
}

inline void backendSynchronize ()
{
#if defined(__CUDACC__)
    cudaDeviceSynchronize();
#endif
}

#endif
