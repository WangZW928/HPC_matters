#ifndef BASE_TESTS_2DHEAT_PARALLELFOR2D_H_
#define BASE_TESTS_2DHEAT_PARALLELFOR2D_H_

#include "Backend.H"

template <typename F>
BASE_GPU_HOST_DEVICE
inline void call_f_2d (F const& f, int i, int j) noexcept
{
    f(i, j);
}

#if defined(__CUDACC__)

template <typename F>
__global__ void parallelFor2DKernel (int nx, int ny, F f)
{
    int icell = blockIdx.x * blockDim.x + threadIdx.x;
    int ncell = nx * ny;

    if (icell < ncell) {
        int j = icell / nx;
        int i = icell % nx;
        call_f_2d(f, i, j);
    }
}

#endif

template <typename F>
void parallelFor2D (int nx, int ny, F f) noexcept
{
#if defined(__CUDACC__)
    int ncell = nx * ny;
    int threads_per_block = 256;
    int blocks = (ncell + threads_per_block - 1) / threads_per_block;
    parallelFor2DKernel<<<blocks, threads_per_block>>>(nx, ny, f);
#else
#if defined(BASE_TESTS_USE_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            call_f_2d(f, i, j);
        }
    }
#endif
}

#endif
