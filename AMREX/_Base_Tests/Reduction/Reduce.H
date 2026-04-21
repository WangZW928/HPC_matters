#ifndef BASE_TESTS_REDUCTION_REDUCE_H_
#define BASE_TESTS_REDUCTION_REDUCE_H_

#include "Backend.H"
#include "Buffer.H"

#include <algorithm>
#include <tuple>
#include <vector>

struct Vec2
{
    double x = 0.0;
    double y = 0.0;
};

// The two tuple entries carry the same local value through two different
// reduction channels:
//   entry 0 -> sum(|v|)
//   entry 1 -> max(|v|)
using ReduceTuple = std::tuple<double, double>;

BASE_GPU_HOST_DEVICE
inline ReduceTuple reduceIdentity () noexcept
{
    return {0.0, 0.0};
}

BASE_GPU_HOST_DEVICE
inline void reduceCombine (ReduceTuple& dst, ReduceTuple const& src) noexcept
{
    // Combine the first channel with sum and the second channel with max.
    std::get<0>(dst) += std::get<0>(src);
    std::get<1>(dst) = (std::max)(std::get<1>(dst), std::get<1>(src));
}

class ReduceData
{
public:
    explicit ReduceData (int nblocks)
        : m_partial(nblocks, defaultMemorySpace()),
          m_nblocks(nblocks)
    {}

    ReduceTuple* devicePtr () noexcept { return m_partial.data(); }
    int nBlocks () const noexcept { return m_nblocks; }

    ReduceTuple value () const
    {
        auto host = m_partial.copyToHostVector();
        ReduceTuple result = reduceIdentity();
        for (auto const& v : host) {
            reduceCombine(result, v);
        }
        return result;
    }

private:
    Buffer<ReduceTuple> m_partial;
    int m_nblocks = 0;
};

#if defined(__CUDACC__)

template <typename F>
__global__ void reduceKernel (int n, ReduceTuple* partial, F f)
{
    __shared__ double sh_sum[256];
    __shared__ double sh_max[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    ReduceTuple local = reduceIdentity();
    if (i < n) {
        local = f(i);
    }

    sh_sum[tid] = std::get<0>(local);
    sh_max[tid] = std::get<1>(local);
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_max[tid] = (sh_max[tid] > sh_max[tid + stride]) ? sh_max[tid] : sh_max[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = {sh_sum[0], sh_max[0]};
    }
}

#endif

class Reducer
{
public:
    using Result_t = ReduceTuple;

    template <typename F>
    Result_t eval (int n, F f) const
    {
#if defined(__CUDACC__)
        int threads_per_block = 256;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        ReduceData reduce_data(blocks);
        reduceKernel<<<blocks, threads_per_block>>>(n, reduce_data.devicePtr(), f);
        backendSynchronize();
        return reduce_data.value();
#else
        ReduceTuple result = reduceIdentity();
#if defined(BASE_TESTS_USE_OPENMP)
#pragma omp parallel
#endif
        {
            ReduceTuple local = reduceIdentity();
#if defined(BASE_TESTS_USE_OPENMP)
#pragma omp for nowait
#endif
            for (int i = 0; i < n; ++i) {
                reduceCombine(local, f(i));
            }
#if defined(BASE_TESTS_USE_OPENMP)
#pragma omp critical
#endif
            reduceCombine(result, local);
        }
        return result;
#endif
    }
};

#endif
