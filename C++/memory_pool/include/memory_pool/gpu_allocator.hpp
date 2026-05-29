#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MEMORY_POOL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace memory_pool {

#ifdef MEMORY_POOL_ENABLE_CUDA

class CudaError : public std::runtime_error {
public:
    explicit CudaError(cudaError_t error)
        : std::runtime_error(cudaGetErrorString(error)) {}
};

inline void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw CudaError(error);
    }
}

template <typename T>
class CudaDeviceBuffer {
public:
    explicit CudaDeviceBuffer(std::size_t size)
        : size_(size) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }

    ~CudaDeviceBuffer() {
        cudaFree(data_);
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    std::size_t size() const {
        return size_;
    }

    std::size_t bytes() const {
        return size_ * sizeof(T);
    }

private:
    std::size_t size_{0};
    T* data_{nullptr};
};

template <typename T>
class CudaManagedAllocator {
public:
    using value_type = T;

    CudaManagedAllocator() noexcept = default;

    template <typename U>
    CudaManagedAllocator(const CudaManagedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFree(ptr);
    }

    template <typename U>
    bool operator==(const CudaManagedAllocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const CudaManagedAllocator<U>&) const noexcept {
        return false;
    }
};

template <typename T>
using CudaManagedVector = std::vector<T, CudaManagedAllocator<T>>;

#else

template <typename T>
class CudaDeviceBuffer;

template <typename T>
class CudaManagedAllocator;

template <typename T>
using CudaManagedVector = std::vector<T, CudaManagedAllocator<T>>;

#endif

} // namespace memory_pool
