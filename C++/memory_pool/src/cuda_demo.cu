#include "memory_pool/gpu_allocator.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__global__ void square_kernel(int* values, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        values[index] = values[index] * values[index];
    }
}

void demo_device_buffer() {
    constexpr int size = 8;
    std::vector<int> host_in{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> host_out(size);

    memory_pool::CudaDeviceBuffer<int> device_values(size);

    memory_pool::check_cuda(cudaMemcpy(device_values.data(),
                                       host_in.data(),
                                       device_values.bytes(),
                                       cudaMemcpyHostToDevice));

    square_kernel<<<1, 32>>>(device_values.data(), size);
    memory_pool::check_cuda(cudaGetLastError());
    memory_pool::check_cuda(cudaDeviceSynchronize());

    memory_pool::check_cuda(cudaMemcpy(host_out.data(),
                                       device_values.data(),
                                       device_values.bytes(),
                                       cudaMemcpyDeviceToHost));

    std::cout << "=== CudaDeviceBuffer<T> Demo ===\n";
    std::cout << "squared values:";
    for (int value : host_out) {
        std::cout << ' ' << value;
    }
    std::cout << "\n\n";
}

void demo_managed_vector() {
    memory_pool::CudaManagedVector<int> values;
    values.reserve(8);
    for (int i = 0; i < 8; ++i) {
        values.push_back(i + 1);
    }

    square_kernel<<<1, 32>>>(values.data(), static_cast<int>(values.size()));
    memory_pool::check_cuda(cudaGetLastError());
    memory_pool::check_cuda(cudaDeviceSynchronize());

    std::cout << "=== CudaManagedVector<T> Demo ===\n";
    std::cout << "managed vector values:";
    for (int value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

int main() {
    demo_device_buffer();
    demo_managed_vector();
    return 0;
}
