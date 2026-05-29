#include "memory_pool/gpu_allocator.hpp"
#include "memory_pool/object_pool.hpp"
#include "memory_pool/pool_allocator.hpp"

#include <iostream>
#include <memory>

struct Particle {
    float x{};
    float y{};
    float z{};
    int id{};
};

void demo_object_pool() {
    memory_pool::ObjectPool<Particle> pool(4);

    std::cout << "=== RAII ObjectPool<T> Demo ===\n";
    std::cout << "block size = " << pool.block_size() << " bytes\n";
    std::cout << "alignment  = " << pool.block_alignment() << " bytes\n";
    std::cout << "capacity   = " << pool.capacity() << " blocks\n";
    std::cout << "used       = " << pool.used_count() << " blocks\n\n";

    auto p1 = pool.make_handle(Particle{1.0F, 2.0F, 3.0F, 101});
    auto p2 = pool.make_handle(Particle{4.0F, 5.0F, 6.0F, 102});

    std::cout << "after creating two RAII handles:\n";
    std::cout << "p1 address = " << p1.get() << ", id = " << p1->id << '\n';
    std::cout << "p2 address = " << p2.get() << ", id = " << p2->id << '\n';
    std::cout << "used       = " << pool.used_count() << " blocks\n\n";

    void* old_p1_address = p1.get();
    p1.reset();

    std::cout << "after p1.reset():\n";
    std::cout << "used       = " << pool.used_count() << " blocks\n\n";

    auto p3 = pool.make_handle(Particle{7.0F, 8.0F, 9.0F, 103});

    std::cout << "after creating p3:\n";
    std::cout << "p1 old address = " << old_p1_address << '\n';
    std::cout << "p3 address     = " << p3.get() << ", id = " << p3->id << '\n';
    std::cout << "notice: p3 often reuses p1's old address.\n\n";

    std::cout << "leaving this function will automatically destroy p2 and p3.\n";
}

void demo_vector_allocator() {
    auto vector_memory = std::make_shared<memory_pool::LinearMemoryPool>(1024);
    memory_pool::PoolAllocator<int> allocator(vector_memory);
    memory_pool::PoolVector<int> values(allocator);

    values.reserve(16);
    for (int i = 0; i < 10; ++i) {
        values.push_back(i * i);
    }

    std::cout << "\n=== PoolVector<T> Demo ===\n";
    std::cout << "vector values:";
    for (int value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';

    std::cout << "vector data address = " << static_cast<const void*>(values.data()) << '\n';
    std::cout << "pool capacity       = " << vector_memory->capacity() << " bytes\n";
    std::cout << "pool used           = " << vector_memory->used_bytes() << " bytes\n";
}

void explain_gpu_allocator() {
    std::cout << "\n=== Optional GPU Allocator Notes ===\n";
    std::cout << "include/memory_pool/gpu_allocator.hpp provides optional CUDA helpers.\n";
    std::cout << "They are compiled only when MEMORY_POOL_ENABLE_CUDA is defined.\n";
    std::cout << "Device memory is not ordinary host memory, so do not use cudaMalloc memory\n";
    std::cout << "as a normal std::vector backing store. Use CudaManagedVector<T> only with\n";
    std::cout << "CUDA unified memory, or use CudaDeviceBuffer<T> for kernel-facing buffers.\n";
}

int main() {
    demo_object_pool();
    demo_vector_allocator();
    explain_gpu_allocator();

    return 0;
}
