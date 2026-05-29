#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

namespace memory_pool {

class LinearMemoryPool {
public:
    explicit LinearMemoryPool(std::size_t byte_capacity)
        : capacity_(byte_capacity),
          buffer_(::operator new(byte_capacity, std::align_val_t{alignof(std::max_align_t)})) {}

    ~LinearMemoryPool() {
        ::operator delete(buffer_, std::align_val_t{alignof(std::max_align_t)});
    }

    LinearMemoryPool(const LinearMemoryPool&) = delete;
    LinearMemoryPool& operator=(const LinearMemoryPool&) = delete;

    void* allocate(std::size_t bytes, std::size_t alignment) {
        auto* base = static_cast<std::byte*>(buffer_);
        std::uintptr_t current = reinterpret_cast<std::uintptr_t>(base + offset_);
        const std::size_t adjustment = alignment_adjustment(current, alignment);

        if (offset_ + adjustment + bytes > capacity_) {
            throw std::bad_alloc{};
        }

        offset_ += adjustment;
        void* result = base + offset_;
        offset_ += bytes;
        return result;
    }

    void deallocate(void*, std::size_t) {
        // Linear pools release all allocations together when the pool is reset or destroyed.
    }

    void reset() {
        offset_ = 0;
    }

    std::size_t capacity() const {
        return capacity_;
    }

    std::size_t used_bytes() const {
        return offset_;
    }

private:
    static std::size_t alignment_adjustment(std::uintptr_t address, std::size_t alignment) {
        const std::size_t remainder = address % alignment;
        return remainder == 0 ? 0 : alignment - remainder;
    }

    std::size_t capacity_;
    std::size_t offset_{0};
    void* buffer_{nullptr};
};

} // namespace memory_pool
