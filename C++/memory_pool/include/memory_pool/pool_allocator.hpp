#pragma once

#include "memory_pool/linear_memory_pool.hpp"

#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace memory_pool {

template <typename T>
class PoolAllocator {
public:
    using value_type = T;

    explicit PoolAllocator(std::shared_ptr<LinearMemoryPool> pool) noexcept
        : pool_(std::move(pool)) {}

    template <typename U>
    PoolAllocator(const PoolAllocator<U>& other) noexcept
        : pool_(other.pool()) {}

    T* allocate(std::size_t n) {
        if (pool_ == nullptr) {
            throw std::bad_alloc{};
        }

        void* raw = pool_->allocate(n * sizeof(T), alignof(T));
        return static_cast<T*>(raw);
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        pool_->deallocate(ptr, n * sizeof(T));
    }

    std::shared_ptr<LinearMemoryPool> pool() const noexcept {
        return pool_;
    }

    template <typename U>
    bool operator==(const PoolAllocator<U>& other) const noexcept {
        return pool_ == other.pool();
    }

    template <typename U>
    bool operator!=(const PoolAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    std::shared_ptr<LinearMemoryPool> pool_;
};

template <typename T>
using PoolVector = std::vector<T, PoolAllocator<T>>;

} // namespace memory_pool
