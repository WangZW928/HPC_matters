#pragma once

#include "memory_pool/fixed_block_memory_pool.hpp"

#include <memory>
#include <utility>

namespace memory_pool {

template <typename T>
class ObjectPool {
public:
    struct Deleter {
        ObjectPool* pool{};

        void operator()(T* ptr) const {
            if (pool != nullptr) {
                pool->destroy(ptr);
            }
        }
    };

    using Handle = std::unique_ptr<T, Deleter>;

    explicit ObjectPool(std::size_t capacity)
        : memory_pool_(sizeof(T), capacity, alignof(T)) {}

    template <typename... Args>
    T* create(Args&&... args) {
        void* raw = memory_pool_.allocate();
        try {
            return new (raw) T(std::forward<Args>(args)...);
        } catch (...) {
            memory_pool_.deallocate(raw);
            throw;
        }
    }

    void destroy(T* ptr) {
        if (ptr == nullptr) {
            return;
        }

        ptr->~T();
        memory_pool_.deallocate(ptr);
    }

    template <typename... Args>
    Handle make_handle(Args&&... args) {
        return Handle(create(std::forward<Args>(args)...), Deleter{this});
    }

    std::size_t capacity() const {
        return memory_pool_.capacity();
    }

    std::size_t used_count() const {
        return memory_pool_.used_count();
    }

    std::size_t block_size() const {
        return memory_pool_.block_size();
    }

    std::size_t block_alignment() const {
        return memory_pool_.block_alignment();
    }

private:
    FixedBlockMemoryPool memory_pool_;
};

} // namespace memory_pool
