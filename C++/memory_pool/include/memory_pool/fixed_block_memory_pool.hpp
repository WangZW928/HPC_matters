#pragma once

#include <cstddef>
#include <new>

namespace memory_pool {

class FixedBlockMemoryPool {
public:
    FixedBlockMemoryPool(std::size_t block_size,
                         std::size_t block_count,
                         std::size_t block_alignment = alignof(std::max_align_t))
        : block_alignment_(block_alignment < alignof(FreeBlock) ? alignof(FreeBlock) : block_alignment),
          block_size_(round_up(block_size < sizeof(FreeBlock) ? sizeof(FreeBlock) : block_size,
                               block_alignment_)),
          block_count_(block_count),
          buffer_(::operator new(block_size_ * block_count_,
                                 std::align_val_t{block_alignment_})) {
        build_free_list();
    }

    ~FixedBlockMemoryPool() {
        ::operator delete(buffer_, std::align_val_t{block_alignment_});
    }

    FixedBlockMemoryPool(const FixedBlockMemoryPool&) = delete;
    FixedBlockMemoryPool& operator=(const FixedBlockMemoryPool&) = delete;

    void* allocate() {
        if (free_list_ == nullptr) {
            throw std::bad_alloc{};
        }

        FreeBlock* block = free_list_;
        free_list_ = free_list_->next;
        ++used_count_;
        return block;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) {
            return;
        }

        auto* block = static_cast<FreeBlock*>(ptr);
        block->next = free_list_;
        free_list_ = block;
        --used_count_;
    }

    std::size_t block_size() const {
        return block_size_;
    }

    std::size_t block_alignment() const {
        return block_alignment_;
    }

    std::size_t capacity() const {
        return block_count_;
    }

    std::size_t used_count() const {
        return used_count_;
    }

private:
    struct FreeBlock {
        FreeBlock* next;
    };

    static std::size_t round_up(std::size_t value, std::size_t alignment) {
        const std::size_t remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    }

    void build_free_list() {
        free_list_ = nullptr;
        auto* bytes = static_cast<std::byte*>(buffer_);
        for (std::size_t i = 0; i < block_count_; ++i) {
            void* address = bytes + i * block_size_;
            auto* block = static_cast<FreeBlock*>(address);
            block->next = free_list_;
            free_list_ = block;
        }
    }

    std::size_t block_alignment_;
    std::size_t block_size_;
    std::size_t block_count_;
    std::size_t used_count_{0};
    void* buffer_{nullptr};
    FreeBlock* free_list_{nullptr};
};

} // namespace memory_pool
