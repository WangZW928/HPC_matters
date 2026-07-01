#include <cstddef>
#include <functional>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 教学版哈希表：
// - 使用模板支持任意可哈希的 Key 和任意 Value
// - 使用链地址法解决冲突：每个桶是一个 std::list
// - 为了便于学习，代码更重视清晰性，而不是极致性能
template <typename Key, typename Value>
class HashTable {
private:
    using Entry = std::pair<Key, Value>;

    std::vector<std::list<Entry>> buckets_;
    std::size_t size_ = 0;
    double max_load_factor_ = 0.75;
    std::hash<Key> hasher_;

    std::size_t bucket_index(const Key& key) const {
        return hasher_(key) % buckets_.size();
    }

public:
    explicit HashTable(std::size_t bucket_count = 8) {
        if (bucket_count == 0) {
            throw std::invalid_argument("bucket_count must be greater than 0");
        }

        buckets_.resize(bucket_count);
    }

    // 插入键值对。
    // 如果 key 已经存在，则更新对应 value。
    void insert(const Key& key, const Value& value) {
        if ((static_cast<double>(size_ + 1) / buckets_.size()) > max_load_factor_) {
            rehash(buckets_.size() * 2);
        }

        std::size_t index = bucket_index(key);

        for (auto& entry : buckets_[index]) {
            if (entry.first == key) {
                entry.second = value;
                return;
            }
        }

        buckets_[index].push_back({key, value});
        ++size_;
    }

    // 查找 key。
    // 找到时返回 value 的地址，没找到时返回 nullptr。
    Value* find(const Key& key) {
        std::size_t index = bucket_index(key);

        for (auto& entry : buckets_[index]) {
            if (entry.first == key) {
                return &entry.second;
            }
        }

        return nullptr;
    }

    // const 版本的查找函数，允许在只读哈希表上调用 find。
    const Value* find(const Key& key) const {
        std::size_t index = bucket_index(key);

        for (const auto& entry : buckets_[index]) {
            if (entry.first == key) {
                return &entry.second;
            }
        }

        return nullptr;
    }

    // 删除 key。
    // 删除成功返回 true；key 不存在返回 false。
    bool erase(const Key& key) {
        std::size_t index = bucket_index(key);
        auto& bucket = buckets_[index];

        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->first == key) {
                bucket.erase(it);
                --size_;
                return true;
            }
        }

        return false;
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    std::size_t bucket_count() const {
        return buckets_.size();
    }

    double load_factor() const {
        return static_cast<double>(size_) / buckets_.size();
    }

    // 重新分配桶，并把所有元素根据新的桶数量重新放置。
    void rehash(std::size_t new_bucket_count) {
        if (new_bucket_count == 0) {
            throw std::invalid_argument("new_bucket_count must be greater than 0");
        }

        std::vector<std::list<Entry>> new_buckets(new_bucket_count);

        for (const auto& bucket : buckets_) {
            for (const auto& entry : bucket) {
                std::size_t new_index = hasher_(entry.first) % new_bucket_count;
                new_buckets[new_index].push_back(entry);
            }
        }

        buckets_ = std::move(new_buckets);
    }

    // 打印内部桶分布，便于观察哈希冲突和 rehash 效果。
    void print_buckets() const {
        for (std::size_t i = 0; i < buckets_.size(); ++i) {
            std::cout << "bucket[" << i << "]: ";

            if (buckets_[i].empty()) {
                std::cout << "(empty)";
            } else {
                for (const auto& entry : buckets_[i]) {
                    std::cout << "(" << entry.first << ", " << entry.second << ") ";
                }
            }

            std::cout << '\n';
        }
    }
};

int main() {
    std::cout << "=== 字符串 key 的哈希表示例 ===\n";

    HashTable<std::string, int> scores;

    scores.insert("Alice", 95);
    scores.insert("Bob", 88);
    scores.insert("Charlie", 91);
    scores.insert("Diana", 86);

    std::cout << "插入 4 个学生后：\n";
    scores.print_buckets();
    std::cout << "size = " << scores.size() << '\n';
    std::cout << "bucket_count = " << scores.bucket_count() << '\n';
    std::cout << "load_factor = " << scores.load_factor() << "\n\n";

    std::cout << "查找 Alice：";
    if (int* score = scores.find("Alice")) {
        std::cout << *score << '\n';
    } else {
        std::cout << "not found\n";
    }

    std::cout << "更新 Alice 的分数为 99\n";
    scores.insert("Alice", 99);
    if (int* score = scores.find("Alice")) {
        std::cout << "Alice = " << *score << "\n\n";
    }

    std::cout << "删除 Bob：";
    if (scores.erase("Bob")) {
        std::cout << "success\n";
    } else {
        std::cout << "not found\n";
    }

    std::cout << "删除后：\n";
    scores.print_buckets();
    std::cout << "size = " << scores.size() << "\n\n";

    std::cout << "继续插入更多元素，观察自动 rehash：\n";
    scores.insert("Eve", 77);
    scores.insert("Frank", 83);
    scores.insert("Grace", 92);
    scores.insert("Heidi", 89);
    scores.insert("Ivan", 81);
    scores.insert("Judy", 94);

    scores.print_buckets();
    std::cout << "size = " << scores.size() << '\n';
    std::cout << "bucket_count = " << scores.bucket_count() << '\n';
    std::cout << "load_factor = " << scores.load_factor() << "\n\n";

    std::cout << "=== 整数 key 的哈希表示例 ===\n";

    HashTable<int, std::string> id_to_name(4);
    id_to_name.insert(1001, "Zhang San");
    id_to_name.insert(1002, "Li Si");
    id_to_name.insert(1003, "Wang Wu");

    id_to_name.print_buckets();

    const HashTable<int, std::string>& readonly = id_to_name;
    if (const std::string* name = readonly.find(1002)) {
        std::cout << "id 1002 -> " << *name << '\n';
    }

    std::cout << "手动 rehash 到 16 个桶\n";
    id_to_name.rehash(16);
    id_to_name.print_buckets();

    return 0;
}
