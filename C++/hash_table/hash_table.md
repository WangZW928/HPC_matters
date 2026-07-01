# Hash Table（哈希表）C++ 教程

哈希表是一种通过“键 key”快速定位“值 value”的数据结构。它的核心思想是：用哈希函数把 key 映射成数组下标，然后把数据存到对应位置。

例如，要保存学生学号到姓名的映射：

```cpp
2024001 -> "Alice"
2024002 -> "Bob"
```

如果能把 `2024001` 直接转换成数组下标，就不需要从头到尾查找，查询速度通常可以接近 `O(1)`。

## 1. 基本原理

### 1.1 什么是哈希表

哈希表通常由两部分组成：

- 桶数组（bucket array）：一个数组，每个位置叫一个桶。
- 哈希函数（hash function）：把 key 转换成桶下标。

基本流程：

```cpp
index = hash(key) % bucket_count;
```

其中：

- `hash(key)` 得到一个整数哈希值。
- `bucket_count` 是桶的数量。
- `% bucket_count` 保证下标落在数组范围内。

### 1.2 哈希函数

哈希函数的目标是把 key 尽量均匀地分布到不同桶中。

最简单的整数哈希方式：

```cpp
std::size_t hash_int(int key, std::size_t bucket_count) {
    return key % bucket_count;
}
```

这个方法容易理解，但如果 key 本身有规律，可能导致大量冲突。例如桶数量是 10，而 key 经常是 `10, 20, 30, 40`，它们都会落到桶 0。

字符串可以使用 FNV-1a 这类简单且分布较好的哈希算法：

```cpp
std::size_t fnv1a_hash(const std::string& s) {
    const std::size_t offset_basis = 14695981039346656037ull;
    const std::size_t fnv_prime = 1099511628211ull;

    std::size_t hash = offset_basis;
    for (unsigned char c : s) {
        hash ^= c;
        hash *= fnv_prime;
    }
    return hash;
}
```

在实际 C++ 程序中，更常用的是标准库提供的 `std::hash<Key>`。它支持很多常见类型，也可以为自定义类型特化。

### 1.3 哈希冲突

不同 key 可能得到相同桶下标，这叫哈希冲突。

例如有 5 个桶：

```text
hash("cat") % 5 == 2
hash("dog") % 5 == 2
```

`"cat"` 和 `"dog"` 都想放进桶 2，这时必须有冲突解决策略。

### 1.4 冲突解决：链地址法

链地址法（chaining）让每个桶保存一个链表。同一个桶里的多个元素挂在同一条链表上。

```text
bucket[0]: empty
bucket[1]: (Tom, 90)
bucket[2]: (cat, 1) -> (dog, 2)
bucket[3]: empty
bucket[4]: (Bob, 85)
```

查找时：

1. 先算出桶下标。
2. 只在该桶的链表中查找。

优点：

- 实现直观。
- 删除操作简单。
- 负载因子超过 1 也能工作。

缺点：

- 每个桶中的链表太长时，性能会下降。
- 链表节点可能带来额外内存开销。

### 1.5 冲突解决：开放寻址法

开放寻址法（open addressing）不使用链表，所有元素都存放在桶数组本身。如果目标桶已经被占用，就继续寻找下一个可用桶。

线性探测（linear probing）是最常见的开放寻址方法之一：

```cpp
index = hash(key) % bucket_count;

while (bucket[index] is occupied) {
    index = (index + 1) % bucket_count;
}
```

示意：

```text
原始下标为 2，但 bucket[2] 已被占用：

bucket[0]: empty
bucket[1]: empty
bucket[2]: A
bucket[3]: B
bucket[4]: empty  <- 插入这里
```

优点：

- 数据集中存储，缓存友好。
- 不需要额外链表节点。

缺点：

- 删除操作更复杂，通常需要 tombstone 标记。
- 负载因子过高时性能下降明显。
- 必须保证表中有空桶。

### 1.6 时间复杂度

在哈希函数分布较好、负载因子合理的情况下：

| 操作 | 平均复杂度 | 最坏复杂度 |
| --- | --- | --- |
| 插入 | `O(1)` | `O(n)` |
| 查找 | `O(1)` | `O(n)` |
| 删除 | `O(1)` | `O(n)` |

最坏情况通常发生在大量 key 都落到同一个桶中，此时哈希表退化成链表查找。

## 2. 实现步骤

下面逐步构建一个教学版哈希表。

### 2.1 设计哈希函数

第一步是把 key 转成桶下标。

简单取模：

```cpp
std::size_t index_for_int(int key, std::size_t bucket_count) {
    return static_cast<std::size_t>(key) % bucket_count;
}
```

更通用的写法使用 `std::hash`：

```cpp
template <typename Key>
std::size_t index_for_key(const Key& key, std::size_t bucket_count) {
    return std::hash<Key>{}(key) % bucket_count;
}
```

对于字符串，可以了解 FNV-1a 的思想：从一个初始值开始，逐字节混合字符。

```cpp
std::size_t fnv1a_hash(const std::string& s) {
    std::size_t hash = 14695981039346656037ull;

    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ull;
    }

    return hash;
}
```

教学实现中，我们使用 `std::hash<Key>`，这样模板类可以支持 `int`、`std::string` 等多种 key。

### 2.2 用链表桶实现链地址法

我们让每个桶都是一个 `std::list`，链表中存储键值对。

```cpp
template <typename Key, typename Value>
class HashTable {
private:
    using Entry = std::pair<Key, Value>;
    std::vector<std::list<Entry>> buckets;
    std::size_t count;
};
```

桶下标计算：

```cpp
std::size_t bucket_index(const Key& key) const {
    return std::hash<Key>{}(key) % buckets.size();
}
```

### 2.3 插入操作

插入时需要处理两种情况：

- key 已存在：更新 value。
- key 不存在：插入新键值对。

```cpp
void insert(const Key& key, const Value& value) {
    std::size_t index = bucket_index(key);

    for (auto& entry : buckets[index]) {
        if (entry.first == key) {
            entry.second = value;
            return;
        }
    }

    buckets[index].push_back({key, value});
    ++count;
}
```

### 2.4 查找操作

查找时先定位桶，再遍历桶里的链表。

```cpp
Value* find(const Key& key) {
    std::size_t index = bucket_index(key);

    for (auto& entry : buckets[index]) {
        if (entry.first == key) {
            return &entry.second;
        }
    }

    return nullptr;
}
```

返回指针的好处是：

- 找到时返回 value 的地址。
- 没找到时返回 `nullptr`。

### 2.5 删除操作

删除同样先找到桶，然后在链表中删除对应节点。

```cpp
bool erase(const Key& key) {
    std::size_t index = bucket_index(key);
    auto& bucket = buckets[index];

    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (it->first == key) {
            bucket.erase(it);
            --count;
            return true;
        }
    }

    return false;
}
```

### 2.6 开放寻址法中的线性探测

本教程配套源码使用链地址法。为了理解另一种方案，下面给出线性探测的插入伪代码：

```cpp
std::size_t index = std::hash<Key>{}(key) % buckets.size();

while (buckets[index] is occupied) {
    if (buckets[index].key == key) {
        buckets[index].value = value;
        return;
    }

    index = (index + 1) % buckets.size();
}

buckets[index] = {key, value};
```

开放寻址法删除时不能简单地把桶设为空，否则会打断后续探测路径。通常要使用一个特殊状态，例如：

```text
EMPTY      从未使用
OCCUPIED   正在使用
DELETED    曾经使用，现已删除
```

`DELETED` 也叫 tombstone。

### 2.7 负载因子和 rehash

负载因子（load factor）表示元素数量和桶数量的比例：

```cpp
load_factor = size / bucket_count
```

链地址法中，负载因子越高，每个桶的平均链表长度越长，查找越慢。

通常会设置最大负载因子，例如 `0.75`。当插入后负载因子超过阈值，就扩容并重新分布元素，这个过程叫 rehash。

```cpp
if (load_factor() > max_load_factor) {
    rehash(buckets.size() * 2);
}
```

rehash 的核心步骤：

1. 创建更多桶。
2. 遍历旧桶中的所有元素。
3. 按新桶数量重新计算下标。
4. 把元素放进新桶。

```cpp
void rehash(std::size_t new_bucket_count) {
    std::vector<std::list<Entry>> new_buckets(new_bucket_count);

    for (auto& bucket : buckets) {
        for (auto& entry : bucket) {
            std::size_t new_index =
                std::hash<Key>{}(entry.first) % new_bucket_count;
            new_buckets[new_index].push_back(entry);
        }
    }

    buckets = std::move(new_buckets);
}
```

## 3. 调用方式

配套文件 `hash_table.cpp` 提供了一个模板类：

```cpp
template <typename Key, typename Value>
class HashTable;
```

### 3.1 创建哈希表

```cpp
HashTable<std::string, int> scores;
HashTable<int, std::string> id_to_name(16);
```

构造函数参数是初始桶数量，默认值为 8。

### 3.2 插入或更新

```cpp
scores.insert("Alice", 95);
scores.insert("Bob", 88);
scores.insert("Alice", 99); // key 已存在，更新分数
```

### 3.3 查找

```cpp
if (int* score = scores.find("Alice")) {
    std::cout << "Alice: " << *score << "\n";
} else {
    std::cout << "Alice not found\n";
}
```

如果 key 存在，`find` 返回 value 指针；如果不存在，返回 `nullptr`。

也可以在只读对象上查找：

```cpp
const HashTable<std::string, int>& readonly_scores = scores;

if (const int* score = readonly_scores.find("Bob")) {
    std::cout << *score << "\n";
}
```

### 3.4 删除

```cpp
bool removed = scores.erase("Bob");

if (removed) {
    std::cout << "Bob removed\n";
}
```

`erase` 返回 `true` 表示成功删除，返回 `false` 表示 key 不存在。

### 3.5 查看大小和负载因子

```cpp
std::cout << "size = " << scores.size() << "\n";
std::cout << "load factor = " << scores.load_factor() << "\n";
```

### 3.6 手动 rehash

通常插入时会自动 rehash。也可以手动指定新的桶数量：

```cpp
scores.rehash(32);
```

当你预计会插入很多元素时，可以提前扩容，减少多次 rehash 的成本。

### 3.7 完整使用示例

```cpp
#include <iostream>
#include <string>

int main() {
    HashTable<std::string, int> table;

    table.insert("C++", 95);
    table.insert("Python", 90);
    table.insert("Rust", 88);

    if (int* value = table.find("C++")) {
        std::cout << "C++ score = " << *value << "\n";
    }

    table.erase("Python");

    std::cout << "size = " << table.size() << "\n";
    std::cout << "load factor = " << table.load_factor() << "\n";

    return 0;
}
```

## 4. 学习建议

学习哈希表时，可以重点观察三件事：

- 哈希函数是否把 key 均匀分布到了不同桶。
- 冲突发生后，数据结构如何继续保持正确。
- 负载因子升高后，为什么 rehash 能恢复查询效率。

C++ 标准库中的 `std::unordered_map` 就是工业级哈希表实现。理解本教程中的简化版本后，再阅读和使用 `std::unordered_map` 会更容易。
