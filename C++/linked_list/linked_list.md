# Linked List（链表）—— 现代 C++ 视角

链表是一种通过"节点 + 指针"串起来的线性数据结构。每个节点存放数据和指向下一节点的指针。相比数组，链表的插入/删除通常不需要移动其他元素。

## 1. 为什么还需要链表？

数组（`std::vector`）在大多数场景优于链表，因为：

- 连续内存，缓存友好
- 随机访问 O(1)

但链表有它不可替代的位置：

| 场景 | 数组 | 链表 |
|------|------|------|
| 头部插入/删除 | O(n) | O(1) |
| 中间插入/删除（已知位置） | O(n) | O(1) |
| 随机访问 | O(1) | O(n) |
| 缓存局部性 | 极好 | 差 |
| 迭代中插入不失效 | ❌ 可能 reallocate | ✅ 迭代器稳定 |

HPC 场景中链表较少直接使用，但理解和实现链表是理解更复杂指针结构的基石（skip list、intrusive list、B+ 树、图邻接表等）。

## 2. 单向链表（Singly Linked List）

每个节点：`data + next指针`。

```
head -> [A|●] -> [B|●] -> [C|nullptr]
```

### 2.1 传统 C 风格（裸指针）

```cpp
struct Node {
    int data;
    Node* next;
};
```

这种写法的问题：
- 需要手动管理内存（`new` / `delete`）
- 忘记 `delete` → 内存泄漏
- 异常不安全

### 2.2 现代 C++ 风格（`std::unique_ptr`）

```cpp
#include <memory>

template <typename T>
struct Node {
    T data;
    std::unique_ptr<Node> next;

    Node(T value) : data(std::move(value)), next(nullptr) {}
};
```

`std::unique_ptr` 自动释放节点，RAII 保障异常安全。但注意：递归析构可能爆栈（链表很长时），需要在析构函数里用迭代方式销毁。

## 3. 双向链表（Doubly Linked List）

每个节点：`prev指针 + data + next指针`。

```
head ⇄ [A] ⇄ [B] ⇄ [C] ⇄ tail
```

双向链表的优势：
- 可以从任意节点向前/向后遍历
- 删除任意节点只需 O(1)（因为已有 prev）
- 尾部插入 O(1)（维护 tail 指针）

现代 C++ 实现要点：
- `prev` 用裸指针（没有所有权），`next` 用 `unique_ptr`（拥有权）
- 或者全部用 `unique_ptr` + 手动处理 `prev` 的原始指针

## 4. 关键操作的时间复杂度

### 4.1 单向链表

| 操作 | 头部 | 尾部（无 tail） | 尾部（有 tail） | 任意位置 |
|------|------|-----------------|-----------------|----------|
| 插入 | O(1) | O(n) | O(1) | O(1)* |
| 删除 | O(1) | O(n) | O(n) | O(1)* |
| 查找 | O(n) | O(n) | O(n) | O(n) |

\* 已知节点位置的前提下

### 4.2 双向链表

| 操作 | 头部 | 尾部 | 任意位置 |
|------|------|------|----------|
| 插入 | O(1) | O(1) | O(1)* |
| 删除 | O(1) | O(1) | O(1) |
| 查找 | O(n) | O(n) | O(n) |

## 5. 迭代器设计

现代 C++ 链表应该提供迭代器接口，这样才能用 range-for 和标准算法：

```cpp
for (auto& value : list) {
    std::cout << value << " ";
}
```

迭代器需要实现：
- `operator*` — 解引用
- `operator++` — 前进
- `operator!=` — 比较

## 6. C++ 标准库的链表

### std::forward_list（C++11）
- 单向链表，最小内存开销
- 只支持前向遍历
- `insert_after`, `erase_after`（注意是"之后"）

### std::list
- 双向链表
- `push_front`, `push_back`, `insert`, `erase`
- 迭代器在插入/删除时不会失效
- 额外开销：每个节点两个指针

## 7. 性能注意事项

1. **不要用链表替代 vector 做随机访问** — 每次 `list[i]` 都是 O(n) 遍历
2. **考虑内存碎片** — 大量小节点分配可能导致碎片化
3. **递归析构危险** — `unique_ptr` 链过长时析构会爆栈
4. **HPC 场景优先 vector** — 除非确实需要频繁的中间插入/删除且迭代器稳定

## 8. 本示例程序

`linked_list.cpp` 演示了：

- 用 `std::unique_ptr` 实现单/双向链表
- 迭代器接口
- 常见操作（push_front, push_back, pop_front, erase）
- 与 `std::forward_list` / `std::list` 的对比
- 性能基准：vector vs list 的随机插入

学习建议：运行程序，观察链表的内部结构变化，理解指针操作和所有权语义。
