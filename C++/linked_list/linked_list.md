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
- 拷贝时必须深拷贝节点，不能复制 `unique_ptr`
- 移动时可以直接转移 `head` / `tail` / `size`，但被覆盖的旧链表仍应迭代销毁，避免长链递归析构

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
- iterator traits — 例如 `std::forward_iterator_tag` / `std::bidirectional_iterator_tag`
- `const_iterator` — 让 `const SinglyLinkedList<T>&` 也能安全遍历

本示例中：

- `SinglyLinkedList<T>` 提供前向迭代器和 `ConstIterator`
- `DoublyLinkedList<T>` 提供双向迭代器、`ConstIterator`，并能通过 `std::reverse_iterator` 反向遍历
- `front()` / `back()` 同时提供 const 与非 const 版本

## 6. 拷贝、移动与异常安全

`std::unique_ptr` 让节点所有权非常清晰，但它也意味着默认拷贝被禁用。容器如果要支持拷贝，必须自己写：

- copy constructor：遍历源链表，对每个元素创建新节点
- copy assignment：先拷贝到临时对象，再 `swap`，这样如果中途分配失败，原链表不变
- move constructor / move assignment：转移所有权后，把源链表置空
- `clear()`：迭代销毁节点，避免很长的链表通过 `unique_ptr` 递归析构导致栈溢出

示例代码里还用 `static_assert(std::is_copy_constructible<T>::value, ...)` 提醒：只有元素类型 `T` 可拷贝时，链表深拷贝才成立。C++20 可以进一步用 concepts 写成 `requires std::copy_constructible<T>`，但本项目按 C++17 编译。

## 7. 经典链表练习：reverse 与 merge

### reverse()

单向链表反转是理解指针所有权转移的好练习。核心思想是维护三段：

- `previous`：已经反转好的前缀
- `head_`：当前节点
- `next`：临时保存原来的后继节点

每一步把当前节点的 `next` 指向 `previous`，再整体向前推进。整个过程 O(n)，不分配新节点。

### merge_sorted()

合并两个有序单向链表也是 O(n) 操作。本示例的 `merge_sorted(left, right)` 接收两个链表值参数，然后移动节点到结果链表中：

- 不拷贝元素值
- 不重新分配节点
- 保持稳定排序：相等时优先取左侧链表

这和 `std::forward_list::merge` / `std::list::merge` 的思想接近：链表的强项之一就是可以通过改指针来拼接节点。

## 8. Intrusive List（侵入式链表）

普通容器式链表会为每个元素额外分配一个节点：

```cpp
[node: data + next] -> [node: data + next]
```

侵入式链表把 `next` 指针直接放进业务对象里：

```cpp
struct Particle {
    int id;
    double energy_mev;
    Particle* next;
};
```

优点：

- 不需要额外节点分配
- 对象地址稳定时，插入/删除只改指针
- 常见于内核、游戏引擎、内存池、任务队列等低层系统

代价：

- 链表不拥有对象生命周期，调用者必须保证对象仍然活着
- 一个对象如果要同时进入多个链表，需要多个 hook 指针
- 接口不如标准容器通用，容易误用

## 9. C++ 标准库的链表

### std::forward_list（C++11）
- 单向链表，最小内存开销
- 只支持前向遍历
- `insert_after`, `erase_after`（注意是"之后"）

### std::list
- 双向链表
- `push_front`, `push_back`, `insert`, `erase`
- 迭代器在插入/删除时不会失效
- 额外开销：每个节点两个指针

## 10. 性能注意事项

1. **不要用链表替代 vector 做随机访问** — 每次 `list[i]` 都是 O(n) 遍历
2. **考虑内存碎片** — 大量小节点分配可能导致碎片化
3. **递归析构危险** — `unique_ptr` 链过长时析构会爆栈
4. **HPC 场景优先 vector** — 除非确实需要频繁的中间插入/删除且迭代器稳定
5. **节点池 / allocator** — 如果大量创建销毁节点，可以考虑 `std::pmr`、对象池或自定义 allocator，减少小对象分配开销
6. **批量操作** — `merge`、`splice`、整链拼接这类操作通常比逐元素拷贝更符合链表优势
7. **内存布局** — 链表节点散布在堆上，CPU cache miss 往往比算法复杂度更影响真实性能

## 11. 线程安全说明

本教程代码没有实现内部加锁。一般规则：

- 多个线程只读同一个链表：通常可以，但前提是没有线程同时修改它
- 一个线程写、其他线程读：需要外部同步，例如 `std::mutex`
- 多个线程同时写：需要外部同步，或改用专门设计的并发数据结构
- 迭代器稳定不等于线程安全；节点不搬家，但并发删除仍可能让另一个线程持有悬空迭代器

教学代码刻意不加入锁，因为锁会掩盖链表本身的所有权和指针逻辑。工业代码应在更高层明确同步策略。

## 12. 本示例程序

`linked_list.cpp` 演示了：

- 用 `std::unique_ptr` 实现单/双向链表
- 深拷贝、移动语义、迭代销毁
- 非 const / const 迭代器接口
- 常见操作（push_front, push_back, emplace_front, emplace_back, pop_front, erase）
- 单向链表 `reverse()` 和两个有序链表的 `merge_sorted()`
- 双向链表的反向遍历
- 一个简单 intrusive list 示例
- 与 `std::forward_list` / `std::list` 的对比
- 性能基准：vector vs list 的随机插入

学习建议：运行程序，观察链表的内部结构变化，理解指针操作和所有权语义。
