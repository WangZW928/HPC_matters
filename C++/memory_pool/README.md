# C++ 内存池学习笔记

这个项目用于学习内存池的基本思想。第一版代码实现了一个“固定大小块内存池”，对应文件是 `src/main.cpp`。

现在项目已经拆成了更接近工程结构的形式：

```text
include/memory_pool/fixed_block_memory_pool.hpp  固定大小 block 内存池
include/memory_pool/object_pool.hpp              类型安全对象池
include/memory_pool/linear_memory_pool.hpp       线性内存池
include/memory_pool/pool_allocator.hpp           STL allocator 和 PoolVector
include/memory_pool/gpu_allocator.hpp            可选 CUDA GPU allocator 辅助工具
src/main.cpp                                     示例程序
```

## 1. 为什么需要内存池

普通的 `new` / `delete` 很方便，但它们通常会经过通用堆分配器。通用分配器要处理各种大小、各种生命周期、各种线程场景，因此开销不一定小。

当程序频繁申请和释放大量小对象时，可能出现几个问题：

1. 分配和释放成本高。
2. 内存碎片增多。
3. 对象分布分散，缓存局部性变差。
4. 高频路径里 `new` / `delete` 的时间不够稳定。

内存池的基本思路是：提前申请一大块内存，然后自己管理这块内存中的小块。

## 2. 内存池和内存申请、释放的关系

普通内存申请方式：

```text
程序
  |
  | new / malloc
  v
通用堆分配器
  |
  | 向操作系统申请或复用堆内存
  v
操作系统 / 虚拟内存

程序
  |
  | delete / free
  v
通用堆分配器
  |
  | 归还到堆分配器管理结构中
  v
后续再被其他申请复用
```

使用内存池之后：

```text
初始化阶段：

程序
  |
  | 一次性申请较大内存
  v
内存池
  |
  | 底层仍然可能调用 new / malloc / mmap
  v
通用堆分配器或操作系统


运行阶段：

程序
  |
  | pool.allocate()
  v
内存池空闲链表
  |
  | 弹出一个空闲 block
  v
返回一小块可用内存

程序
  |
  | pool.deallocate(ptr)
  v
内存池空闲链表
  |
  | 把 block 插回链表
  v
等待下一次复用
```

核心区别：

```text
普通 new/delete:
每次对象申请释放都可能触达通用堆分配器。

内存池:
先批量拿一大块内存，之后小对象申请释放主要在池内部完成。
```

所以内存池不是绕开内存申请，而是把“很多次小申请”变成“少数几次大申请 + 池内快速复用”。

## 3. 内存池在程序中的调用流程

以本项目中的 `Particle` 为例，完整流程如下：

```text
1. 创建内存池

FixedBlockMemoryPool pool(sizeof(Particle), 4);

        |
        v

内存池申请一整块 buffer

+---------+---------+---------+---------+
| block 0 | block 1 | block 2 | block 3 |
+---------+---------+---------+---------+

        |
        v

把空闲 block 串成 free list

free_list -> block 3 -> block 2 -> block 1 -> block 0 -> null
```

申请对象时：

```text
2. 申请原始内存

void* raw = pool.allocate();

        |
        v

free_list 弹出一个 block

返回 raw 指针
```

构造对象时：

```text
3. 在 raw 上构造对象

Particle* p = new (raw) Particle{...};

        |
        v

raw 指向的原始内存变成真正的 Particle 对象
```

释放对象时：

```text
4. 先析构对象

p->~Particle();

        |
        v

Particle 生命周期结束，但那块内存还在
```

归还内存时：

```text
5. 把 block 还给内存池

pool.deallocate(p);

        |
        v

block 回到 free list，等待下次 allocate 复用
```

可以把它记成一句话：

```text
内存池管内存：allocate / deallocate
对象生命周期管对象：placement new / 显式析构
```

## 4. 内存池的核心思想

一个简单的固定块内存池通常这样工作：

1. 一次性申请一大片连续内存。
2. 把它切成很多等大小 block。
3. 用一个空闲链表记录哪些 block 还没被使用。
4. 分配时，从空闲链表弹出一个 block。
5. 释放时，把 block 放回空闲链表。

这样做的好处是：

- 分配和释放都接近 O(1)。
- 同类对象挨得更近，缓存友好。
- 可以减少通用堆分配器的压力。
- 对生命周期规律的对象更容易统一管理。

## 5. 固定大小块内存池

本项目里的 `FixedBlockMemoryPool` 是最适合入门的一种内存池。

它只负责分配固定大小的 block：

```cpp
FixedBlockMemoryPool pool(sizeof(Particle), 4);
```

这表示：

- 每个 block 至少能容纳一个 `Particle`。
- 总共有 4 个 block。
- 最多同时放 4 个 `Particle` 对象。

它不关心对象类型，只管理原始内存：

```cpp
void* raw = pool.allocate();
pool.deallocate(raw);
```

真正构造对象需要 placement new：

```cpp
auto* p = new (raw) Particle{1.0F, 2.0F, 3.0F, 101};
```

对象销毁需要手动调用析构函数：

```cpp
p->~Particle();
pool.deallocate(p);
```

这点非常重要：内存池负责“内存”，对象构造和析构仍然要正确处理。

## 6. 空闲链表是什么

释放状态下的 block 本身没有对象，所以可以把它的前几个字节借来存一个指针，指向下一个空闲 block。

代码里是：

```cpp
struct FreeBlock {
    FreeBlock* next;
};
```

每个空闲 block 都临时看作 `FreeBlock`。这样空闲 block 串起来，就形成了 free list。

分配时：

```text
free_list -> block A -> block B -> block C
```

取走 A 后：

```text
free_list -> block B -> block C
```

释放 A 时，再把 A 插回链表头。

### 栈式空闲链表会提高 cache hit 吗

通常会有帮助，但要看具体访问模式。

本项目里的空闲链表采用的是“栈式 LIFO”策略：

```text
deallocate(ptr):
    把刚释放的 block 放到 free_list 头部

allocate():
    从 free_list 头部取出一个 block
```

这意味着：刚释放的 block 很可能下一次马上又被分配出去。

从 cache 的角度看，刚被使用过的 block 可能还在 L1/L2 cache 中。如果下一次分配立刻复用它，就更有机会命中 cache：

```text
释放 p1
  |
  v
p1 对应 block 放回 free_list 头部
  |
  v
下一次 allocate() 优先拿回这个 block
  |
  v
更可能复用仍在 cache 中的数据地址
```

这叫时间局部性：最近访问过的数据，短时间内再次访问的概率较高。

不过也要注意：

- 如果对象释放后很久才复用，cache 可能早就被其他数据替换了。
- 空闲链表本身是指针结构，遍历链表时可能有 pointer chasing 成本。
- 固定大小 block 连续存放时，空间局部性通常也不错。
- 对象是否 cache 友好，还取决于程序后续是否按连续顺序访问这些对象。

所以更准确的说法是：

```text
栈式 free list 能提高“最近释放、马上复用”场景下的 cache 命中机会。
但整体 cache 表现还要看对象访问模式，而不只看分配器结构。
```

## 7. placement new 是什么

普通 `new Particle{...}` 做两件事：

1. 申请内存。
2. 在这块内存上构造对象。

placement new 只做第二件事：

```cpp
new (address) Particle{...};
```

意思是：不要重新申请内存，就在 `address` 这块已经给定的内存上构造 `Particle`。

所以内存池经常和 placement new 一起出现。

## 8. 内存池适合什么场景

适合：

- 游戏中的粒子、子弹、组件。
- 网络服务器里的连接对象、请求对象。
- 编译器/解释器中的 AST 节点。
- 高频创建销毁的小对象。
- 生命周期相近的一批对象。

不适合：

- 对象大小差异很大。
- 对象数量很少。
- 生命周期完全不可预测。
- 代码简单性比性能更重要。

内存池是性能工具，不是默认选择。先写清楚，再在热点路径中使用。

## 9. 需要注意的坑

对齐问题。

内存池返回的地址必须满足对象类型的对齐要求。这个入门示例适合学习思路，但严格工程实现还要使用 `std::align` 或按最大对齐值分配。

析构问题。

如果对象有析构函数，归还内存前必须手动调用析构函数，否则资源可能泄漏。

越界问题。

固定容量内存池用完后，要么抛出异常，要么扩容，要么返回 `nullptr`。本项目选择抛出 `std::bad_alloc`。

重复释放问题。

同一块内存重复归还会破坏空闲链表。工程实现通常需要 debug 检查或更严格的所有权设计。

线程安全问题。

本项目不是线程安全的。多线程场景需要锁、无锁结构、线程本地池，或者每个线程一个池。

## 10. 类型安全的 `ObjectPool<T>`

内存池管理的是原始内存：

```cpp
void* allocate();
void deallocate(void*);
```

对象池管理的是对象生命周期：

```cpp
Particle* create(...);
void destroy(Particle*);
```

本项目现在已经在 `FixedBlockMemoryPool` 之上封装了一个类型安全的 `ObjectPool<T>`：

```cpp
template <typename T>
class ObjectPool {
public:
    explicit ObjectPool(std::size_t capacity);

    template <typename... Args>
    T* create(Args&&... args);

    void destroy(T* ptr);
};
```

使用方式从原来的：

```cpp
void* raw = pool.allocate();
auto* p = new (raw) Particle{...};

p->~Particle();
pool.deallocate(p);
```

变成：

```cpp
ObjectPool<Particle> pool(4);

Particle* p = pool.create(Particle{1.0F, 2.0F, 3.0F, 101});
pool.destroy(p);
```

这层封装的价值是：

- 调用方拿到的是 `Particle*`，不是 `void*`。
- `create(...)` 内部负责 `allocate()` 和 placement new。
- `destroy(ptr)` 内部负责调用析构函数和 `deallocate()`。
- 构造函数如果抛异常，会把已经申请的 block 还给内存池，避免泄漏。

关系图：

```text
调用方
  |
  | pool.create(args...)
  v
ObjectPool<T>
  |
  | memory_pool_.allocate()
  | placement new T(args...)
  v
返回 T*

调用方
  |
  | pool.destroy(ptr)
  v
ObjectPool<T>
  |
  | ptr->~T()
  | memory_pool_.deallocate(ptr)
  v
block 回到 free list
```

可以这样理解：

```text
FixedBlockMemoryPool:
    只知道 block，不知道 T。

ObjectPool<T>:
    知道 T，负责把 block 变成对象，再把对象还原成 block。
```

## 11. RAII 句柄：自动调用 `destroy()`

手动调用 `destroy(ptr)` 仍然有一个风险：如果中途 `return`、抛异常，或者程序员忘了写，就可能泄漏对象生命周期。

所以项目里给 `ObjectPool<T>` 增加了 RAII 句柄：

```cpp
using Handle = std::unique_ptr<T, Deleter>;
```

现在可以这样使用：

```cpp
ObjectPool<Particle> pool(4);

auto p = pool.make_handle(Particle{1.0F, 2.0F, 3.0F, 101});
```

`p` 离开作用域时，会自动调用：

```cpp
pool.destroy(p.get());
```

调用流程：

```text
pool.make_handle(...)
  |
  | create(...)
  v
构造 T 对象
  |
  v
返回 unique_ptr<T, Deleter>
  |
  | 离开作用域 / reset()
  v
Deleter 自动调用 pool.destroy(ptr)
  |
  v
析构对象，并把 block 还给 free list
```

需要注意：RAII 句柄内部保存了 `ObjectPool<T>*`，所以池对象必须比句柄活得更久。

## 12. 接入 STL allocator：让 `std::vector` 使用内存池

`std::vector` 和 `ObjectPool<T>` 的需求不一样。

`ObjectPool<T>` 一次分配一个对象：

```text
allocate one block -> construct one T
```

但 `std::vector<T>` 经常一次申请连续的 `n` 个元素：

```text
allocate n * sizeof(T) bytes -> store T[0], T[1], ..., T[n-1]
```

因此项目里新增了一个 `LinearMemoryPool` 和 `PoolAllocator<T>`。它更像一个线性分配器。

为了让使用方式更像“自己的 vector”，项目里还定义了一个类型别名：

```cpp
template <typename T>
using PoolVector = std::vector<T, PoolAllocator<T>>;
```

它不是继承 `std::vector`，而是使用标准容器本身提供的 allocator 扩展点。这样做更符合 C++ 标准库的设计方式：容器行为仍然来自 `std::vector`，内存申请则交给 `PoolAllocator<T>`。

使用方式：

```cpp
auto memory = std::make_shared<LinearMemoryPool>(1024);
PoolAllocator<int> allocator(memory);

PoolVector<int> values(allocator);
values.reserve(16);
values.push_back(1);
values.push_back(2);
```

调用关系：

```text
PoolVector<int>
  |
  | reserve / 扩容
  v
PoolAllocator<int>::allocate(n)
  |
  | pool.allocate(n * sizeof(int), alignof(int))
  v
LinearMemoryPool
  |
  | 从连续 buffer 中切出一段连续内存
  v
返回给 std::vector 使用
```

这个 allocator 的重要特点：

- 支持 `std::vector` 需要的连续内存。
- 分配速度很快，只移动一个 offset。
- `deallocate()` 不会立刻回收单次分配，内存随整个池一起释放。
- 适合“一批容器一起创建，一起销毁”的场景。

这和固定块 free list 是两种不同池：

```text
FixedBlockMemoryPool:
    适合 ObjectPool<T>，一次复用一个固定大小 block。

LinearMemoryPool:
    适合 std::vector 这类容器，一次分配一段连续内存。
```

## 13. GPU 端 allocator 有必要吗

有学习价值，但是否在当前项目里“必须使用”，要看目标。

如果你的目标是理解 C++ allocator 机制、对象池和 CPU 端内存管理，那么 GPU allocator 不是必须的。它会引入 CUDA/HIP、设备内存、统一内存、host/device 拷贝、kernel 访问等额外概念。

如果你的目标是 HPC 或 GPU 编程，那么很值得补。因为 GPU 内存和 CPU 内存不是同一种东西，allocator 的语义会更明显：

```text
CPU allocator:
    返回 CPU 可以直接读写的地址。

GPU device allocator:
    返回 GPU device pointer，CPU 通常不能像普通指针一样解引用。

CUDA managed allocator:
    返回 unified memory 指针，CPU/GPU 都能访问，但访问时可能发生页迁移。
```

项目里新增了 `include/memory_pool/gpu_allocator.hpp`。它默认不启用 CUDA，避免没有 CUDA SDK 的机器编译失败。

启用方式是定义：

```cpp
#define MEMORY_POOL_ENABLE_CUDA
#include "memory_pool/gpu_allocator.hpp"
```

或者在编译参数里定义：

```bash
-DMEMORY_POOL_ENABLE_CUDA
```

目前这个头文件提供两类东西：

```cpp
memory_pool::CudaDeviceBuffer<T>
memory_pool::CudaManagedAllocator<T>
memory_pool::CudaManagedVector<T>
```

项目现在也提供了一个真正的 CUDA 示例：

```text
src/cuda_demo.cu
```

它做了两件事：

```text
1. CudaDeviceBuffer<int>
   CPU vector -> cudaMemcpy 到 GPU -> kernel 平方 -> cudaMemcpy 回 CPU

2. CudaManagedVector<int>
   使用 cudaMallocManaged 分配统一内存 -> 直接传给 kernel -> CPU 读取结果
```

`CudaDeviceBuffer<T>` 使用 `cudaMalloc` / `cudaFree`，适合把指针传给 CUDA kernel：

```cpp
memory_pool::CudaDeviceBuffer<float> buffer(1024);
float* device_ptr = buffer.data();
```

注意：`device_ptr` 是 GPU 端指针，不应该在普通 CPU C++ 代码里直接 `device_ptr[0] = 1.0F`。

`CudaManagedAllocator<T>` 使用 `cudaMallocManaged`，可以和 `std::vector` 的 allocator 机制结合：

```cpp
memory_pool::CudaManagedVector<int> values;
values.push_back(1);
values.push_back(2);
```

但它用的是统一内存，不是纯 device memory。统一内存更方便学习和原型开发；高性能 GPU 程序仍然要认真考虑数据迁移、预取、同步和访问模式。

最重要的边界：

```text
不要把 cudaMalloc 得到的纯 device memory 直接当作普通 std::vector 的后备内存。

std::vector 会在 CPU 端构造、析构、移动元素。
纯 device pointer 对 CPU 来说不是普通可解引用地址。
```

所以这个项目保留了两个方向：

- CPU 侧：`PoolVector<T>` 使用 `LinearMemoryPool`。
- GPU 侧：`CudaDeviceBuffer<T>` 管理 device memory，`CudaManagedVector<T>` 演示 unified memory allocator。

如果 `nvcc` 不在 PATH，但安装在常见位置 `/usr/local/cuda/bin/nvcc`，当前 CMake 会尝试自动找到它。也可以手动指定：

```bash
CUDACXX=/usr/local/cuda/bin/nvcc cmake -S . -B build
```

运行 CUDA 示例：

```bash
cmake -S . -B build
cmake --build build
./build/memory_pool_cuda_demo
```

如果不想构建 CUDA 示例：

```bash
cmake -S . -B build -DMEMORY_POOL_BUILD_CUDA=OFF
```

## 14. 如何运行

在 WSL 或 Linux 环境中：

```bash
cd /home/wangz/MyProject/HPC_matters/C++/memory_pool
cmake -S . -B build
cmake --build build
./build/memory_pool_demo
```

也可以直接用 g++：

```bash
g++ -std=c++17 -Wall -Wextra -pedantic src/main.cpp -o memory_pool_demo
./memory_pool_demo
```

## 15. 后续可以继续扩展

下一步可以把这个项目扩展成：

1. 支持自动扩容的内存池。
2. 增加 debug 检查，发现重复释放和非本池指针。
3. 给 `ObjectPool<T>` 增加遍历或统计能力。
4. 改成 C++17 `std::pmr::memory_resource` 风格。
5. 增加真正的 CUDA 示例，例如 kernel 读写 `CudaDeviceBuffer<T>`。
