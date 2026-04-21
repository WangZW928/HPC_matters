# ParallelFor Example Design Notes

这份文档说明 `_Base_Tests/ParallelFor` 这个最小项目的设计哲学，以及它是如何一步步向 AMReX 的调用方式靠拢的。

## 目标

这个小项目不是为了完整复刻 AMReX，而是为了抽出一条最小但清晰的学习链：

1. 用户只看到一个统一的并行接口
2. 同一份用户代码可以同时服务 CPU 和 GPU 编译
3. 数据容器和数据 view 的关系尽量贴近 `BaseFab / Array4`
4. `ParallelFor` 只负责遍历和执行，不负责顺手分配内存
5. CPU/GPU 差异尽量往底层收，而不是散落在用户代码里

所以它更像一个“AMReX 思想缩略模型”，而不是一个通用并行库。

## 最核心的调用形状

当前最重要的代码形状在 [`main.cpp`](./main.cpp) 里：

```cpp
Buffer<double> buffer(ncell, defaultMemorySpace());
buffer.fillZero();

auto view = buffer.view2D(nx, ny);

parallelFor2D(nx, ny, [=] BASE_GPU_DEVICE (int i, int j) noexcept {
    view(i, j) = 100.0 + i + 10.0 * j;
});
```

这段代码故意长得像下面这类 AMReX 写法：

```cpp
auto const& arr = fab.array();
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
    arr(i,j,k) = ...;
});
```

也就是说，这个小项目最想保留的不是“功能像不像 AMReX”，而是“调用直觉像不像 AMReX”。

## 设计哲学

### 1. 用户层只保留一个接口

在 [`ParallelFor2D.H`](./ParallelFor2D.H) 里，对外只有：

```cpp
parallelFor2D(nx, ny, lambda)
```

不会暴露：

- `parallelFor2D_cpu(...)`
- `parallelFor2D_gpu(...)`

这是因为 AMReX 的核心风格就是：

- 用户写统一接口
- 后端分发藏在实现内部

所以即使内部仍然有 CPU 路径和 GPU 路径，用户层也不应该直接感知到这两个名字。

### 2. 数据容器和数据 view 分开

在 [`Buffer.H`](./Buffer.H) 里，`Buffer<T>` 负责：

- 分配内存
- 释放内存
- host/device 区分
- 把数据拷回 host

在 [`Array2DView.H`](./Array2DView.H) 里，`Array2DView<T>` 只负责：

- 保存指针
- 保存尺寸
- 用 `operator()(i,j)` 做索引访问

这和 AMReX 里“容器”和“view”分开的思想是很接近的：

- `BaseFab` / `FArrayBox` / `MultiFab` 更像 owning container
- `Array4` 更像 non-owning view

这里的 `Buffer + Array2DView`，就是一个最小化的类比模型。

### 3. 内存创建放在外层，ParallelFor 只负责执行

这是这次收敛里很关键的一步。

我们最终把内存申请放在 [`main.cpp`](./main.cpp) 外层，而不是放在某个 helper 里：

```cpp
Buffer<double> buffer(...);
auto view = buffer.view2D(nx, ny);
parallelFor2D(nx, ny, ...);
```

原因是这更符合 AMReX 的真实调用习惯：

1. 先创建容器
2. 再拿 view
3. 再 launch `ParallelFor`

也就是说，`ParallelFor` 的职责应该是：

- 遍历 index space
- 调用 kernel body

而不是：

- 分配数据
- 管理对象生命周期
- 隐藏容器来源

这会让代码的控制流更自然，也更容易扩展成多块数据、多次 launch 的场景。

### 4. 后端差异往底层收，不散落在用户代码里

在 [`Backend.H`](./Backend.H) 里，我们把这几类东西集中起来了：

- `BASE_GPU_DEVICE`
- `BASE_GPU_HOST_DEVICE`
- `backendSynchronize()`
- `backendLabel()`

在 [`Buffer.H`](./Buffer.H) 里，还集中定义了：

- `defaultMemorySpace()`

这样 [`main.cpp`](./main.cpp) 就不需要自己再写一段 `#if defined(__CUDACC__)` 去判断：

- 当前是不是 GPU 编译
- 默认该分配哪种 memory space
- 该打印什么 label

这种收法更接近 AMReX 的设计哲学：

- 用户层尽量少写后端判断
- 基础设施层负责吸收平台差异

### 5. 一份用户源码，同时服务 CPU 和 GPU

当前仓库里只保留一份入口文件：[`main.cpp`](./main.cpp)。

这份源码同时可以：

- 用普通 C++ 编译，`BASE_GPU_DEVICE` 变成空宏
- 用 CUDA 编译，`BASE_GPU_DEVICE` 变成 `__device__`

这很像 AMReX 的感觉，因为用户看到的是同一份调用代码。

需要注意的是，GPU 目标在构建时仍然需要 `.cu` 编译域。这里的处理方式是：

- 源码仓库里只有一个 `main.cpp`
- 如果检测到 CUDA，CMake 会在 build 目录复制出 `main_cuda.cu`
- GPU 目标编译 build 目录里的那份生成文件

这样做的目的，是同时满足两件事：

- 用户阅读时只有一份统一源码
- 构建系统仍然能正确进入 CUDA 编译流程

## 为什么删掉了多余的 helper

这个项目中间曾经出现过一层 `runParallelForDemo(...)` helper。它在重构初期有一定价值，因为它帮助我们先统一 CPU/GPU 的共享流程。

但继续保留它，会开始偏离 AMReX 的真实使用方式，因为它把下面这些操作包得太深了：

- 建 buffer
- 拿 view
- 调 `parallelFor2D`
- 同步
- 打印

而 AMReX 里更常见的直觉是：

```cpp
auto arr = fab.array();
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (...) {
    arr(...) = ...;
});
```

所以最终我们选择把 helper 去掉，让 `main.cpp` 直接呈现最核心的调用结构。

这一步的意义是：

- 代码更短
- 调用关系更直接
- 和 AMReX 的类比更清楚

## 线性化思路

在 [`ParallelFor2D.H`](./ParallelFor2D.H) 里，GPU 路径采用最简单的 2D 线性化：

```text
icell = i + j * nx
```

反解时：

```text
j = icell / nx
i = icell % nx
```

这和 AMReX 在 `Box` 上的 GPU 线性化思想是一致的，只不过这里把问题缩到了最小的二维整数网格。

这也是为什么这个例子适合作为学习入口：

- 你可以先看懂最小的线性化
- 再去对照 AMReX 的 `BoxIndexerND`
- 最后再回到真正的 `ParallelFor(box, lambda)`

## 文件职责链

当前各个文件的职责可以概括为：

- [`main.cpp`](./main.cpp)
  用户层调用入口，负责组织“容器 -> view -> ParallelFor”这条主线。
- [`ParallelFor2D.H`](./ParallelFor2D.H)
  提供统一并行接口，并在内部切换 CPU/GPU 后端。
- [`Buffer.H`](./Buffer.H)
  提供最小 owning container，模拟 host/device 内存管理。
- [`Array2DView.H`](./Array2DView.H)
  提供最小 non-owning view，模拟 `Array4` 风格的索引访问。
- [`Backend.H`](./Backend.H)
  收纳后端相关宏和少量后端 helper。
- [`Print2D.H`](./Print2D.H)
  仅负责把结果打印出来，便于验证。

## 这份例子不打算做什么

为了保持教学清晰，这个小项目刻意没有去做：

- 完整的 box 类型
- ghost cell
- component 维
- 多个 tile 或多个 block 容器
- 更复杂的 launch policy
- 完整仿真 AMReX 的 memory arena 系统

因为这些内容一旦都加进来，就会冲淡它当前最重要的教学目标：

“先建立统一接口、view、后端分发、线性化这几件事之间的基本感觉。”

## 推荐的下一步

如果要继续把这个例子往 AMReX 靠近，最自然的下一步是：

1. 加一个最小 `Box2D`
2. 把 `parallelFor2D(nx, ny, ...)` 改成 `parallelFor(box, ...)`
3. 把 `i/j` 的范围从默认 `0...nx-1` 改成由 box 决定

这样你后面再去看真正的 `amrex::ParallelFor(box, lambda)`、`BoxND`、`BoxIndexerND`，就会很顺。

