# Minimal ParallelFor2D Project

这个小项目现在只保留一份用户入口源码：`main.cpp`。

它被改造成了一个更接近 AMReX 风格的最小实验：

- 用户层只有一个统一接口：`parallelFor2D(nx, ny, lambda)`
- `Array2DView` 负责统一索引访问
- `Buffer<T>` 负责统一 host/device 内存抽象
- `Backend.H` 负责统一后端宏和同步接口
- `main.cpp` 是唯一的入口源码

## 先看哪份文档

- 设计哲学和思路链：[`Design_Notes.md`](./Design_Notes.md)

## 文件说明

- `Backend.H`
  一个很轻量的后端抽象层，定义了：
  - `BASE_GPU_DEVICE`
  - `BASE_GPU_HOST_DEVICE`
  - `backendSynchronize()`
- `Array2DView.H`
  一个最小的 2D 视图类型，内部按 `i + j * nx` 做线性索引。
- `Buffer.H`
  一个很小的内存抽象：
  - `Buffer<T>(n, MemorySpace::Host)`
  - `Buffer<T>(n, MemorySpace::Device)`
- `ParallelFor2D.H`
  只提供一个对外接口：
  - `parallelFor2D(nx, ny, lambda)`

  实现内部：
  - CPU 编译时走 OpenMP 双层循环
  - CUDA 编译时走最简单的线性 kernel launch
- `main.cpp`
  唯一的用户入口源码。
- `Print2D.H`
  共用的 2D 打印小工具
- `CMakeLists.txt`
  最小构建脚本

## 为什么去掉了额外 helper

这个项目现在故意不再保留额外 demo helper，因为我们希望最核心的调用形状直接暴露在 `main.cpp` 里，便于和 AMReX 的：

- `fab.array()`
- `amrex::ParallelFor(...)`

直接对照。

## 现在为什么更像 AMReX

这个结构已经更接近 AMReX 的组织方式了：

- 用户看到的是统一接口 `parallelFor2D(...)`
- 用户入口源码也只有一份 `main.cpp`
- 内存容器在外层先创建
- view 在外层先拿好
- `parallelFor2D(...)` 只负责遍历和执行
- CPU/GPU 分发藏在实现内部
- host/device callable 的差异收进 `Backend.H`

## 这份 `main.cpp` 是怎么同时服务 CPU/GPU 的

关键点有两个：

1. `BASE_GPU_DEVICE` 在非 CUDA 编译下会展开为空
2. 默认 memory space 和默认 label 已经分别收进：
   - [`Buffer.H`](./Buffer.H)
   - [`Backend.H`](./Backend.H)

所以 `main.cpp` 本身不再需要显式写 `#if defined(__CUDACC__)`。

构建时：

- CPU 目标直接编译 `main.cpp`
- 如果检测到 CUDA，CMake 会在构建目录里自动复制出一个 `main_cuda.cu`
- GPU 目标编译那份构建产物，而源码仓库里仍然只保留一个 `main.cpp`
- 因为 GPU 路径使用了带 `BASE_GPU_DEVICE` 的 lambda，CUDA 目标需要开启 `--extended-lambda`

## 线性化方式

这里的 2D 线性化规则是：

```text
icell = i + j * nx
```

反解时：

```text
j = icell / nx
i = icell % nx
```

这和 AMReX 在 `Box` 上做的 GPU 线性化思想是一致的，只是这里简化成了纯 2D 整数网格。

## 构建

### CPU

```bash
cmake -S . -B build
cmake --build build -j
./build/parallelfor2d_cpu
```

### GPU

如果系统里有 CUDA 编译器，CMake 会自动生成 `parallelfor2d_gpu`：

```bash
cmake -S . -B build
cmake --build build -j
./build/parallelfor2d_gpu
```

## 当前环境验证

当前这台机器里：

- CPU 目标已成功编译并运行
- 我已经修正了 nvcc 下常见的两个问题：
  - `__global__` kernel 不能带 `noexcept`
  - device lambda 需要 `--extended-lambda`
- 本会话环境里仍然无法直接用 `nvcc` 实机编译验证，但你本地有 CUDA 时应当可以继续测试
