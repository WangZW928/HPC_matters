# AMReX `ParallelFor(box, lambda)` 到 `launch_global` 的 GPU 调用链

## 1. 我们要回答的问题

当我们写：

```cpp
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
    // work
});
```

GPU 构建下，代码到底是怎么一步一步走到真正的 GPU kernel launch 的？

这篇笔记就只回答这条链：

```text
ParallelFor(box, lambda)
-> ParallelFor(..., KernelInfo, ...)
-> GPU implementation
-> AMREX_LAUNCH_KERNEL(...)
-> launch_global(...)
-> device lambda 真正执行
```

## 2. 先看最外层用户入口

最外层用户入口在：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1537
  `void ParallelFor (BoxND<dim> const& box, L&& f) noexcept`

它的内容非常短，基本就是：

```cpp
ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box, std::forward<L>(f));
```

所以这一步做了两件事：

1. 给线程块大小补上默认值 `AMREX_GPU_MAX_THREADS`
2. 给 launch 信息补上默认值 `Gpu::KernelInfo{}`

也就是说，用户写的最简单接口，其实只是一个“默认参数包装层”。

## 3. 第二层：带 `KernelInfo` 的转发入口

下一层在：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1318
  `ParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept`

这个版本做的事情也很直接：

```cpp
ParallelFor<AMREX_GPU_MAX_THREADS>(info, box, std::forward<L>(f));
```

也就是说，这一层只是把：

- 用户传进来的 `box`
- 用户传进来的 `lambda`
- 默认或显式给定的 `KernelInfo`

继续转发给真正的模板实例实现。

## 4. 第三层：真正准备 launch 的 GPU 实现

真正干活的核心实现，在 CUDA/HIP 路径里是：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1014
  `ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, L const& f) noexcept`

这层代码已经开始做真正的 kernel launch 准备。它的核心步骤可以按顺序看：

### 4.1 空 box 直接返回

```cpp
if (amrex::isEmpty(box)) { return; }
```

### 4.2 构造 box 的线性索引器

```cpp
const BoxIndexerND<dim> indexer(box);
```

位置：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1017

这一步是后面 GPU 线性化的关键。

### 4.3 生成 execution config

```cpp
const auto& nec = Gpu::makeNExecutionConfigs<MT>(box);
```

它会根据 box 大小、线程块大小等信息，给出一组 launch 配置片段。

### 4.4 遍历每一段 execution config 并 launch

```cpp
for (auto const& ec : nec) {
    const auto start_idx = std::uint64_t(ec.start_idx);
    AMREX_LAUNCH_KERNEL(...)
}
```

这说明 AMReX 并不一定只发一个 kernel 配置，有时会把大 box 分成多个 launch 片段。

## 5. 第四层：真正发起 GPU kernel 的地方

这一层就是最关键的“入口中的入口”。

发 kernel 的宏在：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):37
  CUDA 下：
  `amrex::launch_global<MT><<<...>>>(...)`
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):42
  HIP 下：
  `hipLaunchKernelGGL(launch_global<MT>, ...)`

也就是说，真正的 GPU kernel launch 点就是：

```cpp
AMREX_LAUNCH_KERNEL(...)
```

而这个宏最终会展开成对 `launch_global` 的调用。

这就是“真正从 C++ 主机侧跳到 GPU kernel”的那个地方。

## 6. 第五层：`launch_global` 是什么

`launch_global` 定义在：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):56
  `AMREX_GPU_GLOBAL void launch_global (L f0, Lambdas... fs)`

实现非常短：

```cpp
AMREX_GPU_GLOBAL void launch_global (L f0, Lambdas... fs) {
    f0();
    call_device(fs...);
}
```

它本质上是一个通用 GPU kernel 包装器。

你可以把它理解成：

- `launch_global` 自己是 `__global__` kernel
- 它接收一个或多个 device-callable lambda/functor
- kernel 启动后，在 device 上执行这些 lambda

所以，`ParallelFor` 并不是直接把你的 `(i,j,k)` lambda 当作一个裸 kernel 发出去，而是把它包在一个“真正的全局 kernel 外壳” `launch_global` 里面。

## 7. 第六层：你的 lambda 是什么时候执行的

在 [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1021 附近，传给 `AMREX_LAUNCH_KERNEL` 的其实是一个新的 device lambda：

```cpp
[=] AMREX_GPU_DEVICE () noexcept {
    auto icell = ...;
    if (icell < indexer.numPts()) {
        auto iv = indexer.intVect(icell);
        detail::call_f_intvect_handler(f, iv, ...);
    }
}
```

注意这里有两个 lambda：

### 外层 lambda

这是传给 `AMREX_LAUNCH_KERNEL` 的 lambda。
它不带 `(i,j,k)` 参数，而是在 device 上自己算：

- 当前线程的 `icell`
- `icell -> iv`
- 再去调用用户 lambda

### 内层用户 lambda

也就是你写的：

```cpp
[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }
```

它不是直接作为全局 kernel 被 launch 的，而是在外层 launch lambda 内部，通过：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1026
  `detail::call_f_intvect_handler(f, iv, ...)`

被真正调用。

所以要非常明确：

```text
你写的 lambda
!=
真正被 <<< >>> 发出的那个全局 kernel
```

真正被发出的，是 `launch_global`；
而你写的 lambda，是 `launch_global` 内部再调用的 device callable 对象。

## 8. 整条调用链画成图

可以把 `ParallelFor(box, lambda)` 的 GPU 路径画成：

```text
用户代码
  |
  v
ParallelFor(box, user_lambda)
  |
  v
ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box, user_lambda)
  |
  v
ParallelFor(Gpu::KernelInfo, box, user_lambda) 的 GPU 实现
  |
  | 构造 BoxIndexerND
  | 构造 execution config
  v
AMREX_LAUNCH_KERNEL(..., launch_lambda)
  |
  v
launch_global<<<...>>>(launch_lambda)
  |
  v
GPU 上执行 launch_lambda
  |
  | 计算 icell
  | icell -> iv
  v
call_f_intvect_handler(user_lambda, iv, ...)
  |
  v
最终执行 user_lambda(i,j,k)
```

## 9. 为什么 AMReX 要这么设计

这种设计有几个明显好处：

### 9.1 把用户接口和后端 launch 细节隔开

用户只写：

```cpp
ParallelFor(box, lambda)
```

不需要自己处理：

- 线程块大小
- grid 配置
- 线性索引到 `(i,j,k)` 的映射
- CUDA/HIP 差异

### 9.2 `launch_global` 可以复用

`launch_global` 是 AMReX 通用的 GPU kernel 外壳，不只服务一个 `ParallelFor` 变体。

### 9.3 用户 lambda 保持“算法视角”

用户可以只关心：

```cpp
(i,j,k) 这个 cell 要做什么
```

而不必关心：

```cpp
这个 cell 在 GPU 线程空间里对应哪个 thread
```

## 10. 一句话总结

`ParallelFor(box, lambda)` 在 GPU 构建下的真正入口链是：

“先由 `ParallelFor(box, lambda)` 做默认参数转发，再进入 GPU 实现，随后通过 `AMREX_LAUNCH_KERNEL` 发起真正的 `launch_global` kernel；而你写的 `(i,j,k)` lambda，是在这个 kernel 内部根据 `icell -> IntVect` 映射后才被调用的。”
