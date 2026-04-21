# AMReX `ParallelFor(box, lambda)` 重载学习笔记

## 1. 我要研究的对象是什么

这里聚焦的是最常见的一类调用：

```cpp
amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
    // kernel body
});
```

用户通常写的是 `Box`，但在实现层它对应的是：

- [`AMReX_BaseFwd.H`](../AMReX_BaseFwd.H):30
  `using Box = BoxND<AMREX_SPACEDIM>;`

所以 `ParallelFor(box, lambda)` 的真正模板形态其实是：

```cpp
ParallelFor (BoxND<dim> const& box, L&& f)
```

## 2. 相关源码文件

这一组接口主要分布在：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
  总入口头文件，按后端选择具体实现。
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
  GPU 构建时使用的实现，包含 CUDA/HIP/SYCL 路径。
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
  非 GPU 构建时使用的 CPU 递归循环实现。
- [`AMReX_TypeTraits.H`](../AMReX_TypeTraits.H)
  提供 `MaybeDeviceRunnable` 等 traits，参与 GPU 重载选择。

在 [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H) 里，具体接线位置是：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):294
  `#include <AMReX_GpuLaunchFunctsG.H>`
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):297
  `#include <AMReX_GpuLaunchFunctsC.H>`
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):333
  `ParallelForOMP(Box const&, L const&)`
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):366
  OpenMP CPU 实现版本的 `ParallelForOMP(Box const&, L const&)`

## 3. 最常见的重载有哪些

如果只看 `ParallelFor(box, lambda)` 这一支，最重要的是下面几类。

### 3.1 CPU 版本入口

定义在 [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)：

- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):293
  `void ParallelFor (BoxND<dim> const& box, L const& f) noexcept`
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):302
  `void ParallelFor (BoxND<dim> const& box, L&& f) noexcept`
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):309
  `void ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, L&& f) noexcept`
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):315
  `void ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, L&& f) noexcept`

这些重载的特点是：

- `L const&` 那个是真正干活的 CPU 实现
- `L&&` 版本只是做转发
- 带 `Gpu::KernelInfo` 的版本在 CPU 路径里基本只是继续转发，并不会真的发 kernel

### 3.2 GPU 版本用户入口

定义在 [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1537
  `void ParallelFor (BoxND<dim> const& box, L&& f) noexcept`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1547
  `void ParallelFor (BoxND<dim> const& box, L&& f) noexcept`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1318
  `ParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept`

这几层的关系很清楚：

1. 你平时调用 `ParallelFor(box, lambda)`
2. 它先补上默认线程配置 `AMREX_GPU_MAX_THREADS`
3. 再补上默认的 `Gpu::KernelInfo{}`
4. 最后进入真正后端实现

也就是说，用户看见的是一个很简洁的接口，但内部其实会先走一层默认参数装配。

### 3.3 GPU 版本的核心执行实现

这一层是真正把 `box` 映射成 GPU kernel 执行的地方。

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):396
  `ParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L const& f) noexcept`
  这是 GPU/SYCL 路径中的一个核心实现。
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1013
  `ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, L const& f) noexcept`
  这是 CUDA/HIP 路径里的核心实现之一。

这一层的共同特点是：

- 如果 `box` 为空，直接返回
- 把 `box` 包装成 `BoxIndexerND<dim>`
- 以 `box.numPts()` 为基础生成 execution config
- 在 GPU 上用线性线程索引 `icell`
- 再由 `BoxIndexerND` 把 `icell` 反解成空间索引 `iv`
- 最后再调用你的 lambda

## 4. lambda 可以是什么签名

这一点很重要。

`ParallelFor(box, lambda)` 并不是只接受一种 lambda 签名。它会通过辅助函数把 `IntVectND<dim>` 展开成多个整数参数，再尝试调用你的 lambda。

相关辅助在：

- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):76
  `call_f_intvect_handler`（CPU 路径）
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):86
  `call_f_intvect_handler`（GPU 路径）
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):56
  `call_f_intvect_inner`

核心思路是：

```cpp
f(iv[0], iv[1], iv[2], ...)
```

也就是把 `IntVectND` 展开成独立参数传进去。

对 `Box` 这种最常见情况，常见签名有：

```cpp
(int i, int j, int k)
```

或者：

```cpp
(int i, int j, int k, Gpu::Handler const& handler)
```

后者更底层一些，通常在需要 reduction、warp 信息、或更细粒度控制时才会用到。

## 5. CPU 路径是怎么执行的

CPU 路径非常直接，它不会“launch GPU kernel”，而是做递归展开循环。

核心实现：

- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):261
  `ParallelFor_impND`

逻辑可以概括成：

1. 取出 `lo = lbound_iv(box)` 和 `hi = ubound_iv(box)`
2. 准备一个 `IntVectND<dim> iv`
3. 递归生成 `dim` 重循环
4. 最内层调用 `call_f_intvect_handler(f, iv)`

在 1D/2D/3D 特化路径里，还会尽量让最内层 `i` 循环带上 `AMREX_PRAGMA_SIMD`。

所以 CPU 路径的本质是：

```text
Box -> 多重 for 循环 -> 组装 iv -> 调 lambda
```

## 6. GPU 路径是怎么执行的

GPU 路径不是三重循环，而是先把 box 扁平化成线性索引空间。

关键点：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):404
  `BoxIndexerND<dim> indexer(box)`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1017
  `BoxIndexerND<dim> indexer(box)`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1026
  `detail::call_f_intvect_handler(f, iv, ...)`

逻辑可以概括成：

1. 把 box 看成 `numPts()` 个 cell 的线性区间
2. 每个 GPU thread 拿到一个 `icell`
3. 用 `indexer.intVect(icell)` 反解出 `(i,j,k)`
4. 再调用 lambda

所以 GPU 路径的本质是：

```text
Box -> 线性线程索引 icell -> indexer.intVect(icell) -> lambda(i,j,k)
```

这也是为什么用户接口看起来像三重循环，但 GPU 实现并不真的在设备端写三重 for。

## 7. `Gpu::KernelInfo` 在这里起什么作用

`Gpu::KernelInfo` 是 AMReX 给 kernel launch 传附加信息的结构。

在 `ParallelFor(box, lambda)` 最普通的调用中，你通常不会自己传它，而是使用默认的：

```cpp
Gpu::KernelInfo{}
```

用户入口会自动补上它，例如：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1537
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H):309

如果你以后看到更底层写法：

```cpp
ParallelFor(Gpu::KernelInfo{}, box, lambda)
```

它本质上是在显式地走同一条重载链。

## 8. `MaybeDeviceRunnable` 为什么重要

在 GPU 路径里，并不是所有 lambda 都能直接在 device 上跑。

相关 trait：

- [`AMReX_TypeTraits.H`](../AMReX_TypeTraits.H):104
  `struct MaybeDeviceRunnable : std::true_type {};`

GPU 版本很多重载都带这个约束：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1013
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1318

这表示 AMReX 会在编译期筛选“这个 lambda 是否可能作为 device callable 对象使用”。

这也是为什么我们平时会写：

```cpp
[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }
```

而不是随便写一个 host-only lambda 塞进去。

## 9. `ParallelForOMP(box, lambda)` 和它的关系

如果你看到 `ParallelForOMP(box, lambda)`，它并不是另一套完全不同的 API，而是一个“在 CPU+OpenMP 情况下更积极并行”的包装。

相关位置：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):333
  如果不是 OpenMP CPU 特化路径，它直接转发到 `ParallelFor(box, f)`
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H):366
  在 CPU+OpenMP 路径下，它会自己展开成 OpenMP for 循环

所以可以把它理解成：

- `ParallelFor` 是主接口
- `ParallelForOMP` 是带 OpenMP 偏向的 host 包装

## 10. 我建议你先记住的重载链

如果你现在只是想抓住主线，先记住这一条就够了：

### GPU 构建时

```text
ParallelFor(box, lambda)
-> ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box, lambda)
-> GPU backend implementation
-> BoxIndexerND 把线性线程索引映射成 (i,j,k)
-> 调 lambda(i,j,k)
```

### 非 GPU 构建时

```text
ParallelFor(box, lambda)
-> CPU overload
-> ParallelFor_impND
-> 多重 for 循环
-> 调 lambda(i,j,k)
```

## 11. 一句话总结

`ParallelFor(box, lambda)` 的重载设计，本质上是在给用户提供一个统一接口：

“你写一个像三重循环体一样的 lambda；AMReX 根据当前后端，把它转发到 CPU 递归循环、OpenMP 循环，或者真正的 GPU kernel launch 实现里去执行。”

## 12. `ParallelFor(box, lambda)` 的线性化是怎么做的

如果你现在主要关心 GPU launch kernel，那么最关键的不是重载链本身，而是这一步：

```text
(i,j,k) 这种逻辑上的三维 cell 索引
        ->
GPU 上的一维线程索引 icell
        ->
再反解回 (i,j,k)
```

这就是 AMReX 在 GPU 路径里做的“线性化”。

### 12.1 先把 `Box` 看成一段线性区间

在 GPU 路径里，`ParallelFor` 先把 `box` 包装成：

- [`AMReX_Box.H`](../AMReX_Box.H):2151
  `struct BoxIndexerND`

而真正使用它的位置，比如：

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1017
  `const BoxIndexerND<dim> indexer(box);`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1023
  `auto icell = std::uint64_t(MT)*blockIdx.x+threadIdx.x + start_idx;`
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H):1025
  `auto iv = indexer.intVect(icell);`

也就是说，kernel 内部线程先拿到的不是 `(i,j,k)`，而是一个线性编号 `icell`。

### 12.2 `icell` 的来源

以 CUDA/HIP 路径为例，每个线程的全局线性编号大意是：

```cpp
icell = MT * blockIdx.x + threadIdx.x + start_idx;
```

其中：

- `MT` 是每个 block 的线程数
- `blockIdx.x` 是 block 编号
- `threadIdx.x` 是 block 内线程编号
- `start_idx` 是当前 execution config 片段的起始 cell 偏移

所以从 GPU 角度看，整个 box 先被当成一个长度为 `box.numPts()` 的一维数组。

### 12.3 `BoxIndexerND` 里是怎么存线性化信息的

`BoxIndexerND` 的构造函数在：

- [`AMReX_Box.H`](../AMReX_Box.H):2156

关键逻辑是：

```cpp
std::uint64_t mult = 1;
for (int i=0; i<dim-1; ++i) {
    mult *= box.length(i);
    fdm[i] = Math::FastDivmodU64(mult);
}
```

这里的含义是：

- `box.length(0)` 也就是 `nx`
- `box.length(1)` 也就是 `ny`
- `box.length(2)` 也就是 `nz`

于是它会预先建立这样的“除数”：

- 2D 时：`fdm[0] = nx`
- 3D 时：
  - `fdm[0] = nx`
  - `fdm[1] = nx * ny`

这和 `Array4` 的 stride 其实很像，本质上都是 Fortran 风格顺序：

- `i` 变化最快
- `j` 次之
- `k` 最慢

### 12.4 `icell -> (i,j,k)` 的反解公式

核心函数在：

- [`AMReX_Box.H`](../AMReX_Box.H):2169
  `IntVectND<dim> intVect (std::uint64_t icell) const`

源码逻辑是：

```cpp
IntVectND<dim> retval = lo;

for (int i=dim-1; i>0; --i) {
    std::uint64_t quotient, remainder;
    fdm[i-1](quotient, remainder, icell);
    retval[i] += quotient;
    icell = remainder;
}

retval[0] += icell;
```

它做的事情就是不断地“整除 + 取余”：

#### 2D 情况

如果 box 尺寸是 `nx, ny`，那么：

```text
j = icell / nx
i = icell % nx
```

最后：

```text
i_real = ilo + i
j_real = jlo + j
```

#### 3D 情况

如果 box 尺寸是 `nx, ny, nz`，那么：

```text
k  = icell / (nx * ny)
r1 = icell % (nx * ny)

j  = r1 / nx
i  = r1 % nx
```

最后：

```text
i_real = ilo + i
j_real = jlo + j
k_real = klo + k
```

所以它的线性化顺序可以写成：

```text
icell = (i-ilo)
      + (j-jlo) * nx
      + (k-klo) * nx * ny
```

你会发现，这和 `Array4` 的空间部分偏移公式完全一致。

### 12.5 为什么说这是 Fortran 风格顺序

从上面的公式可以看到：

- `i` 每加 1，`icell` 加 1
- `j` 每加 1，`icell` 加 `nx`
- `k` 每加 1，`icell` 加 `nx * ny`

也就是：

```text
i fastest
j next
k slowest
```

这和 `Array4` 的空间索引布局是一致的，所以 AMReX 的数据访问和 kernel 线性化在思路上是统一的。

### 12.6 一个 3D 小例子

假设：

- `ilo=10, jlo=20, klo=30`
- `nx=4, ny=3, nz=2`

那么：

```text
icell = 0  -> (10,20,30)
icell = 1  -> (11,20,30)
icell = 2  -> (12,20,30)
icell = 3  -> (13,20,30)
icell = 4  -> (10,21,30)
```

再比如：

```text
icell = 12
k  = 12 / (4*3) = 1
r1 = 12 % 12 = 0
j  = 0 / 4 = 0
i  = 0 % 4 = 0
```

所以：

```text
icell = 12 -> (10,20,31)
```

### 12.7 `BoxND::atOffset` 其实是同一套公式

你还可以对照另一个实现：

- [`AMReX_Box.H`](../AMReX_Box.H):1071
  `BoxND<dim>::atOffset(Long offset) const noexcept`

它的逻辑和 `BoxIndexerND::intVect` 本质上是同一件事，也是把一个线性 offset 反解回多维索引。

所以你可以把 `BoxIndexerND` 理解成 GPU launch 场景下更专门、更轻量的 `atOffset` 工具。

## 13. 一句话抓重点

`ParallelFor(box, lambda)` 在 GPU 路径下的线性化，本质上就是：

“先把整个 `Box` 按 `i` 最快、`j` 次之、`k` 最慢的顺序压成一个一维 `icell` 空间；然后每个 GPU thread 拿一个 `icell`，再通过 `BoxIndexerND::intVect(icell)` 反解回真实的 `(i,j,k)`。”
