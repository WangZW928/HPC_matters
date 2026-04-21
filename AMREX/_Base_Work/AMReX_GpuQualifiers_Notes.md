# AMReX_GpuQualifiers.H 移植笔记

这份笔记面向 AMReX GPU 移植预研，聚焦 [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H) 在调用链中的角色和适配要点。

## 1. 这个文件在 GPU 栈里的定位

`AMReX_GpuQualifiers.H` 是 AMReX GPU 语义的“最上游宏契约层”。

它主要负责：

- 把 CUDA/HIP/SYCL/CPU 的限定符差异封装成统一宏。
- 提供 host/device 编译域分支宏。
- 提供是否处于 device 编译域的统一判定。
- 提供设备全局变量的统一声明宏。

如果这层映射不正确，`ParallelFor`、`Reduce`、`MFParallelFor` 等上层调用都会连锁受影响。

## 2. 统一限定符宏

在 `AMREX_USE_GPU && !AMREX_USE_SYCL` 下，核心宏映射为：

- `AMREX_GPU_HOST` -> `__host__`
- `AMREX_GPU_DEVICE` -> `__device__`
- `AMREX_GPU_GLOBAL` -> `__global__`
- `AMREX_GPU_HOST_DEVICE` -> `__host__ __device__`
- `AMREX_GPU_CONSTANT` -> `__constant__`
- `AMREX_GPU_MANAGED` -> `__managed__`
- `AMREX_GPU_DEVICE_MANAGED` -> `__device__ __managed__`

非 GPU 构建时这些宏会变为空。

移植意义：

- 这是 device lambda、kernel wrapper、device function 能否编译的入口。
- 新后端首先要保证这些宏与编译器语义一一对应。

## 3. host/device 条件执行宏

关键宏：

- `AMREX_IF_ON_DEVICE(CODE)`
- `AMREX_IF_ON_HOST(CODE)`

作用：在同一份源码里根据编译域选择执行分支。

实现特点：

- CUDA + PGI/NVHPC 下优先走 `NV_IF_TARGET`。
- 其余场景通过 `__CUDA_ARCH__ / __HIP_DEVICE_COMPILE__ / __SYCL_DEVICE_ONLY__` 判定。
- 内部用 `AMREX_IMPL_STRIP_PARENS` 处理宏参数括号，确保 `CODE` 能安全展开。

移植意义：

- 这是“单源代码双域执行”语义是否成立的关键。
- 如果这组宏行为异常，常见症状是 host/device 分支反了或都执行不到。

## 4. device 编译域统一判定

关键宏：

- `AMREX_DEVICE_COMPILE`

定义：

```text
(__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || __SYCL_DEVICE_ONLY__)
```

移植意义：

- 上层代码会依赖它判断当前是否在 device 编译域。
- 自研后端如果没有兼容这些内建宏，需要提供等价判定路径。

## 5. 设备全局变量宏

关键宏：

- `AMREX_DEVICE_GLOBAL_VARIABLE(...)`
- 内部分流：`AMREX_DGV` / `AMREX_DGVARR`

按后端展开：

- SYCL: `device_global`
- CUDA/HIP: `__device__`
- CPU: 普通全局变量

移植意义：

- 这组宏影响设备端全局状态声明与链接语义。
- 需要尽早验证标量/数组两种声明都能通过编译与链接。

## 6. 对移植预研的直接建议

优先验证这 4 组能力：

1. `AMREX_GPU_HOST_DEVICE` / `AMREX_GPU_GLOBAL` 宏映射正确。
2. `AMREX_IF_ON_DEVICE/HOST` 行为正确。
3. `AMREX_DEVICE_COMPILE` 判定正确。
4. `AMREX_DEVICE_GLOBAL_VARIABLE`（标量+数组）可编可链。

如果以上 4 项通过，后续 `GpuLaunch` 和 `ParallelFor` 适配会顺很多。

## 7. 这份文件在调用链中的关系

可与下列文件联动阅读：

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)

阅读顺序建议：

1. 先读 `AMReX_GpuQualifiers.H`（语义契约）
2. 再读 `GpuLaunch*`（执行链）
3. 最后读 `MFParallelFor`（上层真实调用）

