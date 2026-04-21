# AMReX_GpuLaunch.H 移植笔记

对应源码：[`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)

## 1. 文件定位

`AMReX_GpuLaunch.H` 是 AMReX GPU 执行链中的发射枢纽，主要职责如下：

- 定义后端 launch 宏（CUDA/HIP）。
- 定义 kernel wrapper（`launch_global`）。
- 拼接 CPU/GPU 两条 `ParallelFor` 路径（通过 `GpuLaunchFunctsG/C`）。
- 提供执行配置辅助（`ExecutionConfig`）。

高层调用链可概括为：

```text
ParallelFor(...)
-> GpuLaunchFuncts*.H 模板分发
-> GpuLaunch.H 中的 launch 宏与 wrapper
-> 后端 runtime 发射 kernel
```

## 2. 最终 launch 宏

CUDA/HIP 构建下核心宏：

- `AMREX_LAUNCH_KERNEL(...)`
- `AMREX_LAUNCH_KERNEL_NOBOUND(...)`

后端映射：

- CUDA：`<<<blocks, threads, sharedMem, stream>>>`
- HIP：`hipLaunchKernelGGL(...)`

这层若无法正确映射，上层 `ParallelFor` 即使能编译，也无法真正发射执行。

## 3. Kernel Wrapper

主要 wrapper：

- `launch_global<MT>(...)`
- `launch_global(...)`

配套函数：

- `call_device(...)`：设备侧按顺序调用可变参数 lambda。
- `launch_host(...)`：主机侧包装路径。

该层用于把多 lambda、模板化调用收敛为可发射的 kernel 入口。

## 4. ExecutionConfig 作用

`Gpu::ExecutionConfig` 与 `makeExecutionConfig` 负责执行参数组织：

- 由 `Box` 或 `N` 计算 `numBlocks/numThreads`。
- 施加线程约束（`AMREX_GPU_MAX_THREADS`、warp 约束）。
- 通过 `makeNExecutionConfigs` 处理大规模 `N` 的分段发射。

移植检查重点：

- block/grid 计算不溢出。
- 发射维度不越界。
- 大规模分段发射结果正确。

## 5. CPU/GPU 路径拼接

文件尾部包含决定执行路径：

- GPU：`AMReX_GpuLaunchMacrosG.H` + `AMReX_GpuLaunchFunctsG.H`
- 非 GPU：`AMReX_GpuLaunchMacrosC.H` + `AMReX_GpuLaunchFunctsC.H`
- 同时包含：`AMReX_GpuLaunchFunctsSIMD.H`

这套机制保证同一 `ParallelFor` 接口能落到不同后端实现。

## 6. 落地判定口径

### 6.1 编译可落地

- `AMREX_GPU_GLOBAL` kernel 声明可编译。
- `AMREX_LAUNCH_KERNEL` 能展开为目标后端支持语法。
- device lambda 与 wrapper 组合可编译。

### 6.2 运行可落地

- `ParallelFor(box, lambda)` 可发射并执行。
- `ParallelFor(box, ncomp, lambda)` component 维行为正确。
- stream 传参与同步语义正确。

### 6.3 正确性可落地

- 索引映射与 `Array4` 访问语义一致。
- 大 `N` 分段路径正确。
- CPU/GPU 在确定性算例上结果一致。

## 7. 常见风险

- launch 宏映射通过，但 stream/sharedMem 语义不一致。
- wrapper 可编译，但 device lambda 约束不满足。
- block/grid 超限导致 silent failure 或越界。
- 仅小规模通过，`ncomp` 或大规模路径失败。

## 8. 实施顺序

1. 打通最小 `AMREX_LAUNCH_KERNEL` 发射。
2. 验证多 lambda 的 `launch_global` 包装路径。
3. 验证 `ExecutionConfig` 的边界与分段场景。
4. 回到上层验证 `ParallelFor/MFParallelFor` 真实路径。

## 9. 联动阅读

- [`AMReX_GpuQualifiers_Notes.md`](./AMReX_GpuQualifiers_Notes.md)
- [`GPU_Launch_Chain_Overview.md`](./GPU_Launch_Chain_Overview.md)
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
