# GPU 启动链总览

这份文档聚焦一件事：`AMReX 用户调用 -> GPU kernel 真正 launch` 的完整路径。

目标不是覆盖所有实现细节，而是让移植时先抓住最关键的调用链和适配点。

## 1. 一条主链先记住

```text
ParallelFor / MFParallelFor / TagParallelFor
-> AMReX_GpuLaunchFunctsG.H (GPU 路径模板)
-> AMReX_GpuLaunch.H (launch 包装 + 宏)
-> AMReX_GpuLaunchGlobal.H (global kernel 声明)
-> 后端 runtime kernel launch
```

CPU fallback 主链是：

```text
ParallelFor / MFParallelFor
-> AMReX_GpuLaunchFunctsC.H
-> 串行 / OpenMP / SIMD 路径
```

## 2. 关键文件分层

### 2.1 用户入口层

- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)
- [`AMReX_MFParallelForG.H`](../AMReX_MFParallelForG.H)
- [`AMReX_MFParallelForC.H`](../AMReX_MFParallelForC.H)
- [`AMReX_TagParallelFor.H`](../AMReX_TagParallelFor.H)
- [`AMReX_CTOParallelForImpl.H`](../AMReX_CTOParallelForImpl.H)

作用：把 `Box/Array4/FabArray` 上的遍历意图表达成统一 `ParallelFor` 调用。

### 2.2 launch 分发层

- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
- [`AMReX_GpuLaunchFunctsSIMD.H`](../AMReX_GpuLaunchFunctsSIMD.H)

作用：模板重载分发、lambda 包装、索引线性化与 CPU/GPU 路径切换。

### 2.3 backend launch 层

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
- [`AMReX_GpuLaunchGlobal.H`](../AMReX_GpuLaunchGlobal.H)
- [`AMReX_GpuLaunchMacrosG.H`](../AMReX_GpuLaunchMacrosG.H)
- [`AMReX_GpuLaunchMacrosC.H`](../AMReX_GpuLaunchMacrosC.H)
- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H)

作用：统一 `AMREX_GPU_GLOBAL` / `AMREX_GPU_DEVICE` / `AMREX_LAUNCH_KERNEL` 等抽象，映射到底层后端。

## 3. 移植时最先看什么

1. `AMReX_GpuQualifiers.H`：限定符宏是否可映射到目标后端编译器模型。
2. `AMReX_GpuLaunch.H`：最终 launch 宏和 kernel wrapper 能否落地。
3. `AMReX_GpuLaunchFunctsG.H`：`ParallelFor(box, lambda)` GPU 分支是否全部可编译。
4. `AMReX_GpuLaunchFunctsC.H`：CPU fallback 是否可作为 early bring-up 保底路径。

## 4. 最小启动链 smoke case（建议）

- case A: `ParallelFor(box, lambda(i,j,k))` 单分量写入。
- case B: `ParallelFor(box, ncomp, lambda(i,j,k,n))` 带 component 维。
- case C: `MFParallelFor` + `Array4` 访问。

验证目标：

- 能编译（限定符、device lambda、kernel wrapper）。
- 能 launch（grid/block 计算 + runtime 调用）。
- 能正确回写（索引线性化正确）。

## 5. 典型风险点

- device lambda 语法能力不完整（例如需要额外编译选项）。
- `__global__`/kernel wrapper 语义和现有后端不匹配。
- 索引线性化与 `Array4` 内存布局不一致。
- launch 参数类型（grid/block/stream）映射不一致。

## 6. 交付导向检查表

- [ ] `AMREX_GPU_DEVICE` / `AMREX_GPU_HOST_DEVICE` / `AMREX_GPU_GLOBAL` 可用
- [ ] `AMREX_LAUNCH_KERNEL` 可触发真实后端 launch
- [ ] `ParallelFor` 三类入口（box / box+ncomp / MF）可编可跑
- [ ] CPU fallback 可运行并与 GPU 结果一致（至少小规模）

