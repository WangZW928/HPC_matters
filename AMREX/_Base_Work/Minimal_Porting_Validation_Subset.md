# 最小移植验证子集

这份文档给出一套“先跑通再扩展”的最小验证闭环，目标是尽快判断目标后端芯片后端是否具备 AMReX 可移植性。

## 1. 为什么要做最小子集

全量适配通常周期长、反馈慢。最小子集的价值是：

- 快速定位阻塞层（编译模型 / launch / memory / 原语）。
- 先建立可执行闭环，再迭代扩功能与性能。

## 2. 子集范围（建议）

### 2.1 编译与限定符

- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H)
- [`AMReX_GpuTypes.H`](../AMReX_GpuTypes.H)

通过标准：device lambda、host-device 函数、global kernel 声明可编译。

### 2.2 启动链

- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)

通过标准：`ParallelFor(box, lambda)` 和 `ParallelFor(box, ncomp, lambda)` 可跑通。

### 2.3 数据访问

- [`AMReX_Array4.H`](../AMReX_Array4.H)
- [`AMReX_FabArray.H`](../AMReX_FabArray.H)
- [`AMReX_MFIter.H`](../AMReX_MFIter.H)

通过标准：`Array4(i,j,k,n)` 读写正确，线性化与 stride 一致。

### 2.4 规约

- [`AMReX_Reduce.H`](../AMReX_Reduce.H)
- [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)
- [`AMReX_GpuReduce.H`](../AMReX_GpuReduce.H)

通过标准：本地 `sum/max` 与 tuple reduction 正确。

### 2.5 内存路径

- [`AMReX_GpuMemory.H`](../AMReX_GpuMemory.H)
- [`AMReX_GpuAllocators.H`](../AMReX_GpuAllocators.H)
- [`AMReX_Arena.H`](../AMReX_Arena.H)

通过标准：alloc/copy/free 与异步生命周期正确。

### 2.6 原子和基础原语

- [`AMReX_GpuAtomic.H`](../AMReX_GpuAtomic.H)

通过标准：atomic add/min/max 等最小集合可用。

## 3. 推荐验证顺序

1. 限定符与最小 kernel 编译。
2. `ParallelFor` 启动链跑通。
3. `Array4` 访问正确性验证。
4. `MFParallelFor` 真实上层入口验证。
5. 本地规约 `Reduce/ParReduce`。
6. 内存与生命周期稳定性。
7. 原子操作覆盖验证。

这个顺序能保证每一步都建立在前一步可运行基础上。

## 4. 最小通过标准（DoD）

- [ ] 编译通过：device lambda + launch 宏
- [ ] 运行通过：`ParallelFor` 基础 case
- [ ] 正确性通过：`Array4` / `MFParallelFor` 结果与 CPU 一致
- [ ] 规约通过：`sum/max` + tuple reduction 一致
- [ ] 内存通过：异步场景无生命周期错误
- [ ] 原子通过：基础 atomic case 稳定

## 5. 可延后项（预研阶段）

- CUDA Graph 相关（[`AMReX_CudaGraph.H`](../AMReX_CudaGraph.H)）
- 高级容器适配（[`AMReX_GpuContainers.H`](../AMReX_GpuContainers.H)）
- 复杂数与边缘路径（[`AMReX_GpuComplex.H`](../AMReX_GpuComplex.H)）

先把最小闭环做稳，再补这些高级能力，通常更高效。

