## 子文档导航

为便于直接推进移植预研，这份总览已拆成 3 个可执行子文档：

1. [`GPU 启动链总览`](./GPU_Launch_Chain_Overview.md)
2. [`GPU 内存与生命周期`](./GPU_Memory_and_Lifecycle.md)
3. [`最小移植验证子集`](./Minimal_Porting_Validation_Subset.md)

建议阅读顺序：`启动链 -> 内存生命周期 -> 最小验证子集`。

---
# AMReX GPU Porting Overview

这份笔记面向“把 AMReX 移植到公司自研芯片架构”的预研阶段，重点不是把所有文件逐个解释完，而是先建立一张 **GPU 调用栈地图**：

- AMReX 的 GPU 调用链从哪里开始
- 哪些文件属于“前端接口”
- 哪些文件属于“后端抽象层”
- 哪些文件是在移植时最优先需要读透和适配的

## 1. 先给一个总判断

列出来的文件整体方向是对的，而且已经覆盖了 **AMReX Base 层 GPU 支撑体系的大部分核心文件**。

如果从“移植工作量”和“后端适配价值”看，这些文件大致可以分成 6 层：

1. 顶层入口与限定符
2. 启动与 kernel 调度
3. 设备管理与执行控制
4. 内存与生命周期管理
5. 基础并行原语
6. 上层使用者与典型调用方

对芯片移植来说，真正最关键的不是把文件列表背下来，而是先抓住：

```text
限定符宏
-> launch 宏/函数
-> device/stream 控制
-> memory/arena
-> atomic/reduce
-> 上层 ParallelFor / MFParallelFor / Array4 / Reduce 调用方
```

这条链才是移植时真正会反复碰到的主线。

---

## 2. 这份清单的大方向是否准确

### 2.1 列出的核心 GPU 文件，方向是准确的

列的这些文件，基本都属于 Base 层 GPU 支撑体系的主干：

- `AMReX_Gpu.H`
- `AMReX_GpuQualifiers.H`
- `AMReX_GpuTypes.H`
- `AMReX_GpuControl.H/.cpp`
- `AMReX_GpuDevice.H/.cpp`
- `AMReX_GpuMemory.H`
- `AMReX_GpuAllocators.H`
- `AMReX_GpuError.H`
- `AMReX_GpuAssert.H`
- `AMReX_GpuPrint.H`
- `AMReX_GpuKernelInfo.H`
- `AMReX_GpuElixir.H/.cpp`
- `AMReX_GpuAsyncArray.H/.cpp`
- `AMReX_GpuBuffer.H`
- `AMReX_GpuContainers.H`
- `AMReX_GpuRange.H`
- `AMReX_GpuUtility.H/.cpp`
- `AMReX_GpuComplex.H`
- `AMReX_GpuAtomic.H`
- `AMReX_GpuReduce.H`
- `AMReX_GpuLaunchGlobal.H`
- `AMReX_GpuLaunch.H`
- `AMReX_GpuLaunch.nolint.H`
- `AMReX_GpuLaunchMacrosC.H/.nolint.H`
- `AMReX_GpuLaunchMacrosG.H/.nolint.H`
- `AMReX_GpuLaunchFunctsC.H`
- `AMReX_GpuLaunchFunctsG.H`
- `AMReX_GpuLaunchFunctsSIMD.H`
- `AMReX_CudaGraph.H`

这批文件足以构成一条完整的 GPU 执行支撑链。

### 2.2 但如果目标是“做移植预研”，还建议补看这些文件

除了列的 GPU 核心文件，建议再补上这些“调用方 / 上层入口 / 配套基础设施”文件：

- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)
- [`AMReX_MFParallelForC.H`](../AMReX_MFParallelForC.H)
- [`AMReX_MFParallelForG.H`](../AMReX_MFParallelForG.H)
- [`AMReX_TagParallelFor.H`](../AMReX_TagParallelFor.H)
- [`AMReX_CTOParallelForImpl.H`](../AMReX_CTOParallelForImpl.H)
- [`AMReX_Reduce.H`](../AMReX_Reduce.H)
- [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)
- [`AMReX_ParallelReduce.H`](../AMReX_ParallelReduce.H)
- [`AMReX_Arena.H`](../AMReX_Arena.H)
- [`AMReX_Arena.cpp`](../AMReX_Arena.cpp)
- [`AMReX_Array4.H`](../AMReX_Array4.H)
- [`AMReX_FabArray.H`](../AMReX_FabArray.H)
- [`AMReX_MFIter.H`](../AMReX_MFIter.H)
- [`AMReX_FabArrayUtility.H`](../AMReX_FabArrayUtility.H)

原因很简单：

- 前一批文件告诉“后端怎么工作”
- 这一批文件告诉“上层到底是怎么用后端的”

移植时必须同时理解这两层，否则容易只把后端接口补齐，却不知道真实调用压力集中在哪里。

---

## 3. 从移植视角看，AMReX GPU 栈怎么分层

## 3.1 顶层入口与限定符层

### 相关文件

- [`AMReX_Gpu.H`](../AMReX_Gpu.H)
- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H)
- [`AMReX_GpuTypes.H`](../AMReX_GpuTypes.H)
- [`AMReX_GpuKernelInfo.H`](../AMReX_GpuKernelInfo.H)

### 这层干什么

这层主要解决：

- 当前是否启用 GPU 后端
- host/device 限定符怎么统一抽象
- `dim3`、stream、handler、launch 信息等基础类型怎么统一

### 移植预研重点

这层是 **编译模型适配入口**。

对自研芯片来说，目标后端通常首先要回答：

1. 有没有 CUDA/HIP/SYCL 兼容编译模型？
2. 是否需要定义一套新的 qualifiers 宏？
3. `AMREX_GPU_DEVICE` / `AMREX_GPU_HOST_DEVICE` / `AMREX_GPU_GLOBAL` 这些宏最终应该展开成什么？
4. kernel launch 所需的最小类型体系是否已有？

如果这一层没有设计好，后面 launch、memory、reduce 都会变得很被动。

---

## 3.2 Kernel 启动与调度层

### 相关文件

- [`AMReX_GpuLaunchGlobal.H`](../AMReX_GpuLaunchGlobal.H)
- [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
- [`AMReX_GpuLaunch.nolint.H`](../AMReX_GpuLaunch.nolint.H)
- [`AMReX_GpuLaunchMacrosC.H`](../AMReX_GpuLaunchMacrosC.H)
- [`AMReX_GpuLaunchMacrosC.nolint.H`](../AMReX_GpuLaunchMacrosC.nolint.H)
- [`AMReX_GpuLaunchMacrosG.H`](../AMReX_GpuLaunchMacrosG.H)
- [`AMReX_GpuLaunchMacrosG.nolint.H`](../AMReX_GpuLaunchMacrosG.nolint.H)
- [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
- [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
- [`AMReX_GpuLaunchFunctsSIMD.H`](../AMReX_GpuLaunchFunctsSIMD.H)
- [`AMReX_CudaGraph.H`](../AMReX_CudaGraph.H)

### 这层干什么

这层是 AMReX GPU 运行模型的核心。

它主要解决：

- `ParallelFor(...)` 最终怎么变成真实 kernel launch
- CPU fallback 路径怎么走
- GPU 路径下 block/thread/grid 怎么计算
- 不同后端 launch 宏怎么组织
- 是否支持 graph capture 等高级执行模式

### 移植预研重点

如果只能优先啃一层，建议 **先啃这层**。

因为对芯片移植来说，最关键的问题往往是：

- 能不能把 AMReX 的 lambda kernel 正确 launch 起来？
- 能不能支持 AMReX 当前依赖的 launch 语义？
- 能不能兼容它的 CPU/GPU 双路径分发？

### 具体建议

先重点看：

1. [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
2. [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
3. [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
4. [`AMReX_GpuLaunchGlobal.H`](../AMReX_GpuLaunchGlobal.H)

这几份文件能帮建立：

```text
用户调用 ParallelFor
-> 进入 launch 函数模板
-> 进入 CPU/GPU 分发
-> 进入 global kernel / 后端具体 launch
```

这条最关键的调用链。

---

## 3.3 设备管理与执行控制层

### 相关文件

- [`AMReX_GpuControl.H`](../AMReX_GpuControl.H)
- [`AMReX_GpuControl.cpp`](../AMReX_GpuControl.cpp)
- [`AMReX_GpuDevice.H`](../AMReX_GpuDevice.H)
- [`AMReX_GpuDevice.cpp`](../AMReX_GpuDevice.cpp)
- [`AMReX_GpuUtility.H`](../AMReX_GpuUtility.H)
- [`AMReX_GpuUtility.cpp`](../AMReX_GpuUtility.cpp)

### 这层干什么

这层主要负责：

- 当前设备选择
- stream 管理
- device 属性查询
- launch region 开关
- 同步行为
- 一些执行期辅助函数

### 移植预研重点

对自研芯片来说，这一层是 **运行时适配的核心**。

目标后端要尽快确认：

- 设备枚举模型是什么
- stream/queue 模型是什么
- 是否支持多 stream
- 是否需要兼容 external stream
- 设备属性查询能不能映射到 AMReX 当前接口
- 同步语义是否和 CUDA/HIP 接近

如果这层不能稳定提供语义，`ReduceData`、`AsyncArray`、Elixir、launch graph 这些机制都会受影响。

---

## 3.4 内存与生命周期管理层

### 相关文件

- [`AMReX_GpuMemory.H`](../AMReX_GpuMemory.H)
- [`AMReX_GpuAllocators.H`](../AMReX_GpuAllocators.H)
- [`AMReX_GpuElixir.H`](../AMReX_GpuElixir.H)
- [`AMReX_GpuElixir.cpp`](../AMReX_GpuElixir.cpp)
- [`AMReX_GpuAsyncArray.H`](../AMReX_GpuAsyncArray.H)
- [`AMReX_GpuAsyncArray.cpp`](../AMReX_GpuAsyncArray.cpp)
- [`AMReX_GpuBuffer.H`](../AMReX_GpuBuffer.H)
- [`AMReX_Arena.H`](../AMReX_Arena.H)
- [`AMReX_Arena.cpp`](../AMReX_Arena.cpp)

### 这层干什么

这层主要负责：

- device / managed / pinned 等内存分配策略
- allocator 封装
- 临时对象生命周期延长（Elixir）
- 异步数组与 stream 相关资源管理
- Arena 抽象

### 移植预研重点

这层对移植来说通常是第二优先级。

目标后端要先回答：

1. 自研芯片后端支持哪些内存类型？
2. 是否有 pinned memory / unified memory / managed memory 的等价概念？
3. 是否支持异步 memcpy？
4. 对象生命周期如果跨异步执行，怎么保证有效？
5. Arena 体系是否可以直接复用，还是需要定制 allocator backend？

很多移植项目最后卡住，不是在 launch，而是在“内存模型和 CUDA 不完全等价”这一层。

---

## 3.5 基础并行原语层

### 相关文件

- [`AMReX_GpuAtomic.H`](../AMReX_GpuAtomic.H)
- [`AMReX_GpuReduce.H`](../AMReX_GpuReduce.H)
- [`AMReX_GpuComplex.H`](../AMReX_GpuComplex.H)
- [`AMReX_GpuContainers.H`](../AMReX_GpuContainers.H)
- [`AMReX_GpuRange.H`](../AMReX_GpuRange.H)
- [`AMReX_GpuPrint.H`](../AMReX_GpuPrint.H)
- [`AMReX_GpuAssert.H`](../AMReX_GpuAssert.H)
- [`AMReX_GpuError.H`](../AMReX_GpuError.H)

### 这层干什么

这层是“kernel 内部能不能顺利运行”的基础设施：

- 原子操作
- block reduce / warp reduce
- 复数类型支持
- GPU-friendly 容器和范围
- 设备端 assert / print / error

### 移植预研重点

这层是“后端功能完整性检查”的关键。

尤其要重点验证：

- 原子加 / 原子最值 / CAS 是否齐全
- block reduction 有没有高效实现
- 是否支持设备端 `printf`
- 设备端断言机制如何映射
- 复杂数和标准库支持够不够

如果这层缺一块，上层很多算子都能编，但会在运行时 silently wrong。

---

## 3.6 上层调用方与真实压力来源

### 相关文件

- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)
- [`AMReX_MFParallelForC.H`](../AMReX_MFParallelForC.H)
- [`AMReX_MFParallelForG.H`](../AMReX_MFParallelForG.H)
- [`AMReX_TagParallelFor.H`](../AMReX_TagParallelFor.H)
- [`AMReX_CTOParallelForImpl.H`](../AMReX_CTOParallelForImpl.H)
- [`AMReX_Array4.H`](../AMReX_Array4.H)
- [`AMReX_Reduce.H`](../AMReX_Reduce.H)
- [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)
- [`AMReX_FabArrayUtility.H`](../AMReX_FabArrayUtility.H)

### 为什么这些也必须看

因为移植最终不是为了“后端文件能通过编译”，而是为了：

- `ParallelFor` 真能跑
- `Array4` 访问真能跑
- `MFParallelFor` 真能跑
- `Reduce/ParReduce` 真能跑

所以这批文件是验证“后端适配是否真的覆盖到上层真实用法”的观测窗口。

特别建议重点看：

- `ParallelFor(box, lambda)`
- `MFParallelFor`
- `Reduce / ParReduce`

这三类用法，因为它们是最典型也最常用的 GPU 执行入口。

---

## 4. 从调用链看，AMReX GPU 最核心的主线是什么

如果只抓最核心的一条链，可以这样记：

```text
用户层
ParallelFor / MFParallelFor / Reduce / ParReduce

->

launch / reducer 抽象层
AMReX_GpuLaunch*.H / AMReX_Reduce.H / AMReX_ParReduce.H

->

后端控制层
AMReX_GpuDevice.* / AMReX_GpuControl.* / AMReX_GpuUtility.*

->

底层原语层
AMReX_GpuAtomic.H / AMReX_GpuReduce.H / AMReX_GpuMemory.H / allocators

->

芯片运行时 / 编译器 / queue / memory / kernel model
```

对于移植项目，这条链的每一层都要能对上。

---

## 5. 预研阶段最建议优先读的文件顺序

如果目标是“尽快评估可移植性”，建议这个顺序：

1. [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H)
   先看编译模型入口和限定符宏。
2. [`AMReX_GpuLaunch.H`](../AMReX_GpuLaunch.H)
   看 launch 主接口。
3. [`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)
   看 GPU 路径的真实执行模板。
4. [`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)
   看 CPU fallback 模型。
5. [`AMReX_GpuDevice.H`](../AMReX_GpuDevice.H) + [`AMReX_GpuDevice.cpp`](../AMReX_GpuDevice.cpp)
   看 runtime 控制面。
6. [`AMReX_GpuMemory.H`](../AMReX_GpuMemory.H) + [`AMReX_GpuAllocators.H`](../AMReX_GpuAllocators.H)
   看内存模型。
7. [`AMReX_GpuAtomic.H`](../AMReX_GpuAtomic.H) + [`AMReX_GpuReduce.H`](../AMReX_GpuReduce.H)
   看并行原语能力。
8. [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H) + [`AMReX_Reduce.H`](../AMReX_Reduce.H)
   回到上层真实调用方。

这个顺序的好处是：

- 先看“能不能编”
- 再看“能不能 launch”
- 再看“能不能跑稳”
- 最后看“能不能覆盖真实使用场景”

---

## 6. 对这份清单，建议额外补充的几点

## 6.1 `MFParallelFor` 相关文件值得单列

如果的目标是芯片移植，这几份文件值得从“上层真实调用者”角度单列出来：

- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)
- [`AMReX_MFParallelForC.H`](../AMReX_MFParallelForC.H)
- [`AMReX_MFParallelForG.H`](../AMReX_MFParallelForG.H)

因为很多应用真正大量使用的是它们，而不是直接手写底层 launch。

## 6.2 `Reduce/ParReduce` 也值得列进 GPU 调用图谱

虽然它们不一定被直觉归到“GPU launch 文件”，但从真实执行路径看，它们就是 GPU 规约的重要调用方：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H)
- [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)
- [`AMReX_ParallelReduce.H`](../AMReX_ParallelReduce.H)

尤其如果目标后端自研芯片的 block reduce / atomic / tuple reduce 支持不完整，这一层会很快暴露问题。

## 6.3 `Arena` 体系最好尽早拉进来

如果后端内存模型和 CUDA 不完全等价，那么：

- [`AMReX_Arena.H`](../AMReX_Arena.H)
- [`AMReX_Arena.cpp`](../AMReX_Arena.cpp)

最好尽早看。

因为 AMReX 很多 GPU 内存路径最终都和 Arena 体系发生关系。

---

## 7. 预研阶段最值得先回答的问题

移植预研阶段可优先回答以下问题：

1. 目标后端的编译器是否支持 AMReX 当前依赖的 device lambda / host-device 函数模型？
2. 目标后端是否有 CUDA-like kernel launch 抽象，还是需要重做 launch backend？
3. 目标后端的 queue/stream 模型是否能映射到 AMReX 的 stream 语义？
4. 目标后端的内存类型是否能支撑 pinned / async / device alloc 这些路径？
5. 原子操作集合是否齐全？
6. block reduction / warp reduction 是否有可用实现？
7. 设备端调试能力是否足够支持 assert / print / error reporting？
8. graph capture 是否必须支持，还是预研阶段可以先屏蔽？
9. CPU fallback 路径能否作为初期保底路径共存？
10. 上层 `ParallelFor` / `MFParallelFor` / `Reduce` 的典型 case 能不能先跑通一个最小子集？

这 10 个问题，基本决定了目标后端后面的移植路线会是：

- 轻量适配
- 中等重构
- 还是需要新后端层

---

## 8. 当前预研工作建议

如果是预研阶段，建议不要一开始就试图“全量支持 AMReX GPU 层”。

更务实的路线是：

1. 先让限定符和最小 kernel launch 走通
2. 先让最小 `ParallelFor` 跑通
3. 再让最小 `Array4` + `MFParallelFor` 跑通
4. 再让最小 `Reduce/ParReduce` 跑通
5. 最后再补内存优化、graph capture、复杂容器等高级能力

也就是：

```text
先建立可执行最小闭环
-> 再扩能力
-> 再补性能和高级特性
```

这通常比从文件列表逐个啃更有效。

---

## 9. 结论

### 的清单

- 大方向是准确的
- 已经覆盖了 Base 层 GPU 支撑体系的大多数核心文件

### 建议额外补充

- `MFParallelFor*`
- `Reduce / ParReduce / ParallelReduce`
- `Arena`
- `Array4` / `FabArray` / `MFIter` 这些真实调用方

### 预研阶段最应该先抓住的主线

```text
Qualifiers
-> Launch
-> Device/Stream Control
-> Memory/Arena
-> Atomic/Reduce
-> ParallelFor / MFParallelFor / Reduce 的真实调用方
```




