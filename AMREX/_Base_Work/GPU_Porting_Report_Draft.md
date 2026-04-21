# GPU 可移植性调研报告草稿（20 天）

该文档为报告主草稿，不绑定单一源码文件，用于持续补充调研证据与结论。

## 1. 目标与范围

- 目标：评估 AMReX 在目标自研 GPU 架构上的可移植性、风险和落地路径。
- 范围：以 `Src/Base` 为主，覆盖编译模型、launch、内存、原语、上层调用。
- 非范围（当前阶段后置）：
  - 全量性能优化
  - 图捕获高级能力全覆盖
  - 长尾非关键子模块兼容

## 2. 总体结论框架（待填）

- 可移植性等级：`可用 / 部分可用 / 需重构`
- 当前最强证据：
- 当前主要阻塞：
- 预估落地路径：

## 3. 关键调研维度

### 3.1 编译模型与限定符

- 关注点：
  - `AMREX_GPU_HOST_DEVICE` / `AMREX_GPU_GLOBAL` 映射正确性
  - `AMREX_IF_ON_DEVICE/HOST` 分支行为
  - `AMREX_DEVICE_COMPILE` 判定稳定性
- 风险：host/device 语义错位导致模板分支误判
- 当前状态（待填）：
- 证据（待填）：

### 3.2 Kernel Launch 与执行模型

- 关注点：
  - `ParallelFor` 到 launch wrapper 的闭环
  - 单 lambda 与多 lambda wrapper（variadic launch）覆盖
  - device lambda 与编译选项依赖
  - grid/block/stream 参数 ABI 对齐
- 风险：可编译但运行路径不完整，或参数语义不一致
- 当前状态（待填）：
- 证据（待填）：

#### 多 lambda wrapper 专项（建议必测）

- 风险原因：单 lambda 通过不代表 variadic wrapper 可用。
- 建议样例：
  1. 单 lambda launch（基线）
  2. 双 lambda launch（同一 kernel 顺序执行）
  3. 三 lambda launch（混合只读/写入捕获）
- 记录指标：编译、运行、CPU 对照、额外编译选项依赖。

### 3.3 内存模型与生命周期

- 关注点：alloc/free、copy/sync、async 时序，managed/pinned 可用性，Arena/AsyncArray/Elixir 一致性
- 风险：异步释放导致不稳定错误
- 当前状态（待填）：
- 证据（待填）：

### 3.4 原子与规约原语

- 关注点：atomic add/min/max/CAS 完备性，block/warp reduction 一致性，tuple reduction 正确性
- 风险：silently wrong（结果错误但不中断）
- 当前状态（待填）：
- 证据（待填）：

### 3.5 上层真实调用覆盖

- 关注点：`ParallelFor` / `MFParallelFor` / `Reduce/ParReduce`，`Array4` 线性化一致性
- 风险：仅底层 demo 通过，真实路径不可用
- 当前状态（待填）：
- 证据（待填）：

### 3.6 诊断能力与可维护性

- 关注点：device assert/print/error 可观测性，故障回传可定位性
- 风险：定位效率低，调试成本高
- 当前状态（待填）：
- 证据（待填）：

## 4. 20 天执行节奏

### D1-D5：编译模型与最小 launch

- 输出：宏契约映射清单，最小 `ParallelFor` 可编可跑证据

### D6-D12：内存与原语

- 输出：alloc/copy/sync 生命周期测试，atomic/reduce 对照结果

### D13-D20：上层覆盖与风险收敛

- 输出：`MFParallelFor` / `Reduce` 路径结果，阻塞项分类，最终等级结论与后续计划

## 5. 风险分级模板

- 风险标题：
- 严重度：`高 / 中 / 低`
- 触发条件：
- 影响范围：
- 当前规避方案：
- 根因修复建议：
- 预计修复成本：

## 6. 证据索引（待持续补充）

- 构建日志：
- 运行日志：
- 正确性对照：
- 性能观测：
- 代码位置：

## 7. 相关笔记入口

- [`AMReX_GpuQualifiers_Notes.md`](./AMReX_GpuQualifiers_Notes.md)
- [`GPU_Launch_Chain_Overview.md`](./GPU_Launch_Chain_Overview.md)
- [`GPU_Memory_and_Lifecycle.md`](./GPU_Memory_and_Lifecycle.md)
- [`Minimal_Porting_Validation_Subset.md`](./Minimal_Porting_Validation_Subset.md)
- [`AMReX_GPU_Porting_Overview.md`](./AMReX_GPU_Porting_Overview.md)
