# AMReX_GpuMemory.H + AMReX_GpuAllocators.H 移植笔记

对应源码：

- [`AMReX_GpuMemory.H`](../AMReX_GpuMemory.H)
- [`AMReX_GpuAllocators.H`](../AMReX_GpuAllocators.H)

## 1. 文件职责

该组文件定义 GPU 内存分配与分配器适配层，主要覆盖：

- device/managed/pinned 等分配路径。
- allocator 与容器/上层对象的衔接。
- 与 Arena 体系的协同。

## 2. 必查项

- 分配/释放接口与后端 runtime 的一致性。
- async copy 与 stream 语义匹配性。
- managed 机制可用性与降级策略。
- 内存对齐策略是否满足 kernel 访问要求。

## 3. 高风险问题

- 生命周期错位（释放早于异步任务完成）。
- managed 语义不完整导致正确性或性能异常。
- 对齐策略不当导致随机数值问题。

## 4. 最小验证建议

1. alloc -> kernel write -> copy back -> free。
2. async copy + delayed sync 时序验证。
3. 随机大小分配压力测试（碎片与泄漏观察）。
