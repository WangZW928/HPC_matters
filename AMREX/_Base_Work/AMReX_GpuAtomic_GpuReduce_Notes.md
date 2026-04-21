# AMReX_GpuAtomic.H + AMReX_GpuReduce.H 移植笔记

对应源码：

- [`AMReX_GpuAtomic.H`](../AMReX_GpuAtomic.H)
- [`AMReX_GpuReduce.H`](../AMReX_GpuReduce.H)

## 1. 文件职责

该层属于 GPU 并行原语层，主要包括：

- 原子操作（add/min/max/CAS 等）。
- 设备侧规约（block/warp 级）。

该层是 `Reduce`、统计算子、残差计算等上层能力的基础。

## 2. 必查项

- 原子操作集合是否完整。
- 浮点与整数原子语义是否正确。
- 不同 `block size` 下规约结果是否稳定。
- tuple/多通道规约是否可用（若上层依赖）。

## 3. 高风险问题

- 可运行但结果偏差（最常见风险）。
- 不同优化级别下行为不一致。
- 非整 block 的边界规模下规约错误。

## 4. 最小验证建议

1. 固定输入的 atomic 累加与 CPU 对照。
2. `min/max/sum` 三类规约全覆盖。
3. 多次重复运行验证稳定性。
