# AMReX_GpuLaunchFunctsC.H 移植笔记

对应源码：[`AMReX_GpuLaunchFunctsC.H`](../AMReX_GpuLaunchFunctsC.H)

## 1. 文件职责

该文件提供 CPU fallback 路径的 `ParallelFor` 模板实现，主要用于：

- GPU 不可用时保持统一接口可运行。
- 在 CPU 侧展开 `ParallelFor` 的各类重载。
- 与 OpenMP/SIMD 路径协同执行。

## 2. 移植阶段关注原因

即使 GPU 为主要目标，CPU fallback 仍是必要基线：

- 作为调试保底路径。
- 作为 CPU/GPU 结果对照参考。
- 用于验证接口语义在双路径上的一致性。

## 3. 必查项

- 与 `AMReX_GpuLaunchFunctsG.H` 同名重载的语义一致性。
- `Box`、`ncomp`、多 lambda 路径覆盖完整性。
- OpenMP 打开/关闭时结果一致性与稳定性。

## 4. 最小验证建议

1. 同一输入分别运行 CPU fallback 与 GPU 路径。
2. 对比 `sum/max/L2` 等统计量。
3. 固定输入与随机种子，重复运行确认确定性行为。
