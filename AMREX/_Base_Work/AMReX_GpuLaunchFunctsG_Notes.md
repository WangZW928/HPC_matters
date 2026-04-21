# AMReX_GpuLaunchFunctsG.H 移植笔记

对应源码：[`AMReX_GpuLaunchFunctsG.H`](../AMReX_GpuLaunchFunctsG.H)

## 1. 文件职责

该文件是 GPU 路径下 `ParallelFor` 的核心模板执行层，负责：

- `ParallelFor(...)` 重载分发（`N`、`Box`、`Box+ncomp`、多 lambda）。
- 索引线性化与反线性化（`icell -> (i,j,k[,n])`）。
- 调用 `AMREX_LAUNCH_KERNEL` 触发 GPU 发射。
- 大规模 `N` 场景的分段发射。

## 2. 必查项

- device lambda 是否可被完整推导与转发。
- `Box` 与 `ncomp` 路径是否全部可编译。
- 线性化逻辑是否与 `Array4` 布局一致。
- 分段发射路径在大规模场景下是否正确。

## 3. 常见误判

- 仅验证 `ParallelFor(N, ...)`，未覆盖 `Box` 与 `ncomp`。
- 仅验证小规模，未覆盖分段发射路径。
- 编译通过但索引映射错误，导致数值偏差。

## 4. 最小验证建议

1. `ParallelFor(N, ...)`：线性数组写入与校验。
2. `ParallelFor(box, ...)`：2D/3D 网格写入与校验。
3. `ParallelFor(box, ncomp, ...)`：component 维校验。
4. 超大 `N`：分段发射累计结果校验。
