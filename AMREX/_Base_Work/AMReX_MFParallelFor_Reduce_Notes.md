# AMReX_MFParallelFor.H + AMReX_Reduce.H 移植笔记

对应源码：

- [`AMReX_MFParallelFor.H`](../AMReX_MFParallelFor.H)
- [`AMReX_Reduce.H`](../AMReX_Reduce.H)

## 1. 文件职责

该层属于上层真实调用入口：

- `MFParallelFor`：面向 `MultiFab/FabArray` 的并行遍历。
- `Reduce`：本地规约与多通道规约抽象（sum/min/max/tuple）。

## 2. 为什么作为验收层

`Qualifiers/Launch/Memory/Atomic` 通过后，仍需由该层确认真实业务路径可用。该层通过通常才代表上层调用可落地。

## 3. 必查项

- `Array4` 访问与 `Box` 索引映射一致性。
- `MFParallelFor` 在 GPU 路径的稳定发射能力。
- `Reduce` 多通道规约结果正确性。
- CPU/GPU 对照结果在容许误差内一致。

## 4. 最小验证建议

1. `MFParallelFor` 写入 `MultiFab` 标量场并校验。
2. `Reduce` 计算 `sum + max` 双通道并对照 CPU。
3. 组合用例：`MFParallelFor` 产出后执行 `Reduce` 汇总。
