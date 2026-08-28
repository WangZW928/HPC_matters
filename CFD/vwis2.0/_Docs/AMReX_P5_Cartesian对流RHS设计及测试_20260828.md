# AMReX P5-001 Cartesian 对流 RHS 设计及测试（2026-08-28）

## 范围与结论

本增量只实现单层、均匀 Cartesian 的保守对流 RHS。它复用 P2 的
`Ucat/Ucont` 数据契约、face ownership，以及 P3 的 halo/物理 ghost 流程；没有
实现粘性、曲线 metric/19 点算子、时间推进、LES 或 IBM。在 P5-001 本次验收时
`advance_one_step()` 仍为 no-op；其后新增的显式时间基线见
`AMReX_P5-004_时间推进设计及测试_20260828.md`，不反向扩大本报告的 P5-001 结论。

CPU 使用锁定的 AMReX 26.04、SHA
`9219ba416b7ba2073dd1b12bf19fdce27391f17b`，clean configure/build 成功，完整
CTest 13/13 PASS。P5-001 因 MPI/CUDA runtime 与 CFD case 尚缺，在主清单中记为
“进行中”。

## 离散与 legacy 对照

对方向 `d` 的积分体积通量和速度分量 `m`，代码计算

```text
F[d,m]_face = Ucont[d]_face * (Ucat[L,m] + Ucat[R,m]) / 2
RHS_adv[m] = -sum_d(F[d,m]_hi - F[d,m]_lo) / cell_volume
```

旧 `RHSSolver.C` 的 `-second_order` 分支在面上存
`-ucont * 0.5*(ucat_L+ucat_R)`，随后取高低面差；上式把负号放在最终散度外，
Cartesian 结果可比。`Ucont` 仍为速度乘面面积，故只除一次 cell volume。
这不是旧曲线 metric、IBM 邻域分支或四阶分支的等价实现。

kernel 在 `OverrideSync`/`FillBoundary` 及所需物理 ghost 后读取 stencil。每个 cell
只写一次调用方提供的三分量 cell `MultiFab`，没有共享 face 写竞争，也没有 host
容器被 device lambda 捕获。

## 回归结果

周期制造场取 `U=(0.75,sin(2*pi*x),0)`，解析对流 RHS 为
`(0,-0.75*2*pi*cos(2*pi*x),0)`。粗细网格均为多 Box：

|x cells|dx|离散 stencil L∞ 误差|连续解 L∞ 误差|
|---:|---:|---:|---:|
|16|0.0625|3.55e-15|1.178784173e-1|
|32|0.03125|7.55e-15|3.007572841e-2|

连续误差比为 3.919，符合二阶中心格式；常量 x/z 被输运分量 RHS 为 roundoff
零。非周期测试使用 x-low uniform inflow、x-high constrained outflow、其余 no-slip
wall，`max_grid_size=4`；常量单位贯流的 RHS max error 为 0，同时经过 P3 physical
ghost/边界 face 流程。

验证命令与结果：

```text
bash amrex_port/tests/static_contract_check.sh                 PASS
cmake configure/build (Release, locked AMReX 26.04 CPU)        PASS
ctest --test-dir /tmp/vwis-p5-port-build.aFeX8i --output-on-failure
                                                               PASS 13/13
```

未运行 MPI/CUDA；因此没有把单 rank 多 Box 推断成多 rank/GPU 证据。
