# AMReX P5-003 C1 identity adapter 增量报告

_AVWiS 曲线坐标移植 G0.2 · 2026-08-31_

## 结论

P5-003/C1 已完成：AVWiSSolver 在自身的 `BoxArray/DistributionMapping` 上构造并只读持有 identity `MetricData`，并将现有 integrated `Ucont` divergence 路径的 cell-volume 来源切换到一个窄 identity metric adapter。

这只关闭 C1/G0.2 子模块，不关闭 P5-003、G0 总门，也不代表 AVWiS 已支持一般曲线坐标。

## 实施范围

- 新增 `AVWiSMetricAdapter.H/.cpp`，只接受 identity mapping，检查 metric epoch、cell/face `BoxArray`、`DistributionMapping`、组件和布局
- `AVWiSSolver` 增加只读 `MetricData`/epoch 访问，并在构造阶段建立与 solver 完全一致的 identity metric
- `AVWiSProjection.cpp` 的散度路径使用 `MetricData::cell_volume_cc()`；`Ucont` 仍是已积分的面体积通量，不重复乘面积
- 新增 `tests/curvilinear/MetricIdentityAdapterContract.cpp`，覆盖单 Box、多 Box、常速度零散度、非零仿射通量、Cartesian reference 和 stale epoch 拒绝

未实施：解析曲线 mapping、弱/一般非正交算子、曲线对流/粘性、19 点压力、曲线 BC/ghost、曲线 checkpoint、MPI/GPU runtime 和物理曲线 case。

## 验证

环境为锁定 AMReX 26.04、CPU double、MPI/CUDA OFF。实际结果：

| 验证项 | 结果 |
| --- | --- |
| clean configure/build：`build/amrex_port_p5c1_final` | PASS |
| focused identity metric contract | PASS |
| focused identity adapter contract | PASS |
| static contract | PASS |
| 全量 CTest | 28/28 PASS |
| `avwis_lid_driven_cavity_sanity` | 1/1 PASS |
| `git diff --check` | PASS |
| MPI/CUDA runtime | NOT TESTED；所用 AMReX 包为 MPI/CUDA OFF |

主要命令：

```bash
cmake --build build/amrex_port_p5c1_final -j2
bash amrex_port/tests/static_contract_check.sh
ctest --test-dir build/amrex_port_p5c1_final --output-on-failure
ctest --test-dir build/amrex_port_p5c1_final -R avwis_lid_driven_cavity_sanity --output-on-failure
git diff --check
```

## 影响与下一步

identity 下现有 Cartesian 散度结果保持紧容差一致；方腔、对流、粘性、压力投影、checkpoint 和输出 schema 未改变。下一步为 C2/G1：实现参数化解析正交 mapping，并在三网格 MMS、GCL、free-stream、投影、多 Box、restart 和 MPI 证据下验收；不得把 C1 结果外推为曲线坐标支持。

## 依据

- [AVWiS 曲线坐标实现规格](./AVWiS曲线坐标实现规格.md)
- [AMReX 迁移方案](./AMReX迁移方案.md#当前状态剩余工作与下一步路线)
- [AMReX 移植任务清单](./AMReX移植任务清单.md)
- [P5-003 G0.1 报告](./AMReX_P5-003_G0_identity_metric_20260831.md)
