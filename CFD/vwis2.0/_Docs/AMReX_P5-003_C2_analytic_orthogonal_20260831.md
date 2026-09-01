# AMReX P5-003 C2 analytic orthogonal 增量报告

_C2.1/G1 geometry 基线日期 2026-08-31；验收执行 2026-09-01_

---

## 📋 结论

P5-003/C2.1 已完成：生产 `CoordinateMapping` 增加参数化 separable analytic
orthogonal provider，现有 `MetricData::build()` 可直接构造并验证其正 Jacobian、离散
体积、共享有向面面积、cell cofactor/逆 metric 和 face gradient metric。独立 G1
geometry contract 已覆盖 identity 极限、强拉伸、多 Box/ghost/owner、GCL、互反、
constant-velocity geometric divergence 与三网格 `Jx` 二阶收敛。

P5-003/C2 与 G1 总门仍为进行中。C2.2 的 metric-aware gradient/divergence、
`Ucat↔Ucont` mapped 变换和 orthogonal projection 尚未实现；analytic mapping 未接入
`AVWiSSolver` 配置，也未进入现有 Cartesian pressure/advection/viscous/BC/time 路径。

## 🎯 实施范围

每个方向使用同一种一维生产映射：

$$
x_m=\xi_{m,lo}+s_mL_m\left[t_m+
\frac{a_m}{2\pi}\sin(2\pi t_m)\right],\qquad
t_m=\frac{\xi_m-\xi_{m,lo}}{L_m}.
$$

其导数为

$$
\frac{\partial x_m}{\partial\xi_m}
=s_m\left[1+a_m\cos(2\pi t_m)\right].
$$

生产参数接口要求有限 `scale=s_m>0`，并以 roundoff margin 强制
`abs(stretch=a_m)<1`，因此三个方向单调且 $J_x>0$。`scale=1,stretch=0` 精确退化到
C0 identity。映射公式和 GPU-safe evaluator 位于生产 `src`，测试只调用生产接口，
不持有 test-only mapping 实现。

严格 factory 仅接受 `identity` 与 `analytic_orthogonal`；未知类型、NaN/Inf、非正
尺度及可能使导数归零的拉伸参数均在构造期拒绝。`fill_nodes()` 的 device lambda
只捕获标量和 `Array4`，不捕获 host 多态对象。

```mermaid
flowchart LR
    accTitle: C2 Increment Boundary
    accDescr: C2.1 completes the analytic orthogonal geometry provider and contracts, while C2.2 remains responsible for mapped operators and solver integration.

    c0([✅ C0 identity metric]) --> c1([✅ C1 identity adapter])
    c1 --> c21([✅ C2.1 geometry])
    c21 --> c22[⚙️ C2.2 mapped operators]
    c22 --> g1([🏁 G1 acceptance])

    classDef complete fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef pending fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f

    class c0,c1,c21 complete
    class c22,g1 pending
```

## 📊 Contract 覆盖

| 契约 | C2.1 证据 |
| --- | --- |
| Identity 极限 | analytic 默认参数与 C0 的 node/cell/face 全字段紧容差比较 |
| 强拉伸正几何 | `stretch=(0.94,-0.82,0.67)`、`scale=(1.4,0.75,1.2)`；`min(Jx)>0`、`min(V)>0`、有向对角面面积为正 |
| Orthogonality | cell `grad_xi_cc/area_cofactor_cc` 与 face `S/Q` 的非对角分量为 roundoff |
| Reciprocity/GCL | 复用 `MetricData::validate()` 的 reciprocal、cofactor reciprocity 与逐 cell 面闭合 |
| Free stream | 常速度构造 `Ucont=u·S`，净积分面通量除 `Vc` 的几何散度在容差内为零 |
| Refinement | `N=12,24,48` 的 cell-centered `Jx` 对解析导数；$L_\infty$ 阶数不低于 1.8 |
| 解析精确量 | separable face area 与同节点多面体 volume 对生产 evaluator 的离散解析值达到 roundoff |
| Layout | 单 Box、many Box、node ghost 解析延拓、共享 face `OwnerMask` 唯一计数 |
| 严格拒绝 | 零/负/Inf scale，`stretch=±1`/NaN，不可表示的 Jacobian 上下界，identity 携带 analytic 参数，未知 mapping type |

## 🧪 验收结果

环境为 locked AMReX 26.04、Release、CPU double、MPI/CUDA OFF；正式验收使用全新
`build/amrex_port_p5c2_final` 目录。没有 runtime 证据的后端不记作通过。

| 验证项 | 结果 |
| --- | --- |
| `bash amrex_port/tests/static_contract_check.sh` | PASS |
| clean configure/build | PASS；locked 26.04，全目标构建完成 |
| focused C0/C1/G1 | 3/3 PASS |
| G1 三网格 `Jx` | $L_\infty$ error = 0.1053657896, 0.02783728089, 0.007055648417；order = 1.920316322, 1.980167719 |
| 完整 CTest | 29/29 PASS |
| `avwis_lid_driven_cavity_sanity` | 1/1 PASS |
| `git diff --check` | PASS |
| MPI runtime | NOT TESTED；验收包 MPI OFF |
| CUDA runtime | NOT TESTED；验收包 CUDA OFF |

## ⚠️ 能力边界与下一步

C2.1 只证明 analytic orthogonal mapping 和 `MetricData` 几何语义。它不证明 mapped
scalar gradient、动量散度、压力 operator/correction、曲线物理 BC、restart provenance
或时间推进，也不包含弱/一般非正交 mapping 和 19 点压力项。

C2.2 应增加 metric-aware gradient/divergence 与只使用对角 `Q` 的 orthogonal
projection，并用常量/线性场、三网格 MMS、投影散度下降及 identity 紧容差回归验收。
只有这些路径完整后才可增加显式 `avwis.mapping.type=identity|analytic_orthogonal`；默认
必须保持 identity，analytic 模式不得落入未改造的 Cartesian 算子。

## 🔗 依据

- [AVWiS 曲线坐标实现规格](./AVWiS曲线坐标实现规格.md)
- [AMReX 迁移方案](./AMReX迁移方案.md#当前状态剩余工作与下一步路线)
- [AMReX 移植任务清单](./AMReX移植任务清单.md)
- [P5-003 C1 identity adapter 报告](./AMReX_P5-003_C1_identity_adapter_20260831.md)
