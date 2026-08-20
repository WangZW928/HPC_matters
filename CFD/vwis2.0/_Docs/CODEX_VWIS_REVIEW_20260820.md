# vwis2.0 源码—算法—数据流—AMReX 审阅报告

审阅日期：2026-08-20。范围为仓库根目录下 `vwis2.0/` 与本目录；结论分为“源码已确认”“合理推断”“待验证”。本报告不修改源码，也不把历史 VFS 手册能力当作当前树能力。

## 1. 覆盖与结论摘要

- 源码文件：39/39 已递归审阅，约 26,630 行；包括 17 个 `.C`、2 个 `.c`、18 个 `.h`、1 个 `.py` 与 1 个 `makefile`（按仓库实际文件计数）。
- 原有 Markdown：8/8 已审阅并修改；本报告为新增第 9 个 Markdown。没有“已审阅但未修改”的原有文档。
- 当前实现可确认：静态结构化曲线网格、PETSc DMDA ghost/MPI、单相不可压缩投影、LES、sharp-interface 三角面 IBM、可选分区 FSI、PETSc binary/HDF5 Vec I/O。
- 未确认/未实现于本树：level set、两相 rho/mu、表面张力、风机 actuator、AMR、AMReX、GPU kernel、以 CFL 自动改写时间步。原文涉及这些的部分均已标记为手册背景或迁移建议。

## 2. 可追溯实现图

```text
main.C
  CurvGrid::ReadGrid/ReadBC/FormMetrics -> DMDA + metric/Jacobian
  UData::InitializeData                -> global/local Vec（Ucont/Ucat/P/Nvert）
  IBMRead/Search/Interpolation          -> Nvert + IBMInfo
  FlowSolver::Solve
    RHSSolver::Solve + LES/WallModel    -> Rhs/Rhs_o
    Integrator::Solve/SolveFunction     -> SNES, predicted Ucont
    PoissonSolver::Solve                -> HYPRE Phi2 -> Phi -> P, Projection(Ucont)
    BcsUtility::IbBC + UData::Contra2Cart -> boundary/IBM and Ucat
  UData::CopyLastStep/WriteData; Probe/PlaneExtraction
```

数据布局（源码已确认）：`CurvGrid` 建立标量 `da` 与三分量 `fda`；`UData::InitializeData` 以 `DMCreateGlobalVector/DMCreateLocalVector` 分配 `Ucont/Ucat` 与 `lUcont/lUcat`，标量 `P/Nvert` 同理。`Cmpnts***` 是 `DMDAVecGetArray` 的 host 视图，i/j/k 邻域依赖依靠 `DMGlobalToLocalBegin/End` 填 ghost。`Ucont` 被用作三个方向的离散通量，但它不是三个独立、类型化 face DM。

## 3. 公式核验与实现映射

### 3.1 连续性、投影与守恒

源码诊断采用

$$
D_{ijk}(U)=A_{j,ijk}[(U^1_{ijk}-U^1_{i-1,j,k})+(U^2_{ijk}-U^2_{i,j-1,k})+(U^3_{ijk}-U^3_{i,j,k-1})].
$$

证据是 `FlowSolver::CalculateDivergence`。`PoissonSolver::PoissonRHS2_hypre` 对可解流体点组装同一通量差（周期端点另行回绕），乘 $-Stc_t/\Delta t$；`PoissonLHS` 用曲线 metric 组装最多 19 个非零元；`Projection` 再改写 `Ucont`，`UpdatePressure` 令 $P\leftarrow P+\Phi$。故可安全写成

$$
L_g\Phi=-\frac{Stc_t}{\Delta t}D(U^*),\qquad
U^{n+1}=U^*-\mathcal G_g\Phi,
$$

并以 $D(U^{n+1})\approx0$ 为目标。$L_g$ 和 $\mathcal G_g$ 不能未经边界行/metric 逐项比较而替代为 Cartesian 7 点 Laplacian。更重要的是，散度诊断主动跳过外边界和 `Nvert` 邻域，不能单独证明 IBM 邻域守恒。

### 3.2 动量、时间推进、LES 与稳定性

`RHSSolver::Solve` 从局部 `Ucont/Ucat/metric/Nvert` 构建对流和粘性项；`-second_order` 影响对流插值。`Integrator::SolveFunction` 的 `timeCoeff≈1` 分支含一阶时间差分和新旧 RHS 各半，另一分支含

$$
\frac{-1.5U^n+2U^{n-1}-0.5U^{n-2}}{\Delta t},
$$

并把相应 RHS 缩放 $1/1.5$。残差由 SNES 解到其配置容差；这不是独立的线性 CFL 证明。`CalculateMinimumDt` 计算 $|U^q A_j|^{-1}$ 量级与最小网格尺度后只保存/输出，`main.C` 从 options 读取固定 `dt`；因此稳定性条件是运行前验算，不是代码强制。

`LESModel::ComputeSmagorinksyConstant/ComputeEddyViscosity` 是 SGS 路径。可由代码支持的闭合形式为 $\nu_t=C_s\Delta^2|S|$，其中体积尺度来自 Jacobian；具体滤波、裁剪和 IBM 例外应以所用 options 分支复查。没有 rho/phase 字段，所以不可从本树推出变密度动量、level set 或表面张力方程。

### 3.3 边界、IBM、FSI 和线性可解性

物理边界由 `BcsUtility::FormBcs/InitializeFlowField/IbBC` 与 `Integrator::SolveFunction` 的面通量置零共同完成；类型宏位于 `BcsUtility.h`，但还有算例特定整数分支，必须用实际 `bcs.dat/control.dat` 复核。IBM 从 `ImmersedBoundary::IBMRead/IBMSearchAdvanced/IBMInterpolationAdvanced` 的三角面、分桶、相交、最近面与 image/interpolation 路径进入 `Nvert/IBMInfo`；它是 sharp-interface 插值法，不是连续 delta 体力 IBM。

`PoissonSolver::RemoveNullspace` 对不是特殊 `BC(3)==-10` 的情况移除 RHS 均值；压力零空间和出口参考仍需算例检查。HYPRE 是 IJMatrix/IJVector/ParCSR，默认 GMRES + BoomerAMG、宏 `PCG_POISSON` 时 PCG。`StructSolver::CheckConvergence` 的“差大于容差即 converge”以及旋转支路容差赋值是静态风险；`-str>1` 只能称有限次分区迭代，不能称已验证强耦合。

## 4. 源码覆盖清单与证据

| 组 | 已审阅文件 | 关键证据 |
|---|---|---|
| 驱动/网格/字段 | `main.C`, `CurvGrid.C/.h`, `UData.C/.h`, `makefile`, `data.py` | 生命周期、DMDA、metric、I/O |
| 流场/求解器 | `FlowSolver.C/.h`, `RHSSolver.C/.h`, `Integrator.C/.h`, `PoissonSolver.C/.h` | RHS、SNES、HYPRE、投影、散度 |
| 物理/边界 | `BcsUtility.C/.h`, `LESModel.C/.h`, `WallModel.C/.h`, `WallFunctions.C/.h` | BC、LES、近壁通量 |
| IBM/FSI | `ImmersedBoundary.C/.h`, `FSI.C/.h`, `StructSolver.C/.h`, `ibm_functions.c/.h` | 几何、插值、力、运动、收敛 |
| 工具 | `functions.c/.h`, `PointProbe.C/.h`, `PlaneExtraction.C/.h`, `Timer.C/.h` | metric/导数、归约、探针、抽面、计时 |

静态风险（未改源码）：`PoissonRHS2_hypre` 的 `JAj=getlKAj()`；`WallFunctions::utau_wf` 光滑分支无返回；`StructSolver::CheckConvergence` 逻辑；以及 Poisson 对移动 IBM 的重组装分支被注释。需要编译告警、单元测试和实际算例确认影响范围。

## 5. AMReX 可行性矩阵

| 类别 | 可直接复用 | 重写/适配层 | 风险与验收 |
|---|---|---|---|
| `Geometry/BoxArray/DistributionMapping/MultiFab` | 字段命名、metric 输入、`Ucont/Ucat/P/Nvert` 语义 | 建立 level 数据；`Ucont` 拆为三个 face `MultiFab`，`Ucat/P/Nvert` 为 cell `MultiFab` | 多 Box/AMR 不能照搬 DMDA 索引；制造解验证 |
| `MFIter`/ghost/BC | 各边界物理意图 | 用 `MFIter`/`Array4` 重写 stencil；`FillBoundary/FillPatch` 和 BC functor | AMReX 不自动填 VFS 例外 ghost；逐面通量回归 |
| Poisson | 投影语义、HYPRE/PETSc 经验 | Cartesian 用 `MLMG/MacProjector`；曲线 19 点做自定义 `MLLinOp` 或临时 PETSc/Hypre bridge | 非正交交叉项、零空间和可解性；比较 Phi/U/散度 |
| EB/IBM/AMR | 三角面、`Nvert`、力积分测试 | 在 EB2 或 custom sharp IBM 二选一，提供 `build/classify/apply/force` 接口 | EB 小 cut-cell 与移动体；AMR 跨层插值/双计数 |
| GPU/MPI | 算子数学、诊断 | `ParallelFor` device kernel、连续 IBM 数据、GPU-aware MPI | `MultiFab` 不自动消除 host conversion；profile/缩放 |
| I/O/build/test | 输出字段及物理检查 | AMReX plotfile/checkpoint、CMake、CI 回归 | 重启、regrid 后 metric/IBM mask，MPI 不变性 |

推荐阶段：单层 Cartesian MAC projection → 规则 LES/RHS → 静态 sharp IBM 或 EB（二选一）→ 曲线 metric operator → 移动 IBM/FSI → AMR。保留 PETSc 的合理位置是小规模 FSI 非线性系统或尚未实现的曲线算子；禁止长期把热路径反复转换 `MultiFab <-> PETSc Vec <-> HYPRE`。

## 6. 文档修改摘要

| 文档 | 修改 |
|---|---|
| `00_整体架构.md` | 加实现边界并校正静态风险表述 |
| `01_模块详解.md` | 说明 DMDA/通量语义，标出 Poisson 不重组与 CFL 诊断 |
| `02_数据流与求解流程.md` | 校正时间循环/FSI 语义，补 ghost、MPI、HYPRE 数据流 |
| `03_关键算法.md` | 收紧单相范围，补投影和时间离散可追溯公式 |
| `04_编译与运行.md` | 区分依赖说明和已验证构建，补数值运行清单 |
| `05_可移植性与GPU优化分析.md` | 降级未实测 GPU 断言，增加装配/拷贝风险与基准门槛 |
| `VFS_算法详解.md` | 将两相/level-set 等标为手册背景，替换时间步中的错误实现陈述 |
| `AMReX迁移方案.md` | 增加 AMReX 不自动保证声明与迁移验收矩阵 |

## 7. 后续必须做的数值/性能验证

1. 制造解或 Taylor–Green：确认空间/时间阶、`timeCoeff` 分支与压力投影残差。
2. 通道/腔流与周期盒：确认 BC、零空间和全局质量守恒；IBM 邻域另算净通量。
3. 静态/运动 IBM：无穿透、力/力矩、`Nvert` 改变后的 Poisson 矩阵有效性和 FSI 停止逻辑。
4. 1/多 MPI rank 与重启：字段范数、积分量、统计量和输出一致性。
5. 迁移/GPU：每个阶段比较 L2/L∞、通量、SNES/HYPRE 迭代、时间分解、host-device copy、halo、强弱缩放；没有基准数据不得给性能数字或科学等价性结论。
