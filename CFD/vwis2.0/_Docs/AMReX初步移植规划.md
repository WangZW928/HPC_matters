# AMReX 初步移植规划（证据驱动）

> 范围：这是对 `vwis2.0/` 的增量移植规划和 `amrex_port/` 独立骨架说明，**不替换、也不修改**当前 PETSc/HYPRE 求解器。证据只来自入口和关键调用链，未全量展开源码。状态：AMReX 骨架尚未在本机编译（见验证）。

## 1. 当前框架、模块地图与依赖

当前程序是一个 MPI/PETSc 的曲线结构网格 CFD 求解器。`main.C` 初始化 PETSc，创建所有对象；网格读入、metric 和数据分配先行，之后 IBM/FSI 初始化；每一外层时间步可包含强耦合结构内迭代。共识别 12 个运行模块（4 个基础、8 个复杂/耦合），另有壁面模型、探针和截面提取等辅助模块。

|模块（分类）|职责|源码证据|直接依赖/AMReX 起点|
|---|---|---|---|
|驱动（基础）|参数、对象生命周期、时间/强耦合循环、输出调度|`vwis2.0/main.C:31-211`|`amrex::Initialize/Finalize`、`ParmParse`、C++ solver facade|
|网格与 metric（复杂）|读曲线网格/BC、创建 DMDA、形成 `Csi/Eta/Zet/Aj` 与面 metric|`CurvGrid.C:115-118,270-304,339-1193`|P1 先 `Geometry/BoxArray/DM` Cartesian；曲线算子后置|
|字段管理（基础）|全局/局部 PETSc Vec、历史场、读写、Cartesian 转换|`UData.C:142-190,294-327,530-548,553-572`|cell `MultiFab` 与三方向 face `MultiFab`、`FillBoundary`|
|RHS/LES（复杂）|动量 RHS、压力梯度、metric 导数、Smagorinsky 黏性|`RHSSolver.C:33,58,472,686-1179`; `FlowSolver.C:70-90`|`MFIter` + `ParallelFor`；先 Cartesian stencil 再迁 LES|
|时间推进/SNES（复杂）|隐式自由动量残差和 SNES/KSP 配置|`Integrator.C:41-170,326`|先显式/半隐式可验证步；SNES 语义需单独复现|
|Poisson/HYPRE（复杂）|组装曲线 pressure 矩阵、PETSc↔HYPRE 向量转换、AMG/Krylov、投影|`PoissonSolver.h:29-105`; `.C:469-616,891-1598,2352,2799`|Cartesian 先用 `MLMG`/`MacProjector`；不能照搬 HYPRE 数据路径|
|BC（基础）|入口流量、物理/IBM 边界、初值|`FlowSolver.C:49-66,114-118`; `BcsUtility.C`|`BCRec` + `FillPatch`/物理边界 functor|
|IBM（复杂）|三角面读入、搜索、切点/插值模板、`Nvert` 标记|`ImmersedBoundary.C:87-141,225-682,1006-1485`|先静态 EB2 或独立静态 IBM；sharp IBM 后置|
|FSI（复杂）|结构初始化、重启、强耦合内循环的收敛状态|`main.C:145-159`; `FSI.C:85`、`StructSolver.C`|先定义接口和收敛契约，最后接入|
|I/O/诊断（基础）|restart/场输出、LES 输出、IBM 输出、probe/plane、散度/KE|`main.C:178-189`; `UData.C:432-495`; `FlowSolver.C:110-126`|plotfile/checkpoint + reduction diagnostics|
|壁面模型（复杂）|壁面黏性，且与 LES/IBM 相连|`main.C:75,84-88`; `WallModel.*`|RHS/LES 验证后接入|
|探针/平面（复杂辅助）|采样与截面输出|`main.C:67-70,186-188`|plotfile 后再实现采样接口|

依赖关系（箭头表示运行数据/控制依赖）：

```text
main/PETSc+MPI
  -> CurvGrid(DMDA, metric) -> UData(Vec: Ucont/Ucat/P/Nvert)
  -> IBM + FSI 初始化
  -> FlowSolver: BC -> RHS/LES/wall -> Integrator(SNES) -> Poisson(HYPRE)+Projection
  -> BC/Contra2Cart -> diagnostics + UData/LES/IBM/probe/plane output
```

## 2. 当前调用链与数据流

`main` 的构造顺序为 `CurvGrid → UData → IBM/FSI → BC/Poisson/LES/wall/RHS/Integrator → FlowSolver/StructSolver`（`main.C:46-93`）。启动阶段严格按 `ReadGrid/ReadBC/InitializeVecs/FormMetrics`、`InitializeData`、IBM 搜索/插值、初值/历史场推进执行（`main.C:105-145`）。每个时间步的结构强耦合内循环先 `StructSolver::Solve`，再 `FlowSolver::Solve`（`main.C:154-173`）。

`FlowSolver::Solve` 的实链为：最小步长 → 首步 IBM BC → 入流平面/流量 → 首步 RHS/wall 初始化 → 可选 LES → `RHSSolver::Solve(Rhs_o)`/压力梯度/wall → `Integrator::Solve`（`SNESSolve`）→ `PoissonSolver::Solve`（`SolvePoisson → UpdatePressure → Projection`）→ 散度检查 → BC、局部转全局、`Contra2Cart` 与 KE（`FlowSolver.C:35-126`; `PoissonSolver.C:40-49`; `Integrator.C:168`）。随后主程序复制历史、平均和输出。

数据含义和并行方式如下。

- `CurvGrid::CreateDM` 以 `DMDACreate3d(..., DMDA_STENCIL_BOX, ...)` 建立标量 `da` 和向量 `fda` 的分区；periodic 由 DMDA boundary type 设置（`CurvGrid.C:270-304`）。`UData::InitializeData` 为全局和 local Vec 成对分配（`UData.C:142-165`）。
- local Vec 是显式 ghost 工作区：`DMGlobalToLocalBegin/End` 反复用于 `Ucont/Ucat/Nvert/P`（例如 `UData.C:305-325,541-548`）。AMReX 对应为 `MultiFab` 的 grow cells、`FillBoundary`（同层/MPI/周期）和跨层 `FillPatch`；物理 ghost 仍须 BC 代码填充。
- `Ucont` 是积分面体积通量 $\mathbf u\cdot\mathbf A_f$，`Ucat` 是物理
  Cartesian 速度；证据是 `FormMetrics` 的面积叉积、`Contra2Cart` 的线性方程
  及 `Aj × 面差` 散度（`UData.C:553-572,704-725`）。AMReX 将 `Ucont` 拆为
  x/y/z 三个 face-centred `MultiFab`；需要面速度的 API 必须显式除以面积。
  `P/Phi/Nvert` 为 cell-centred `MultiFab`。
- 曲线连续性诊断在 `FlowSolver.C:133-...` 使用 metric/Jacobian。其最小形式为

  `D(U)=A_j[(U^1_i-U^1_{i-1})+(U^2_j-U^2_{j-1})+(U^3_k-U^3_{k-1})]`。

  若括号内原始净通量记为 `R(U)`，legacy RHS 实际写入
  `+(St*c_t/dt) R(U*)`（不乘 `Aj`），Projection 再按源码系数修正通量。
  符号、`St/timeCoeff` 缩放和边界行必须在 P4 逐项验证；`L_g` 的 19 点曲线
  stencil **不能**直接用 Cartesian `MLMG` 替换。

## 3. AMReX 对应物与边界

|现有概念|AMReX 对应|迁移说明|
|---|---|---|
|DMDA 网格/坐标分区|`Geometry` + `BoxArray` + `DistributionMapping`|P1 用规则 Cartesian；`Geometry` 本身不是曲线 metric 容器|
|global/local Vec 与 ghost|`MultiFab`（nGrow）、`MFIter`、`FillBoundary`/`FillPatch`|AMReX/MPI halo 管理由 `MultiFab`；物理边界单独填充|
|`Ucont`|每方向 face `MultiFab`|积分面体积通量；共享面归约须去重，MacProjector 面速度需显式 adapter|
|`P/Phi/Nvert`|cell `MultiFab`|`Nvert` 仅占位，尚无几何赋值|
|循环与 MPI 局部遍历|`MFIter` + `ParallelFor`|天然可迁 GPU；避免 PETSc Array 视图语义|
|BC 编码|`BCRec` + `FillPatch` + 边界 functor|需将每个现有 BC code 显式分类/测试|
|Poisson/投影|Cartesian `MLMG` 或 `MacProjector`|仅适用于一致的 Cartesian 离散；曲线 `L_g` 要自定义算子/线性求解策略|
|sharp IBM|EB2（几何可表达时）或自定义 IBM|EB2 非 sharp-IBM 的自动等价替代|
|输出/restart|plotfile/checkpoint|先为新字段命名并建立对比导出|
|makefile|CMake + `find_package(AMReX CONFIG REQUIRED)`|骨架已采用，无额外第三方依赖|

## 4. 分阶段任务与完成定义（复核后顺序）

原表将字段、物理边界和 Cartesian 投影压缩在 P1/P2，将 AMR 排在静态 IBM 之前；这不利于定位 MAC 布局、边界和压力零空间错误。以下顺序与当前调用链一致：先冻结可比基线，再建立**无物理算法**的字段骨架，随后分别验证字段/metric、边界数据流和 Cartesian 投影；静态 IBM 与 FSI 在单层路径正确后接入；AMR/EB/GPU 最后。阶段号以 `_Docs/AMReX移植任务清单.md` 为后续唯一进度口径。

|阶段|输入 → 输出|依赖|验收测试|风险|完成定义|
|---|---|---|---|---|---|
|P0 基线与接口冻结|固定 PETSc case/输入 → 版本、字段/单位/索引/BC、守恒/散度/压力/力与重启基准|现有 case 可编译运行（尚待实测）|记录网格、`dt`、通量、散度、压力参考、力/力矩、采样点和 restart 差异|没有基线就无等价判据|可重跑的参考包与容差表|
|P1 基础框架|P0 契约 → `Geometry/BA/DM`、`MultiFab`、ghost、`BCRec` 接口、日志/计时、plotfile/checkpoint 框架|P0|单/多 rank 的零场、制造场与 periodic ghost；CMake 配置/编译|将骨架误报为物理移植|框架测试通过；无 RHS/投影时明确标注|
|P2 字段与网格|P1 → `P/Phi/Nvert`、三 face `Ucont`、cell `Ucat`、Cartesian metric/布局契约|P1|索引、单位、face/cell 位置和 `Ucont↔Ucat` 常量场测试|将 DMDA 三分量视图错误等同为 type-safe MAC 面场|单层 Cartesian 字段语义已测；曲线 metric 不在本阶段声明完成|
|P3 边界与数据流|P2 + BC 字典 → 周期/非周期 ghost、入口/出口/壁面更新次序|P2、P0 BC case|逐面制造解、halo/MPI 一致性、净质量通量|`FillBoundary` 不会填物理 ghost；现有 BC 整数码有算例分支|物理 BC 与 halo 责任边界及测试固定|
|P4 压力投影|P3 预测面通量/压力 BC → RHS、Phi、投影与零空间报告|P3、压力基准|Cartesian 周期制造解、投影后散度、压力常数不变性、MPI 一致性|符号/缩放/面 BC；曲线 19 点算子不能直接替换为 Cartesian 算子|Cartesian `MLMG`/`MacProjector` 路径通过；曲线路径另立设计门|
|P5 动量 RHS 与时间推进|P4 + 冻结 stencil/时间层 → 对流、粘性、LES、推进与 CFL 报告|P2-P4|对流/扩散制造解、时间收敛、无 LES 回归、CFL 与能量检查|把 SNES 非线性语义默默替换为显式步|明确推进法及 SNES 去留，且通过 Cartesian benchmark|
|P6 IBM|P4/P5 + 三角面 → 分类、插值或 EB、无滑移、力/力矩/通量|IBM 路线决策|静态平板/圆柱、网格加密、质量与力诊断|EB2 不是 sharp IBM 的自动等价替代；小 cut-cell 稳定性|只完成选定静态路线，不宣称原 IBM 逐位等价|
|P7 FSI|P6 + 结构状态/力 → 移动几何、耦合迭代、重建|P0 收敛契约、P6|每步残差/松弛/失败路径，运动周期守恒与重启|当前 `CheckConvergence` 逻辑疑点；移动后旧 Poisson 重组装分支被注释|收敛语义修正并在运动 reference case 端到端验证|
|P8 I/O 与诊断|P1-P7 字段 → restart、输出、probe/plane、平均及性能诊断|各字段契约|重启/单多 rank 一致性和诊断闭合|PETSc binary/HDF5 与 AMReX 格式不兼容|兼容策略和新格式 schema 均有回归|
|P9 AMR/EB/GPU|稳定的单层物理路径 → tagging、regrid、coarse-fine、EB、GPU/MPI 优化|P4/P5；P6 若含 EB|细网格对照、reflux/投影、CPU/GPU/MPI 与扩展性|AMR 放大 metric/IBM/负载均衡问题|正确性先于性能，性能目标有实测数据|
|P10 验收与发布|P0-P9 结果 → 验收报告、文档、代码评审|全部目标路径|制造解、流动/IBM/FSI、restart、MPI、性能回归|只以骨架或单一算例声称科学等价|验收矩阵全部闭环并留存证据|

### 不可直接迁移项（必须单独设计/验证）

1. **曲线 19 点算子**：`PoissonLHS` 依赖 `Csi/Eta/Zet/Aj` 和邻域编号（`PoissonSolver.C:891-1598`）；不是 Cartesian 7 点 Laplacian，不能直接替为 `MLMG`。
2. **sharp IBM**：现实现做三角形搜索、射线/最近面、插值点及系数选择（`ImmersedBoundary.C:225-682,1006-1485`）。EB2 是替代建模路线，不是逐算法移植。
3. **移动几何后的矩阵重组**：当前 Poisson 在 `SolvePoisson` 中按状态建/销 HYPRE matrix/vector/solver（`PoissonSolver.C:469-531`）；AMReX 需要重新定义 operator、EB/系数、ghost 和缓存失效策略。
4. **FSI 收敛语义**：主循环的 `sisteps` 和 `isConverged` 控制强/弱耦合（`main.C:154-173`），不可仅搬成固定子循环；必须冻结残差、松弛、失败处理和时间层状态。
5. **HYPRE 数据路径**：现有实现把 PETSc Vec 按 global indices 填入 `HYPRE_IJVector` 再拷回（`PoissonSolver.C:373-405,550-616`）。AMReX `MultiFab` 没有该索引布局；Cartesian 路径优先原生 MLMG，曲线路径另行评估自定义求解器/桥接成本。

## 5. 最小可行路线与延后项

最小可行路线是 **P0 → P1 → P2 → P3（无 LES 或固定模型）→ P4 静态几何**：先在单层 Cartesian、固定几何上得到可复现实验的 MAC 质量守恒和动量基线，再决定 EB2 与 custom IBM 的取舍。`amrex_port/` 当前完成到限定 P2 Cartesian 数据/变换/布局 contract；`advance_one_step()` 仍是明确的 no-op，尚无质量守恒或动量基线。

不应过早做：AMR tagging/重网格、GPU 性能调优、曲线 metric 的强行 `MLMG` 替换、移动 IBM、FSI 强耦合、以及复刻 PETSc→HYPRE 复制路径。它们均会在基线、离散和 BC 尚未固定时掩盖数值差异。

## 6. 本次新增物与验证范围

- 规划：`_Docs/AMReX初步移植规划.md`。
- 独立骨架：`amrex_port/CMakeLists.txt`、`amrex_port/src/main.cpp`、`amrex_port/src/VwisAmrExSolver.H/.cpp`、`amrex_port/README.md`。
- 骨架显式使用 `Geometry`、`BoxArray`、`DistributionMapping`、cell `MultiFab`、face `MultiFab`、`MFIter`、`ParallelFor`、`FillBoundary` 和 CMake AMReX package；保留后续 `BCRec`/`FillPatch`、`MLMG`/`MacProjector`、EB2/custom IBM、plotfile/checkpoint 的接口路线。
- 未实现：全部物理 BC、RHS、LES、SNES 风格动量求解、Poisson/MAC 投影、IBM/EB、FSI、曲线 metric、AMR、GPU 专项优化、plotfile/checkpoint。没有 AMReX 包时仅做静态源码/CMake 检查，不能把该状态表述为编译通过。
