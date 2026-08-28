# AMReX 移植任务清单

> 主清单（2026-08-20）：这是后续移植进度的唯一状态记录；“骨架存在”仅表示数据布局骨架，不表示任何物理算法已移植。状态取值为 `未开始`、`进行中`、`已完成`。除有明确证据的既有审阅/骨架项外，初始状态均为 `未开始`。

## 总体状态摘要

|状态|数量|说明|
|---|---:|---|
|已完成|10|P0-001/002、P2-001--005、P3-001、P4-005 与 P5-002 的限定工程/设计契约；不代表 CFD 求解器完成|
|进行中|17|P0/P1、P3-002--004、P4-001--004、P5-001 与 P5-004；部分已有 CPU runtime，MPI/restart/CFD 数值验收仍缺|
|未开始|26|其余物理实现、数值验证、AMR/IBM/FSI、I/O 和发布任务|
|合计|53|按下列任务 ID（含 P10）计数|

### 当前已完成的真实项与证据边界

- `P0-001`：已审阅本计划相关源码、三份规划/审阅文档；证据为 `_Docs/CODEX_VWIS_REVIEW_20260820.md` 与本次针对 `main.C`、`CurvGrid.C`、`UData.C`、`FlowSolver.C`、`Integrator.C`、`PoissonSolver.C`、`ImmersedBoundary.C`、`StructSolver.C` 的符号/函数复核。
- `P0-002`：两份规划已复核并修订，尤其是阶段依赖、曲线 19 点压力算子和骨架边界；证据为 `_Docs/AMReX初步移植规划.md`、`_Docs/AMReX迁移方案.md` 的本次改动。
- `P0-003`：AMReX 26.04 已以 CPU、MPI（OpenMPI 4.1.6）和 CUDA（arch 89）构建/安装并记录；旧 PETSc/HYPRE/HDF5 ABI、旧 solver 和 CFD 基准仍未验证，因此仍进行中。
- `P0-004/P0-005`：已形成 provisional 字段/时间层/BC/datum/I/O 契约；P2 review 已从 legacy 冻结 `Ucont=Ucat·面面积余因子` 的体积通量语义及 `Aj=det(∂ξ/∂x)`，但没有 case/压力基准/BC 表/输出，P0-005 仍不能签字完成。
- `P1-001`：`amrex_port` 已有 `Initialize/Finalize`、`ParmParse`、`Geometry`、`BoxArray`、`DistributionMapping`、cell/face `MultiFab`、字段注册、`MFIter`/GPU-safe `ParallelFor`、`FillBoundary`、`BCRec` 元数据、日志/计时和 schema-only metadata。后续 P4/P5 已在此骨架上增加限定 Cartesian 投影/RHS/显式推进，但 P1 的多后端运行与真实 I/O 契约仍未闭环，故保持“进行中”。
- `P2-001--005`：限定为单层均匀 Cartesian 数据/变换/布局契约。`Ucont[d]` 是积分面体积通量 $u_d A_d$，`Ucat` 是 cell 速度；散度回归使用净面通量/单元体积。唯一面求和用 `OwnerMask/sum_unique`，face ghost 仅覆盖 inter-Box/MPI/周期，非周期 domain face 的邻接单元复制只是 P2 外推闭合，不是物理 BC。CPU/MPI 与单卡 CUDA 历史结果须以 P2 review 后的定向复测记录为准；这些都不是 P3/P4/P5 数值验收。
- `P3-001`：已从旧 `BcsUtility/CurvGrid` 抽取 1/3/4/5 与独立 periodic 的证据，建立逐面 named/legacy adapter；旧 0/2/6/8/10/11/12/13/14/-1/-2 明确拒绝。2026-08-24 clean 复测为 CPU CTest 7/7 PASS、MPI configure/build/link PASS、CUDA compile/device-link/link PASS；MPI 因执行环境禁止 PMIx socket、CUDA 因设备/驱动不可见而在进入应用代码前 BLOCKED，不能记作 runtime PASS。P3-002--004 保持进行中；详见 `_Docs/AMReX_P3_边界与诊断设计及测试_20260824.md`。
- `P4-001--004`：已实现单层均匀 Cartesian 的积分面通量散度/RHS、显式 pressure BC/compatibility/datum、AMReX `MLPoisson`/MLMG 求解、带面积适配的 face correction 和 `Ucat` 同步。2026-08-26 locked 26.04 CPU CTest 10/10 PASS；周期、封闭 Neumann、入口/定压出口制造测试的散度均降至约 $10^{-10}$ 或更低。该 P4 证据不包含后续 P5 动量/时间推进或 CFD case，MPI runtime 仍缺，故保持进行中。
- `P4-005`：已明确选择“Cartesian `MLPoisson` 仅作为 P4 基线，legacy 曲线非正交 19 点算子延期决策”；禁止把两者视为替换关系。曲线 metric、交叉项、自定义算子/延迟修正和验证 case 留 P5 原型，故设计门完成但曲线实现未开始。
- `P5-001`：已实现保守 Cartesian 对流 RHS，面通量为 `Ucont[d]` 乘 `Ucat` 两侧中心平均，cell RHS 为负净面动量通量除一次体积；与 legacy `-second_order` 分支可比但不宣称曲线等价。2026-08-28 locked 26.04 CPU CTest 13/13 PASS；16/32 周期制造场误差比约 3.92，物理入口/出口多 Box 常量贯流 RHS 为零。MPI/CUDA runtime 与 CFD case 未验，故仍为进行中。
- `P5-002`：已实现单层均匀 Cartesian 常系数粘性 RHS，使用 `nu` 乘三方向 cell-centred 二阶中心 Laplacian；周期/inter-Box halo 与物理 no-slip ghost 分开处理，非周期方向要求显式 BC。2026-08-28 locked 26.04 CPU clean build、静态契约检查和 CTest 16/16 PASS；周期 16/32 制造解、动量守恒、负能量率及物理边界多 Box 通量平衡均 PASS。变系数、曲线 metric、IBM/EB、时间推进、MPI/CUDA runtime 和完整 CFD case 不在本任务范围。
- `P5-004`：选择显式 Euler 预测加现有 Cartesian 投影作为最小基线，投影时间系数固定为 1；`Ucat/Ucont` 均显式维护 `n/n-1/n-2`，并记录 time/step/history depth。2026-08-28 locked 26.04 CPU clean build与 CTest 18/18 PASS；周期剪切扩散 `dt/dt/2` 误差比 2.0026、动量漂移约 $1.39\times10^{-17}$、散度/history 误差为 0，超限扩散数被预期拒绝。该路径不等价于 legacy SNES；半隐式/BDF2 留 P5-005。MPI build/link PASS 但 2-rank PMIx runtime BLOCKED，checkpoint payload/真实 restart 与 CFD case 缺失，故保持进行中。详见 `_Docs/AMReX_P5-004_时间推进设计及测试_20260828.md`。

## 关键路径与门禁

```text
P0 基线/接口
  -> P1 框架 -> P2 字段/Cartesian 网格 -> P3 BC/数据流 -> P4 Cartesian 投影
  -> P5 RHS/推进 -> P6 静态 IBM
  -> P7 FSI ─┐
  -> P8 I/O ├-> P9 AMR/EB/GPU -> P10 验收/发布
             ┘

曲线 metric/19 点算子：P2 metric 语义冻结 -> P4 设计决策 -> P5 验证性原型；
它不是 Cartesian MLMG/MacProjector 的后端替换，也不应成为 P4 的验收前提。
```

门禁：P4 前不得实现生产 RHS；P6 前必须已有单层、Cartesian、带物理边界的质量守恒基线；P7 前必须冻结 FSI 收敛判据；P9 前必须使目标单层物理路径和 restart 正确。任何阶段通过都需要单/多 MPI rank 证据（适用时）。

## 任务明细

列含义：**依赖**为所需上游证据；若上游任务因更宽范围仍“进行中”，下游只能在备注中明确已验收的子契约，不能反向宣称上游整体完成。**I/O** 为任务输入/输出；**文件**为当前源码证据或预期目标位置（目标文件名仅作规划，不创建）；**验收**为可判定结果。

### P0 基线与接口冻结

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P0-001|审阅|提取调用链、字段、离散与风险证据|无|源码/旧文档 → 审阅结论|`vwis2.0/{main,CurvGrid,UData,FlowSolver,Integrator,PoissonSolver,ImmersedBoundary,StructSolver}.C`|证据边界已写入审阅报告|未运行算例不能替代数值证据|已完成|只读相关函数，未全量输出源码|
|P0-002|规划|复核并修订迁移顺序与结论强度|P0-001|三份文档 → 修订计划|`_Docs/AMReX初步移植规划.md`、`AMReX迁移方案.md`|曲线算子、骨架、阶段依赖已校正|后续源码/算例可能改变结论|已完成|本清单为后续主记录|
|P0-003|构建基线|冻结 PETSc/HYPRE/MPI/编译器、编译选项和 AMReX 分支/CMake 版本|P0-001|环境与命令 → lockfile/构建说明|`vwis2.0/makefile`、`amrex_port/{CMakeLists.txt,CMakePresets.json,amrex_version.lock}`|可复现旧程序和骨架的 configure/build 命令|依赖 ABI/精度/GPU 后端不一致|进行中|AMReX 26.04 SHA、CPU/MPI/CUDA arch-89 build/install、CMake/compilers/options/commands 已实测并记录；OpenMPI 4.1.6 MPI run 与单 rank CUDA runtime 已通过。旧 PETSc/HYPRE ABI、CUDA-aware MPI 和 CFD 基准仍未验证|
|P0-004|数值基准|选择固定 Cartesian、曲线、IBM/FSI（若可运行）参考 case，记录网格、dt、容差|P0-003|case/控制文件 → 基准 manifest|`control.dat`、`bcs.dat`、输出目录（待定位）|每 case 输入可重跑且版本化|原树 case/外部网格缺失|进行中|提供 P1 smoke/multibox 输入模板；旧 reference case、控制/BC 文件和结果未提供，数值部分 blocked|
|P0-005|接口冻结|字段字典：单位、索引、时间层、BC 码、pressure datum、输出命名和 I/O 策略|P0-004|基准字段 → interface contract|`UData.C`、`BcsUtility.C`、`PoissonSolver.C`、`_Docs/AMReX_P0P1_设计说明.md`|评审签字；所有 P1+ 字段可追溯|`Ucont` 旧布局并非类型化 face MAC|进行中|已冻结 P2 所需的 Cartesian `Ucont/Ucat` 与 `Aj` 语义；物理单位换算、BC 整数码、datum、I/O/case 仍缺失，不能整体签字|

### P1 基础框架

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P1-001|骨架|保持 AMReX 初始化、ParmParse、Geometry/BA/DM 与基础字段布局|P0-001|输入参数 → 单层对象|`amrex_port/src/*`|静态代码存在，见上方证据|无物理算法|进行中|AMReX 26.04 CPU/MPI/CUDA builds compile; MPI and single-rank CUDA runtime contracts pass. cell/face/历史层/ownership 仍只是 P1 framework，不等同 CFD 移植|
|P1-002|构建|为骨架固定 AMReX 分支、CMake preset、CPU/MPI 选项和最小 inputs|P0-003|依赖版本 → 可配置工程|`amrex_port/{CMakeLists.txt,CMakePresets.json,inputs/*}`|clean configure/build 成功|AMReX package discovery/编译器不匹配|进行中|26.04 lock/SHA；CPU/MPI/CUDA arch-89 configure/build 成功。MPI consumer 因 AMReX package 导出 C MPI dependency，最小修复为 `project(... LANGUAGES C CXX)`；CUDA device link 和单 rank runtime 均通过，CUDA-aware MPI 未测试|
|P1-003|并行框架|验证 MultiFab 多 Box、MFIter、nGrow、periodic `FillBoundary` 与 MPI halo|P1-002|制造场 → halo 对比|`amrex_port/src/VwisAmrExSolver.cpp`、inputs|1/多 rank bitwise 或容差一致|physical ghost 不由 FillBoundary 填充|进行中|CPU MPI `p1_contract.in` 在 2/4 ranks PASS；`p1_multibox.in` 在 2 ranks PASS。CUDA 单 rank 同类 contract PASS；CUDA-aware MPI 未测试|
|P1-004|BC/运行设施|建立 `BCRec` 映射接口、物理 BC functor、日志/计时和错误报告|P0-005,P1-002|BC 字典 → 边界元数据/日志|`amrex_port/src/VwisAmrExSolver.*`|六面 BC 分类完整，未实现类型显式拒绝|旧整数码有算例特例|进行中|已建立 `ext_dir` BCRec、参数校验、rank 输出/计时；旧码未逐面映射，物理 functor 未实现|
|P1-005|基础 I/O|建立 plotfile/checkpoint 元数据、字段注册与版本头|P1-002,P0-005|字段注册 → 空场 plot/checkpoint|`amrex_port/src/VwisAmrExSolver.*`|空场可读回并校验 schema|与 PETSc 文件格式不兼容|进行中|已注册字段并写 schema-only JSON (`payload_written=false`)；未写 plot/checkpoint payload，不能恢复|

### P2 字段与网格

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P2-001|字段|定义 `P/Phi/Nvert` cell MultiFab、时间层和 component 命名|P0-005,P1-003|契约 → scalar fields|`amrex_port/src/VwisAmrExSolver.*`、`inputs/p2_contract.in`|IndexType/nGrow/单位语义字段断言；未决物理换算显式标出|把 `Nvert` 当 EB volume fraction|已完成|字段契约含 component/time-layer/units；`P/Phi` 同为 legacy 无量纲压力尺度，物理换算/pressure datum 仍属 P0/P4；`Nvert` 保持分类语义|
|P2-002|速度|定义三个 face `Ucont` 与 cell 三分量 `Ucat`，冻结体积通量和 face ownership|P2-001|字段 → MAC/cell 布局|`amrex_port/src/VwisAmrExSolver.*`、`inputs/p2_contract.in`|常量速度给出 $Ucont_d=u_dA_d$；唯一面数/通量和正确|把 legacy 积分通量当 MacProjector 面速度|已完成|三个独立 face `MultiFab`；共享面最低 global box 为 owner，归约用 `OwnerMask/sum_unique`；legacy 依据为 `FormMetrics/Contra2Cart/CalculateDivergence`|
|P2-003|变换|实现/测试 Cartesian `Ucont↔Ucat`；曲线变换显式留为未实现边界|P2-002|Ucat/Ucont → 同步场|`VwisAmrExSolver.*`|常量/线性场按 face area 乘除且单位一致|曲线 metric 不能以 Cartesian 面积代替|已完成|cell→face：法向速度线性平均后乘面积；face→cell：相邻通量和除以 `2A_d`；共享 valid face `OverrideSync` 后再 halo；曲线实现仍在 P5|
|P2-004|网格|建立单层 Cartesian 坐标、dx、cell volume 和 metric 数据命名|P1-003,P0-005|domain/n_cell → Geometry/metric metadata|`VwisAmrExSolver.*`|$A_d\Delta x_d=V$ 与 Geometry 一致；`Aj` 定义不含糊|`Aj`/J/1/J 混淆|已完成|记录 dx、volume、face area；仅登记未分配的 `legacy_Aj_equivalent=1/V`（unit-index 计算坐标），不声称曲线 metric/Jacobian 场|
|P2-005|布局回归|多 Box/MPI 上验证 cell/face ghost、边界 face 与 derived field 一致性|P2-001--004|制造场 → layout report|`inputs/p2_contract.in`、`p2_boundary_face.in`|1/多 rank 范数、唯一 face sum、face ghost 和 divergence stencil 一致|跨 Box face 双计数|已完成|周期/非周期 face ghost、OwnerMask 唯一计数/通量和、非周期边界邻接单元外推、净面通量/体积 derived divergence；只是 manufactured layout regression，不是全域守恒或物理 BC|

### P3 边界与数据流

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P3-001|BC 分类|把旧 BC 整数码与每个面速度/压力/标量条件逐项映射，标出未支持项|P0-005,P1-004|BC 文件 → 显式 BC 表|`BcsUtility.C:672-1406`,`Integrator.C`、`amrex_port/src/*`、P3 设计记录|每个已用 case 无隐式默认分支|入口/出口与压力条件耦合|已完成|旧 1/3/4/5 和独立 periodic 有显式映射；新 named slip 不冒充旧码；0/2/6/8/10/11/12/13/14/-1/-2 显式拒绝；无旧 case，4/5 仅承诺文档中的 Cartesian 子集|
|P3-002|ghost 流程|实现周期/非周期 `FillBoundary`、物理 ghost 填充和调用顺序|P2-005,P3-001|fields+BC → valid ghost|`amrex_port/src/VwisAmrExSolver.*`、`inputs/p3_*`|逐面制造解和跨 rank halo 一致|误把 FillBoundary 用作物理 BC|进行中|已分离 halo 与 physical fill，固定 OverrideSync/FillBoundary→physical→diagnostics，并有 epoch stale 拒绝；2026-08-24 CPU CTest 7/7 PASS，MPI build/link PASS 但 2-rank 在 `MPI_Init` 前被 PMIx socket 权限阻塞；多层 FillPatch 留 P9|
|P3-003|流量 BC|实现入口流量/平面输入与出口压力/通量约束的最小 Cartesian 路径|P3-002|profile/target flux → face Ucont|`amrex_port/src/VwisAmrExSolver.*`、`inputs/p3_cartesian_boundary.in`|全局净通量、面 profile 和压力 datum 达容差|旧 case 特定分支|进行中|实现 uniform/linear-plane MPI 全局归一化、fixed pressure ghost、可选等流量出口；不是压力投影/datum 求解；CPU P3 PASS，MPI multi-rank runtime BLOCKED；CUDA compile/device-link PASS 但无设备 runtime|
|P3-004|一致性诊断|在每个阶段核算 halo、边界面、全局质量通量和 ghost freshness|P3-002,P2-005|fields → diagnostic report|`amrex_port/src/VwisAmrExSolver.*`、`inputs/p3_*`|错误注入能被检测；MPI reductions 正确|只报 interior divergence 掩盖边界/IBM|进行中|实现 stage epoch、boundary owner-unique flux、全域 divergence integral/Linf、显式 sum/max MPI reduction 和 stale 注错；无 IBM；CPU PASS，MPI reduction runtime 因 PMIx socket BLOCKED，CUDA runtime 因设备/驱动不可见 BLOCKED|

### P4 压力投影

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P4-001|散度/RHS|从积分 face `Ucont` 以净通量/体积计算 Cartesian 散度和 pressure RHS，冻结符号/时间系数|P3-004|U* → div/RHS|`FlowSolver.C:133+`,`PoissonSolver.C:209+`|解析场 L2/L∞、全域通量恒等式、积分 RHS 可解性通过|把 volume flux 再除 dx；旧诊断跳过边界/IBM 邻域|进行中|实现 $rhs=(\alpha/dt)\,div(U^*)$；CPU 制造场通过，MPI/CFD case 尚缺；曲线 19 点不在此替换|
|P4-002|压力 BC/零空间|定义 Dirichlet/Neumann/周期组合、压力参考和 RHS 去均值策略|P3-001,P4-001|BC/RHS → solvable system|`PoissonSolver.C:134+`|常数压力不改速度；兼容性条件被记录|`BC(3)==-10` 语义需以 case 复核|进行中|周期/封闭 Neumann 显式验 RHS mean 并拒绝不兼容，不减 RHS 均值；只去 converged Phi 均值选 gauge；定压出口为 Phi=0 Dirichlet|
|P4-003|Cartesian 求解器|评估并实现一个固定版本的 `MacProjector` 或 MLMG+自写修正路径|P4-001,P4-002|RHS/BC → Phi|目标 `Projection.*`|残差、迭代数和 CPU/MPI 可复现|API 面速度与 `Ucont` 积分通量语义不匹配|进行中|选择 locked 26.04 `MLPoisson`/MLMG + 自写 correction；`getFluxes` operator flux 经显式面积适配，CPU PASS，MPI runtime 待环境|
|P4-004|速度修正|按相同 face metric/系数修正 Ucont，并同步 Ucat|P4-003,P2-003|Phi/U* → U(n+1)|`PoissonSolver.C:2352+`|投影后 div 降至容差；质量守恒|符号、dt 系数、边界 face|进行中|同一 face/volume 离散，修正 Ucont 后同步 Ucat；不重新覆盖 pressure-outlet correction；CPU 三类 BC PASS|
|P4-005|曲线算子决策|用现有 `PoissonLHS`/RHS/Projection 比对 19 点项、对称性、零空间和可选实现|P4-001,P0-005|metric/stencil → design record|`PoissonSolver.C:891-1598,2352+`|明确 custom operator、deferred correction 或临时 PETSc/HYPRE 对照的门槛|将 19 点曲线算子直接替换 Cartesian 算子|已完成|P4 只选 Cartesian `MLPoisson` 基线；曲线 19 点交叉 metric 需 P5 自定义算子/deferred correction 原型及 case 后再定，不以 Cartesian 路径替换|

### P5 动量 RHS 与时间推进

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P5-001|对流|实现 Cartesian 对流通量与旧 `-second_order` 分支的可比 stencil 说明|P3-004,P2-002|Ucat/Ucont → advective RHS|`RHSSolver.C`、`amrex_port/src/VwisAmrExAdvection.cpp`|平流制造解与网格收敛|面/cell 插值位置错误|进行中|保守 face flux=`Ucont[d]`×两侧 `Ucat` 中心平均，RHS 为负净通量/体积；locked 26.04 CPU 13/13 PASS，16→32 制造场 L∞ 误差比 3.92，边界多 Box 常量贯流 RHS=0；MPI/CUDA/CFD case 未验，且不宣称曲线等价|
|P5-002|粘性|实现常系数粘性、边界通量与动量守恒测试|P3-002,P5-001|Ucat,nu → viscous RHS|`amrex_port/src/VwisAmrExViscosity.cpp`|扩散制造解与能量耗散|wall model 与基本粘性混淆|已完成|locked 26.04 CPU clean build、静态契约检查、CTest 16/16 PASS；周期 16/32、物理边界多 Box 均 PASS；变系数/metric/时间推进后置|
|P5-003|metric 原型|实现曲线 metric 读入/计算、导数/面 metric，并以制造解验证|P2-003,P2-004,P4-005|曲线网格 → metric fields/operator evidence|`CurvGrid.C:312+`,`functions.*`|metric identity、GCL/体积、19 点项逐项对照|非正交交叉项和 Jacobian 缩放|未开始|只有通过后才可称支持曲线网格|
|P5-004|时间推进|比较显式、半隐式与 BDF2；固定历史场和时间系数|P5-001,P5-002,P4-004|RHS/history → U*|`Integrator.C:326+`、`amrex_port/src/VwisAmrExTime.cpp`|时间收敛、CFL、守恒、restart 时间层一致|旧 SNES 分支不可被默默抹平|进行中|2026-08-28 选择显式 Euler+投影系数 1 的最小 Cartesian 基线；CPU clean/CTest 18/18 PASS，时间误差比 2.0026、守恒/散度/history 与 CFL 拒绝 PASS；不等价于 legacy SNES。MPI runtime、checkpoint/restart payload 与 CFD case BLOCKED；半隐式/BDF2 留 P5-005|
|P5-005|SNES 策略|决定保留 PETSc SNES 过渡接口或以明确线性/非线性 AMReX 语义替代|P5-004,P0-005|残差/Jacobian 契约 → 决策|`Integrator.C:88,168,326+`|残差、停止条件、失败处理有回归|“显式一步”等同 SNES 收敛的错误结论|未开始|结构 SNES/非线性子系统另行评估|
|P5-006|LES|迁移 Smagorinsky/壁面相关最小路径、二阶插值和 SGS 诊断|P5-001--005|速度/metric → nu_t/RHS|`LESModel.C`,`RHSSolver.C`,`WallModel.C`|无 LES 回归、Cs/nu_t 剖面对比|滤波/裁剪/IBM 例外未冻结|未开始|AMR dynamic LES 留 P9|

### P6 IBM

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P6-001|路线决策|在 EB2 与自定义 static sharp IBM 中选择首条路线，冻结目标等价层级|P4-005,P5-004|几何/需求 → ADR|`ImmersedBoundary.C`、`amrex_port` 目标接口|性能、运动频率、几何表达、守恒准则有决策|EB2 不等于 sharp IBM|未开始|不在同一首版混用两路线|
|P6-002|几何输入|读三角面、建立空间搜索和可并行 SoA 几何表示|P6-001|mesh → distributed geometry|`ImmersedBoundary.C:87-682`|分类与最近面查询对照|host 链表/GPU 不可用|未开始|EB2 路线需定义可靠几何转换|
|P6-003|分类/模板|建立 Nvert 或 EB 几何、image/interpolation stencil 和 ghost 失效策略|P6-002,P3-004|geometry → mask/stencil|`ImmersedBoundary.C:1737+`|静态平板/球分类与插值误差|Nvert 非严格体积分数；stencil 跨 rank|未开始|只做静态|
|P6-004|边界/投影耦合|把无滑移、IBM 速度修正与 pressure projection 顺序固定|P6-003,P4-004,P5-004|mask/stencil,U* → corrected U/Phi|`BcsUtility.C:1893+`,`PoissonSolver.C`|无穿透、质量通量和散度报告|投影后重新引入 slip/divergence|未开始|必须报告 IBM 邻域而非掩盖|
|P6-005|载荷验证|计算压力/黏性力、力矩、通量；执行静态圆柱/平板网格收敛|P6-004|fields/geometry → loads|`ImmersedBoundary.C`,`FSI.C`|力/力矩/质量及细化趋势|EB 小 cut-cell/锐界插值误差|未开始|不声称与旧 IBM 逐点等价|

### P7 FSI

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P7-001|结构接口|冻结状态、载荷、位移/速度、时间层、重启和单位契约|P0-005,P6-005|fluid loads ↔ FSIState|`FSI.C/.h`,`StructSolver.C/.h`|独立结构/接口测试|结构模型外部依赖|未开始|不以现有逻辑直接作为正确规范|
|P7-002|收敛语义|修正并测试强/弱耦合的残差、松弛、最大迭代与失败语义|P7-001|迭代历史 → converged/failure|`main.C:145-173`,`StructSolver.C:112+`|残差单调/停止条件测试|当前 `CheckConvergence` 存逻辑疑点|未开始|`-str>1` 不能自动称强耦合收敛|
|P7-003|移动几何|定义“结构更新→几何→分类/stencil→Poisson/operator→flow”重建事务|P7-002,P6-004,P4-005|FSIState → geometry epoch|`FSI.C:432+`,`PoissonSolver.C:469+`|每 epoch 无旧 stencil/operator；运动周期质量守恒|旧移动矩阵重组装分支被注释|未开始|EB2 重建成本须实测|
|P7-004|FSI 验收|运行运动 IBM/FSI case，比较位移、频率、相位、载荷和重启|P7-003,P8-002|case → validation report|目标测试/文档|迭代/残差及物理量通过容差|added-mass/不稳定耦合|未开始|先单层/单 MPI，再扩展|

### P8 I/O 与诊断

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P8-001|字段输出|定义 plotfile/HDF5（如启用）字段名、派生量、时刻、单位与旧输出映射|P0-005,P1-005|fields → schema|`UData.C:419+`、目标 `IO.*`|可视化与 schema regression|PETSc Vec I/O 不可直接读取|未开始|兼容策略必须明示转换或弃用|
|P8-002|restart|checkpoint 恢复所有流体时间层、metric/IBM/FSI metadata 与版本迁移|P8-001,P5-004|checkpoint → identical state|`UData.C:283+`,`FSI.C`|连续跑 vs 重启的范数/积分量一致|漏存 U(n-1)/geometry epoch|未开始|P7 使用本任务前需先完成基础流体 restart|
|P8-003|采样/统计|实现 probe、plane、平均量、力/力矩、质量、散度、动能与压力诊断|P3-004,P5-006|fields → CSV/plot diagnostics|`PointProbe.C`,`PlaneExtraction.C`,`UData.C`|固定点/平面与 MPI reduction 对照|采样跨 Box/AMR 覆盖|未开始|IBM 邻域需单独统计|
|P8-004|性能诊断|分解 kernel、Poisson、I/O、halo、host-device copy 与内存计时|P1-004,P5-004|timers → performance log|`Timer.C`、目标 `Diagnostics.*`|无负/漏计；多 rank 汇总正确|没有基准时不可下性能结论|未开始|P9 性能模型输入|

### P9 AMR/EB/GPU

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P9-001|AMR 数据流|实现 tagging、regrid、FillPatch、AverageDown 与 level metadata|P4-004,P8-002|tag criteria → hierarchy|目标 `AmrCore`/`MeshManager.*`|regrid 后场/metadata 完整|曲线 metric 与 AMR 关系未设计|未开始|先规则 Cartesian、no-subcycling|
|P9-002|coarse-fine 守恒|实现动量 flux register/reflux 与 composite projection 次序|P9-001,P5-004|level fluxes → corrected state|目标 `FluxRegister`/`Projection.*`|细网格对照、CF divergence/质量通过|reflux 后重生散度|未开始|压力与动量同步是硬门禁|
|P9-003|EB/小 cut-cell|若选 EB，验证 EB 数据、稳定化、载荷和 regrid/moving 策略|P6-001,P6-005,P9-001|EB geometry → stable advance|目标 EB handler|cut-cell case 不发散且守恒/力收敛|小体积分数 CFL/质量泄漏|未开始|custom IBM 路径则记录不适用|
|P9-004|GPU/MPI|将热 RHS/投影/IBM（适用）写为 ParallelFor，维持数据驻留和 GPU-aware halo|P5-006,P8-004|CPU implementation → GPU path/profile|目标 kernels|CPU/GPU 数值回归、无意外 host copy|设备 lambda/几何搜索/通信|未开始|先正确后优化|
|P9-005|性能模型|测量单/多 rank、CPU/GPU 强弱扩展、halo/Poisson/I/O 占比并设目标|P9-002,P9-004|profiles → scaling report|目标 benchmark scripts/docs|可复现实测数据与资源说明|case 太小/噪声掩盖瓶颈|未开始|不得从骨架推断性能|

### P10 验收与发布

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P10-001|制造解|Taylor–Green/解析投影、对流/扩散空间时间收敛与压力零空间回归|P4-004,P5-004|tests → convergence report|目标 tests|预声明阶数/容差达标|用单一范数隐藏局部错误|未开始|Cartesian 路径最低科学门槛|
|P10-002|基准流动|通道/腔流、入口出口、质量/动能/压力与单多 MPI rank 回归|P5-006,P8-003|cases → regression baselines|目标 tests/docs|所有 P0 指标达标|BC 与诊断不闭合|未开始|以冻结 case 为准|
|P10-003|IBM/FSI|静态/运动 IBM 与 FSI 载荷、无穿透、耦合残差和 restart 验收|P6-005,P7-004,P8-002|cases → validation report|目标 tests/docs|网格趋势/耦合语义/重启闭环|只验证静态却声称运动可靠|未开始|无 FSI 路径可标为不适用，不得跳过声明|
|P10-004|性能/可靠性|单/多 rank、CPU/GPU（适用）、restart 一致性与性能基线|P8-004,P9-005|profiles/checkpoints → release evidence|目标 CI/benchmark|资源、版本、命令、结果可复现|硬件/编译器漂移|未开始|性能不是科学等价的替代|
|P10-005|发布|代码评审、文档、已知限制、变更日志与任务状态归档|P10-001--004|evidence → release candidate|本清单、设计/用户文档|验收矩阵关闭且无“骨架=物理移植”表述|未关闭决策项|未开始|发布前复核任务状态|

## 各模块 Definition of Done

|模块|Definition of Done|
|---|---|
|P0 基线与接口|环境、case、字段/单位/索引/BC/压力基准和全部比较容差可重跑、可追溯。|
|P1 基础框架|固定 AMReX 版本的 CMake 构建通过；单/多 rank 字段、ghost、日志和空 I/O 测试通过；无物理算法的边界明确。|
|P2 字段与网格|cell/face IndexType、时间层、体积通量单位、转换、唯一面归约和 ghost 在多 Box/MPI 制造场通过；只冻结 Cartesian 数据/代数语义，不包含 CFD 方程验收。|
|P3 边界与数据流|每个目标 BC 的物理 ghost、halo、入口/出口净通量与调用顺序均有逐面测试。|
|P4 压力投影|Cartesian RHS、BC、零空间、solve、face 修正共用离散；投影后散度/质量/MPI 指标达标；曲线路径单独有设计结论。|
|P5 RHS 与推进|对流、粘性、metric（如启用）、LES 和推进法在制造解/基准 case 达到预设收敛、守恒、CFL 和时间层要求。|
|P6 IBM|选定 EB2 或 custom sharp IBM 的静态几何、无滑移、压力/黏性力、力矩和局部质量诊断随网格收敛。|
|P7 FSI|收敛/失败语义正确；移动后的几何、mask/stencil/operator 事务化重建；指定运动 case 与 restart 通过。|
|P8 I/O/诊断|plotfile/checkpoint、schema 版本、probe/plane/平均/积分与性能计时可读、可重启、MPI 一致。|
|P9 AMR/EB/GPU|AMR coarse-fine 守恒与 composite projection 正确；EB cut-cell（如用）稳定；CPU/GPU/MPI 正确性与性能均有实测。|
|P10 发布|制造解、基准流动、IBM/FSI、重启和性能验收闭环；文档、限制、代码评审与可复现命令齐全。|

## 当前阻塞项与决策项

|决策|必须回答的问题|最晚门禁|暂定原则|
|---|---|---|---|
|AMReX 版本|目标 tag、DIM/precision、MPI/GPU 后端、HDF5/EB 选项是否锁定？|P1-002|不以“系统已装”代替锁定版本；配置后记录。|
|Cartesian 与曲线 metric|首个可交付是否仅 Cartesian？曲线 J/Aj、face metric、GCL 与 19 点项如何逐项验证？|P4-005|Cartesian 是独立基线；曲线算子不可被 7 点/标准 projector 直接代替。|
|EB2 与自定义 IBM|几何是否可稳定转 EB？运动频率、sharp 插值继承和小 cut-cell 稳定性哪个优先？|P6-001|首版二选一，统一 `IBMHandler` 接口。|
|MLMG 与自定义算子|完整非正交 19 点算子是否可在锁定版本中正确实现/求解？过渡期是否保留 PETSc/HYPRE 对照？|P4-005|先做离散/零空间对照；不预先承诺 custom `MLLinOp`。|
|SNES 是否保留|动量隐式残差、线性化、容差和失败处理是否必须与 PETSc SNES 同义？|P5-005|显式/半隐式替代必须是明示的数值方案变更。|
|I/O 兼容策略|保留 PETSc binary/HDF5 读取、一次性转换，还是仅 AMReX checkpoint/plotfile？|P8-001|先定义 schema 和转换验证；不可声称格式天然兼容。|

## 建议的下一步（按顺序）

1. 执行 `P0-003`：锁定并实测 PETSc/HYPRE/MPI 与 AMReX/CMake 构建环境，补齐可复现命令。
2. 执行 `P0-004` 与 `P0-005`：选定至少一个固定 Cartesian 基准，冻结 `Ucont/Ucat/P/Nvert`、`Aj`、BC、压力基准和 restart 指标。
3. 执行 `P1-002`、`P1-003`：配置编译当前 `amrex_port`，以 1/多 MPI rank 验证多 Box periodic ghost；确认仍无物理算法。
4. 复核 P2 review 后 CPU/MPI/CUDA 的短 contract 结果；只把 P2 称为“Cartesian 数据/变换/布局契约完成”。
5. 执行 `P3-001` 至 `P4-005`：先固定实际 case 的边界/零空间，再建立独立 Cartesian 投影与曲线算子设计门；不得用 P2 derived divergence 代替投影或守恒验收。
