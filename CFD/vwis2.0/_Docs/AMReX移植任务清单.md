# AMReX 移植任务清单

> 生产推进增量（2026-10-12）：共用的面通量权威/增量压力时间推进已接入全周期
> Cartesian identity 和 separable orthogonal（含 stretch）网格；压力求解按坐标分派
> MLPoisson、MLABecLaplacian 或原 shear deferred。拉伸正交预测插值补齐体积加权守恒，
> 八步 mixed-mode 的动量漂移由3.81995e-8降至2.22045e-16，原残差/散度门不变。
> 新增两份真实 solve 输入，并复用压力波应用验证 Cartesian/正交三网格近二阶收敛。
> 新路径物理边界与 checkpoint/restart 明确拒绝，压力时间收敛与一般19点仍开放；
> 53项统计14/20/19保持，未新增CTest。详见 [使用及离散说明](../amrex_port/README.md#periodic-cartesian-and-orthogonal-incremental-pressure)。

> 生产能力增量（2026-10-12）：真实 `advance_one_step()` 新增 cell 三分量空间体力场，
> 可与常量体力叠加；新增 `case_mapped_pressure_wave` 应用模式，使用独立解析力驱动
> 非零压力、非零衰减速度，并输出逐步物理误差与最终场 CSV。CPU 16/32/64 网格的
> 速度、显式 n 层压力、压力梯度和连续动量平衡误差均呈近二阶收敛（误差比约3.94–3.99）。
> 这是受限 affine shear 受迫波的非零物理压力空间参考；非线性流动、压力时间收敛、
> 一般19点与历史后端精度门仍开放。未新增 CTest 项，53项统计14/20/19保持。
> 使用方法和证据见 [应用说明](../amrex_port/README.md#mapped-pressure-wave-case)。

> 最新生产算法增量（2026-10-12）：新增 opt-in `face_authoritative + incremental_pressure`，真实 `advance_one_step()` 使用旧 P 的面梯度 predictor，并以移位未知量 Q=P_old+Phi 求解现有 deferred MLMG；Phi 为当步压力增量，P 仅在全部候选通过原残差/散度门后提交。直接小增量 RHS 候选触及内层舍入底，已隔离；未放宽容差。默认 dual-state 与 instantaneous P=Phi 路径保持。连续物理压力、历史首步 CPU布局/CUDA P/Phi 门仍未关闭，C4 未开始、53项统计14/20/19不变。见 [增量压力实施报告](./AMReX_M2_增量压力状态与移位求解_20261012.md)。


> 最新增量（2026-10-12）：复现并否决两种未安全关闭 P/Phi 门的 stencil 舍入候选，已恢复原实现。最终仅 face_authoritative 投影复用当次 Ucat 三分量最大范数，每次少六次局部扫描/全局 max 归约；公式、原门限与 fatal 语义不变。CPU/MPI/CUDA 定向 CTest 为18/19/18项全 PASS，旧路径各126字段及面权威全快照与本轮基线逐值一致。CPU布局/CUDA首步 P/Phi 仍 BLOCKED（1.11620e-12 / 1.19446e-12），MPI4限定337字段 PASS；M2/C3/M4 未关闭，14/20/19与C4未开始保持。见 [候选审计与范数复用报告](./AMReX_M2_后端舍入候选审计与M4范数复用_20261012.md)。

> 最新生产增量（20261011 积分 RHS 统一）：仅 face_authoritative 的 MLMG 输入与 deferred cross action 改为直接积分面净通量，消除 net/V*V 往返。CPU/MPI/CUDA build 与定向 CTest 各12/12 PASS，旧路径各126字段逐值不变。337字段中 MPI4 全部PASS；CPU布局/CUDA仍仅首步P/Phi BLOCKED（1.11620e-12 / 1.19446e-12），严格<=1e-12门不变。M2/P5-005/M4/C3/C7进行中、C4未开始；主统计维持14/20/19。物理压力与整步rollback未完成；见 [本轮实施报告](./AMReX_M2_积分RHS统一与首步压力门_20261011.md)。

> 主清单（2026-08-20）：这是后续移植进度的唯一状态记录；“骨架存在”仅表示数据布局骨架，不表示任何物理算法已移植。状态取值为 `未开始`、`进行中`、`已完成`。除有明确证据的既有审阅/骨架项外，初始状态均为 `未开始`。

## 总体状态摘要

|状态|数量|说明|
|---|---:|---|
|已完成|14|P0-001/002、P2-001--005、P3-001、P4-005、P5-002、P8-001--003、P10-001 的限定工程/设计契约；不代表完整 CFD 求解器完成|
|进行中|20|P0/P1、P3-002--004、P4-001--004、P5-001/P5-003/P5-004/P5-005 与 P10-002；C7 已补限定正交 mapped 的 1/2/4-rank 和 CPU/CUDA 对照，完整 CFD/性能/I/O 验收仍缺|
|未开始|19|其余物理实现、数值验证、AMR/IBM/FSI、I/O 和发布任务|
|合计|53|按下列任务 ID（含 P10）计数|

2026-10-11 更新：C7 的正交 mapped 数值对照子门已补齐，CPU 47/47、MPI 48/48、
CUDA 38/38 CTest PASS；CPU/CUDA 全场相对 L2 差最大约 `5.03e-14`。本轮不改变任何 P 级任务
状态，该 C7 增量当时的统计为 **14 已完成、19 进行中、20 未开始**；随后 P5-005
转为进行中，当前统计以页首表及任务行为准：**14 已完成、20 进行中、19 未开始**。C7 保持进行中，仍缺
kernel/MLMG/halo/同步分项 profiling、CUDA-aware MPI 和 MPI/GPU restart；实施与
CUDA 完整适用集结果见 [`AMReX_C7_MPI_CPU-GPU对照_20261011.md`](./AMReX_C7_MPI_CPU-GPU对照_20261011.md)。
下文带日期的旧 `BLOCKED/未验` 记载保留当时证据边界；本次多 rank 数值对照只覆盖
上述正交 mapped 子契约，不据此关闭旧 Cartesian/完整 CFD/MPI I/O 验收。

2026-10-11 C3 后续增量：新增固定 `analytic_shear` mapping、完整 cross face flux、
共享压力/黏性 stencil、全残差驱动的 periodic deferred projection、剪切速度重构和稳定性
guard。CPU 三网格 MMS、正负 epsilon 扫描、零剪切对照、独立 affine action、守恒、
压力恢复、失败状态保护与 100 步自由流通过。P5-005 从“未开始”转为“进行中”，
因为源码残差审计与显式算法选择/拒绝接口已落地；SNES/半隐式替代尚未完成。
当前统计为 **14 已完成、20 进行中、19 未开始**。C3 为“进行中：受限周期弱剪切
数值切片已验证”，C4 仍未开始；最终全套 CPU 54/54、MPI 55/55、CUDA 45/45 PASS，
另有 16 组 C3 多 rank 契约及十组 CPU/CUDA 范数对照通过。M2/M3–M6 决策及边界见 [`AMReX_C3_弱剪切算子与算法决策_20261011.md`](./AMReX_C3_弱剪切算子与算法决策_20261011.md)。

2026-10-11 C3 第二增量：非零 shear 的 deferred scratch 由 solver 持久复用，
全部 face/cell correction 在 flow commit 前准备；补齐四步混合 xy/xz 模态实际推进、
状态/history/halo 一致性与 checkpoint 读写拒绝回归。64x64x4 后续步在 inner 800 次
仍持续收敛，增加工作上限到 1600；inner 1e-12、full residual 1e-11、散度 1e-10
门槛保留。有限的双离散表示继续使用 Ucat 增量梯度和 Ucont 守恒通量；M2 的单一
权威/per-field epoch 总门仍未关闭。状态统计仍为 14/20/19，C3 保持进行中、C4 未开始。
最终验证与成本见 [`AMReX_C3_多步状态与workspace_20261011.md`](./AMReX_C3_多步状态与workspace_20261011.md)。

2026-10-11 C3 第三增量：真实 32/64/128 步固定时间及 256 步长程审计，速度/面通量
时间步差约一阶；去 datum 的累计 P 差误差比约 0.5，物理压力时间门 BLOCKED。
新增 `state_contract=experimental_dual_state` 明确拒绝单一权威/物理压力等未验收声明。
MPI/CPU-CUDA Phi 的细 dt 全场 L2 对照未全部过门，保留 BLOCKED；未放宽 residual/divergence/
backend 门。生产数值公式、workspace、C7 专用文件及历史报告保持。状态/统计仍为
14/20/19，详见 [时间审计及 M2 决策](./AMReX_C3_时间审计与M2拒绝契约_20261011.md)。

2026-10-11 P5-005 拒绝契约增量：修复 `solve` 等模式接受却忽略非 1
`projection_time_coefficient` 的入口缺口；仅四个独立投影契约允许可调系数，
其余模式在 solver 构造前明确拒绝。CPU 17/17、MPI 18/18、CUDA 17/17 定向
CTest PASS，包含默认/显式 1 的真实 16 步逐值一致性及四 rank/空闲 rank。
生产数值公式未变，累计压力时间门与细 dt Phi 后端门继续 BLOCKED；
P5-005/M2/M4/C3/C7 保持进行中、C4 未开始，主统计仍为 14/20/19（53 项）。
见 [投影时间系数拒绝契约报告](./AMReX_P5-005_投影时间系数拒绝契约_20261011.md)。

### 当前已完成的真实项与证据边界

- `P0-001`：已审阅本计划相关源码、三份规划/审阅文档；证据为 `_Docs/CODEX_VWIS_REVIEW_20260820.md` 与本次针对 `main.C`、`CurvGrid.C`、`UData.C`、`FlowSolver.C`、`Integrator.C`、`PoissonSolver.C`、`ImmersedBoundary.C`、`StructSolver.C` 的符号/函数复核。
- `P0-002`：两份规划已复核并修订，尤其是阶段依赖、曲线 19 点压力算子和骨架边界；证据为 `_Docs/AMReX初步移植规划.md`、`_Docs/AMReX迁移方案.md` 的本次改动。
- `P0-003`：AMReX 26.04 已以 CPU、MPI（OpenMPI 4.1.6）和 CUDA（arch 89）构建/安装并记录；旧 PETSc/HYPRE/HDF5 ABI、旧 solver 和 CFD 基准仍未验证，因此仍进行中。
- `P0-004/P0-005`：新增 `p0_cartesian_benchmark.in` 与 CTest runner，locked AMReX 26.04 CPU 实际运行 8 步并生成逐步 JSON（散度、净通量、动量、动能、计时和构建信息）。这是 manufactured/contract baseline，不是 legacy vwis 对照；真实 legacy case、曲线/IBM/FSI reference 和物理单位仍缺，因此两项保持“进行中”。详见 `_Docs/AMReX_P0-004_P0-005_Cartesian基准设计及结果_20260829.md`。
- `P1-001`：`amrex_port` 已有 `Initialize/Finalize`、`ParmParse`、`Geometry`、`BoxArray`、`DistributionMapping`、cell/face `MultiFab`、字段注册、`MFIter`/GPU-safe `ParallelFor`、`FillBoundary`、`BCRec` 元数据、日志/计时和 schema-only metadata。后续 P4/P5 已在此骨架上增加限定 Cartesian 投影/RHS/显式推进，但 P1 的多后端运行与真实 I/O 契约仍未闭环，故保持“进行中”。
- `P2-001--005`：限定为单层均匀 Cartesian 数据/变换/布局契约。`Ucont[d]` 是积分面体积通量 $u_d A_d$，`Ucat` 是 cell 速度；散度回归使用净面通量/单元体积。唯一面求和用 `OwnerMask/sum_unique`，face ghost 仅覆盖 inter-Box/MPI/周期，非周期 domain face 的邻接单元复制只是 P2 外推闭合，不是物理 BC。CPU/MPI 与单卡 CUDA 历史结果须以 P2 review 后的定向复测记录为准；这些都不是 P3/P4/P5 数值验收。
- `P3-001`：已从旧 `BcsUtility/CurvGrid` 抽取 1/3/4/5 与独立 periodic 的证据，建立逐面 named/legacy adapter；旧 0/2/6/8/10/11/12/13/14/-1/-2 明确拒绝。2026-08-24 clean 复测为 CPU CTest 7/7 PASS、MPI configure/build/link PASS、CUDA compile/device-link/link PASS；MPI 因执行环境禁止 PMIx socket、CUDA 因设备/驱动不可见而在进入应用代码前 BLOCKED，不能记作 runtime PASS。P3-002--004 保持进行中；详见 `_Docs/AMReX_P3_边界与诊断设计及测试_20260824.md`。
- `P4-001--004`：已实现单层均匀 Cartesian 的积分面通量散度/RHS、显式 pressure BC/compatibility/datum、AMReX `MLPoisson`/MLMG 求解、带面积适配的 face correction 和 `Ucat` 同步。2026-08-26 locked 26.04 CPU CTest 10/10 PASS；周期、封闭 Neumann、入口/定压出口制造测试的散度均降至约 $10^{-10}$ 或更低。该 P4 证据不包含后续 P5 动量/时间推进或 CFD case，MPI runtime 仍缺，故保持进行中。
- `P4-005`：已明确选择“Cartesian `MLPoisson` 仅作为 P4 基线，legacy 曲线非正交 19 点算子延期决策”；禁止把两者视为替换关系。该设计门完成时曲线实现尚未开始；其后 P5-003/C0 已完成 G0.1 identity metric 数据模型，但交叉项、自定义算子/延迟修正和验证 case 仍未开始。
- `P5-001`：已实现保守 Cartesian 对流 RHS，面通量为 `Ucont[d]` 乘 `Ucat` 两侧中心平均，cell RHS 为负净面动量通量除一次体积；与 legacy `-second_order` 分支可比但不宣称曲线等价。2026-08-28 locked 26.04 CPU CTest 13/13 PASS；16/32 周期制造场误差比约 3.92，物理入口/出口多 Box 常量贯流 RHS 为零。MPI/CUDA runtime 与 CFD case 未验，故仍为进行中。
- `P5-002`：已实现单层均匀 Cartesian 常系数粘性 RHS，使用 `nu` 乘三方向 cell-centred 二阶中心 Laplacian；周期/inter-Box halo 与物理 no-slip ghost 分开处理，非周期方向要求显式 BC。2026-08-28 locked 26.04 CPU clean build、静态契约检查和 CTest 16/16 PASS；周期 16/32 制造解、动量守恒、负能量率及物理边界多 Box 通量平衡均 PASS。变系数、曲线 metric、IBM/EB、时间推进、MPI/CUDA runtime 和完整 CFD case 不在本任务范围。
- `P5-003`：C0/G0.1、C1/G0.2、C2.1/C2.2 与 C5.1/C5.2 已建立 separable analytic orthogonal metric、变换、正交投影和受限物理边界。2026-09-18 的 C2.3 又把 conservative mapped advection、metric-aware constant-coefficient viscosity、稳定性 guard 与显式 Euler/正交投影串接起来；16/32/64 三网格对流与黏性均呈二阶误差比，非平凡 predictor 和 free stream 一步误差为 0。C6 同日完成 schema-v3 mapping provenance checkpoint/restart。C8-G1 随后冻结正交 stretched-channel Poiseuille 物理 reference，速度、稳态残差与流量三网格近二阶，压力 datum-free 误差和散度为零，并追溯 legacy profile/pressure-gradient 源码。2026-10-11 C7 补齐周期与入口/出口正交 mapped 的 1/2/4-rank、不同 Box 分解和 CPU/CUDA 紧容差对照；一般非正交/曲面、MPI/GPU restart、分项性能与 legacy 整机复演仍未完成，因此 P5-003 与完整 G1 保持进行中。
- `P5-004`：选择显式 Euler 预测加现有 Cartesian 投影作为最小基线，投影时间系数固定为 1；`Ucat/Ucont` 均显式维护 `n/n-1/n-2`，并记录 time/step/history depth。2026-08-28 locked 26.04 CPU clean build与 CTest 18/18 PASS；周期剪切扩散 `dt/dt/2` 误差比 2.0026、动量漂移约 $1.39\times10^{-17}$、散度/history 误差为 0，超限扩散数被预期拒绝。该路径不等价于 legacy SNES；半隐式/BDF2 留 P5-005。后续 P8-001/002 已补齐受限 Cartesian CPU 单 rank checkpoint/restart，但 MPI runtime 和完整 CFD 验收仍缺，故保持进行中。详见 `_Docs/AMReX_P5-004_时间推进设计及测试_20260828.md`。
- `P8-001/002`：实现单层均匀 Cartesian、单进程 CPU 的版本化 checkpoint/restart。使用 AMReX `VisMF` 保存 `Ucat/Ucat_old/Ucat_older`、三个方向 `Ucont` 的三层 history、`P/Phi/Nvert`，并保存/校验时间步、网格、ghost/layout/components、BC、精度、维度和 AMReX 版本。连续运行与第 K 步 checkpoint 后恢复至 N 步的状态比较为 bitwise 一致；损坏 Header 被严格拒绝。`build/amrex_port_p8` 的 clean build、静态检查、P8 2/2 和完整 CTest 20/20 PASS。当前不宣称 plotfile、MPI restart、GPU restart、AMR、IBM/FSI 或曲线网格支持；详见 `_Docs/AMReX_P8-001_P8-002_checkpoint_restart_20260829.md`。
- `P8-003`：实现单层均匀 Cartesian、单进程 CPU 的 cell-containing point probe、plane 统计与确定性 CSV、全域/边界 JSON 诊断，并将等间隔 post-step 时间平均接入 physical channel。`4x3x2` 多 Box 制造契约精确检查 probe、6 行 plane、散度、section flow、动量、动能及压力统计；clean focused 4/4、完整 CTest 23/23 PASS。该证据是 contract test，不是 CFD validation；不包含插值 probe、MPI/GPU、AMR coverage、IBM 力/力矩或 plotfile。详见 `_Docs/AMReX_P8-003_采样统计诊断_20260829.md`。
- `P10-001`：统一 `benchmark_cartesian_mms` 运行 Taylor–Green 制造解，16/32/64 三网格的对流和黏性 $L_2$ 误差比均大于 3.2；每个网格又以 `dt/dt/2` 验证显式时间误差下降。独立转录 legacy `RHSSolver.C` 的 `-second_order` Cartesian conservative stencil，逐点最大差为 0；该 reference 只证明该离散分支，不等于旧 PETSc 程序整机等价。CPU 与 CUDA `sm_89` 回归均纳入 CTest。
- `P10-002`：新增 named `moving_wall` 最小 Cartesian 边界及 `32x32x1`、周期 z、`Re=100` 方腔工程演示。显式 Euler 基线运行至 `tU/L=10`，保存字段/中心线/历史 CSV、JSON 与无平滑 Matplotlib 图；壁速重构误差和净壁面通量为 0，投影后散度最大值约 $4.69\times10^{-12}$。这是单网格 CPU 单 rank demonstration，不含 Ghia/legacy 对照、网格/时间步收敛或压力验证，故 P10-002 仅转为进行中；证据见 `run_cases/lid_driven_cavity/README.md`。

### 当前推进顺序（2026-10-12 核对）

本次逐行核对 53 个 P 级任务，统计为 14 已完成、20 进行中、19 未开始。
当前生产起点是受限全周期 affine shear 的 `face_authoritative + incremental_pressure`；
它已验证压力增量与数值失败提交保护；本轮补齐受迫 affine 波的非零压力空间参考，
一般流动的连续物理压力和时间层验收仍开放。最新范围和历史
后端精度门见 [增量压力报告](./AMReX_M2_增量压力状态与移位求解_20261012.md)。

|顺序|工作与交付|验收要求|依赖与边界|
|---|---|---|---|
|1，当前主线|建立非零解析压力 MMS/reference，明确显式 n 层压力与 n+1 存储槽的对应；记录去 datum 的 P、面压力梯度、动量残差及速度重构误差|先验证参考自身的散度与动量平衡；分别做空间三网格和固定终止时间的三档 dt 收敛，明确时间比较点；误差预期由独立推导冻结，不能沿用自比较通过来认定物理正确|先限定 CPU、全周期 affine shear、现有 Euler/component Laplacian；是否需要修正生产离散由参考结果决定|
|2，独立精度问题|定位初始面状态舍入经 1/dt 放大后的首步 P/Phi 差异；以现有真实基线比较 CPU 分块、MPI4 和 CUDA|保留原 relative L2≤1e-12 门、首步与完整字段；候选须通过旧路径兼容、独立 full residual/散度及成本检查|历史 CPU布局/CUDA 门仍 BLOCKED；局部配对差分/FMA 已否决，不能重复视为待交付方案|
|3，生产成本|分开计时 diagonal MLMG、cross action、halo、归约和 workspace；审查 multigrid hierarchy/smoothing|在同一硬件与输入下隔离计时，报告迭代数、内存及各部分时间；任何优化须保持数值验收|可先采集成本；性能结论以测量为准，范数复用减少六次归约不等于整体加速门完成|
|4，扩大可用范围|按目标 case 冻结非正交物理边界/pressure datum，再决定一般19点算子的 custom AMReX 或 PETSc/HYPRE 过渡路线；随后补 legacy 整机、restart 和工程验证|一般算子须有独立 action/stencil、BC/null space、网格收敛及跨 rank/backend 证据；restart 按后端单独验收|先有压力参考与收敛/成本依据；C4、SNES/半隐式/BDF2、IBM/FSI、AMR 不以受限切片通过而关闭|

本次实际复跑 `ctest --test-dir build/c3-20261011-cpu -R '^avwis_m2_pressure_increment$' --output-on-failure`，1/1 PASS；CPU 审计仍记录离散压力时间细化比约 2.00143、
`physical_pressure_convergence=NOT_TESTED`。本次未重跑 MPI/CUDA 或全套测试，
不更新其历史验收状态，也不改变任务状态。

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
|P0-004|数值基准|选择固定 Cartesian、曲线、IBM/FSI（若可运行）参考 case，记录网格、dt、容差|P0-003|case/控制文件 → 基准 manifest|`amrex_port/inputs/p0_cartesian_benchmark.in`、`tests/cartesian_benchmark.cmake`、JSON 输出|Cartesian manufactured case 可重跑并版本化|原树真实 case/外部网格缺失|进行中|Cartesian contract baseline 已可实际 build/run；8 步 CPU CTest PASS。明确不宣称 CFD 正确或 legacy 等价；曲线/IBM/FSI/真实 reference 仍 blocked|
|P0-005|接口冻结|字段字典：单位、索引、时间层、BC 码、pressure datum、输出命名和 I/O 策略|P0-004|基准字段 → interface contract|`amrex_port/benchmarks/CartesianBenchmarkCase.cpp`、`inputs/p0_cartesian_benchmark.in`、JSON manifest|基准字段与输出可追溯|`Ucont` 旧布局并非类型化 face MAC|进行中|benchmark JSON 固化网格/dt/nu/步数/BC/zero-mean datum、逐步指标和 AMReX/compiler 信息；物理单位换算、legacy BC 整数码映射和真实 case 仍未冻结，不能整体签字|

### P1 基础框架

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P1-001|骨架|保持 AMReX 初始化、ParmParse、Geometry/BA/DM 与基础字段布局|P0-001|输入参数 → 单层对象|`amrex_port/src/*`|静态代码存在，见上方证据|无物理算法|进行中|AMReX 26.04 CPU/MPI/CUDA builds compile; MPI and single-rank CUDA runtime contracts pass. cell/face/历史层/ownership 仍只是 P1 framework，不等同 CFD 移植|
|P1-002|构建|为骨架固定 AMReX 分支、CMake preset、CPU/MPI 选项和最小 inputs|P0-003|依赖版本 → 可配置工程|`amrex_port/{CMakeLists.txt,CMakePresets.json,inputs/*}`|clean configure/build 成功|AMReX package discovery/编译器不匹配|进行中|26.04 lock/SHA；CPU/MPI/CUDA arch-89 configure/build 成功。MPI consumer 因 AMReX package 导出 C MPI dependency，最小修复为 `project(... LANGUAGES C CXX)`；CUDA device link 和单 rank runtime 均通过，CUDA-aware MPI 未测试|
|P1-003|并行框架|验证 MultiFab 多 Box、MFIter、nGrow、periodic `FillBoundary` 与 MPI halo|P1-002|制造场 → halo 对比|`amrex_port/src/AVWiSSolver.cpp`、inputs|1/多 rank bitwise 或容差一致|physical ghost 不由 FillBoundary 填充|进行中|CPU MPI `p1_contract.in` 在 2/4 ranks PASS；`p1_multibox.in` 在 2 ranks PASS。CUDA 单 rank 同类 contract PASS；CUDA-aware MPI 未测试|
|P1-004|BC/运行设施|建立 `BCRec` 映射接口、物理 BC functor、日志/计时和错误报告|P0-005,P1-002|BC 字典 → 边界元数据/日志|`amrex_port/src/{CartesianBoundaryConfig,AVWiSBoundary,AVWiSDiagnostics}.*`|六面 BC 分类完整，未实现类型显式拒绝|旧整数码有算例特例|进行中|已建立 `ext_dir` BCRec、受限 Cartesian/orthogonal mapped 物理 ghost/face pipeline、参数校验和 rank 输出/计时；未支持旧码显式拒绝，完整 legacy 映射和性能计时仍缺|
|P1-005|基础 I/O|建立 plotfile/checkpoint 元数据、字段注册与版本头|P1-002,P0-005|字段注册 → 空场 plot/checkpoint|`amrex_port/src/AVWiSSolver.*`|空场可读回并校验 schema|与 PETSc 文件格式不兼容|进行中|字段/schema-only JSON 已注册；P8-001/002 已有受限 CPU 单 rank VisMF checkpoint/restart payload，但 plotfile/HDF5 与通用后端 I/O 仍缺|

### P2 字段与网格

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P2-001|字段|定义 `P/Phi/Nvert` cell MultiFab、时间层和 component 命名|P0-005,P1-003|契约 → scalar fields|`amrex_port/src/AVWiSSolver.*`、`inputs/p2_contract.in`|IndexType/nGrow/单位语义字段断言；未决物理换算显式标出|把 `Nvert` 当 EB volume fraction|已完成|字段契约含 component/time-layer/units；`P/Phi` 同为 legacy 无量纲压力尺度，物理换算/pressure datum 仍属 P0/P4；`Nvert` 保持分类语义|
|P2-002|速度|定义三个 face `Ucont` 与 cell 三分量 `Ucat`，冻结体积通量和 face ownership|P2-001|字段 → MAC/cell 布局|`amrex_port/src/AVWiSSolver.*`、`inputs/p2_contract.in`|常量速度给出 $Ucont_d=u_dA_d$；唯一面数/通量和正确|把 legacy 积分通量当 MacProjector 面速度|已完成|三个独立 face `MultiFab`；共享面最低 global box 为 owner，归约用 `OwnerMask/sum_unique`；legacy 依据为 `FormMetrics/Contra2Cart/CalculateDivergence`|
|P2-003|变换|实现/测试 Cartesian `Ucont↔Ucat`；曲线变换显式留为未实现边界|P2-002|Ucat/Ucont → 同步场|`AVWiSSolver.*`|常量/线性场按 face area 乘除且单位一致|曲线 metric 不能以 Cartesian 面积代替|已完成|cell→face：法向速度线性平均后乘面积；face→cell：相邻通量和除以 `2A_d`；共享 valid face `OverrideSync` 后再 halo；曲线实现仍在 P5|
|P2-004|网格|建立单层 Cartesian 坐标、dx、cell volume 和 metric 数据命名|P1-003,P0-005|domain/n_cell → Geometry/metric metadata|`AVWiSSolver.*`|$A_d\Delta x_d=V$ 与 Geometry 一致；`Aj` 定义不含糊|`Aj`/J/1/J 混淆|已完成|记录 dx、volume、face area；仅登记未分配的 `legacy_Aj_equivalent=1/V`（unit-index 计算坐标），不声称曲线 metric/Jacobian 场|
|P2-005|布局回归|多 Box/MPI 上验证 cell/face ghost、边界 face 与 derived field 一致性|P2-001--004|制造场 → layout report|`inputs/p2_contract.in`、`p2_boundary_face.in`|1/多 rank 范数、唯一 face sum、face ghost 和 divergence stencil 一致|跨 Box face 双计数|已完成|周期/非周期 face ghost、OwnerMask 唯一计数/通量和、非周期边界邻接单元外推、净面通量/体积 derived divergence；只是 manufactured layout regression，不是全域守恒或物理 BC|

### P3 边界与数据流

|ID|模块|任务描述|依赖|I/O|文件|验收|风险|状态|备注|
|---|---|---|---|---|---|---|---|---|---|
|P3-001|BC 分类|把旧 BC 整数码与每个面速度/压力/标量条件逐项映射，标出未支持项|P0-005,P1-004|BC 文件 → 显式 BC 表|`BcsUtility.C:672-1406`,`Integrator.C`、`amrex_port/src/*`、P3 设计记录|每个已用 case 无隐式默认分支|入口/出口与压力条件耦合|已完成|旧 1/3/4/5 和独立 periodic 有显式映射；新 named slip 不冒充旧码；0/2/6/8/10/11/12/13/14/-1/-2 显式拒绝；无旧 case，4/5 仅承诺文档中的 Cartesian 子集|
|P3-002|ghost 流程|实现周期/非周期 `FillBoundary`、物理 ghost 填充和调用顺序|P2-005,P3-001|fields+BC → valid ghost|`amrex_port/src/AVWiSSolver.*`、`inputs/p3_*`|逐面制造解和跨 rank halo 一致|误把 FillBoundary 用作物理 BC|进行中|已分离 halo 与 physical fill，固定 OverrideSync/FillBoundary→physical→diagnostics，并有 epoch stale 拒绝；2026-08-24 CPU CTest 7/7 PASS，MPI build/link PASS 但 2-rank 在 `MPI_Init` 前被 PMIx socket 权限阻塞；多层 FillPatch 留 P9|
|P3-003|流量 BC|实现入口流量/平面输入与出口压力/通量约束的最小 Cartesian 路径|P3-002|profile/target flux → face Ucont|`amrex_port/src/AVWiSSolver.*`、`inputs/p3_cartesian_boundary.in`|全局净通量、面 profile 和压力 datum 达容差|旧 case 特定分支|进行中|实现 uniform/linear-plane MPI 全局归一化、fixed pressure ghost、可选等流量出口；不是压力投影/datum 求解；CPU P3 PASS，MPI multi-rank runtime BLOCKED；CUDA compile/device-link PASS 但无设备 runtime|
|P3-004|一致性诊断|在每个阶段核算 halo、边界面、全局质量通量和 ghost freshness|P3-002,P2-005|fields → diagnostic report|`amrex_port/src/AVWiSSolver.*`、`inputs/p3_*`|错误注入能被检测；MPI reductions 正确|只报 interior divergence 掩盖边界/IBM|进行中|实现 stage epoch、boundary owner-unique flux、全域 divergence integral/Linf、显式 sum/max MPI reduction 和 stale 注错；无 IBM；CPU PASS，MPI reduction runtime 因 PMIx socket BLOCKED，CUDA runtime 因设备/驱动不可见 BLOCKED|

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
|P5-001|对流|实现 Cartesian 对流通量与旧 `-second_order` 分支的可比 stencil 说明|P3-004,P2-002|Ucat/Ucont → advective RHS|`RHSSolver.C`、`amrex_port/src/AVWiSAdvection.cpp`|平流制造解与网格收敛|面/cell 插值位置错误|进行中|保守 face flux=`Ucont[d]`×两侧 `Ucat` 中心平均，RHS 为负净通量/体积；locked 26.04 CPU 13/13 PASS，16→32 制造场 L∞ 误差比 3.92，边界多 Box 常量贯流 RHS=0；MPI/CUDA/CFD case 未验，且不宣称曲线等价|
|P5-002|粘性|实现常系数粘性、边界通量与动量守恒测试|P3-002,P5-001|Ucat,nu → viscous RHS|`amrex_port/src/AVWiSViscosity.cpp`|扩散制造解与能量耗散|wall model 与基本粘性混淆|已完成|locked 26.04 CPU clean build、静态契约检查、CTest 16/16 PASS；周期 16/32、物理边界多 Box 均 PASS；变系数/metric/时间推进后置|
|P5-003|metric 原型|实现曲线 metric 读入/计算、导数/面 metric，并以制造解验证|P2-003,P2-004,P4-005|曲线网格 → metric fields/operator evidence|`amrex_port/src/AVWiSCoordinateMapping.*`、`AVWiSMetricData.*`、`AVWiSMappedOperators.*`、`AVWiSDeferredProjection.cpp`、`AVWiSBoundary.cpp`、`AVWiSProjection.cpp`、`AVWiSAdvection.cpp`、`AVWiSViscosity.cpp`、`AVWiSTime.cpp`、`benchmarks/MappedChannelReferenceCase.*`、`tests/contracts/AVWiSC{2,3,5,7}ContractChecks.cpp`、`tests/c7_backend_compare.py`、`CurvGrid.C:312+`|metric identity、GCL/体积、正交算子、mapped RHS/time、受限 mapped BC 与正交物理 reference；19 点项仍后置|非正交交叉项和 Jacobian 缩放|进行中|C0/C1、C2.1--C2.3、C5.1/C5.2、C6 CPU restart 及 C8-G1 已完成；2026-10-11 C7 周期/入口出口 1/2/4 rank 与 CPU/CUDA 全场对照 PASS，CPU 47/47、MPI 48/48；2026-10-11 C3 新增受控全周期 xy shear/cross production flux、黏性/project/step，三网格 MMS、退化/守恒/失败保护及 MPI/CUDA 子集通过；GPU/MPI restart、分项 profiling、C3 总门/C4、一般曲面和 legacy 整机仍缺；见 [`AMReX_C7_MPI_CPU-GPU对照_20261011.md`](./AMReX_C7_MPI_CPU-GPU对照_20261011.md)|
|P5-004|时间推进|比较显式、半隐式与 BDF2；固定历史场和时间系数|P5-001,P5-002,P4-004|RHS/history → U*|`Integrator.C:326+`、`amrex_port/src/AVWiSTime.cpp`|时间收敛、CFL、守恒、restart 时间层一致|旧 SNES 分支不可被默默抹平|进行中|2026-08-28 选择显式 Euler+投影系数 1 的最小 Cartesian 基线；CPU clean/CTest 18/18 PASS，时间误差比 2.0026、守恒/散度/history 与 CFL 拒绝 PASS；不等价于 legacy SNES。MPI runtime 与 CFD case 仍 BLOCKED；checkpoint/restart 已由 P8-001/002 覆盖；半隐式/BDF2 留 P5-005|
|P5-005|SNES 策略|决定保留 PETSc SNES 过渡接口或以明确线性/非线性 AMReX 语义替代|P5-004,P0-005|残差/Jacobian 契约 → 决策|`Integrator.C:88,168,326+`|残差、停止条件、失败处理有回归|“显式一步”等同 SNES 收敛的错误结论|进行中|2026-10-11 源码审计冻结可达半权重 nonlinear residual 与不可达 BDF2 分支（getTimeCoeff=1）；新增 time_integrator=explicit_euler 与 viscous_operator=component_laplacian 的明确选择/拒绝接口。此为限定决策/接口完成，第三增量补 M2 状态声明拒绝及累计压力时间不收敛反例；SNES 残差/Jacobian、停止与失败回归及最终隐式方案仍开放；见 [`AMReX_C3_弱剪切算子与算法决策_20261011.md`](./AMReX_C3_弱剪切算子与算法决策_20261011.md)|
|P5-006|LES|迁移 Smagorinsky/壁面相关最小路径、二阶插值和 SGS 诊断|P5-001--005|速度/metric → nu_t/RHS|`LESModel.C`,`RHSSolver.C`,`WallModel.C`|无 LES 回归、Cs/nu_t 剖面对比|滤波/裁剪/IBM 例外未冻结|未开始|AMR dynamic LES 留 P9|

#### P5-003 曲线路线子模块状态

下表是 P5-003 的唯一子模块状态记录；它不改变页首 53 个 P 级任务的计数。
子模块共 9 个：C0、C1、C2.1--C2.3、C6 和 C8-G1 已完成，C5.1/C5.2 的
separable-orthogonal 边界子集已完成；完整 G1 仍等待 C5 一般曲面范围、C7 剩余验收
和 legacy 整机证据；C3 已进入受限周期弱剪切实现，C4 尚未开始。P5-003 因此仍为“进行中”；正交子集完成不代表
solver 已支持一般曲线坐标。

| 子模块 | 范围 | 依赖 | 验收摘要 | 状态 |
| --- | --- | --- | --- | --- |
| C0 / G0.1 | 独立 identity `CoordinateMapping/MetricData` 数据模型与 CPU 契约 | P2-003/004、P4-005 | 正体积/互反/GCL、identity、`Ucont=u·S`、多 Box/ghost；证据见 [`AMReX_P5-003_G0_identity_metric_20260831.md`](./AMReX_P5-003_G0_identity_metric_20260831.md) | 已完成 |
| C1 / G0.2 | Identity metric adapter 接入一条 solver `Ucat↔Ucont`/散度路径 | C0、P2、P4 Cartesian 基线 | adapter 前后逐值或紧容差一致，完整 Cartesian CTest 不退化；epoch/layout/非 identity 拒绝；尚需后续 G0 投影接受项才能关闭 G0 | 已完成 |
| C2 / G1 | 解析正交 mapping 与对角 metric 算子 | C1 | C2.1/C2.2 geometry、gradient/divergence、transform、投影；C2.3 三网格 mapped 对流/黏性二阶收敛、非平凡一步推进及 free stream；C8-G1 已补物理 reference，完整 G1 仍由 C5/C7/legacy 证据关闭 | C2.1--C2.3 已完成；完整 G1 进行中 |
| C3 / G2 | 弱非正交 mapping、cross flux、metric-aware 对流/黏性 | C2、M2/M4/M6 | 2026-10-11 固定 xy shear（abs(epsilon)≤0.25）全周期生产算子/RHS/project/step；16/32/64 MMS、epsilon=0 回到正交投影、±0.1/±0.25 扫描、affine 任意向量 11 点 action、守恒、全残差、失败无 flow-state 修正与 100 步自由流 PASS；见 [`AMReX_C3_弱剪切算子与算法决策_20261011.md`](./AMReX_C3_弱剪切算子与算法决策_20261011.md) | 进行中：限定弱剪切切片已验证；一般 G2/M4 成本/物理 case 未关闭；第二增量 workspace/四步契约见 [续报](./AMReX_C3_多步状态与workspace_20261011.md)；第三增量长程/压力时间门 BLOCKED 见 [时间审计](./AMReX_C3_时间审计与M2拒绝契约_20261011.md) |
| C4 / G3 | 一般非正交与完整 19 点压力路径 | C3、M3/M4/M5 | operator action、逐项 stencil、BC/null space、legacy case、MPI/收敛/性能；C3 的 xy-shear 11 点子集不构成 legacy 19 点等价，M5 custom/PETSc 选择仍开放 | 未开始 |
| C5 | 曲线 BC/ghost/face owner 流水线 | C1；曲线验收依赖 C2/C3、M3 | C5.1/C5.2 已验 no-slip、inlet/outlet、tangential moving、slip/symmetry、周期平移、角 ghost、多 Box owner、stale/非法配置拒绝和匹配投影；C7 已补周期/入口出口及 slip/symmetry 的 MPI 分块对照；一般曲面仍后置 | 进行中：C5.2 separable-orthogonal 子集已完成 |
| C6 | 曲线 checkpoint/restart 与 mapping provenance | C1、C2、C5 | schema-v3 保存 mapping/type/参数/checksum/metric 版本；同分解 bitwise continuation、跨 Box 分解恢复、错配严格拒绝 | 已完成：CPU 单 rank；MPI/GPU/AMR restart 不在本子门范围 |
| C7 | 曲线 MPI/GPU runtime | 各模块 CPU 参考；最终依赖 C1–C6 | 1/2/4 rank、分块不变性、CPU/GPU 紧容差及硬件/性能记录 | 进行中：2026-10-11 正交周期/入口出口 MPI 12 组、CPU/CUDA 6 组跨后端对照 PASS；CPU 47/47、MPI 48/48；全场相对 L2 最大 `5.03e-14`；完整 CUDA 结果见报告。分项 profiling、CUDA-aware MPI、MPI/GPU restart 尚缺 |
| C8 | 物理曲线 case 与回归 | C2、C5、C6；一般曲线声明再依赖 C3/C4/C7 | 正交 stretched-channel Poiseuille 三网格速度/残差/流量近二阶，压力 datum-free 误差与散度为零；壁面剪切至少一阶；legacy 源码语义已追溯，整机复演尚缺 | 进行中：C8-G1 已完成；一般 curved/legacy executable 未完成 |

详细目标、主要源码位置与执行顺序见
[`AMReX迁移方案.md` 的“当前状态、剩余工作与下一步路线”](./AMReX迁移方案.md#当前状态剩余工作与下一步路线)；
数学/API 门禁仍以 [`AVWiS曲线坐标实现规格.md`](./AVWiS曲线坐标实现规格.md) 为准。

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
|P8-001|字段输出/checkpoint|定义并实现单层 Cartesian checkpoint 字段名、时间层、网格/BC 元数据与版本化 payload|P0-005,P1-005|fields → VisMF payload + Header|`amrex_port/src/AVWiSCheckpoint.cpp`、`amrex_port/src/AVWiSSolver.H`|所有基础流体字段可写出并由 schema 校验读回|漏存 U(n-1)/布局或误称 plotfile 兼容|已完成|2026-08-29；仅单层均匀 Cartesian、单进程 CPU；不包含 plotfile/HDF5、MPI/GPU、metric/IBM/FSI metadata；见 P8 报告|
|P8-002|restart|恢复基础流体所有时间层并验证连续跑 vs 重启的一致性|P8-001,P5-004|checkpoint → identical state|`amrex_port/src/AVWiSCheckpoint.cpp`、`amrex_port/tests/p8_*`|连续跑 vs 重启 bitwise 一致；损坏 Header 拒绝|未来扩展时漏存 geometry epoch/IBM/FSI|已完成|2026-08-29；CPU CTest 20/20，P8 2/2；只验收单进程基础流体 restart，P7/P9/P10 仍需扩展证据|
|P8-003|采样/统计|实现 probe、plane、平均量、力/力矩、质量、散度、动能与压力诊断|P3-004,P5-006|fields → CSV/plot diagnostics|`amrex_port/src/AVWiSDiagnostics.cpp`、`benchmarks/PhysicalChannelCase.cpp`|固定点/平面与 MPI reduction 对照|采样跨 Box/AMR 覆盖|已完成|2026-08-29 限定完成 CPU 单 rank 均匀 Cartesian subset：cell probe、plane/CSV、等间隔时间平均、散度/质量/出口/动量/动能/压力；制造 contract 与 physical integration PASS。MPI/AMR/GPU、插值、IBM 力/力矩及 plotfile 明确后置|
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
|P10-001|制造解|Taylor–Green/解析投影、对流/扩散空间时间收敛与压力零空间回归|P4-004,P5-004|tests → convergence report|`amrex_port/benchmarks/TaylorGreenMMSCase.cpp`、`tests/p10_taylor_green_convergence.cmake`|预声明阶数/容差达标|用单一范数隐藏局部错误|已完成|16/32/64 三网格对流/黏性二阶；各网格 `dt/dt/2` 时间误差下降；periodic pressure datum；legacy `-second_order` source-stencil 逐点一致；CPU/CUDA PASS。不是旧程序整机等价|
|P10-002|基准流动|通道/腔流、入口出口、质量/动能/压力与单多 MPI rank 回归|P5-006,P8-003|cases → regression baselines|`run_cases/lid_driven_cavity/`、目标 tests/docs|所有 P0 指标达标|BC 与诊断不闭合|进行中|2026-08-29 已有 Re=100 方腔 CPU 单 rank 工程演示与基本 sanity；无 reference/网格收敛/MPI，不能标记完成|
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

### 2026-10-11 算法门追记

|门|本轮明确决策与实现|未关闭条件|
|---|---|---|
|M2|固定 shear 的 normalized 六面 LS 三角特化；常量精确，法向条件数上界约 1.641；零 shear 回到原对角平均|一般面法向/legacy 三 cofactor 对照仍开放；第三增量明确双表示/累计压力契约，压力时间门 BLOCKED、single authority/per-field epoch 未冻结|
|M3|首个 shear 路径只接受 periodic，检查积分 RHS compatibility，Phi 以物理体积加权 zero mean；物理 BC 配置拒绝|legacy case pressure BC/datum 及其均值修补不能从当前通用 BC 推定|
|M4|共享 full/cross flux 的 deferred correction；全物理相对残差≤1e-11，最大 40 次外迭代；不足即失败。受控剪切扫描 PASS|已补 256 步混合模态与时间步审计；2026-10-12 opt-in 投影消除六次重复速度范数扫描/归约，全场 exact 回归；实测分项成本/内存、强拉伸、物理 case 与普适生产许可未关闭|
|M5|C3 暂用对角 MLABecLaplacian + 显式 cross；已验证 affine 独立 action|C4 custom MLLinOp 或 PETSc/HYPRE 过渡仍 OPEN，未建立 custom 多层原型|
|M6|明确保持常系数 component Laplacian，压力/黏性共享 face stencil；legacy RHSSolver 转置梯度已审计，unsupported stress 模式拒绝|完整应力散度、变黏度、legacy 逐项等价与其 MMS 未完成|

依据、数学式与命令见 [`AMReX_C3_弱剪切算子与算法决策_20261011.md`](./AMReX_C3_弱剪切算子与算法决策_20261011.md)。以上限定决策不把任何完整 M3–M6 门记为完成。

## 建议的下一步（按顺序）

1. C3 已补 256 步与固定时间 dt/dt/2/dt/4 审计；先解决累计压力时间门 BLOCKED 的数学决策：预测步旧压力梯度、P/Phi 的物理语义、DI(Ucat) 与 DUcont 的约束接缝和投影幂等性，再冻结单一权威/per-field epoch。保留细 dt 的 Phi backend BLOCKED，不以速度一阶或共同 halo epoch 代替这些门。M4 继续拆分 diagonal/cross/halo/分配及浅 hierarchy 成本，不先扩大支持范围。
2. 冻结 C3 pressure 与 Ucat/Ucont 权威同步策略，设计非正交物理 BC；保持 epsilon=0 与 C2 紧回归。随后以证据决定 M5 的 C4 custom 19 点或过渡路线。
3. 继续 P5-005 隐式 residual/Jacobian/停止/失败契约与 M6 完整应力分支；获取 legacy 输入、网格、reference 与 PETSc/HYPRE 依赖，不以源码/解析解替代整机复演。
4. C7 剩余 profiling、CUDA-aware MPI、MPI/GPU restart 与 P10-002/P6-001 作为独立后续门；本轮已保留 C7，未重复扩充其 comparator。

### 2026-10-11 Pressure/Phi reference 诊断增量

新增最小压力 reference、独立串行 DG/DIC/diagonal/cross action 与同 RHS 重放，
保持 `experimental_dual_state`、显式 Euler alpha=1、P 累计 correction 和原拒绝规则。
初态 `DI(Ucat)=0.085266493` 与 `DG−DIC` 的约束缺陷一致；非恒定旧 P 扰动不改变
真实一步的 Ucat/Phi。Box 布局、MPI4、CUDA 的短轨迹 Phi 原相对门仍 BLOCKED，
但输入逐值相同 RHS 时差仅约 1e-16；把候选 RHS 交回 CPU 可恢复其 Phi 至同等精度。
证据把本反例定位到状态/RHS 的舍入差经散度和 1/dt 放大，未发现需要换一般算子的依据。
可选 MLMG trace 只观察既有残差/迭代数，默认无新增归约、容差或停止条件变更。
这是诊断/契约进展，两个原门继续 BLOCKED；不改任何 P 级任务状态，
53 项仍为 14 已完成、20 进行中、19 未开始。CPU 14/14、MPI 15/15、CUDA 14/14
定向 CTest 与三后端单网格 C3 失败恢复契约通过；没有重复 480 步长测。
详见 [Pressure/Phi reference 根因诊断](./AMReX_Pressure-Phi_reference根因诊断_20261011.md)。


### 2026-10-11 M2 一致初态与失败事务诊断增量

新增 test-only affine shear 离散 streamfunction 初态：cell 权威一次构造 Ucont=I(Ucat)，
初态 DI/D 均约 6.62e-15，非零 DG multiplier 与独立 Fourier reference 通过；
DIC source residual 约0.14645，even grid有8个checkerboard null modes，不能把DG当DIC。
4/8/16步固定T=0.002的Ucat/三面差比近2，累计P约0.49999、末Phi约0.30083，仍BLOCKED。
直接face重投影的flow字段幂等PASS但Phi被清零，全字段幂等BLOCKED；先I重建再投影不幂等。
真实整步deferred exhaustion准确拒绝，current/history/epoch已改变，整步rollback BLOCKED。
新affine CPUlayout16/MPI4/CUDA六transport字段对照PASS；CUDA全部诊断字段仍13项BLOCKED。
这不是原stretched pressure/Phi门PASS：原门保留历史BLOCKED，本轮未重跑长矩阵。
本轮生产源码保持起始hash；定向CTest CPU13/13、MPI14/14、CUDA13/13通过，
数学决策strict driver准确exit=1并保留完整证据。当前主统计仍14/20/19（53项），
M2/P5-005/M4/C3/C7进行中、C4未开始。
详见 [M2一致初态与失败事务诊断](./AMReX_M2_一致初态与失败事务诊断_20261011.md)。


### 2026-10-11 M2 opt-in 面守恒权威生产增量

新增真实生产 `advance_one_step()` 分支，显式选择
`state_contract=face_authoritative`、`pressure_contract=instantaneous_pressure`、
`failure_contract=fatal`。仅支持非零 affine `analytic_shear`（stretch=0 0 0）、
全周期、无物理 BC、现有 explicit Euler/component Laplacian；默认仍为
`experimental_dual_state` + `accumulated_correction`，没有默认切换。
新面 predictor 为 Ucont[n]+dt*I(advection+viscosity+force)，保持面权威；成功 DG
投影后以现有六面归一化剪切重构 R 得 Ucat=R(Ucont)，不会先以 I(Ucat) 覆盖面状态。
`instantaneous_pressure` 严格指该次 DG 投影的离散 face 约束乘子，P=Phi；其 RHS
在显式 n 层评估，不能宣称发布于 n+1 的 P 已通过连续物理压力/时间层验收。
旧累计输出为 P+=Phi correction history，旧 P 不消费于 predictor。

投影面/cell 候选全部准备后才提交。真实 exhaustion 不提交 P/Ucat/投影修正，
但面 predictor、history、epoch 可改变；整步仍 fatal，`failure_contract=rollback`
准确拒绝。没有全 advance rollback、C4/SNES/BDF2/完整应力/IBM/FSI。
固定 T=.002 的 CPU 4/8/16 步速度/面差比近2，P/Phi 离散乘子差比2.00143；
限定时间子门通过不替代物理压力验收。layout16/MPI4/CUDA 完整337字段中
首步 P/Phi 原 relative<=1e-12 门仍 BLOCKED，其余335字段 PASS；原 stretched
压力/Phi门未重跑，保留 BLOCKED。主统计仍 **14 已完成、20 进行中、19 未开始**，
M2/P5-005/M4/C3/C7 进行中，C4 未开始。实现、最终验证及边界见
[面守恒权威生产增量报告](./AMReX_M2_面守恒权威生产增量_20261011.md)。

### 2026-10-12 当前优先顺序与 M4 安全增量

两种切向 stencil 舍入候选分别使 MPI4 首步子门回退或 CUDA 差异增大；第一种
还改变 CUDA 旧默认路径的101个字段，均不予保留。不能据微小舍入改善推进 C4。
本轮转为当次面权威投影的速度范数复用；缓存仅在 Ucat 不变的尺度计算块有效，
整步失败后重试仍重新计算。没有新增数值测试矩阵，原 CTest 和 comparator 保留。
下一步仍先设计初始 projection/归约/面修正的可复现精度方案并评估成本；M4 的
六次归约消除只是局部工作量减少，分项计时、普适性能许可和 M2/C3 总门保持开放。

### 2026-10-12 当前主线：增量压力生产推进

本轮优先顺序覆盖上一条“先做精度/范数复用”的建议：已交付受限 opt-in
`incremental_pressure` 的真实 Euler predictor、当步增量和累计 P 状态机；默认路径
保持。直接小增量 RHS 在 dt/2 第二步出现内层相对残差舍入底，选择同算子的移位
未知量 Q=P_old+Phi；并重新审计实际提交压力的 full residual，未改原门限。
M2/C3/P5-005 继续进行中；该接口只关闭有限范围的压力增量状态子契约，不关闭
连续物理压力及其时间层验收。C4/SNES/BDF2 均未进入。
下一项算法工作应建立显式 n 层面乘子的物理压力 reference 和时间层对应，检验
空间重构/压力梯度与动量残差；首步 CPU布局/CUDA 压力门保留 BLOCKED，不能以本轮
压力增量子门通过替代它。命令、失败边界及结果见上述实施报告。
