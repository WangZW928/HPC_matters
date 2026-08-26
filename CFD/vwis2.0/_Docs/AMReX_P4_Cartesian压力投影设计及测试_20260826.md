# AMReX P4 Cartesian 压力投影设计及测试（2026-08-26）

## 1. 结论与范围

本增量实现并验证了一个最小、单层、均匀 Cartesian 压力投影子契约：从积分面
体积通量 `Ucont[d]` 形成散度/RHS，经 AMReX 26.04 `MLPoisson`/MLMG 求解压力
修正 `Phi`，修正同一个 face flux，再同步 `Ucat`。CPU clean CTest 10/10 PASS；
其中原 P0--P3 七项全部保持 PASS，新增周期、封闭 Neumann、入口/定压出口三项
P4 runtime test。

这不是完整 P4/CFD 验收：没有动量 predictor、时间推进、物理 reference case、
MPI runtime PASS、曲线 metric/19 点算子、IBM/FSI、LES、AMR 或 restart payload。
因此交付结论是“verified P4 Cartesian sub-contract”，不是完整 VWiS CFD 或完整
P4 发布门通过。

## 2. legacy 证据与离散约定

`PoissonSolver.C:209+` 的旧 RHS 从三个 `Ucont` 方向做净通量差，并乘
`-1/dt * St * timeCoeff`；`Projection:2352+` 再以 `dt*St/timeCoeff` 和曲线
metric 修正 `Ucont`。`PoissonLHS:891+` 明确组装含非正交交叉项的 19 点矩阵。
P4 不逐行复制旧符号，而冻结下列自洽 Cartesian 约定（本路径取 `St=1`）：

$$
D(U^*)_c = \frac{\sum_d(U^d_{f+}-U^d_{f-})}{V_c},\qquad
\nabla^2\Phi = \frac{\alpha}{\Delta t}D(U^*),
$$

$$
U^d_f \leftarrow U^d_f-\frac{\Delta t}{\alpha}A_d
\left(\frac{\partial\Phi}{\partial x_d}\right)_f.
$$

其中 `alpha=vwis.projection_time_coefficient>0`，`dt>0`。`Ucont[d]` 始终是
$u_dA_d$；散度只除一次 cell volume，不再把它当速度除以 `dx[d]`。AMReX
`MLMG::getFluxes` 返回线性算子的 signed face flux（`-grad(Phi)` 约定）；适配层
按其符号相加，并显式乘 $A_d$，等价于上式。由此 AMReX 接口和 VWiS 存储之间
不存在隐式 velocity/volume-flux 语义切换。

求解完成后 `P += Phi`，共享面执行 `OverrideSync`/`FillBoundary`，随后从修正后
的两个相邻积分面通量除面积并平均得到 `Ucat`。P3 的 pre-projection outlet
总流量约束不会在投影后再次覆盖 pressure-outlet face correction。

## 3. pressure BC、compatibility 与 datum

|速度/Geometry 边界|`Phi` 条件|face correction|
|---|---|---|
|periodic|periodic|周期梯度修正|
|noslip/slip/symmetry|homogeneous Neumann|法向边界通量不变|
|inflow|homogeneous Neumann|规定入口通量不变|
|fixed-pressure outflow|homogeneous Dirichlet `Phi=0`|允许修正出口法向通量|

P4 支持 P3 的“一入口+一定压出口”组合，并新增显式封闭 no-penetration 组合用于
全 Neumann 投影；其他入口/出口计数组合拒绝。未开启 `vwisbcs` 的非周期 Geometry
也拒绝，避免隐式 pressure BC。

全周期或全 Neumann 系统保留常数 null space。代码先计算 MPI-global RHS mean，
用与 RHS max norm/机器精度相关的容差检查 compatibility，并关闭 MLMG 的自动
solvability repair；不兼容 RHS 明确报错，绝不自动减 RHS 均值。兼容 solve 后只
减 `Phi` 的 global mean 选择零均值 gauge。存在定压出口时系统非奇异，该出口
提供 correction-pressure datum。

## 4. runtime 测试

- `p4_periodic_projection.in`：三方向周期、多 Box；常压/零通量不改变速度；正弦
  integrated flux 制造散度；非零均值 RHS 注错必须被拒绝。
- `p4_closed_neumann_projection.in`：六面显式 no-slip、法向通量为零；兼容正弦
  模式、compatibility 拒绝和零均值 gauge。
- `p4_inflow_outflow_projection.in`：P3 x-low uniform inflow、x-high fixed pressure
  与 constrained pre-projection flux、其余 wall；验证 Dirichlet outlet 可修正，
  Neumann 面保持规定通量且散度下降。

CPU 定向输出：

|case|max divergence before|max divergence after|结果|
|---|---:|---:|---|
|periodic sine|6.211657082|7.790620001e-12|PASS|
|closed Neumann sine|6.211657082|3.956213135e-11|PASS|
|inflow/fixed-P outflow|18|8.498268755e-11|PASS|

## 5. 构建与执行证据

锁定源码为 AMReX tag `26.04` / commit
`9219ba416b7ba2073dd1b12bf19fdce27391f17b`，double、3D、C++17。本轮在 detached
Git checkout `/tmp/amrex-2604-git.xDMDBR` 构建，以确保 AMReX CMake 从 tag 导出
`AMReX_VERSION=26.04`。

|证据|命令/目录|结果|
|---|---|---|
|静态契约|`bash CFD/vwis2.0/amrex_port/tests/static_contract_check.sh`|PASS|
|CPU AMReX install|`/tmp/amrex-2604-git-install.3VEY8J`；MPI/Omp/GPU OFF，linear solvers ON|PASS|
|CPU clean configure/build|`/tmp/vwis-p4-final-cpu-build.MO2uwj`，Release，locked package|PASS|
|CPU CTest|`ctest --test-dir /tmp/vwis-p4-final-cpu-build.MO2uwj --output-on-failure`|PASS 10/10|
|CPU direct P4|periodic 与 inflow/outflow inputs|PASS；数值见上表|
|MPI AMReX install|`/tmp/amrex-2604-mpi-install.iJspRt`；OpenMPI 4.1.6，MPI ON|configure/build/install PASS；wrapper 启动时报告受限 socket，但编译/link 完成|
|MPI port build|`/tmp/vwis-p4-mpi-build.Kk1LZ0`|configure/compile/link PASS|
|MPI singleton P4 CTest|三项 P4|BLOCKED 0/3；均在 `MPI_Init` 前因 `socket() errno=1`、PMIx listener 无法启动|
|MPI 2-rank periodic|`mpiexec --oversubscribe -n 2 ... p4_periodic_projection.in`|BLOCKED，exit 1；同一 PMIx/socket 边界，未进入应用代码|

MPI 结果不是 FAIL 的数值证据，也不是 PASS；它只证明 MPI configuration/build/link，
runtime 被当前 sandbox 的本机 socket 权限阻塞。未进行 CUDA 构建/runtime：P4 最低
要求为 CPU CTest，当前宿主先前已记录 GPU 设备不可见；不能由 CPU 推断 GPU。

## 6. 曲线算子决策与剩余限制

P4 选择 `MLPoisson` 只作为 Cartesian 七点 Laplacian 基线。legacy
`PoissonLHS` 的非对角 metric 交叉项产生 19 点 stencil；标准 Cartesian projector
既不表达这些项，也不处理旧 `Nvert` fluid-DOF 压缩、IBM/FSI 或 case-specific
pressure datum。因此禁止把本路径作为 legacy 曲线算子的 drop-in replacement。

P4-005 的设计门结论是延期曲线实现：P5 metric 原型必须先提供 cell/face metric、
逐项 stencil 对照、边界/零空间语义和 curved manufactured/reference case，再在
custom `MLLinOp`、deferred non-orthogonal correction 或临时 PETSc/HYPRE 对照层间
作可验证选择。当前代码不创建推测性 curved scaffolding。
