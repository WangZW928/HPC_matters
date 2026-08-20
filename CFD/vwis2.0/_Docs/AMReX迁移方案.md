# VFS-Wind 到 AMReX 迁移方案

> 审阅校正（实现边界）：AMReX 提供 `Geometry/BoxArray/DistributionMapping/MultiFab`、`MFIter`、ghost 填充、MLMG、I/O 和 EB/AMR 基础设施，但不会自动保持本代码的曲线 metric、IBM 插值、压力零空间、守恒或稳定性。仓库已有未编译的 `amrex_port/` 单层数据布局骨架；它不含 RHS、投影、物理 BC、IBM、FSI 或 I/O，不能视为算法已移植。下文 API 是建议的适配目标。

本文面向 VFS-Wind 当前基于 PETSc DMDA/HYPRE 的结构化曲线网格求解器，说明如何把核心算法迁移到 AMReX 生态。目标不是逐行翻译旧代码，而是在保持 `Ucont/Ucat/P/Nvert` 等数值语义可追踪的前提下，逐步替换为 AMReX 原生的数据结构、AMR 层级、几何多重网格和 GPU-friendly kernel。

迁移原则：

- 先复现单层 Cartesian 不可压缩投影求解器，再增加曲线坐标、AMR、IBM 和多物理模块。
- 保留 VFS 的有限体积通量思想：压力投影应修正面通量 `Ucont`，而不是只修正单元中心速度。
- 优先使用 AMReX 的 `MultiFab`、`MFIter`、`MLMG`、`FluxRegister` 和 regrid 机制；只有在 AMReX 抽象不能表达 VFS sharp-interface IBM 或特殊曲线坐标算子时，才保留自定义实现。
- 每个阶段都要有独立验证算例，避免在 AMR、曲线坐标和 FSI 同时开启时才暴露基础离散错误。

## 1. 架构映射

### 1.1 主要组件对应关系

| VFS Component | AMReX Equivalent | Notes |
|---|---|---|
| PETSc DMDA structured grid | `BoxArray` + `DistributionMapping` + `Geometry` | AMReX native structured AMR |
| PETSc Vec (scalar/vector fields) | `MultiFab` | Multiple components per cell |
| Contravariant fluxes `Ucont` | `MultiFab` with face-centered data | Use `FaceLinear`/`IndexType` |
| CurvGrid metrics (`Csi`, `Eta`, `Zet`) | cell/face `MultiFab` metric fields | 位置由离散式决定；面通量所需 metric 不可仅以 cell 数据替代 |
| Poisson solver (HYPRE/GMRES+AMG) | `MLMG` (geometric multigrid) | Use EB-aware if needed |
| Ghost exchange (`DMGlobalToLocal`) | `FillBoundary` + `FillPatch` | AMReX handles automatically |
| PointProbe / PlaneExtraction | `amrex::ParticleContainer` or custom | Simple interpolation kernel |
| IB geometry (triangular mesh) | `amrex::EB2` (Embedded Boundary) | OR custom IB on `MultiFab` |

这个映射的关键变化是：VFS 把“单层结构化网格 + ghost 区 + 全局向量”封装在 DMDA/Vec 中，AMReX 则把空间域拆成多个 `Box`，用 `BoxArray` 描述网格块，用 `DistributionMapping` 描述 MPI/GPU 分配，用 `MultiFab` 存储每个 box 上的多分量数据。原来的局部三重指针访问：

```cpp
ucont[k][j][i].x
```

应迁移为 AMReX kernel 内的 `Array4` 访问：

```cpp
ucont_x(i,j,k)
```

或多分量形式：

```cpp
u(i,j,k,comp)
```

长期建议对热路径采用 SoA 风格：`u`, `v`, `w` 或 face-centered `u_mac[0..2]` 分开存储。AMReX 的 `MultiFab` 本身是组件维度连续的结构，适合 GPU 上按 component 分块计算。

### 1.2 网格、变量和度量

VFS 中 `CurvGrid` 同时承担网格读入、边界类型、DMDA 创建、metric 计算和 Jacobian 存储。AMReX 版本建议拆成两个层次：

- `MeshManager`：管理 `Geometry`、`BoxArray`、`DistributionMapping`、AMR level、regrid 触发和 coarse-fine 关系。
- `MetricData`：为每个 level 分配并填充曲线坐标 metric，例如 `csi`, `eta`, `zet`, `iaj`, `jaj`, `kaj`, `jac`。

单元中心变量：

- `P`、`Phi`、`rho`、`mu`、`levelset`、`Cs`、`nu_t`、`Nvert` 存储为 cell-centered `MultiFab`。
- `Ucat` 可存为 cell-centered 3-component `MultiFab`，用于对流插值、LES、输出和探针。

面变量：

- `Ucont` 应拆成三组 face-centered `MultiFab`，即 AMReX 常见的 `Array<MultiFab, AMREX_SPACEDIM>`。
- $U^1$ 存在 x-face index type，$U^2$ 存在 y-face index type，$U^3$ 存在 z-face index type。
- 若需要保留曲线坐标面面积向量，可为每个方向存储 face-centered metric，例如 `face_metric[dir]` 的 3 个 component。

这样可以让散度和投影写成 AMReX 原生 MAC 形式：

$$
(\nabla_\xi \cdot U)_{i,j,k}
=
\frac{U^1_{i+1/2,j,k}-U^1_{i-1/2,j,k}}{\Delta \xi^1}
+
\frac{U^2_{i,j+1/2,k}-U^2_{i,j-1/2,k}}{\Delta \xi^2}
+
\frac{U^3_{i,j,k+1/2}-U^3_{i,j,k-1/2}}{\Delta \xi^3}.
$$

### 1.3 Poisson 求解器选择

VFS 当前 Poisson 路线是手写 HYPRE IJMatrix，默认 GMRES/PCG + BoomerAMG，并对 IBM 固体压力自由度做压缩。AMReX 的优先方案是 `MLMG`：

- Cartesian 单层和 AMR 规则区域：使用 `MLABecLaplacian` 或自定义 `MLLinOp`。
- 有 EB 几何：使用 AMReX EB-aware linear operator，例如 EB Laplacian 路线。
- 曲线坐标广义 Poisson：若可写为各向异性扩散形式，优先实现为 variable coefficient operator；若包含明显交叉导数，则可能需要自定义 `MLLinOp`。

压力修正形式可以写成：

$$
\mathcal{L}\phi
=
\frac{\alpha}{\Delta t}\nabla_\xi \cdot U^*,
$$

其中 $\alpha$ 是时间离散系数。Cartesian 情况下 $\mathcal{L}=\nabla \cdot ( \beta \nabla )$，$\beta$ 可取 $1/\rho$ 或投影系数。曲线坐标下应保留 metric tensor：

$$
\mathcal{L}\phi
=
J\frac{\partial}{\partial \xi^m}
\left(
\frac{g^{mk}}{J}\frac{\partial \phi}{\partial \xi^k}
\right),
\qquad
g^{mk}=\nabla \xi^m \cdot \nabla \xi^k.
$$

若 `g^{mk}` 的非对角项很重要，AMReX 标准 cell-centered Laplacian 不能直接完整表达 19 点模板，需要自定义算子或先采用“正交/弱非正交近似 + deferred correction”。生产迁移不建议一开始就复刻 VFS 的完整曲线 Poisson 模板，应先用可验证的正交曲线网格建立基线。

### 1.4 IBM：AMReX EB 还是自定义 IB

AMReX EB2 适合以下场景：

- 固体边界静止或低频重建。
- 几何可由 implicit function、CSG 或可转为 EB level set 的闭合曲面描述。
- 需要 AMReX 原生 EB cut-cell 体积分数、面积分数、边界法向和 EB-aware `MLMG`。
- 目标是稳健地处理复杂固体边界附近的守恒通量和压力投影。

VFS 自定义 sharp-interface IB 更适合以下场景：

- 已有三角面、镜像点、image point 和壁函数逻辑需要最大限度继承。
- 几何随 FSI 高频运动，EB 重建和 AMR regrid 成本过高。
- 希望保留 `Nvert`、`IBMInfo`、边界插值和体表速度设置的原有数值行为。
- 固体边界不希望被 cut-cell 小体积分数稳定性约束限制。

推荐路线是双轨接口：

- `IBMHandler` 定义统一接口：`buildGeometry(levels)`、`classifyCells()`、`applyVelocityBC()`、`computeForces()`。
- `EBIBMHandler` 使用 AMReX EB2 和 EB `MultiFab`。
- `SharpIBMHandler` 使用 VFS 风格三角网格搜索，在 AMR hierarchy 上维护 `Nvert` 和插值 stencil。

第一版迁移不应把 EB2 和自定义 IB 混在同一个算例中。先选一个路径完成验证，再考虑互操作。

### 1.5 PETSc 与 AMReX 线性求解器取舍

优先切换到 AMReX `MLMG` 的情况：

- 标准 Poisson、Helmholtz、diffusion 或可表达为 AMReX linear operator 的投影问题。
- 需要 AMR coarse-fine 一致性、reflux、mask 和 EB 支持。
- 目标平台包含 GPU，且希望避免 PETSc Vec 和 AMReX `MultiFab` 间反复拷贝。

保留 PETSc 的情况：

- FSI 结构方程、刚体/弹性体自由度、全局小规模非线性系统。
- 曲线坐标 Poisson 暂时必须使用 VFS 原 19 点模板，且 AMReX 自定义 `MLLinOp` 尚未完成。
- 需要 PETSc SNES/TS 提供稳健非线性求解或隐式多物理耦合。

不要在热路径长期采用 `MultiFab -> PETSc Vec -> HYPRE -> MultiFab` 的数据转换。这个方案可用于阶段性对照验证，但会削弱 AMReX 的 AMR/GPU 优势。

## 2. 算法适配

### 2.1 曲线坐标变换

#### metric 数组存储

VFS 的 `Csi/Eta/Zet` 可在 AMReX 中拆为以下 `MultiFab`：

- `metric_cc[level]`：cell-centered，9 个 component，存储 $\xi^i_l=\partial \xi^i/\partial x_l$。
- `jac_cc[level]`：cell-centered，1 个 component，存储 $J$ 或 $1/J$，命名必须明确。
- `metric_fc[dir][level]`：face-centered，建议存储该 face 所需的面积向量或 contravariant metric。

若只把 metric 存在 cell center，面通量计算时要插值到 face。为了避免每个 kernel 重复插值，建议在 `MetricData::BuildFaceMetrics()` 中预计算 face metric。曲线网格静止时只需初始化一次；动网格或 FSI deforming mesh 才需要更新。

#### IndexType 选择

AMReX 的 `IndexType` 应和变量物理位置一致：

- `Ucat`：cell-centered，3 component。
- `Ucont[0]`：x-face centered。
- `Ucont[1]`：y-face centered。
- `Ucont[2]`：z-face centered。
- 压力 `P/Phi`：cell-centered；若未来使用 nodal projection，再单独引入 nodal pressure correction。
- viscous/convection face flux：face-centered 临时 `MultiFab`，可以按方向分配。

不要把 `Ucont` 简化成 cell-centered 三分量 `MultiFab` 后再手工偏移索引。这样会破坏 AMReX 的 `MacProjector`、`FluxRegister` 和 coarse-fine 操作语义。

#### 逆变速度变换

从笛卡尔速度到逆变通量：

$$
U^i=\mathbf{u}\cdot \nabla \xi^i.
$$

有限体积实现中建议存储穿过 face 的体积通量：

$$
\hat{U}^i_f
=
\left(\mathbf{u}_f \cdot \mathbf{A}^i_f\right),
\qquad
\mathbf{A}^i_f=\frac{1}{J_f}\nabla \xi^i_f.
$$

其中 $\mathbf{u}_f$ 由 `Ucat` 插值得到，$\mathbf{A}^i_f$ 来自 face metric。压力投影也应直接修正 face 通量：

$$
\hat{U}^{i,n+1}_f
=
\hat{U}^{i,*}_f
-
\Delta t\,\beta_f\,\mathbf{A}^i_f\cdot \nabla \phi_f.
$$

这样散度诊断、Poisson RHS 和投影修正都使用同一套面通量，不会出现 `Ucat` 近似无散但 `Ucont` 不守恒的问题。

### 2.2 投影法/分步求解器

#### fractional step 到 AMReX loops

AMReX-native 时间步可以组织为：

1. `FillPatch` 填充 `Ucat/Ucont/P/rho/mu/levelset` 的 ghost。
2. `ComputeAdvectionFlux` 在 face 上计算对流通量。
3. `ComputeDiffusionFlux` 或隐式 diffusion operator 计算粘性项。
4. `AdvanceMomentum` 得到预测速度 $U^*$ 或 $\mathbf{u}^*$。
5. `ComputeDivergence` 形成 Poisson RHS。
6. `MLMG::solve` 求压力修正 $\phi$。
7. `ProjectFaceVelocity` 修正 `Ucont`。
8. `UpdateCellVelocity` 从 `Ucont` 和 metric 重构或同步 `Ucat`。
9. `AverageDown`、`Reflux`、`FillBoundary`，保证 AMR level 间一致。

对 GPU，步骤 2、3、4、5、7、8 都应写成 `MFIter` + `ParallelFor` kernel。每个 kernel 只处理当前 tile，ghost 有效性由 kernel 前的 `FillBoundary/FillPatch` 保证。

#### NodalProjector、MacProjector 或自定义 projector

AMReX 提供的投影器主要面向 Cartesian MAC 速度：

- `MacProjector`：适合修正 face-centered velocity，使 face divergence 为零。迁移 `Ucont` 的第一选择。
- `NodalProjector`：适合 cell-centered velocity projection，压力 correction 在 nodal 上。若后续采用 collocated velocity，并通过 nodal gradient 修正，可以评估。
- 自定义 projector：曲线坐标、非正交 metric、VFS 19 点 Poisson 或 IBM mask 压缩自由度需要自定义。

建议 Phase 1 使用 `MacProjector` 或直接用 `MLABecLaplacian + 自写 projection kernel`。Phase 2 曲线坐标开始后，将 projector 抽象成：

```cpp
class ProjectionOperator {
public:
  virtual void solve(Vector<MultiFab*>& phi,
                     Vector<Array<MultiFab*, AMREX_SPACEDIM>>& ucont,
                     Real dt) = 0;
};
```

这样 Cartesian projector 和 Curvilinear projector 可以并存，便于回归测试。

#### 非 Cartesian metric 下的 MLMG

曲线坐标 Poisson 最大风险在 operator 表达。若写成：

$$
\nabla \cdot \left(\beta \nabla \phi\right)
$$

只需要 face coefficient $\beta_f$，AMReX 标准 operator 支持较好。但 VFS 的非正交曲线坐标一般包含：

$$
\frac{\partial}{\partial \xi^m}
\left(a^{mk}\frac{\partial \phi}{\partial \xi^k}\right),
\quad m,k=1,2,3,
$$

其中 $m\ne k$ 是交叉导数项。处理策略：

1. 正交或近正交网格：先忽略非对角项，验证主体流程。
2. 非对角项 deferred correction：主 solve 使用对角 SPD operator，交叉项显式放入 RHS，外迭代更新。
3. 原型评估自定义离散与求解路径：先确认 AMReX 版本中的可扩展线性算子接口、边界/粗细层语义和 GPU 可行性；必要时才实现完整 19 点算子及其多层 transfer/边界处理，或在过渡期保留 PETSc/HYPRE 对照路径。

推荐按 1 -> 2 -> 3 推进。这里的第 1 步仅为误差归因实验，不能作为非正交生产算例的数值等价实现；直接上完整自定义算子会把迁移风险集中到最难调试的模块。

### 2.3 Level Set 对流与重初始化

#### AMR hierarchy 上的 level set

Level set $\phi$ 应作为 cell-centered `MultiFab` 存在每个 AMR level。密度和粘度由平滑 Heaviside 函数生成：

$$
\rho(\phi)=\rho_a + (\rho_w-\rho_a)H_\epsilon(\phi),
\qquad
\mu(\phi)=\mu_a + (\mu_w-\mu_a)H_\epsilon(\phi).
$$

`FillPatch` 负责 coarse-fine ghost，物理边界由 `BCRec` 和自定义 boundary functor 处理。界面附近 refinement tagging 可用：

- $|\phi| < c\Delta x$。
- $|\nabla \phi|$ 大。
- 密度/粘度跳变区域。

#### AMR coarse-fine 守恒和 reflux

普通 level set 对流不是严格守恒变量，但两相体积分数或质量相关量需要 coarse-fine 一致。建议：

- 若继续使用 signed distance level set，对流后做 `AverageDown`，重初始化后再同步 coarse-fine ghost。
- 若引入 conservative level set 或 VOF-like 变量，对该守恒变量使用 `FluxRegister` 做 reflux。
- 表面张力、密度和粘度应在 level set 更新并 reinit 后统一重算，避免跨 level 物性不一致。

对一个守恒标量 $q$，fine/coarse flux mismatch 应通过 reflux 修正：

$$
q_c^{n+1}
\leftarrow
q_c^{n+1}
-
\frac{\Delta t}{V_c}
\sum_{f\in \partial c}
\left(F^{fine}_f-F^{coarse}_f\right)A_f.
$$

#### subcycling

若 AMR 采用时间子循环，level $\ell+1$ 通常走 $r$ 个小步匹配 level $\ell$ 一个大步。需要明确：

- `Ucont`、level set 和 body force 在 coarse-fine time interpolation 中的时间位置。
- 压力投影是每个 level 子步单独投影，还是 multilevel composite projection。
- level set reinit 不能在 fine level 独立执行后破坏 coarse-fine 连续性，应在同步点做一次 hierarchy-aware 修正。

Phase 3 建议先使用相同时间步的多 level integration，验证 reflux 和 projection 后，再开启 subcycling。

### 2.4 LES 动态模型

#### 应变率计算

`Ucat` 是 LES 的自然输入。AMReX 实现应分三步：

1. `FillBoundary(Ucat)`，保证速度 ghost 有效。
2. `ComputeGradU` kernel 计算 $\partial u_i/\partial x_j$ 或曲线坐标下的物理梯度。
3. `ComputeStrainAndNuT` kernel 计算：

$$
S_{ij}=\frac{1}{2}
\left(
\frac{\partial u_i}{\partial x_j}
+
\frac{\partial u_j}{\partial x_i}
\right),
\qquad
|S|=\sqrt{2S_{ij}S_{ij}}.
$$

曲线坐标下梯度先在计算空间差分，再用 metric 变换：

$$
\frac{\partial u_i}{\partial x_j}
=
\frac{\partial u_i}{\partial \xi^m}
\frac{\partial \xi^m}{\partial x_j}.
$$

#### AMR level 间测试滤波

动态 Smagorinsky 的测试滤波在单层均匀网格上容易实现，在 AMR 上要额外定义滤波尺度：

- 同一 level 内使用 3x3x3 Simpson 或 box filter。
- coarse-fine 边界附近优先使用 `FillPatch` 后的 ghost 值，避免缺 stencil。
- 若测试滤波尺度跨越 level，应在 composite hierarchy 上定义 filter，这比单 level filter 难得多。

实用建议：第一版 AMR LES 只在每个 level 内做 test filter，并在 coarse-fine buffer 区使用限制策略，例如把动态 `Cs` 限制在邻域平均范围。待 AMR 基础验证通过后，再实现真正的 multilevel filter。

#### `LM/MM` 的 MPI reduction

VFS 动态模型会累积 Germano 关系中的 `LM/MM`，并可在均匀方向做平均。AMReX 中：

- 局部 tile 内用 `ReduceOps` 或 `ParReduce`。
- MPI 层使用 `ParallelDescriptor::ReduceRealSum`。
- 若只在某个方向平均，应先按 plane/bin 累积到 1D host/device buffer，再做 MPI reduction。

公式保持：

$$
C_s =
\frac{1}{2}\frac{LM}{MM+\epsilon},
\qquad
0\le C_s \le C_{s,\max}.
$$

需要沿用旧文档中的注意点：代码里的 `Cs` 更像直接进入 $\nu_t=C_s\Delta^2|S|$ 的组合系数，不一定等同文献中的 Smagorinsky 常数。

### 2.5 FSI 耦合 + AMR

#### regrid 对 FSI 状态的影响

FSI 状态本身通常不是 `MultiFab`，而是结构自由度、顶点坐标、三角面法向和体表速度。AMR regrid 后必须重建所有“网格相关”的派生数据：

- `Nvert` 或 EB cut-cell mask。
- IBM interpolation stencil。
- 体表附近流体采样点。
- force integration 所需的压力/速度梯度采样关系。

但结构状态不能被 regrid 重置。`FSIState` 应只保存物理状态：

$$
\mathbf{q}_s=(\mathbf{x}_s,\mathbf{v}_s,\boldsymbol{\theta},\boldsymbol{\omega})
$$

regrid 后用当前 $\mathbf{q}_s$ 重新生成几何派生数据。

#### IBM 几何变化后的处理

若使用 EB2：

- 几何运动后需要重建 EB index space，成本较高。
- 小幅高频运动会导致频繁 regrid/EB rebuild，不适合直接逐步重建。
- 可考虑 overset-like 局部 refinement 或只在同步点重建 EB。

若使用自定义 sharp IB：

- 每次结构移动后更新三角面位置和速度。
- 在受影响 AMR levels 上重做 cell classification。
- 对 `Nvert` 和 interpolation stencil 建议按 level 存储，并带版本号：`geometry_epoch`。流场 kernel 只使用当前 epoch 的数据。

#### 力的守恒插值

流体力从 Eulerian 网格传到 Lagrangian 结构，结构反力再回到网格时，AMR 上必须避免 coarse/fine 双计数。建议：

- force integration 只在 finest available level 的有效区域执行，使用 AMReX mask 排除被 fine 覆盖的 coarse cells。
- 体力扩散回网格时，先写入各 level 的 `body_force[level]`，再 `AverageDown` 或 reflux 保证动量一致。
- 若 IB 采用插值边界而不是 delta force，仍需确保压力/粘性力采样不重复。

一个简单规则是：任一点 $\mathbf{x}$ 的流固交互只由包含它的最细有效 cell 负责。

### 2.6 波浪生成 + 风机模型

#### 波浪源项

波浪生成、阻尼层和入口回收可统一为 body force 或边界 forcing：

- `wave_force[level]`：cell-centered 3-component `MultiFab`。
- `sponge_coeff[level]`：cell-centered scalar `MultiFab`。
- `target_velocity/target_levelset`：用于松弛区 forcing。

动量方程中加入：

$$
\mathbf{f}_{wave}
=
\sigma(\mathbf{x})
\left(\mathbf{u}_{target}-\mathbf{u}\right),
$$

level set 或自由面也可使用类似 relaxation：

$$
S_\phi
=
\sigma(\mathbf{x})
\left(\phi_{target}-\phi\right).
$$

AMR 下源项只在有效 cell 上计算，coarse 被 fine 覆盖区域不重复施加。

#### 风机 actuator disk/line

风机模型适合用 Lagrangian marker + Eulerian spreading：

- actuator disk：盘面采样点或 annulus bins。
- actuator line：叶片 marker 随时间旋转。
- 对每个 marker，根据所在 AMR level 找 finest valid cell。
- 从 `Ucat` 插值得到局部来流，计算升阻力或推力。
- 将力扩散到 `body_force`，kernel 可用 atomic add 或先按 tile 分桶减少冲突。

插值需要 AMR-aware search：

1. 从 finest level 向 coarse level 查找 marker 所在 box。
2. 若 marker 在 fine valid region，使用 fine level。
3. 若不在，退到 coarse level。

风机力源的总功率和推力应做 MPI reduction，并输出与 VFS 原 `CalculateForces` 类似的诊断量。

## 3. 实施路线图

### Phase 1: 单层 Cartesian 流场求解器 (2-3 months)

实现内容：

- 建立 AMReX 工程骨架：`AmrCore` 或单层 `Geometry/BoxArray/DistributionMapping`。
- 分配 cell-centered `Ucat/P/Phi` 和 face-centered `Ucont`。
- 实现 Cartesian 边界条件、ghost fill、速度初始化和基础输出 plotfile/checkpoint。
- 实现 AMReX-native fractional step：
  - 对流项先用二阶中心/迎风格式。
  - 粘性项先显式或 Crank-Nicolson 简化版本。
  - pressure RHS 使用 face divergence。
  - `MLMG` 求解 Poisson。
  - 投影 kernel 修正 face velocity。
- 暂不引入 IB、LES、FSI、level set、曲线坐标。

关键交付物：lid-driven cavity validation。至少比较中心线速度剖面、最大散度、动能收敛和 Poisson 迭代历史。

工作量估计：2-3 个月。

最大风险：投影变量位置选择错误。若一开始把速度全部做成 cell-centered，后续迁移 `Ucont`、AMR reflux 和 MAC projection 会返工。

### Phase 2: 曲线坐标 + 结构化网格 (2 months)

实现内容：

- 引入 `MetricData`，读入或生成曲线网格坐标。
- 计算 cell-centered 和 face-centered metric、Jacobian、cell volume。
- 实现 `Ucat -> Ucont` 和 `Ucont -> Ucat` 转换。
- 对流项使用 contravariant face flux。
- 粘性项加入物理梯度变换。
- Poisson 先实现正交/弱非正交版本：
  - 对角 metric 进入 face coefficient。
  - 非对角项暂时关闭或 deferred correction。
- 建立曲线网格边界条件和 metric 一致性检查。

关键交付物：curved channel flow validation。比较压力梯度驱动流、壁面剪切、质量守恒和网格加密收敛。

工作量估计：2 个月。

最大风险：Jacobian 定义混乱。VFS 手册中的 $J$ 是逆变换 Jacobian，AMReX 实现必须明确 `jac` 存的是 $J$ 还是 cell volume $1/J$，否则 Poisson RHS 和通量散度会符号/尺度错误。

### Phase 3: AMR + 规则区域 (2-3 months)

实现内容：

- 将 solver 扩展到多 level hierarchy。
- 实现 refinement tagging：涡量、速度梯度、level set 预留接口、用户指定区域。
- `FillPatch`、`AverageDown`、coarse-fine boundary fill。
- 实现 subcycling 时间推进，或先实现 no-subcycling multilevel step。
- 对守恒通量引入 `FluxRegister` 和 reflux。
- 实现 AMR-aware halo exchange 和 multilevel projection。
- 输出每个 level 的散度、CFL、Poisson residual、reflux correction 诊断。

关键交付物：AMR refinement convergence study。用规则区域算例比较单层细网格与 AMR 结果，并检查 coarse-fine 边界无明显压力/速度伪影。

工作量估计：2-3 个月。

最大风险：projection 与 reflux 的顺序。若动量 reflux 后没有做一致的 composite projection，coarse-fine 边界可能重新产生散度误差。

### Phase 4: IB 方法 + AMR (3-4 months)

实现内容：

- 选择 EB2 或自定义 sharp IB 的首个生产路径。
- 若用 EB2：
  - 建立三角面到 implicit/EB 表达的转换或替代几何输入。
  - 使用 EB-aware volume/area fraction 和 EB boundary condition。
  - 接入 EB-aware `MLMG`。
- 若用自定义 IB：
  - 将 `Nvert` 改为 per-level cell-centered `MultiFab`。
  - 将 `IBMInfo` 链表压平成 SoA 数组。
  - 实现 AMR-aware cell classification 和 image point 插值。
  - 在 moving boundary 后按受影响区域重建 stencil。
- force integration 只在 finest valid region 进行。
- IBM 边界条件参与预测步和投影后的速度修正。

关键交付物：cylinder flow / VIV validation。静止圆柱验证阻力、升力、St 数；VIV 验证结构位移、频率和流体力相位。

工作量估计：3-4 个月。

最大风险：移动边界与 AMR regrid 同时发生时状态不一致。必须建立清晰的顺序：结构更新 -> 几何更新 -> regrid -> mask/stencil 重建 -> flow advance。

### Phase 5: 多物理模块 (3-4 months)

实现内容：

- Level set 两相流：
  - 对流、重初始化、物性更新、表面张力。
  - AMR tagging 和 coarse-fine 同步。
- FSI coupling：
  - 结构状态、强/弱耦合循环、力矩积分、重启文件。
  - moving geometry 与 IB/EB 更新接口。
- 波浪生成：
  - 入口波、松弛区、阻尼层、自由面目标场。
- 风机模型：
  - actuator disk/line marker。
  - AMR-aware interpolation/spreading。
  - 推力、功率、扭矩诊断。
- LES 与壁面模型：
  - 动态模型 AMR 版本。
  - 壁面剪切应力或 EB/IB 边界通量。

关键交付物：sloshing tank + wave-structure interaction。至少包含自由面高度、结构力、能量耗散和网格敏感性验证。

工作量估计：3-4 个月。

最大风险：模块耦合顺序失控。level set、FSI、IB、LES 和 wave forcing 都会改动 RHS 或边界状态，必须用统一的 time integrator 管理调用顺序和数据时间层。

### 3.1 阶段汇总

| Phase | Effort | Deliverable | Biggest Risk |
|---|---:|---|---|
| 1. 单层 Cartesian | 2-3 months | Lid-driven cavity | 变量位置和投影语义选错 |
| 2. 曲线坐标 | 2 months | Curved channel flow | Jacobian/metric 定义不一致 |
| 3. AMR 规则区域 | 2-3 months | AMR convergence study | Reflux 与 projection 顺序错误 |
| 4. IB + AMR | 3-4 months | Cylinder/VIV | Moving geometry + regrid 状态不一致 |
| 5. 多物理模块 | 3-4 months | Sloshing + WSI | 多模块时间层和源项耦合混乱 |

总工作量约 12-16 个月，取决于是否完整复刻 VFS 的非正交 Poisson、动态 LES 和 moving sharp-interface IBM。若只迁移 Cartesian/AMR/EB 基础流场，周期可明显缩短。

## 4. 代码设计建议

### 4.1 类层次

建议的核心类结构：

```text
VFSProblem
  ├── MeshManager
  ├── FieldRepository
  ├── MetricData
  ├── FlowSolver
  │     ├── AdvectionOperator
  │     ├── DiffusionOperator
  │     ├── ProjectionOperator
  │     └── TimeIntegrator
  ├── PoissonSolver
  │     ├── MLMGPoissonSolver
  │     └── PetscPoissonSolver
  ├── IBMHandler
  │     ├── EBIBMHandler
  │     └── SharpIBMHandler
  ├── LESModel
  ├── LevelSetSolver
  ├── FSISolver
  ├── WaveForcing
  ├── TurbineModel
  └── Diagnostics
```

`VFSProblem` 只负责生命周期和高层时间循环，不直接写 stencil。所有计算模块通过 `FieldRepository` 获取 `MultiFab`，避免全局变量式耦合。建议保留 VFS 术语作为字段名，例如 `ucont`, `ucat`, `nvert`, `phi`, `pressure`，这样便于和旧文档、旧算例对照。

### 4.2 数据布局：GPU 上优先 SoA

AMReX `MultiFab` 是 GPU-friendly 的核心数据容器。建议：

- face velocity 使用 `Array<MultiFab, AMREX_SPACEDIM>`，天然 SoA。
- cell velocity 可用 3-component `MultiFab` 起步；若热点 kernel 多数只访问单分量，再拆成三个 `MultiFab`。
- metric 建议按用途拆分：
  - `metric_cc` 9 components 用于梯度变换。
  - `area_fc[dir]` 3 components 用于面通量和投影。
  - `volume` 1 component 用于积分和守恒修正。
- IBM 三角面和插值 stencil 必须压平成 SoA，例如 `tri_x0[]`, `tri_y0[]`, `tri_z0[]`, `interp_i[]`, `interp_w[]`，不要保留链表。

AoS 的 `Cmpnts` 风格在旧 CPU 代码中可读，但 GPU 上容易造成不必要加载。迁移时可以先用多 component `MultiFab` 保持语义，优化阶段再拆分热变量。

### 4.3 Kernel fusion

应融合的操作：

- `Ucat -> face interpolation -> advection flux`：若没有其他模块复用 face velocity，可在一个 kernel 中完成。
- `grad U -> strain -> |S| -> nu_t`：LES 中间量不需要输出时可融合。
- `Poisson RHS divergence`：直接从 face `Ucont` 计算 RHS，避免先写临时 divergence。
- `projection -> face velocity update`：每个方向一个 kernel，读 `phi` 和 coefficient，写 `Ucont[dir]`。
- `body force accumulation -> momentum RHS`：规则源项如波浪阻尼可直接进入 RHS。

不建议过早融合的操作：

- 对流通量和散度更新。AMR reflux 需要 face flux，保留显式 flux `MultiFab` 更利于守恒校正。
- LES test filter 和 `LM/MM` reduction。滤波 stencil、边界处理和归约调试复杂，先拆开更安全。
- IBM interpolation 和 force integration。不规则访问多，先保证可验证，再优化。

### 4.4 AMReX 与 PETSc 的边界

建议默认边界：

- 流场大规模网格变量：AMReX `MultiFab`。
- Poisson/Helmholtz：优先 AMReX `MLMG`。
- 小规模结构动力学、全局约束、非线性 FSI correction：可保留 PETSc。
- 旧 VFS Poisson：仅作为过渡验证路径。

若保留 PETSc solver，应通过明确适配层隔离：

```cpp
class PetscAdapter {
public:
  void copyFromMultiFab(const Vector<MultiFab*>& src);
  void solve();
  void copyToMultiFab(Vector<MultiFab*>& dst);
};
```

适配层只能出现在非热点或临时验证路径中。生产主路径不应每个时间步多次做全场格式转换。

### 4.5 构建系统

建议使用 CMake 管理新代码：

- `find_package(AMReX REQUIRED)` 或通过 AMReX submodule/superbuild。
- 编译选项显式区分 `AMReX_SPACEDIM=3`、MPI、OpenMP、CUDA/HIP。
- 可选 PETSc 用 `find_package(PETSc)` 或 `pkg-config`，并通过 CMake option 控制：

```cmake
option(VFS_ENABLE_PETSC "Enable PETSc coupling solvers" OFF)
option(VFS_ENABLE_EB "Enable AMReX EB support" ON)
option(VFS_ENABLE_CUDA "Build with CUDA backend" OFF)
```

目录建议：

```text
Source/
  Core/
  Mesh/
  Flow/
  Projection/
  IBM/
  LES/
  LevelSet/
  FSI/
  Physics/
  Diagnostics/
Exec/
  cavity/
  curved_channel/
  cylinder/
  sloshing/
Tests/
```

每个 `Exec` 应有独立 input file、baseline 输出和 postprocess 脚本。

### 4.6 测试策略

每个阶段都应包含 MMS 和物理基准。

Phase 1：

- MMS：Cartesian incompressible projection，给定解析 $\mathbf{u},p$ 和 forcing。
- 物理：lid-driven cavity、Taylor-Green vortex。
- 指标：$L_2/L_\infty$ 误差、最大散度、Poisson residual。

Phase 2：

- MMS：曲线坐标下制造解，验证 metric、梯度、散度和 Poisson。
- 物理：curved channel、网格非正交性扫描。
- 指标：metric identity、质量守恒、压力梯度误差。

Phase 3：

- MMS：AMR coarse-fine 边界穿越的平滑解析解。
- 物理：局部加密 Taylor-Green 或 shear layer。
- 指标：AMR 与 uniform fine 对比、reflux correction 总量、composite divergence。

Phase 4：

- 几何测试：三角面 classification、EB/IB 法向、体积分数或 `Nvert` mask。
- 物理：圆柱绕流、振荡圆柱、VIV。
- 指标：阻力/升力、St 数、IBM 附近散度、力积分随网格收敛。

Phase 5：

- Level set MMS：纯平移、旋转 Zalesak disk、reinit 后 signed distance 误差。
- 两相物理：sloshing tank、静水压力、毛细波。
- FSI/波浪：wave-structure interaction、actuator disk 推力曲线。
- 指标：质量损失、自由面高度、结构能量、推力/功率守恒。

所有测试都应至少跑 CPU MPI 和一个 GPU 后端配置。回归阈值应区分 bitwise 和 norm-based：AMReX GPU reduction 不应要求逐位一致，但应要求物理误差和守恒量在容差内。

### 4.7 实施注意事项

- 迁移初期不要追求和 VFS 每个时间步逐位一致，应追求离散语义一致和验证算例收敛一致。
- `Ucont` 是守恒核心，任何边界条件、IBM、FSI、波浪源项最终都要检查对 face flux divergence 的影响。
- `Nvert` 如果继续使用浮点标记，建议改为整数 mask 或枚举，避免 GPU kernel 中出现模糊比较。
- 压力零空间要显式处理。全 Neumann 或封闭域下，`MLMG` 需要设置 solvability 或移除 RHS 均值。
- regrid 后必须有统一回调：重新填 metric、物性、IB mask、actuator marker level、diagnostics mask。
- 所有跨 level 的 force、flux、source 都要定义“哪个 level 拥有该物理量”，避免 coarse/fine 双计数。

最实际的迁移切入点是 Phase 1 的 AMReX MAC projection。只要单层 Cartesian 的 `Ucont`、Poisson 和投影语义站稳，后续曲线坐标和 AMR 才有可靠的基座。

## 审阅补充：迁移验收矩阵

| 主题 | 可复用 | 必须重写/适配 | 科学或数值风险 |
|---|---|---|---|
| 网格与字段 | 变量语义、曲线 metric 输入 | `Geometry`、`BoxArray`、`DistributionMapping`、cell/face `MultiFab`、`MFIter` | DMDA 单块与多 Box/AMR 的索引、所有权和 metric 保守恒不等价 |
| ghost/BC/MPI | 边界类型意图、通量诊断 | `FillBoundary`/`FillPatch`、BC functor、coarse-fine ghost | AMReX 不会推导 VFS ghost 或 IBM 例外；逐面通量须验证 |
| 投影/线性系统 | `Ucont` 散度、压力修正语义 | Cartesian `MLMG`/`MacProjector`；曲线 19 点项自定义 `MLLinOp` 或 PETSc/Hypre 适配层 | 非正交交叉 metric、零空间、可解性和 BC 不会自动正确 |
| IBM/FSI | 三角面、`Nvert`、插值/力逻辑 | EB2 或自定义 sharp IBM 二选一；移动重建接口 | cut-cell 稳定性、移动重建、跨层插值和力守恒 |
| GPU/I/O/测试 | 输出变量名、现有基准意图 | `ParallelFor`、设备数据、plotfile/checkpoint、CMake 与回归测试 | 驻留/通信/重启/regrid 后 metric 与 IB mask 一致性 |

验证顺序：均匀 Cartesian 制造解/Taylor–Green（空间时间阶）→ 通道/腔流（BC/投影）→ 静态 IBM（无穿透、力、通量）→ 曲线单层 → 移动 IBM/FSI → AMR/EB。每阶段比较 L2/L∞、质量误差、压力均值、力矩和 MPI 分块不变性；没有这些基准不得声称迁移保持科学结果。
