# vwis2.0 可移植性与 GPU 优化分析

> 证据级别：当前源码没有 CUDA/HIP/Kokkos/OpenMP target 调用，makefile 也是 host PETSc/HYPRE 路径。本页的 GPU 代码和性能判断都是迁移设计，不是现有实现或实测结果；须以目标机基准替代。

本文基于 `_Docs/00_整体架构.md` 到 `_Docs/04_编译与运行.md` 以及 `vwis2.0/` 源码评估 GPU/Kokkos 迁移路径。结论先行：vwis2.0 最合理的路线不是直接重写为 CUDA/HIP，而是先升级 PETSc/构建系统并启用 PETSc GPU Vec/KSP 后端，再把 `RHSSolver`、`LESModel`、`UData::Contra2Cart`、`PoissonSolver::PoissonRHS2_hypre/Projection` 等规则网格热点逐步抽成 Kokkos kernel。IBM 搜索和 FSI 力积分应后置，先保持 CPU 或做混合执行。

## 1. 代码架构评估

### 1.1 当前并行模型

当前代码的并行基础是 PETSc DMDA：

- `CurvGrid` 创建标量 DMDA `da` 和三分量向量 DMDA `fda`。
- 标量场如 `P/Nvert/Aj/Phi/Cs/Nu_t` 存在 `da` 上。
- 三分量场如 `Ucont/Ucat/Csi/Eta/Zet/ICsi/...` 存在 `fda` 上，元素类型是 `Cmpnts {x,y,z}`。
- 所有核心模块通过 `DMDAVecGetArray` 得到 `array[k][j][i]` 或 `array[k][j][i].x/y/z` 形式的本地指针。
- ghost/halo 同步依赖 `DMGlobalToLocalBegin/End`、`DMDALocalToLocalBegin/End` 和 `DMLocalToGlobalBegin/End`。
- Poisson 方程没有用 PETSc Mat/KSP 直接组装，而是手写 HYPRE IJMatrix/IJVector，矩阵底层为 ParCSR，求解器为 GMRES/PCG + BoomerAMG。

这种模型对 MPI 结构化网格很合适，也天然提供局部子域和 ghost 区。但当前实现假定 `DMDAVecGetArray` 返回 CPU 可直接解引用的三重指针。若 Vec 迁移到 GPU，直接在 host 上频繁 `DMDAVecGetArray` 会触发隐式拷贝或根本不适用于 device kernel，因此必须重构数据访问层。

### 1.2 数据布局：AoS 与访问模式

`functions.h` 定义：

```cpp
typedef struct {
    PetscScalar x, y, z;
} Cmpnts;
```

向量场访问形态是：

```cpp
ucat[k][j][i].x
ucont[k][j][i].y
csi[k][j][i].z
```

这属于 AoS（Array of Structs）。在 CPU 上可读性好，三分量同点访问局部性也不错；在 GPU 上则有两个问题：

- 若一个 kernel 同时处理 `x/y/z` 三分量，AoS 尚可接受。
- 若很多 kernel 只读写单分量，例如 `Ucont.x` 面通量、`rhs.y` 或 `Phi` 梯度，AoS 会降低内存合并效率，等价于加载不需要的分量。

当前代码的热点既有三分量同算的梯度/粘性项，也有大量单分量面通量和投影修正。因此迁移时建议采用“兼容层优先，SoA 长期优化”的策略：

- 第一阶段：用 `Kokkos::View<PetscScalar****>` 或轻量包装保持最后一维为 3，语义接近 AoS，降低改动量。
- 第二阶段：对最热路径改成 SoA，例如 `View<double***> u, v, w` 或 `View<double****, LayoutRight>` 并确保 component 维度访问合并。
- IBM/FSI 的 `IBMNodes` 当前是大量裸指针数组，已经接近 SoA，反而更适合 device 化，但链表 `IBMListNode` 不适合 GPU。

### 1.3 主要性能热点

按调用频率和计算量，热点大致如下：

1. `RHSSolver::Solve`
   - 三个方向分别计算对流通量 `d_Div1/2/3` 和粘性通量 `d_Visc1/2/3`。
   - 反复调用 `Compute_du_i/j/k`、读取曲线网格 metric、检查 `Nvert` 与边界条件。
   - 后续将面通量散度投影回 `Rhs`。
   - 这是每次 SNES 残差回调都会执行的核心热点。

2. `LESModel::ComputeSmagorinksyConstant` 与 `ComputeEddyViscosity`
   - 动态 Smagorinsky 包含速度梯度、测试滤波、`LM/MM` 计算、可选均匀方向平均。
   - `ComputeEddyViscosity` 是规则点循环，计算 `Sabs` 和 `nu_t`，GPU 适配性较好。

3. `PoissonSolver::PoissonRHS2_hypre` 与 `Projection`
   - RHS 是标准散度计算。
   - `Projection` 用 `Phi` 梯度修正 `Ucont` 三个面分量，结构类似压力梯度计算，分支多但规则。
   - `PoissonLHS` 只在 setup 或网格/IBM 改变时重建，虽然复杂但不是每步主热点；若 FSI 导致 IBM 频繁重搜，则矩阵重建成本会上升。

4. `UData::Contra2Cart`
   - 将逆变速度 `Ucont` 转换为笛卡尔速度 `Ucat`。
   - 每步多次调用，规则 3D 点循环，算术密度适中，是低风险首批 kernel。

5. `FlowSolver::CalculateDivergence`、`CalculateKE`、`UData::Average`
   - 规则扫描加归约/统计。单次成本低于 RHS，但很适合用 Kokkos reduction 与 PETSc Vec 操作替代。

6. `ImmersedBoundary::IBMSearchAdvanced1`、`IBMInterpolationAdvanced`
   - 搜索包含空间桶、三角面相交、随机射线、链表遍历。
   - 插值遍历 `IBMListNode` 链表，读写非连续格点与 IBM 面数组，是 GPU 上最困难的部分。

7. `FSI::CalculateForces1`
   - 遍历 IBM 插值信息和三角面，计算压力/粘性力、力矩、功率并做全局归约。
   - 比纯 stencil 更不规则，但比几何搜索更容易迁移，因为数据可先压平成数组。

### 1.4 循环结构与依赖分析

代码主体是 `k-j-i` 三重循环，典型范围为：

```cpp
for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
    for (i=lxs; i<lxe; i++)
      ...
```

依赖形态分四类：

- 点式变换：`Contra2Cart`、`ComputeEddyViscosity` 的输出单元只写本点，读邻居。适合 `Kokkos::MDRangePolicy<Rank<3>>`。
- 面通量 stencil：`RHSSolver::Solve` 先生成面通量 `div/visc`，halo 同步后再散度。阶段间有全局/局部同步，适合拆成多个 kernel。
- Poisson stencil：RHS 和投影读 `Phi/Ucont/Nvert/metric` 邻居，写本地面或本地标量；LHS 组装最多 19 点模板，不适合每步在 GPU 手写组装，优先交给 PETSc/HYPRE GPU 后端。
- 不规则 IBM/FSI：链表遍历、三角面搜索、按插值权重访问多个离散点。需要先将链表和结构体数组压平成 device-friendly 数组。

内部循环通常没有写后读依赖，但存在三类同步边界：

- PETSc ghost 更新。
- 周期边界手工拷贝。
- `RHSSolver` 中面通量生成后再做 `DMDALocalToLocal`，随后计算散度。

GPU 迁移必须把这些同步边界显式化：每个 kernel 前保证 ghost 有效，每个 kernel 后只在需要跨子域读写时同步。

## 2. GPU 移植路径分析

### 2.1 Kokkos

#### 适配方式

对 DMDA 本地数组的 Kokkos 化有两种可行做法：

1. PETSc Vec 保持主存储，kernel 前后用 host/device mirror 同步。
   - `VecGetArrayRead/VecGetArray` 取得 host 指针。
   - 包装为 `Kokkos::View<double***, HostSpace, MemoryUnmanaged>` 或拷贝到 device `View`。
   - kernel 执行后拷回 PETSc Vec。
   - 优点是改动小；缺点是 PCIe/HBM 拷贝会吞掉性能，只适合早期验证。

2. PETSc GPU Vec 作为主存储，Kokkos View 直接包装 device pointer。
   - 现代 PETSc 支持 CUDA/HIP/Kokkos/standard GPU Vec，需使用对应 API 获取 device 指针，例如 CUDA 路径使用 `VecCUDAGetArrayRead/Write`，Kokkos 路径使用 PETSc Kokkos Vec API。
   - 对应 Kokkos View 使用 `MemoryUnmanaged` 包装。
   - 优点是避免 host-device 往返；缺点是要求 PETSc 版本升级，接口与后端绑定更紧。

对该代码，推荐第二种作为目标，第一种只用于单 kernel 原型。

#### DMDA 到 Kokkos View 的映射

DMDA 本地数组包含 ghost 区，Kokkos View 应显式保存本地 ghost 尺寸与偏移：

```cpp
using View4D = Kokkos::View<double****, Kokkos::LayoutRight, ExecSpace>;

struct LocalBox {
  int xs, ys, zs;      // owned lower
  int xm, ym, zm;      // owned size
  int gxs, gys, gzs;   // ghost lower
  int gxm, gym, gzm;   // ghost size
};

KOKKOS_INLINE_FUNCTION
int li(int i, int gxs) { return i - gxs; }
```

向量场可映射为：

- `u(k,j,i,0/1/2)` 表示 `Cmpnts.x/y/z`。
- 或长期改为 `u_x(k,j,i)`、`u_y(k,j,i)`、`u_z(k,j,i)`。

标量场映射为 `View3D phi(k,j,i)`。执行区域对应 `lzs:lze`、`lys:lye`、`lxs:lxe`，不包括物理边界。

#### stencil 与 MDRangePolicy

大多数规则网格 kernel 可以写成：

```cpp
Kokkos::parallel_for(
  "projection_i",
  Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    ...
  });
```

`RHSSolver` 的三个方向面通量建议拆成三个 kernel：

- `compute_i_flux`
- `compute_j_flux`
- `compute_k_flux`

然后做 halo update，再执行 `divergence_to_rhs`。这与当前 CPU 代码阶段一致，便于验证数值一致性。

#### PETSc interop 与 HostMirror

早期迁移可用：

```cpp
auto h_u = Kokkos::create_mirror_view(u_dev);
// 从 DMDAVecGetArray 得到的 host 数据填充 h_u
Kokkos::deep_copy(u_dev, h_u);
...
Kokkos::deep_copy(h_rhs, rhs_dev);
// 写回 PETSc host Vec
```

但这只能用于功能验证。生产路径应满足：

- PETSc Vec 在 GPU 内存中常驻。
- DMDA halo exchange 由 PETSc 完成，并尽量使用 GPU-aware MPI。
- Kokkos kernel 直接读写 PETSc Vec 的 device buffer。
- 只有 I/O、旧 HYPRE CPU 路径或 IBM 几何搜索需要 host mirror。

#### 优点

- 可同时覆盖 CUDA/HIP/SYCL/OpenMP，适合 HPC portability。
- 与规则 3D stencil 非常匹配。
- 可逐步迁移：先 `Contra2Cart/LES/Projection`，再 RHS，最后 IBM/FSI。
- `parallel_reduce` 可自然替代 `CalculateKE`、`VolumeFlux`、`LM/MM` 等归约。
- 设备端 helper 函数可复用物理公式，降低 CUDA/HIP 双写成本。

#### 缺点

- 当前 build 是旧 makefile，硬编码 PETSc/HDF5 路径。引入 Kokkos 基本要求 CMake 或至少现代化 make 配置。
- PETSc 3.4 风格代码与现代 PETSc GPU API 差距较大，升级成本不可忽略。
- DMDA array 三重指针与 Kokkos View 抽象不同，需要集中封装索引和局部范围。
- 当前 helper 函数如 `Compute_du_center` 接收 `Cmpnts ***`，不能直接用于 device lambda，需要重写为 `KOKKOS_INLINE_FUNCTION` 版本。
- IBM 链表与裸指针所有权复杂，不能直接搬到 device。

### 2.2 Direct CUDA/HIP

直接 CUDA/HIP 的优势：

- 对内存布局、shared memory、warp-level reduction、stream overlap 有最大控制。
- 对 `RHSSolver` 这种复杂 stencil，可手工调优寄存器和访存。
- 若目标机器明确是 NVIDIA 或 AMD，峰值性能可能高于通用 Kokkos 写法。

但对 vwis2.0 的代价很高：

- 需要为 CUDA/HIP 分别维护 kernel 或引入宏抽象，实际会重新发明一部分 Kokkos。
- PETSc/HYPRE interop 仍然存在，无法绕开 Vec device pointer、halo exchange、KSP backend 配置问题。
- 代码虽约 20 个源模块，但核心 `.C` 文件较长，`RHSSolver.C`、`PoissonSolver.C`、`ImmersedBoundary.C`、`WallModel.C` 逻辑复杂，重写风险远大于行数暗示。
- IBM/FSI 的不规则链表和结构体指针在 CUDA/HIP 中同样需要数据重构。

可行性判断：若项目只服务单一 GPU 平台、人员熟悉 CUDA/HIP 且追求极限性能，可以为少数最终热点写 backend-specific kernel。但作为总体迁移路线不推荐。

### 2.3 OpenMP Offloading

OpenMP target offload 是务实但能力有限的选择：

- 对 `Contra2Cart`、`ComputeEddyViscosity`、`CalculateKE` 等简单循环，可通过 `#pragma omp target teams distribute parallel for collapse(3)` 快速验证 GPU 加速。
- 对现有 C 风格循环侵入较小。
- 可先作为“循环可并行性审计”工具。

限制：

- 复杂 stencil 中大量 helper 函数、分支和多阶段 halo 同步会使 data mapping 难维护。
- 对 PETSc Vec device buffer 的互操作不如 Kokkos/PETSc GPU 后端自然。
- 性能可移植性和调优能力通常弱于 Kokkos，尤其是多个临时 View、归约、分层并行和布局切换。

建议定位：用于 Phase 0/1 的快速实验，不作为长期架构。

### 2.4 Raja

RAJA 与 Kokkos 都能表达 loop policy。对本代码风格：

- RAJA 对“把已有 for 循环策略化”很友好。
- Kokkos 除执行策略外，还提供更完整的 `View`、memory space、mirror、reduction 和生态集成。
- PETSc 对 Kokkos 的接触面和用户经验通常多于 RAJA。

因此本项目更适合 Kokkos。RAJA 可作为机构已有 RAJA 标准栈时的替代，但没有明显优势。

### 2.5 PETSc native GPU 支持

PETSc 现代版本已经支持 GPU Vec、Mat 和求解器后端。对 vwis2.0，这是最低扰动的基础层：

- 保持 DMDA 分解与 MPI halo。
- 将全局 Vec 放在 GPU backend 上。
- Poisson/动量相关线性代数尽可能交给 PETSc GPU 后端。
- 自定义 RHS、LES、IBM kernel 用 Kokkos/CUDA/HIP 读写 Vec 的 device buffer。

运行配置示例：

```bash
mpirun -np 4 ./vwis \
  -vec_type cuda \
  -mat_type aijcusparse \
  -dm_vec_type cuda
```

若使用 HIP：

```bash
mpirun -np 4 ./vwis \
  -vec_type hip \
  -mat_type aijhipsparse \
  -dm_vec_type hip
```

PETSc 配置方向示例：

```bash
./configure \
  --with-cuda=1 \
  --with-kokkos=1 \
  --download-kokkos=1 \
  --download-kokkos-kernels=1 \
  --download-hypre=1 \
  --with-hdf5=1 \
  --download-hdf5=1
```

实际选项需按目标集群模块调整。关键是：不要继续依赖 PETSc 3.4 风格构建。先升级到现代 PETSc，再决定 HYPRE GPU 路径还是 PETSc GAMG/AMGX/cuSPARSE 路径。

## 3. 关键热点分析

| 模块/函数 | 类型 | GPU 目标 | 预期加速 | 难度 | Kokkos 模式 |
|---|---|---:|---:|---:|---|
| `UData::Contra2Cart` | 点式 + 近邻面平均 | GPU | 5-20x | 低 | `parallel_for(MDRangePolicy<Rank<3>>)` |
| `LESModel::ComputeEddyViscosity` | stencil-bound | GPU | 5-25x | 中 | `parallel_for` + device helper |
| `LESModel::ComputeSmagorinksyConstant` 常数模型 | 点式赋值 | PETSc VecSet/GPU | 5-30x | 低 | `deep_copy`/`VecSet` |
| `LESModel::ComputeSmagorinksyConstant` 动态模型 | stencil + filter + reduction | GPU/混合 | 3-15x | 高 | 多 kernel + `parallel_reduce` |
| `RHSSolver::CalculatePressureGradient` | stencil-bound | GPU | 4-15x | 中高 | `parallel_for`，分离边界 kernel |
| `RHSSolver::Solve` 面通量 | stencil-bound | GPU | 4-20x | 高 | 三方向 flux kernel + halo + divergence kernel |
| `PoissonSolver::PoissonRHS2_hypre` | stencil-bound | GPU | 5-20x | 中 | `parallel_for` 写 device RHS，避免逐点 HYPRE host call |
| `PoissonSolver::Projection` | stencil-bound | GPU | 5-20x | 中高 | 三个面方向 kernel |
| `PoissonSolver::PoissonLHS` | 复杂 stencil 组装 | CPU 或 PETSc GPU Mat | 1-5x | 高 | 初期保留 CPU；后期 matrix-free/MatStencil |
| `FlowSolver::CalculateDivergence` | stencil + max reduction | GPU | 5-20x | 低中 | `parallel_reduce(Max)` |
| `FlowSolver::CalculateKE` | reduction-heavy | GPU | 5-30x | 低 | `parallel_reduce(Sum)` |
| `UData::Average` | 点式统计 + stencil | GPU | 3-15x | 中 | 分层 kernel，低阶统计先迁移 |
| `BcsUtility::FormBcs/IbBC` | 边界面循环 + IBM 分支 | CPU/GPU 混合 | 1.5-8x | 高 | 边界专用 2D policies |
| `ImmersedBoundary::IBMInterpolationAdvanced` | irregular | 混合，后期 GPU | 1.5-8x | 很高 | 压平 `IBMInfo` 后 `parallel_for` |
| `ImmersedBoundary::IBMSearchAdvanced1` | irregular geometry search | 初期 CPU | 1-5x | 很高 | 空间桶数组化后可 GPU |
| `FSI::CalculateForces1` | irregular + reduction | 混合，后期 GPU | 2-10x | 高 | `parallel_reduce` over flattened IBM cells |
| `WallModel::CalculateVisc/Solve` | stencil + 壁面分支 | GPU/混合 | 2-12x | 高 | 边界区域 kernel |

### 分类说明

#### Trivially parallelizable

- `UData::Contra2Cart`
- `FlowSolver::CalculateKE`
- `FlowSolver::CalculateDivergence`
- `UData::Average` 中一阶/二阶统计
- 常数 Smagorinsky `VecSet(d_lCs, 0.01)`

建议最先迁移。它们便于验证 Kokkos View 包装、索引约定、PETSc Vec 同步和单元测试。

#### Stencil-bound

- `RHSSolver::Solve`
- `RHSSolver::CalculatePressureGradient`
- `LESModel::ComputeEddyViscosity`
- `PoissonSolver::PoissonRHS2_hypre`
- `PoissonSolver::Projection`
- `PoissonSolver::PoissonLHS`

这些决定主性能。迁移时要把“内部规则区域”和“边界/IBM 附近区域”拆开：规则区域走无分支高吞吐 kernel，边界和 `Nvert` 邻近点走单独 kernel 或 CPU fallback。

#### Reduction-heavy

- `CalculateKE`
- `CalculateDivergence` 的 `VecMax`
- `PoissonSolver::RemoveNullspace`
- `PoissonSolver::VolumeFlux`
- 动态 LES 的 `LM/MM` 平均
- FSI 力/力矩/功率积分

建议用 `Kokkos::parallel_reduce`，并只在必要时做 MPI_Allreduce。当前代码存在多个自定义 `GlobalSum_All/GlobalMax_All`，可集中封装为 GPU local reduction + MPI global reduction。

#### Irregular

- IBM solid marking、最近三角面搜索、射线三角相交。
- `IBMInterpolationAdvanced` 链表遍历。
- FSI 基于 IBM 插值点的力积分。

这些不应作为第一批 GPU 工作。优先重构数据结构：把 `IBMListNode` 链表变成连续 `std::vector<IBMInfo>`，再镜像到 `Kokkos::View<IBMInfo*>`；把三角面、节点坐标、法向、面积整理成 SoA View。

## 4. 移植路线图

### Phase 1：构建系统现代化与 PETSc GPU 后端启用

目标：

- 用 CMake 替代当前硬编码 PETSc/HDF5 路径的 makefile。
- 升级 PETSc 到现代版本，明确 CUDA/HIP/Kokkos/HDF5/HYPRE 组合。
- 保留现有 CPU 路径，建立可重复 regression case。
- 加入 `-Wall -Wextra` 的编译清理计划，先修复已知风险：`WallFunctions::utau_wf` 无返回、`PoissonRHS2_hypre` 中 `JAj = getlKAj()` 疑似错误、`ImmersedBoundary` 构造函数花括号风险、`StructSolver::CheckConvergence` 语义风险。

估计工作量：1.5-3 人月。

### Phase 2：迁移低风险点式 kernel

目标：

- 建立 `DMDAView` 或 `LocalFieldView` 包装层，统一索引、ghost 偏移和 component 访问。
- 迁移 `Contra2Cart`、`CalculateKE`、`CalculateDivergence`、低阶 `Average`。
- 验证 CPU/Kokkos/OpenMP/CUDA 后端数值一致性。

估计工作量：1-2 人月。

### Phase 3：迁移规则 stencil 与 LES

目标：

- 重写 `Compute_du_center`、`Compute_du_i/j/k` 为 `KOKKOS_INLINE_FUNCTION`。
- 迁移 `LESModel::ComputeEddyViscosity`。
- 迁移动态 LES 的局部张量计算，归约与均匀方向平均单独处理。
- 迁移 `RHSSolver::CalculatePressureGradient`。

估计工作量：2-4 人月。

### Phase 4：迁移动量 RHS 与 Poisson RHS/Projection

目标：

- 将 `RHSSolver::Solve` 拆成三方向 flux kernel、halo 同步、散度回写 kernel。
- 把边界/IBM 邻近分支分离，减少 warp divergence。
- `PoissonRHS2_hypre` 避免逐点 host `HYPRE_IJVectorSetValues` 成为瓶颈；优先改成 PETSc Vec RHS + GPU-aware Mat/KSP，或批量 assembly。
- `Projection` 改为 Kokkos kernel。

估计工作量：3-5 人月。

### Phase 5：IBM 与 FSI 混合/全 GPU 集成

目标：

- 将 IBM 链表压平为连续数组。
- `IBMInterpolationAdvanced` 迁移为按 IBM 插值点并行。
- `FSI::CalculateForces1` 改为并行归约。
- 几何搜索初期仍可 CPU 执行；若 FSI 每步都移动且搜索成本主导，再迁移空间桶与三角相交。

估计工作量：4-8 人月。

### Phase 6：性能验证与调优

目标：

- 建立 roofline：RHS/LES/Projection 是内存带宽受限还是算术受限。
- 比较 AoS vs SoA。
- 调整 `MDRangePolicy` tile size，例如 `{4,4,32}` 或后端相关 tile。
- 减少临时 Vec：`d_Div1/2/3`、`d_Visc1/2/3` 可考虑融合或重算，平衡 HBM 带宽与寄存器压力。
- 验证 GPU-aware MPI halo、I/O 路径、HDF5 输出与 restart。

估计工作量：2-4 人月。

总计：约 13.5-26 人月。若只做“LES/RHS/Projection 的单 GPU 加速原型”，约 4-7 人月可得到有意义结果。

## 5. 具体代码示例

以下示例展示迁移形态，不是对当前源文件的直接 patch。真实实现前需要先建立 PETSc Vec 到 Kokkos View 的包装层。

### 5.1 梯度/涡粘计算：`ComputeEddyViscosity`

当前 CPU 形态简化如下：

```cpp
for (k=lzs; k<lze; k++)
  for (j=lys; j<lye; j++)
    for (i=lxs; i<lxe; i++) {
      if (nvert[k][j][i] > 1.1) {
        lnu_t[k][j][i] = 0;
        continue;
      }

      Compute_du_center(i, j, k, mx, my, mz, ucat, nvert,
                        i_periodic, ii_periodic, j_periodic,
                        jj_periodic, k_periodic, kk_periodic,
                        &dudc, &dvdc, &dwdc,
                        &dude, &dvde, &dwde,
                        &dudz, &dvdz, &dwdz);
      Compute_du_dxyz(...);
      lnu_t[k][j][i] = Cs[k][j][i] * filter * filter * Sabs;
    }
```

Kokkos 形态：

```cpp
using Policy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

Kokkos::parallel_for(
  "LES::ComputeEddyViscosity",
  Policy3D({lzs, lys, lxs}, {lze, lye, lxe}),
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    if (nvert(k,j,i) > 1.1) {
      nu_t(k,j,i) = 0.0;
      return;
    }

    GradXi g = compute_du_center_device(
      i, j, k, mx, my, mz, ucat, nvert, periodic);

    GradXYZ d = compute_du_dxyz_device(
      csi(k,j,i,0), csi(k,j,i,1), csi(k,j,i,2),
      eta(k,j,i,0), eta(k,j,i,1), eta(k,j,i,2),
      zet(k,j,i,0), zet(k,j,i,1), zet(k,j,i,2),
      aj(k,j,i), g);

    const double sxx = d.du_dx;
    const double syy = d.dv_dy;
    const double szz = d.dw_dz;
    const double sxy = 0.5 * (d.du_dy + d.dv_dx);
    const double sxz = 0.5 * (d.du_dz + d.dw_dx);
    const double syz = 0.5 * (d.dv_dz + d.dw_dy);

    const double sabs = sqrt(2.0 * (
      sxx*sxx + syy*syy + szz*szz +
      2.0*(sxy*sxy + sxz*sxz + syz*syz)));

    const double delta = cbrt(1.0 / aj(k,j,i));
    nu_t(k,j,i) = cs(k,j,i) * delta * delta * sabs;
  });
```

迁移要点：

- `compute_du_center_device` 不能接收 `Cmpnts ***`，必须改为 View。
- 周期边界逻辑建议封装到 `IndexMap`，避免每个 kernel 重复长分支。
- `Nvert` 邻近点导致的单边差分会产生分支，后期可分离规则区域和 IBM 邻近区域。

### 5.2 Poisson RHS 与 Projection

当前 RHS 逐点计算散度后调用 HYPRE：

```cpp
val = -ucont[k][j][i].x + ucont[k][j][i-1].x
      -ucont[k][j][i].y + ucont[k][j-1][i].y
      -ucont[k][j][i].z + ucont[k-1][j][i].z;
val *= -1.0 / dt * St * coeff;
HYPRE_IJVectorSetValues(B, 1, &idx, &val);
```

GPU 迁移不应在 device kernel 内调用 HYPRE IJ API。建议先写 PETSc/Phi RHS Vec：

```cpp
Kokkos::parallel_for(
  "Poisson::RHS",
  Policy3D({lzs, lys, lxs}, {lze, lye, lxe}),
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    if (nvert(k,j,i) >= poisson_threshold) {
      rhs(k,j,i) = 0.0;
      return;
    }

    const double div =
      ucont(k,j,i,0) - ucont(k,j,i-1,0) +
      ucont(k,j,i,1) - ucont(k,j-1,i,1) +
      ucont(k,j,i,2) - ucont(k-1,j,i,2);

    rhs(k,j,i) = -div / dt * St * time_coeff;
  });
```

然后：

- 若继续使用 HYPRE IJMatrix，做一次 device-to-host 或批量拷贝会成为瓶颈，不适合长期方案。
- 更推荐迁移到 PETSc `Mat`/`KSP`，使用 `-vec_type cuda -mat_type aijcusparse` 或 PETSc/HYPRE GPU 组合。
- 若坚持 HYPRE，需确认目标 HYPRE 版本的 GPU assembly/ParCSR 路径，并避免逐点 SetValues。

Projection kernel 形态：

```cpp
Kokkos::parallel_for(
  "Poisson::ProjectionI",
  Policy3D({lzs, lys, lxs}, {lze, lye, lxe}),
  KOKKOS_LAMBDA(const int k, const int j, const int i) {
    if (i <= 0 || i >= mx-1) return;
    if (nvert(k,j,i) + nvert(k,j,i+1) >= poisson_threshold) return;

    const double dpdc = phi(k,j,i+1) - phi(k,j,i);
    const double dpde = 0.25 * (
      phi(k,j+1,i) + phi(k,j+1,i+1) -
      phi(k,j-1,i) - phi(k,j-1,i+1));
    const double dpdz = 0.25 * (
      phi(k+1,j,i) + phi(k+1,j,i+1) -
      phi(k-1,j,i) - phi(k-1,j,i+1));

    const double g11 = dot(icsi, icsi, k,j,i);
    const double g12 = dot(ieta, icsi, k,j,i);
    const double g13 = dot(izet, icsi, k,j,i);

    ucont(k,j,i,0) -= dt / St / time_coeff *
      (dpdc*g11 + dpde*g12 + dpdz*g13) * iaj(k,j,i);
  });
```

真实代码需保留当前的边界、周期和 IBM 单边差分处理。建议先实现 interior fast path，再实现 boundary slow path。

### 5.3 IBM ghost cell/interpolation 策略

当前 `IBMInterpolationAdvanced` 遍历链表：

```cpp
current = d_ibmlist[ibi].head;
while (current) {
  IBMInfo *ibminfo = &current->ibm_intp;
  current = current->next;
  ...
  Uc = cr1 * lucat[k1][j1][i1]
     + cr2 * lucat[k2][j2][i2]
     + cr3 * lucat[k3][j3][i3];
  ...
}
```

GPU 迁移前应改为压平数组：

```cpp
struct IBMInterpPoint {
  int i, j, k;
  int i1, j1, k1;
  int i2, j2, k2;
  int i3, j3, k3;
  int cell;
  double cr1, cr2, cr3;
  double cs1, cs2, cs3;
  double ds, di;
};

Kokkos::View<IBMInterpPoint*> ibm_points("ibm_points", n_points);
```

kernel：

```cpp
Kokkos::parallel_for(
  "IBM::Interpolation",
  Kokkos::RangePolicy<>(0, n_points),
  KOKKOS_LAMBDA(const int q) {
    const auto p = ibm_points(q);

    const Vec3 uc =
      p.cr1 * load_vec3(lucat, p.k1, p.j1, p.i1) +
      p.cr2 * load_vec3(lucat, p.k2, p.j2, p.i2) +
      p.cr3 * load_vec3(lucat, p.k3, p.j3, p.i3);

    const SurfaceTri tri = tri_data(p.cell);
    const Vec3 ua =
      p.cs1 * body_u(tri.v1) +
      p.cs2 * body_u(tri.v2) +
      p.cs3 * body_u(tri.v3);

    const double sb = p.ds;
    const double sc = p.ds + p.di;
    const Vec3 ub = wall_model_or_linear(ua, uc, sb, sc);

    store_vec3(ucat, p.k, p.j, p.i, ub);
  });
```

要点：

- 链表必须消除，GPU 上用连续数组。
- 若多个 IBM 点写同一个 Euler 网格点，需要定义冲突策略；当前链表通常按近界面单元唯一处理，但迁移前要加断言或预处理去重。
- IBM 几何搜索可先 CPU 执行，只把最终插值列表拷到 GPU。

## 6. 风险与挑战

### 6.1 PETSc DMDA 与 Kokkos interop

最大工程风险是数据所有权。当前所有模块都直接 `DMDAVecGetArray`。GPU 版本必须定义统一接口：

- 获取 host view 还是 device view。
- 何时 ghost update。
- 何时 deep copy。
- 哪个模块拥有临时 View。

如果每个模块自行处理 PETSc/Kokkos interop，后期会出现隐式同步、重复拷贝和难以定位的性能退化。

### 6.2 曲线网格 metric tensor 访存

RHS、LES、Projection 同时读大量 metric：

- 中心 `Csi/Eta/Zet/Aj`
- 面 `ICsi/IEta/IZet/IAj`
- `JCsi/JEta/JZet/JAj`
- `KCsi/KEta/KZet/KAJ`

这会造成高内存带宽压力。优化方向：

- 对规则网格或静态曲线网格，将常用 metric 合并或预计算为 `g11/g12/...`。
- 对 `PoissonLHS` 已经临时生成 `G11..G33`，但每次 VecDuplicate/Destroy 不适合 GPU；应持久化。
- 检查 AoS/SoA 对 metric 的影响。很多地方一次只读 `.x/.y/.z` 后立刻 dot，AoS 可能尚可；但按方向只读某些面 metric 时 SoA 更优。

### 6.3 IBM 不规则访存和分支发散

IBM 是 GPU 迁移最主要的不确定性：

- `Nvert` 分支使规则 stencil 产生 warp divergence。
- `IBMListNode` 链表无法高效迁移。
- 三角面搜索、最近点、射线相交工作量不均匀。
- 插值点访问 Euler 网格是 gather/scatter，不连续。

建议策略：

- RHS/LES 先把 `Nvert` 附近区域剥离，规则内部用 fast kernel。
- IBM 插值列表压平后再 GPU 化。
- 几何搜索仅在 FSI 每步移动且 profiling 证明主导时再迁移。

### 6.4 FSI 耦合开销

FSI 每个强耦合子迭代可能触发：

- 回滚流场。
- 移动 IBM 顶点。
- 清空并重建 `Nvert`。
- 重新 IBM 搜索和插值。
- 多次流场求解。

如果这些阶段在 CPU/GPU 间频繁拷贝，整体性能会很差。Phase 5 前应明确：

- 固定 IBM 算例与运动 IBM 算例分开优化。
- 对弱耦合与强耦合分别 profiling。
- IBM 搜索输出的 `Nvert/IBMInfo` 要么留 CPU 并一次性同步，要么全程 GPU。

### 6.5 当前代码质量阻碍

迁移前需要处理的实际阻碍：

- PETSc 3.4 风格 API 与现代 GPU PETSc API 不匹配。
- makefile 硬编码路径，缺少 feature detection。
- 多处长函数耦合物理选项、边界条件和 stencil，难以单元测试。
- helper 函数不是 header-only/device-callable。
- `PoissonRHS2_hypre` 中 `Vec JAj = d_grid->getlKAj();` 疑似复制错误，`Projection` 中也有类似 `JAj = getlKAj()` 片段，需要核对。
- `DMDAVecGetArray`/`RestoreArray` 成对散布全代码，缺少 RAII 封装，异常或早返回风险高。
- IBM 使用裸指针、链表和大量手工内存管理，device 化前需重构。

## 7. 推荐方案

### 7.1 首选路线

推荐采用“现代 PETSc GPU 后端 + Kokkos 自定义 kernel + IBM/FSI 分阶段混合”的路线：

1. 升级 PETSc 和构建系统，先让 Vec/Mat/KSP 可选择 CUDA/HIP 后端。
2. 建立统一 Kokkos View 包装层，不允许业务模块直接散乱处理 device pointer。
3. 先迁移规则、低风险、高调用频率 kernel：`Contra2Cart`、`CalculateKE`、`CalculateDivergence`、`ComputeEddyViscosity`。
4. 再迁移 `RHSSolver` 和 Poisson RHS/Projection，这是主加速来源。
5. IBM/FSI 先保留 CPU 或混合执行；当规则求解器 GPU 化稳定后，压平 IBM 数据结构并迁移插值/力积分。

不推荐一开始直接 CUDA/HIP 重写，也不推荐只加 OpenMP offload pragma 后期待长期可维护性能。

### 7.2 预计工作量

| 阶段 | 工作量 |
|---|---:|
| Phase 1 构建与 PETSc GPU 后端 | 1.5-3 人月 |
| Phase 2 点式 kernel | 1-2 人月 |
| Phase 3 LES/梯度/stencil helper | 2-4 人月 |
| Phase 4 RHS + Poisson RHS/Projection | 3-5 人月 |
| Phase 5 IBM/FSI | 4-8 人月 |
| Phase 6 性能调优与验证 | 2-4 人月 |
| 合计 | 13.5-26 人月 |

若目标是可发表/可演示的 GPU 原型，而非完整生产迁移，可将范围缩小到 Phase 1-4 的子集，约 4-7 人月。

### 7.3 预期加速范围

实际加速取决于网格规模、IBM/FSI 占比、Poisson 求解器迭代数和 MPI/GPU 通信。合理预期：

- 规则固定边界、无 IBM/弱 IBM、单 GPU 或少量 GPU：整体 3-8x。
- 大网格 LES，RHS/LES/Projection 主导，PETSc/HYPRE GPU 求解器有效：整体 5-15x。
- IBM/FSI 强耦合且几何搜索频繁留在 CPU：整体 1.5-5x。
- 只迁移点式 kernel，不迁移 RHS/Poisson：整体通常低于 2x。

### 7.4 工程决策建议

优先投资顺序：

1. profiling 和 regression case。没有基线，GPU 迁移无法判断正确性和收益。
2. PETSc/CMake 现代化。否则所有 GPU 工作都会被旧构建和旧 API 拖住。
3. 数据访问封装。禁止每个模块各自处理 Kokkos/PETSc 指针。
4. RHS/LES/Projection。这里才是主要性能回报。
5. IBM/FSI 数据结构重构。先压平链表，再谈 GPU。

最终目标架构可采用：DMDA/MPI 负责分解和 halo，PETSc GPU Vec/KSP 负责线性代数和求解器，Kokkos 负责自定义 stencil/IBM/FSI kernel，I/O 和复杂几何预处理按需留在 CPU。这只是候选设计；可移植性能取决于 PETSc/HYPRE 后端、GPU-aware MPI、数据驻留和端到端基准。

## 审阅校正：性能假设与准入门槛

- `PoissonRHS2_hypre` 在 host 逐项 `HYPRE_IJVectorSetValues`，`PetsctoHypreVector/HypretoPetscVector` 又显式复制；它是 GPU 迁移的首要数据流风险，而非可假定的 GPU 热点。
- `RHSSolver`、`LESModel`、`Contra2Cart` 的规则循环应先经 profile 确认为候选；IBM 链表/分桶/分支须先压平数据。
- 文中所有加速比均为待验证估计；准入条件为 CPU、单 GPU、强弱缩放、传输/halo 时间、线性求解占比和数值回归同时通过。
