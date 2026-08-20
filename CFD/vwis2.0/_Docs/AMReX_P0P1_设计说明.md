# AMReX P0/P1 基础工程设计说明

更新：2026-08-20。本说明记录已实现的 P0/P1 工程边界，不修改也不重述
原 `vwis2.0/` 求解器的历史事实。P1 是单层规则 Cartesian 框架，绝不是
任一物理算法、旧曲线网格或 IBM/FSI 的完成声明。

## 1. P0 接口契约与证据边界

旧代码的可追溯来源是 `main.C` 的生命周期、`CurvGrid.C` 的 DMDA/metric、
`UData.C` 的向量及历史层、`BcsUtility.h/.C` 的宏和算例分支、
`FlowSolver.C` 的步骤顺序、`PoissonSolver.C` 的压力更新/零空间；构建来源
是 `makefile`。旧 makefile 硬编码 PETSc 3.4、HDF5、HYPRE、`mpicxx`，并不
构成可重现安装：当前 workspace 也没有找到 `AMReXConfig.cmake`。

P0 提供 `amrex_port/amrex_version.lock`（请求 AMReX 25.02/CMake 3.20/C++17）
和 CMake preset、两份最小输入、静态检查脚本。它们可重放配置意图和失败
诊断，但没有宣称 PETSc case、AMReX、MPI 或基准数据真实可运行。实际配置应
记录 AMReX git SHA、编译器、MPI、GPU backend、PETSc/HYPRE/HDF5 ABI、完整
CMake cache，以及命令/退出码。

|契约项|P1 表达|仍待冻结/验证|
|---|---|---|
|`P`|cell, 1 component, `n`, nGrow=`vwis.nghost`|旧 pressure 的量纲转换和 datum|
|`Phi`|cell, 1 component, workspace|Poisson 符号、单位、BC、零空间|
|`Nvert`|cell, 1 component, legacy 分类|分类整数语义；不是 EB volume fraction|
|`Ucat`|cell, 3 components (x/y/z), `n` 与 `n-1`|无量纲速度标度和 `Ucont` 同步|
|`Ucont[d]`|d-face, 1 normal component, `n/n-1/n-2`|它究竟是速度或体积通量、metric 归一化|
|时间|`dt>0` 由 ParmParse 读取；不旋转历史层|旧 BDF/SNES 分支的时间系数|
|BC|每 component 的 `BCRec(ext_dir)` 元数据|旧六面整数码、入口/出口/墙面具体条件|
|pressure datum|明确未设置|case 驱动的 Dirichlet/Neumann/均值策略|
|I/O|JSON schema metadata，`payload_written=false`|AMReX plot/checkpoint payload 与旧 PETSc 转换|

旧 `UData` 的 `Ucont/Ucat` 都是三分量 DMDA 视图；本框架有意将 `Ucont`
拆成类型化 MAC face 数组，避免把旧表示误当作 face ownership 的证据。

## 2. 框架与内存设计

`VwisAmrExSolver` 以值成员拥有 `Geometry`、`BoxArray`、
`DistributionMapping` 和所有 `MultiFab`，因此没有长寿命的 non-owner
`MultiFab*` 或 PETSc global/local alias。`DistributionMapping` 是唯一的并行
ownership 定义；每个 rank 只经 `MFIter` 访问自己的 FAB。`BoxArray::maxSize`
决定单/多 Box 布局，输入 `p1_smoke.in` 和 `p1_multibox.in` 分别覆盖这两种
结构。多 rank 运行是待 AMReX/MPI 可用后执行的验证命令，尚无结果。

所有 P1 场采用同一 `nGrow`，以便接口先保持一致；这不是未来离散模板必需
的最终选择。cell 数据使用 cell `IndexType`；`Ucont[0/1/2]` 使用 x/y/z-face
`IndexType`，各自仅含 normal component。面上合法的 face ownership 由 AMReX
的 face `BoxArray` 与 `DistributionMapping` 决定，不能自行双写 box 边界面。

初始化 kernel 只写 `validbox()`，Array4 按值捕获并标为
`AMREX_GPU_DEVICE`，故不捕获 host 容器、`this` 或临时 host 指针。长期字段
在构造函数分配；不得在 `MFIter` 内创建长期 owning `MultiFab`。临时工作
场未来应在算法对象外层按明确 owner 生命周期分配，或用 tile-local POD，
并在 kernel 返回前销毁。

## 3. 并行数据流与边界

```text
valid-region write (MFIter/ParallelFor)
  -> FillBoundary(periodicity): inter-Box + MPI + periodic ghost
  -> [future physical BC functor fills ext_dir ghosts]
  -> stencil may read valid + ghost regions
```

`FillBoundary` 不能填非周期物理 ghost；P1 的 `BCRec` 统一为 `ext_dir`，只
声明将来 functor 的责任。对于非周期盒，未被 functor 填的 ghost 不能读取。
跨 AMR level 的 `FillPatch`、coarse-fine ownership 和 physical BC 都不在 P1。

`initialize()` 清零后填 halo；`advance_one_step()` 只作正 `dt` 检查和一次
halo 交换，不更改任一 valid state 或时间层。这保证 P1 的日志/计时可测试，
但不能以零场为守恒、散度、投影或 restart 的数值证据。

## 4. 运行、异常与 I/O 边界

`main` 调用 `amrex::Initialize/Finalize`；所有 `ParmParse` 键置于 `vwis.*`
namespace。构造参数（维度、正网格数、0/1 periodic、正 `dt`）错误会在主函数
捕获后以 rank-safe `amrex::Print` 报告并返回失败。诊断仅由 IO rank 输出，
包括 box/rank 数、零场范数与计时。未用 `Abort` 替代可恢复输入错误，也不在
GPU lambda 内抛异常。

metadata manifest 是独立 JSON，schema 为 `vwis-amrex-p1-metadata-v1`；它列出
字段位置、components、nGrow、单位状态与时间层，特意标记没有 payload。未来
P8 checkpoint 至少要写入所有流体历史层、time/step、units/schema 版本、metric
epoch、IBM/FSI 状态并进行连续跑与 restart 比较；P1 不创建空目录来假装可恢复。

## 5. 验证方法与未完成项

无 AMReX 环境时执行 `bash amrex_port/tests/static_contract_check.sh`，它检查
版本发现/诊断、face history、BCRec、GPU-safe kernel 标记和无物理算子引用。
有锁定 AMReX 后依次 configure/build/ctest，运行 single-box；再以多 Box 和
至少 2 ranks 检查 halo/布局。通过前不得标记 P1-002/003 已完成。P0-004 无
`control.dat`、`bcs.dat` 和基准输出，P0-005 因而不能最终“评审冻结”；本文件
仅建立明确的 provisional contract 和未决项。

下一阶段应首先取得受版本控制的 reference cases 与旧程序构建证据，再逐面
冻结 BC 表和 pressure datum；随后用制造场验证 P1 halo、cell/face mapping，
再进入 P2/P3。不可用 P1 的 Cartesian 零场替代旧曲线 19 点压力算子或 IBM/FSI
验证。
