# KIMI AMReX P0/P1 独立 Review（2026-08-20）

## 1. 总体结论与审阅范围

### 总体结论

本次 P0/P1 交付应被认定为 **“P0/P1 工程契约与 P1 单层 Cartesian 数据布局骨架，未编译、未运行、未完成 CFD 算法”**。代码和文档在主要边界上是相互一致的，尤其明确声明了 `advance_one_step()` 是 no-op、没有 physical BC、RHS、压力投影、IBM/FSI、曲线 metric、AMR 或真实 checkpoint payload。这一点比把框架骨架描述为“求解器已移植”更诚实。

但当前不能给出“可构建”“API 已兼容”“halo 已验证”或“数值语义已冻结”的结论。最重要的门槛仍是：取得锁定版本的 AMReX 包并完成 configure/build/CTest，随后完成单/多 rank 的布局和 halo 制造场测试，再冻结 P0-005 的字段、单位、BC 和压力 datum 契约。

### 审阅范围

已阅读：

- `_Docs/AMReX移植任务清单.md`
- `_Docs/AMReX_P0P1_设计说明.md`
- `_Docs/AMReX_P0P1_实施报告_20260820.md`
- `_Docs/AMReX初步移植规划.md`
- `_Docs/AMReX迁移方案.md`
- `amrex_port/CMakeLists.txt`、`CMakePresets.json`、`amrex_version.lock`、`README.md`
- `amrex_port/inputs/*`、`tests/*`、`src/*`

并对原始 `vwis2.0/` 的字段和迁移边界以现有文档中的源码证据作了交叉检查。本次没有修改原始 `vwis2.0/`，也没有修改实现代码、规划文档或任务清单。

## 2. 分级问题

### Critical

未发现足以在已有代码证据下判定为 Critical 的问题。没有证据表明当前骨架已经产生错误物理结果，因为它尚未实现物理推进，也尚未完成编译运行验证。

### Major

#### M-01：AMReX 版本锁没有形成强制、可审计的版本证据

- **位置**：`amrex_port/CMakeLists.txt:6-24`；`amrex_port/amrex_version.lock:2-5`
- **技术原因**：构建要求 `AMReX_VERSION` 等于 `25.02`，但当 AMReX package 不导出 `AMReX_VERSION` 时，CMake 只发出 warning 并继续配置（`CMakeLists.txt:23-24`）。此时 `amrex_version.lock` 只是请求值，不是实际包的 tag、git SHA 或安装前缀的证明。
- **影响**：即使未来找到一个 API 不同的 AMReX 包且不导出版本变量，配置可能继续到编译阶段；即使编译成功，也不能仅凭当前 lockfile 宣称使用了 25.02。P0-003/P1-002 的“固定版本”验收仍未闭合。
- **建议**：将实际 AMReX git SHA、包安装前缀、编译器、MPI/GPU backend、precision、DIM、CMake cache 和完整 configure 命令写入构建证据；若 package 不导出版本，默认应停止并要求显式提供经审计的版本证明，而不是仅 warning。允许 mismatch 的 exploratory 路径应保留，但必须单独标记为未验证。

#### M-02：CTest 只验证“程序能被调用”，没有验证 P1 的关键 halo/布局契约

- **位置**：`amrex_port/CMakeLists.txt:31-35`；`amrex_port/tests/static_contract_check.sh:1-18`
- **技术原因**：CTest 的两个测试只运行 `p1_smoke.in` 和 `p1_multibox.in`。实现本身只清零字段、调用 `FillBoundary` 并打印诊断；没有制造场、ghost sentinel、跨 Box 复制值、周期边界对照、全局 reduction 或多 rank 一致性断言。静态脚本仅使用文本匹配，不能验证运行时 API 语义。
- **影响**：即便 CTest 将来返回 0，也不能证明 `DistributionMapping`、face BoxArray、nGrow、周期 halo、跨 rank halo 或非周期 physical ghost 责任边界正确。将 CTest 通过作为 P1-003 完成证据会夸大验收强度。
- **建议**：增加独立的运行时契约测试：valid 区写入可识别制造场，检查同一 Box/跨 Box/周期 ghost 的预期值，使用至少 1 和 2 MPI ranks，区分 periodic 与 non-periodic 情况，并对 non-periodic physical ghost 明确检查“未由 FillBoundary 填充、不能被 stencil 读取”。CTest 应分别登记 serial、MPI（若 AMReX/MPI 构建可用）和 schema 输出检查。

#### M-03：BCRec 仅为 cell 字段的未使用元数据，不能代表完整 BC 接口已建立

- **位置**：`amrex_port/src/VwisAmrExSolver.H:82`；`VwisAmrExSolver.cpp:94-105`
- **技术原因**：`m_cell_bcs` 只有 `AMREX_SPACEDIM` 个 `BCRec`，并没有为 `P/Phi/Nvert/Ucat` 的每个 component 或三个 face `Ucont` 字段建立可追踪的 BC 映射；该数组也没有被任何填充 physical ghost 的 functor 使用。所有方向都固定为 `ext_dir`，未表达旧代码六面整数 BC、周期/压力/速度耦合或 face-normal 条件。
- **影响**：P1-004 只能算 BC 元数据占位，不能算 BC 分类接口或物理边界实现。后续若直接在此接口上实现投影或入口/出口条件，容易把 scalar、cell velocity 和 face flux 的边界语义混在一起。
- **建议**：为每类字段/component 或明确的 field group 建立 BC schema，逐面记录旧 BC code 到 AMReX BC 类型及 physical functor 的映射；对未支持的组合显式拒绝。先完成 non-periodic 单层制造场 BC 测试，再进入 P3/P4。

#### M-04：P0-005 仍不能称为字段/单位/时间层已冻结

- **位置**：`amrex_port/src/VwisAmrExSolver.cpp:76-90`；`amrex_port/src/VwisAmrExSolver.H:18-30`；`_Docs/AMReX_P0P1_设计说明.md:21-31`
- **技术原因**：`Ucont` 的 metadata 同时写作“contravariant normal flux/velocity”，并明确 normalization unresolved；`P`、`Phi` 的单位和 pressure datum 未确定；所有字段暂用同一个 nGrow；`time_layer` 是描述字符串，代码没有历史层旋转或推进。
- **影响**：这是合理的 provisional contract，但不能作为旧 `Ucont/Ucat/P/Phi/Nvert` 数值等价或时间积分等价的证据。尤其 `Ucont` 是速度还是包含面积/Jacobian 的体积通量，会直接改变散度、Poisson RHS 和 projection 修正的量纲。
- **建议**：在有版本控制的 Cartesian reference case 上冻结：变量定义、面法向方向、是否含 face area/volume、压力和 correction 的单位、`dt`/时间层系数、BC code、pressure datum、restart 保存层。冻结前保持 P0-005“进行中”。

### Minor

#### m-01：`CMakePresets.json` 只提供 CPU preset，没有显式固定 MPI/编译器/AMReX backend

- **位置**：`amrex_port/CMakePresets.json:3-12`；`amrex_port/amrex_version.lock:5`
- **影响**：preset 可复用性有限；`mpiexec` 命令在 README 中只是配方，无法从 preset 判断目标包是否启用了 MPI，也无法重放 GPU 或精度配置。
- **建议**：P1 CPU 单 rank 可保留当前 preset，但增加并记录 MPI preset 或明确说明 MPI 由 AMReX package 固定；若后续提供 GPU preset，必须记录 backend 和验证状态，不要把未测试选项写成支持矩阵。

#### m-02：输入文件和 CTest 没有显式区分“运行目录”与 metadata 输出契约

- **位置**：`amrex_port/inputs/p1_smoke.in:7`、`p1_multibox.in:7`；`amrex_port/CMakeLists.txt:33-34`
- **影响**：metadata 文件使用相对路径；直接从项目根运行与从 CTest 工作目录运行时输出位置不同，可能造成验证人员找错文件或把旧输出误认为当前结果。
- **建议**：测试使用显式输出目录或在 CTest 中设置 `WORKING_DIRECTORY`，并让测试检查 JSON 的 `schema`、`payload_written=false` 和字段计数，而不是只看进程退出码。

#### m-03：代码证据不足以确认目标 AMReX 25.02 的所有 API/target 名称兼容

- **位置**：`amrex_port/CMakeLists.txt:27-29`；`src/VwisAmrExSolver.cpp:54-68,111-147`；`src/main.cpp:13-45`
- **技术原因**：当前环境没有 `AMReXConfig.cmake`，因此 `AMReX::amrex` target、`Geometry::define` 参数、`MultiFab` 对 converted face `BoxArray` 复用 `DistributionMapping`、`FillBoundary(Periodicity)`、`ParallelFor` lambda 签名和 `RealBox` 初始化均未经过目标版本编译器验证。
- **影响**：不能排除 API 级编译错误或不同 package 导出 target 名称造成的 configure/build 失败。
- **建议**：安装/提供已锁定 AMReX 25.02 package 后，先做 clean configure，再做 CPU compile；将实际 compiler diagnostics 纳入报告。若只支持 3D，应在 CMake 中显式固定 `AMREX_SPACEDIM=3`，否则 DIM 变化时输入文件和 `RealBox`/三维默认值的可移植性需要额外测试。

## 3. CFD 数值语义审阅

### cell/face staggering 与 IndexType

当前设计是正确且清晰的初始方向：`P`、`Phi`、`Nvert`、`Ucat` 为 cell-centered；`Ucont[0/1/2]` 分别由 `convert(m_ba, TheDimensionVector(dir))` 构造 x/y/z face BoxArray。代码没有把 `Ucont` 退化成 cell-centered 三分量伪 MAC，这是重要的正向判断。

但这里只证明了数据结构选择，未证明离散语义：没有散度 kernel、face flux 积分、`Ucont↔Ucat` 变换、pressure gradient 或 projection。`Nvert` 只被当作 legacy 分类并清零，不能解释为 EB volume fraction，也没有 IBM 几何含义验证。

### ghost/nGrow 与 FillBoundary

`nghost` 从 `vwis.nghost` 读取，所有字段采用同一 grow width；`initialize()` 对 cell valid region 做 kernel 写入，对 face MultiFab 使用 `setVal(0.0)`，随后对所有字段调用 `FillBoundary`。文档正确说明 `FillBoundary` 只负责同层 Box/MPI/周期交换，不能填非周期 physical ghost。

风险在于：同一 nGrow 是接口占位而非离散需求；未来不同 stencil、face metric、IBM interpolation 可能需要不同 ghost 宽度。更重要的是，当前没有运行时检查证明“未填的 non-periodic ghost 不会被读取”。P3 之前必须将 valid/halo/physical ghost 的调用顺序和每个字段的 nGrow 固定下来。

### `P/Phi/Nvert/Ucont/Ucat`、时间层和单位

字段注册包含 `pressure`、`phi`、`nvert`、`ucat`/`ucat_old` 以及每方向 `ucont`、`ucont_old`、`ucont_older`，历史层布局与 P1 目标相符。实现不会旋转历史层，也不会改变 valid state；`Phi` 是 workspace。这个 no-op 语义与文档一致。

尚未解决的数值契约包括：`Ucont` 是 normal velocity 还是已含面积/Jacobian 的通量，pressure correction 的单位和符号，旧 BDF/SNES 时间系数，`P` 的 datum 和全 Neumann 零空间，`Nvert` 分类码的整数/浮点语义，以及 `Ucat` 与 `Ucont` 的同步关系。因此不能做守恒、散度、压力或时间收敛结论。

## 4. 内存与并行审阅

- `VwisAmrExSolver` 以值成员持有 `Geometry`、`BoxArray`、`DistributionMapping` 和 `MultiFab`，没有发现长寿命 non-owning `MultiFab*` 或 PETSc alias。ownership 设计方向合理。
- `MFIter` 中取得的 `Array4` 只在 `ParallelFor` 调用内使用，lambda 按值捕获，未捕获 `this`、host vector 或临时 host 指针；GPU-safe 设计证据存在。
- 长期字段在构造函数分配，代码没有在 `MFIter` 中创建 owning `MultiFab`。这是正确的生命周期约束。
- `DistributionMapping` 被用于 cell 和 converted face BoxArray。该用法需要在锁定 AMReX 版本中实际编译，并在多 Box/MPI 运行时检查 face box 的 ownership 和跨 box 边界是否符合预期。
- `FillBoundary` 已覆盖历史层和 face/cell 字段，但没有 physical BC functor；`BCRec` 目前只是未使用的 metadata。不能把 periodic halo 通过等同于 physical BC。
- 未运行 MPI，因此没有证据证明 1/2/多 rank 的 Box 分配一致性、halo 内容、MPI reduction、IO rank 输出和 face ownership。
- 没有 GPU 编译/运行证据。`AMREX_GPU_DEVICE` 和 `ParallelFor` 只证明源码意图，不能证明目标 AMReX 编译器、设备 lambda、内存驻留或 GPU-aware MPI halo 可用。

## 5. 框架/API/CMake 审阅

### 版本锁与 AMReXConfig.cmake

`CMakeLists.txt` 使用 `find_package(AMReX CONFIG QUIET)`，缺包时给出明确的 `AMReX_DIR`/`CMAKE_PREFIX_PATH` 诊断，并禁止 fallback solver。对导出版本不等于 25.02 的包默认失败，exploratory mismatch 需要显式打开，这个策略合理。

不过版本 lock 仍是意图而非证据；见 M-01。特别是 package 不导出 `AMReX_VERSION` 时只 warning，不能称“锁定版本已验证”。

### CTest 输入

CTest 注册了 single-box 和 multi-box 两个命令，输入文件路径为绝对的 source-relative 路径参数，适合作为最小运行配方。但它们都使用零场/no-op，且没有断言 JSON、halo 或 MPI 结果，所以目前是 smoke invocation，不是 P1 acceptance test。

### 可能的编译/API问题

在缺少 AMReX package 的环境中无法对以下接口给出通过结论：

- `AMReX::amrex` imported target 是否由目标安装导出；
- `Geometry::define` 与 `RealBox` 初始化的目标版本签名；
- converted face `BoxArray` 与复用 `DistributionMapping` 的兼容性；
- `MultiFab::FillBoundary(Periodicity)`、`setVal`、`norm0` 的目标版本 API；
- `ParallelFor` 的 `AMREX_GPU_DEVICE` lambda 和 `TilingIfNotGPU()` 组合；
- 目标编译配置下 `AMREX_SPACEDIM`、MPI、GPU backend 与输入文件的匹配。

这些是“因环境缺失未验证”的 API 风险，不应直接写成已确认的实现 bug。

## 6. P0/P1 任务状态是否诚实

总体上是诚实的，且与代码证据相符：

- `P0-003`：版本 lock/preset/CMake 诊断存在，但没有实际 AMReX、MPI、ABI 或编译证据，标记“进行中”正确。
- `P0-004`：没有 reference case、网格、控制文件和基准输出，标记数值 blocked 正确。
- `P0-005`：provisional contract 已有，但单位、BC、`Aj`、pressure datum 未冻结，标记“进行中”正确。
- `P1-001`：框架和字段布局存在，但无编译/运行和物理算法，标记“进行中”正确。
- `P1-002`：缺 `AMReXConfig.cmake`，未配置/构建，标记 blocked/进行中正确。
- `P1-003`：代码有 FillBoundary 路径，但没有 MPI halo 结果，标记进行中正确。
- `P1-004`：BCRec metadata 存在，但 physical functor 和旧 BC 映射不存在，标记进行中正确。
- `P1-005`：只有 schema-only JSON，明确 `payload_written=false`，没有 plotfile/checkpoint/restart，标记进行中正确。

需要继续防止的误读是：静态脚本通过、CMake 诊断正确、字段成员存在，都只能证明工程意图和源码形状，不能把“骨架存在”写成“算法完成”，也不能把 CTest 进程返回 0 写成 halo 或数值验收通过。

## 7. 必须运行的验证命令与 P2/P3 前置条件

### 环境可用后的最低验证顺序

以下命令应在实际锁定的 AMReX 25.02 package、编译器和 MPI 环境中执行，并保存命令、完整输出、退出码和 CMake cache：

```bash
bash amrex_port/tests/static_contract_check.sh
cmake -S amrex_port -B build/amrex_port \
  -G Ninja \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX \
  -DBUILD_TESTING=ON
cmake --build build/amrex_port --verbose
ctest --test-dir build/amrex_port --output-on-failure
./build/amrex_port/vwis_amrex_skeleton amrex_port/inputs/p1_smoke.in
./build/amrex_port/vwis_amrex_skeleton amrex_port/inputs/p1_multibox.in
mpiexec -n 2 ./build/amrex_port/vwis_amrex_skeleton amrex_port/inputs/p1_multibox.in
```

实际路径、MPI launcher 参数和 AMReX 配置应按环境调整；不得把命令配方当成已通过结果。若版本不匹配，必须显式使用 exploratory 选项并把结果标为未验证，而不是替代锁定版本证据。

### P2/P3 前置条件

在进入 P2/P3 前必须完成：

1. AMReX 版本、git SHA、DIM、precision、编译器、MPI/GPU backend 和 CMake 配置可复现。
2. P1 clean configure/build/CTest 成功，至少有单 rank 和 2 rank 运行记录。
3. 用制造场验证 cell/face `IndexType`、Box 分割、nGrow、periodic/inter-Box halo 和 face ownership；比较不同 MPI rank 数的结果。
4. 明确 non-periodic physical ghost 的填充接口和调用顺序，不能继续依赖 `FillBoundary`。
5. 冻结 `Ucont` 的速度/体积通量定义、面方向和单位；冻结 `Ucat` 同步规则。
6. 取得受版本控制的旧 solver case、网格、BC 文件和输出基线，逐面映射旧 BC code。
7. 冻结 `P/Phi` 的符号、单位、pressure datum、零空间和可解性条件，明确 `dt` 与历史层系数。
8. 先完成 Cartesian 字段和 BC 的质量通量/散度诊断，再实现 P4 投影；曲线 19 点算子不能直接用 Cartesian MLMG/MacProjector 代替。

## 8. 检查状态汇总

### 无问题/已通过的检查

- 未发现对原始 `vwis2.0/` 的修改意图；本次审阅未修改该目录。
- P1 no-op 边界在头文件、实现、README、设计说明和实施报告之间基本一致。
- cell/face 字段的初始布局方向与迁移设计一致，未发现把 `Ucont` 伪装为 cell-centered MAC 的代码证据。
- `Nvert` 明确标记为 legacy classification，不冒充 EB volume fraction。
- `FillBoundary` 与 physical BC 的责任边界在文档和实现注释中明确区分。
- `MFIter`/`Array4` 使用方式没有明显的跨迭代器生命周期逃逸证据。
- schema-only metadata 明确写出 `payload_written=false`，没有把 JSON 冒充 checkpoint。
- `bash amrex_port/tests/static_contract_check.sh`：通过。
- `git diff --check`：通过。
- 当前 CMake configure：按预期失败，错误为缺少 `AMReXConfig.cmake`，并输出了 `AMReX_DIR`/`CMAKE_PREFIX_PATH` 诊断；这不是 configure 成功证据。

### 因环境缺失未验证的检查

- AMReX 25.02 实际 package、git SHA 和 imported target/API 兼容性。
- CMake configure 成功、C++ 编译成功和链接成功。
- CTest single-box/multi-box 的运行结果。
- 1/2/多 MPI rank 的 DistributionMapping、halo、周期边界和跨 Box 一致性。
- non-periodic physical BC functor 的行为。
- GPU 编译、GPU kernel、设备内存驻留和 GPU-aware MPI。
- `Ucont/Ucat` 转换、散度、压力 RHS、投影、守恒、时间收敛和任何 CFD 数值正确性。
- plotfile/checkpoint payload、restart 连续性、旧格式转换。
- 曲线 metric、19 点 Poisson、IBM/EB、FSI、AMR、LES、性能扩展性。

## Verdict

**Verdict：可作为“未编译的 P0/P1 Cartesian 工程骨架”继续推进，不可作为已完成 AMReX CFD 移植或数值正确性的交付。**

建议优先级：

1. **最高**：取得并锁定 AMReX 25.02 构建环境，完成 clean configure/build，并补齐真实单/多 rank P1 halo/layout 测试。
2. **高**：冻结 `Ucont/Ucat/P/Phi/Nvert` 的单位、时间层、面通量和 pressure datum 契约，逐面建立 physical BC 映射和 functor 测试。
3. **中**：将 CTest 从 smoke invocation 扩展为可断言的运行时契约测试，再进入 P2 字段回归和 P3 physical BC。
4. **后置**：仅在 Cartesian 基线、投影、质量守恒和 restart 通过后，处理曲线 19 点算子、IBM/FSI、AMR、GPU 和性能优化。
