# Covector Fluids：论文—代码算法设计说明

## 1. 目标与实现边界

《Covector Fluids》把速度一形式（离散实现中是 MAC 面速度）视为随流体运动的 covector。连续层面，流映射 \(\Psi\) 的 pullback 给出

\[
\omega_t = (\Psi_t^{-1})^*\omega_0,
\]

从而保留 Kelvin 环量结构；代码则在规则 MAC 网格上，以前向/后向 particle map、局部 Jacobian 差分和插值实现这一思想。它是对半拉格朗日方法的结构化改进，不等同于任意网格上的严格离散守恒证明。源码还把 CF 与 BFECC、BiMocq 或 MCM 组合，形成可切换的实验框架。

本文以论文精读文档 `Projects/Papers_matters/computer-graphics/covector-fluids.md` 和源码为准，重点说明实际调用链；论文中的连续性质不能直接当作代码已严格满足的离散性质。

## 2. 数据布局

### 2.1 二维

`src/covector2D/CovectorSolver2D.h/.cpp` 使用 MAC 排布：`u(nx+1,ny)` 位于竖直面，`v(nx,ny+1)` 位于水平面；`rho`、`temperature` 和压力位于 cell center。`forward_x/y`、`backward_x/y` 保存速度 map，另有 scalar map；`*_init`、`*_origin`、`*_temp`、`*_change` 用于 BFECC、累计变化和重映射。障碍或 level-set 数据采用与 cell 标量相同的二维数组。

### 2.2 三维与 GPU

`src/covector3D/CovectorSolver.h` 的 MAC 面数组为：`_un(nx+1,ny,nz)`、`_vn(nx,ny+1,nz)`、`_wn(nx,ny,nz+1)`；`_rho`、`_T` 为 cell-centered。`include/fluid_buffer3D.h` 的 `Buffer3D` 使用 8×8×8 block-major 存储，并把物理边界/插值访问统一经过 `.at()` 与线性/三次采样。`Mapping.*` 中 `gpuMapper` 负责 host 面速度到 device 的拷贝、map 更新和重映射。

CUDA kernel 位于 `src/covector3D/GPU_kernel.cu`：线程块通常为 256 threads；u/v/w 三类面分别按自己的有效尺寸发射。`advect_covector_kernel` 通过前后 map 位置差构造局部 pullback；`advect_kernel` 做普通标量/分量平流，`doubleAdvect_kernel` 做 map 混合，compensation kernel 实现 BFECC/误差补偿及局部极值限制。三维压力投影仍主要在 CPU 侧，GPU 负责 mapping 与 advection，因此并非全流程 GPU resident。

## 3. 统一时间步调用链

入口分别是 `src/covector2D/main.cpp` 和 `src/covector3D/main.cpp`：读取 `sim_method`、`sim_setup`，构造 solver，设置烟雾、边界和 VDB 几何，先做初始 projection，再逐 frame 调用 `advance()` 和 `outputResult()`。

每个 scheme 的共同骨架是：

1. 根据 CFL 计算子步数，更新 backward/forward map；2. 用 map 追踪速度、密度和温度；3. 需要时做 CF pullback 或 BFECC/MCM 误差校正；4. 注入浮力、烟雾和外部变化；5. 施加边界并解压力 Poisson 方程，减去压力梯度；6. 估计 map distortion，超过阈值时 reinitialize/remesh；7. 写出图像、能量/涡量统计，或 VDB density/velocity 与 OBJ 边界。

二维的具体分派在 `CovectorSolver2D::advance()`，三维在 `CovectorSolver::advance()`。`advanceCovector()` 先可选半步 `fullAdvect(dt*0.5,...)`，随后完整步，再调用 `estimateDistortion`；二维阈值约为 0.1，三维约为 0.5，超过阈值或达到 delayed frequency 时调用 `velocityReinitialize`/`scalarReinitialize`。这类阈值是实现参数，不是论文定理。

`fullAdvect()` 是 CF 主路径：更新 map，调用 covector velocity/scalar advection，执行 BFECC 或 error compensation，混合边界字段，清理边界，施加 buoyancy，可选扩散，再 projection，并把 change buffer 累计回 mapper。二维对应函数在 `CovectorSolver2D.cpp`，三维由 `Mapping.cpp` 调 CUDA kernel。

## 4. 八种 Scheme

枚举在二维/三维 solver header 中一致：`0 SEMILAG`（普通半拉格朗日）、`1 REFLECTION`（边界 reflection）、`2 SCPF`（trapezoidal/对称 covector 形式）、`3 MACCORMACK`（前后向 MacCormack）、`4 MAC_REFLECTION`、`5 BIMOCQ`（双向 map + BFECC/EC + remesh）、`6 COVECTOR`（CF pullback）、`7 COVECTOR_BIMOCQ`（CF 与 MCM/BiMocq 组合）。

`advanceSemilag()`/`advanceMaccormack()` 只依赖普通 characteristic trace；`advanceSCPF()` 用 backward map 的前后采样做对称速度；`advanceBIMOCQ()` 同时维护 map、advect、correct、buoyancy、projection 和 remesh；`advanceCovector()` 走 `fullAdvect()`，其关键区别是用 map 的局部差分把初始 covector pullback 到当前 MAC 面。scheme 7 由 delayed reinit 与 MCM/EC 开关决定具体混合，不能简单理解为一个独立的新离散定理。

## 5. 压力、边界与稳定性

二维 `projection()` 先调用 `applyVelocityBoundary()`，组装

\[
rhs_{ij}=-(u_{i+1,j}-u_{i,j}+v_{i,j+1}-v_{i,j})/h,
\]

再用 `AMGPCGSolvePrebuilt2D()` 解压力，最后在相邻面减/加压力梯度；`projection_repeat_count` 可重复投影。`buildMultiGrid()` 预组装五点 Laplacian，并用 AMG Galerkin 粗化；纯 Neumann 情况需要去除压力零空间。三维同样是 MAC 散度—Poisson—梯度修正，但求解器与 advection 的 CPU/GPU 分工更混合。

边界通过 `applyVelocityBoundary`、reflection、obstacle mask、level-set/VDB 几何和入口速度组合实现。半拉格朗日 trace 使用 RK 或 DMC；CFL 控制 map 子步。BFECC/MCM 用前向—反向误差估计并做 extrema clamp。扩散（若开启）是约 20 次 red-black Gauss-Seidel。以上机制改善稳定性，但源码没有对所有边界、任意网格或严格离散 Kelvin 守恒给出统一证明。

## 6. setup、输出与复现重点

二维 setup 0–5 依次覆盖 Taylor vortex、leapfrogging、ink/Rayleigh–Taylor、SIGGRAPH logo、inverted Zalesak、von Kármán；典型分辨率为 200–512 或 256²，时间步约 0.01–2，帧数从几十到数千。输出默认在 `../Out_2D`，包括 density、velocity、vorticity、covector BMP 及能量/涡量文本。

三维 `main.cpp` 的 experiment 实际为 0–6：trefoil knot、leapfrogging、smoke plume、pyroclastic cloud、ink jet、delta wing、bunny meteor。默认 `_baseres=128`，按实验放大某一方向，`substeps` 为 2/4/8；输出默认硬编码为 Windows 路径 `H:/BiMocq/Out_3D/`，写 VDB density/velocity 和 OBJ。README 所写“0–7”与实际 switch 不一致。

## 7. 已知构建/代码限制

- 顶层 `CMakeLists.txt` 使用 `project(Covector CXX CUDA)`，即使只想运行二维也会要求 CUDA；README 中“移除 CUDA 即可”的说明与当前顶层配置不完全一致。
- `covector3D/CMakeLists.txt` 仍含 `${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib`、旧 `cuda_add_executable`、固定 `sm_61/sm_75` 和 `CUDA_SAMPLES_DIR`，Linux/新 CUDA/CMake 可能失败。
- 三维输出路径是 Windows 盘符；源码中 `experiment_name` 在参数解析前被检查，属于未初始化输入校验 bug，应先 `atoi(argv[2])` 再范围检查。
- `gpuMapper` 析构函数为空，疑似未释放 device buffer；CUDA 错误检查和同步点也较少。`Buffer3D::at()` 的 clamp 会掩盖部分越界访问。
- 2D/3D 默认实验规模较大，不能把“进程启动/生成首帧”误报成完整实验通过；应记录编译、初始化、若干帧、全程和输出文件四类证据。

## 8. 面向 AMReX/VWiS 的迁移建议

优先保留数学对象而非逐函数搬运：把 MAC face velocity、cell scalar、forward/backward map 和 covector pullback 分别建成明确的 MultiFab/时层 contract；把 periodic/MPI halo 与 physical ghost 分离。`advect_covector_kernel` 的局部 Jacobian/pullback 应成为独立的 face-centered operator，在 AMReX 上用 tile kernel 实现；BFECC clamp、distortion/remesh 和 pressure projection 则分别成为可测试模块。压力 RHS 必须明确边界通量贡献和 null-space 处理，不能把当前 CPU AMG 的隐含假设直接移植。

建议验证顺序为：规则 Cartesian、周期 Taylor vortex；单步 map/pullback 与 CPU reference 对拍；divergence/projection；BFECC extrema；再做 MPI 多 Box、CUDA kernel、VDB/IBM/复杂边界。论文的 Kelvin/covector 解释可指导离散设计，但每个离散 contract 都需要独立的守恒、收敛和并行一致性测试。

## 9. 复现命令（不覆盖源码）

在仓库外建立构建目录可避免污染源码：

```bash
SRC=/home/wangzw/agent-workspace/Projects/HPC_matters/paper_codes/CovectorFluids_code
BUILD=/tmp/CovectorFluids-build
cmake -S "$SRC" -B "$BUILD"
cmake --build "$BUILD" -j"$(nproc)"
```

二维/三维入口分别为：

```bash
cd "$BUILD"
./Covector2D <scheme:0-7> <setup:0-5>
./Covector3D <scheme:0-7> <experiment:0-6>
```

三维运行前需确认 `nvcc`、OpenVDB、TBB、Boost 和 NVIDIA 驱动；若只验证二维，也必须先处理顶层 CUDA 强依赖。默认实验可能耗时很长，建议先用 `timeout` 做启动/首帧 smoke test，并将完整运行单独记录。

本次 Codex 会话探测结果（2026-08-24）：宿主机实际安装了 `/usr/local/cuda-12.6/bin/nvcc`，但 Codex 会话的 `PATH` 没有包含该目录；显式设置 `CUDA_HOME=/usr/local/cuda-12.6` 和 `PATH=$CUDA_HOME/bin:$PATH` 后，CMake 已成功识别 CUDA 12.6。随后配置在自定义 `FindTBB.cmake` 阶段失败，提示缺少 `TBB_LIBRARIES`/`TBB_INCLUDE_DIRS`；当前会话还未发现可用的 OpenVDB/TBB 系统安装，且 `nvidia-smi` 命令不可用。因此此前“宿主机没有 CUDA”的说法不准确，准确结论是“当时 Codex 会话未把 CUDA 暴露给 PATH，且依赖库仍未定位”。日志分别位于 `/tmp/CovectorFluids-build-configure.log` 和 `/tmp/CovectorFluids-build-cuda-configure.log`。用户在自己的完整开发环境中应显式设置 CUDA 路径并提供 TBB/OpenVDB/CUDA 驱动，再重跑上述命令。
