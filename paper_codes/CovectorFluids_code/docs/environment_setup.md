# CovectorFluids 环境部署记录

> 部署日期：2026-08-24  
> 目标平台：Ubuntu 24.04、x86_64

## 当前环境

| 组件 | 当前版本/状态 |
|---|---|
| CMake | 3.28.3 |
| C++ 编译器 | GCC 13.3.0 |
| CUDA Toolkit / NVCC | 12.6 / 12.6.85 |
| TBB | 2021.11.0，本地隔离部署 |
| Boost | 1.83.0，本地隔离部署 |
| OpenVDB | 10.0.1，本地隔离部署 |
| Imath | 3.1.9，本地隔离部署 |
| Python / Pillow | Python 3 / Pillow 12.3.0，用于 PNG 和动画 WebP 编码 |
| NVIDIA GPU | NVIDIA GeForce RTX 4060 Laptop，Compute Capability 8.9，8 GiB |

系统账户没有免密 `sudo` 权限，因此依赖没有安装到 `/usr`。Ubuntu `.deb` 被下载到临时目录并解压到：

```text
.deps/ubuntu24.04-x86_64/
```

该目录约 241 MB，已加入 `.gitignore`。此方案不会修改系统软件包数据库，删除 `.deps/` 即可完整卸载项目级依赖。

## 可复现部署

在仓库根目录执行：

```bash
./scripts/bootstrap_local_deps.sh
./scripts/build_local.sh
```

第一个脚本使用 Ubuntu 24.04 软件源下载并解压 22 个包，主要包括 TBB、Boost filesystem/system、OpenVDB、Imath、Blosc、Snappy 和 Log4cplus。下载缓存使用 `/tmp`，成功或失败退出时都会清理。

完整包清单如下：

| 软件包 | 版本 |
|---|---|
| `libtbb-dev`, `libtbb12`, `libtbbbind-2-5`, `libtbbmalloc2` | 2021.11.0-2ubuntu2 |
| `libboost1.83-dev` | 1.83.0-2.1ubuntu3.2 |
| `libboost-atomic1.83-dev`, `libboost-atomic1.83.0` | 1.83.0-2.1ubuntu3.2 |
| `libboost-filesystem1.83-dev`, `libboost-filesystem1.83.0` | 1.83.0-2.1ubuntu3.2 |
| `libboost-system1.83-dev`, `libboost-system1.83.0` | 1.83.0-2.1ubuntu3.2 |
| `libboost-iostreams1.83.0` | 1.83.0-2.1ubuntu3.2 |
| `libboost-filesystem-dev`, `libboost-system-dev` | 1.83.0.1ubuntu2 |
| `libopenvdb-dev`, `libopenvdb10.0t64` | 10.0.1-2.1build5 |
| `libimath-dev`, `libimath-3-1-29t64` | 3.1.9-3.1ubuntu2 |
| `libblosc-dev`, `libblosc1` | 1.21.5+ds-1build1 |
| `libsnappy1v5` | 1.1.10-1build1 |
| `liblog4cplus-2.0.5t64` | 2.0.8-1.1ubuntu3 |

若机器允许系统级安装，也可以使用：

```bash
sudo apt-get update
sudo apt-get install libtbb-dev libopenvdb-dev libboost-filesystem-dev libboost-system-dev
```

系统安装后仍可使用原始 CMake 流程；本项目当前验证的是隔离部署脚本。

## 构建产物

```text
build/Covector2D
build/Covector3D
```

完成编译与验证后，`build/` 中的 CMake 缓存、对象文件、CUDA 中间链接文件和临时渲染工具已清理，只保留上述两个可执行程序。`/tmp/covector-debs` 安装包缓存也已删除。需要重新编译时运行 `./scripts/build_local.sh`，CMake 会自动重建所需目录和中间文件。

本次为兼容现代环境做了以下最小代码调整：

- 3D 使用 C++14，以满足 OpenVDB 10 的头文件要求。
- 使用 `cudaGetDeviceCount`/`cudaSetDevice` 替代已不再随 CUDA Toolkit 分发的 `helper_cuda.h`。
- 移除重复编译 `GPU_kernel.cu` 且未被链接的 `cuda_lib` 目标。
- 三个几何辅助函数使用 `const Vec3f&` 接受不修改的临时中心坐标。
- 修复 3D 参数解析前读取未初始化变量的问题，并为 2D/3D 增加参数范围检查。

构建仍会产生来自研究代码和旧式 `FindCUDA` 的警告，但两个可执行目标均已成功链接。

## 运行

使用包装脚本可以自动设置本地动态库搜索路径，并确保从 `build/` 启动以满足模型数据的相对路径约定：

```bash
./scripts/run_local.sh 2d 0 4
./scripts/run_local.sh 3d 0 0
```

参数含义：

- 第一个参数：`2d` 或 `3d`。
- 第二个参数：算法编号 `0-7`。
- 第三个参数：2D 实验编号 `0-5`，或 3D 实验编号 `0-6`。

## 本次验证结果

- `Covector2D`：完整运行 `method=0, experiment=4` 成功，退出码为 0。
- 2D 输出：`Out_2D/2D_InvertedZalesak/SF/` 下生成 315 个 BMP 文件，约 37 MB。
- `Covector3D`：完整运行 `method=0, experiment=0`（Stable Fluids + Trefoil Knot）成功，退出码为 0。
- 3D 进度：初始帧 0 与推进帧 1–269 均完成；末帧 CFL 为 0.396103，压力残差约 `1.39e-8`。
- 3D 原始输出：曾在 `Out_3D/TrefoilKnot/SF/` 下生成 810 个 VDB 文件，即每帧 `vel_x/y/z` 各一个，总大小约 11 GB。
- 后处理：从三分量速度计算涡量强度，保留 270 张俯视/侧视投影 PNG、动画 WebP、联系表和逐帧最大值 CSV，共约 5.8 MB。
- 清理：确认可视化有效后，810 个仿真输出 VDB 已按用户要求删除；`modelData/` 中的原始输入 VDB 未删除。
- 可视化位置：`visualizations/TrefoilKnot/SF_vorticity/`。

3D 真实运行需要宿主机 NVIDIA 驱动与 CUDA 12.6 兼容，并向执行环境映射 GPU 设备。当前 GPU 在提升权限的执行环境中可见。

## 3D 可视化后处理

可视化实现由以下两个文件组成：

- `tools/render_vdb_velocity.cpp`：读取三分量速度 VDB，计算涡量强度，生成俯视和侧视最大投影。
- `scripts/render_3d_velocity.sh`：编译渲染工具，将临时 PPM 编码为 PNG 序列、动画 WebP、联系表，并保存逐帧最大涡量 CSV。

使用默认 Trefoil Knot 路径和帧范围：

```bash
./scripts/render_3d_velocity.sh
```

显式指定输入目录、输出目录和帧范围：

```bash
./scripts/render_3d_velocity.sh \
  Out_3D/TrefoilKnot/SF \
  visualizations/TrefoilKnot/SF_vorticity \
  0 269
```

执行可视化前必须保留对应的 `vel_x_render_NNNN.vdb`、`vel_y_render_NNNN.vdb` 和 `vel_z_render_NNNN.vdb`。当前这批 VDB 已在可视化验证后删除；若需重新生成不同视图，必须先重新运行 3D 仿真。
