# Earth-Moon 3D SPH CUDA Simulation

这是一个面向 RTX 4060 的三维弱可压缩 SPH 示例。后端使用 CUDA C++、SoA
粒子数组、uniform grid 空间哈希和 Thrust 排序；前端使用 Matplotlib 显示二进制帧。

## 数值模型

- 地球半径、质量和引力常数被无量纲化为 `R_e=1`、`GM_e=1`
- 保留真实地月质量比 `0.0123` 和距离比 `60.3`
- Cubic Spline 核，支撑半径 `h=0.055`
- Tait EOS，`gamma=7`
- 压力项使用守恒对称形式
- 地球硬壳使用带阻尼的平滑支撑力，并用投影反弹兜底
- Symplectic Euler 时间积分
- 默认将差分潮汐力放大 1,000,000 倍，使其达到地表重力约 10%，突出近月侧和背月侧隆起
- 使用速度和加速度安全上限，阻止局部密度尖峰造成非物理粒子喷射

> 这是研究与教学代码，不是高保真海洋模型。模型忽略地球自转、陆地、海床形状、
> 粘性和表面张力。代码已经减去地球质心处的月球共同加速度；将
> `TIDAL_FORCE_SCALE` 改为 `1.0f` 可恢复真实潮汐强度。默认倍率仅用于清晰展示潮汐方向，
> 不能用于定量预测真实潮位。

## 构建

需要 CUDA Toolkit、支持 CUDA 的 C++ 编译器和 CMake 3.24+。RTX 4060 使用
Ada 架构，因此 CMake 默认设置 `CUDA_ARCHITECTURES=89`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 运行

参数依次是粒子数、步数、导出间隔、输出目录：

```bash
./build/earth_moon_sph 300000 5000 50 output
python3 -m pip install -r requirements.txt
python3 visualize.py "output/frame_*.bin" --stride 8 --color speed
```

首次测试建议先用 `20000 200 20`。默认使用 300,000 粒子，可视化每隔 8 个粒子绘制一个；
需要更多细节时可使用 `--stride 2`，需要更快时可增加到 `--stride 12`。可视化右侧使用真实轨道半径，
但默认将月球显示相位加速 500 倍，以便在短仿真中看清运动；该加速不影响 CUDA 物理计算。

直接生成无背景网格的 GIF：

```bash
python3 render_gif.py "output/frame_*.bin" --output earth_moon.gif --fps 15
```

`render_gif.py` 只依赖 Python 标准库和系统 `ffmpeg`，比 Matplotlib 3D 动画快很多。
GIF 默认最多均匀选取 120 帧、每隔 8 个粒子绘制一个，并将水层径向形变放大 2 倍，
使薄水层流动更容易观察。这些操作只影响显示。可以使用 `--max-frames 0` 导出全部帧，
或用 `--radial-scale 1` 恢复真实径向比例。

GIF 默认按径向位移着色：蓝色表示向地球内部移动，白色接近水层中面，红色表示向外隆起。
这样比速度色标更适合观察潮汐形变。若需要显示速度大小，可使用：

```bash
python3 render_gif.py "output/frame_*.bin" --output speed.gif --color speed
```

每帧包含位置、速度、密度和压力；100,000 粒子时约为 3.2 MB。提高到 500,000 粒子前，
应观察显存占用、邻居数量与时间步稳定性。

## 强潮汐演示

修改潮汐倍率后必须重新编译，并输出到新目录，避免与旧帧混合：

```bash
cmake --build build -j
./build/earth_moon_sph 300000 20000 100 output_strong_tide
python3 render_gif.py "output_strong_tide/frame_*.bin" \
  --output strong_tide.gif --radial-scale 3
```

运行时观察诊断输出。正常情况下 `max_radius` 不应超过安全半径 `1.16`；如果长期贴住
`1.16`，说明演示倍率过强，应减小 `TIDAL_FORCE_SCALE`。
