# register_Occupancy：寄存器与 Occupancy 实验

本项目用于学习：**寄存器压力如何限制并发（occupancy），并影响性能**。

核心观点：

- `registers per thread` 越高，单个 SM 可同时驻留的线程/warp 往往越少
- 并发下降后，隐藏延迟能力变弱，kernel 可能变慢
- 所以“做了更多计算”不一定更快，资源瓶颈可能先出现

## 1. 实验设计

两个 kernel 版本：

- 版本 A（低寄存器）：

```cpp
float x = a[i];
```

- 版本 B（高寄存器）：

```cpp
float tmp[HIGH_REG_TMP_SIZE];
#pragma unroll
for (int k = 0; k < HIGH_REG_TMP_SIZE; k++) tmp[k] = x;
```

其中版本 B 通过局部数组 + 展开循环显著提升寄存器压力。数组长度用宏 `HIGH_REG_TMP_SIZE` 统一控制。

## 2. 你将观察什么

输出中重点关注：

- `regs_per_thread`（每线程寄存器数）
- `theoretical_occupancy`（理论 occupancy）
- `avg_ms` / `std_ms`（kernel 平均时间与波动）

## 3. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── reg_occ_bench.cu
├── scripts/
│   └── plot_results.py
├── results/
├── requirements.txt
└── README.md
```

## 4. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/reg_occ_bench
```

自定义参数：

```bash
./build/reg_occ_bench <output_csv> <repeats> <warmup> <iters>
# 示例
./build/reg_occ_bench results/reg_occ_benchmark.csv 30 8 256
```

## 5. Python 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/reg_occ_benchmark.csv --outdir results
```

会生成：

- `results/registers_per_thread.png`
- `results/occupancy_compare.png`
- `results/runtime_compare.png`
- `results/summary_compare.csv`

## 6. 如何思考“为什么多做计算反而变慢”

可以从这条链路分析：

1. 局部变量增多 -> `regs_per_thread` 上升
2. 每个 SM 可驻留 warp 数下降 -> occupancy 下降
3. 可用于隐藏访存/流水线延迟的 warp 变少
4. 尽管算术操作变多，但总吞吐不一定提升，甚至下降

注意：如果 kernel 完全 compute-bound 且之前 occupancy 远超“够用阈值”，增加寄存器不一定显著变慢。最终要用实测数据判断。

## 7. 可扩展练习

- 修改 `tmp[64]` 为 `tmp[16]`, `tmp[32]`, `tmp[96]`，画出寄存器-occupancy-时间曲线
- 加入 `-Xptxas -v`（或查看 Nsight Compute）对比编译器寄存器分配
- 将 block size 改成 128/256/512，观察 occupancy 与性能是否一致变化


## 8. 批量 Sweep（不同寄存器压力）

项目提供脚本：`scripts/sweep_registers.sh`，用于自动完成：

- 以不同 `HIGH_REG_TMP_SIZE` 重新编译
- 运行 benchmark 并写每轮日志
- 汇总到总 CSV
- 自动调用 Python 画趋势图

### 8.1 一条命令运行

```bash
cd register_Occupancy
bash scripts/sweep_registers.sh
```

### 8.2 可选参数（环境变量）

```bash
TMP_SIZES="8 16 24 32 48 64 80 96 128" REPEATS=20 WARMUP=5 ITERS=256 bash scripts/sweep_registers.sh
```

### 8.3 结果输出

- 汇总 CSV：`results/reg_occ_sweep.csv`
- 每轮日志：`results/logs/build_run_tmp_<size>.log`
- 趋势图：
  - `results/sweep_occupancy_vs_regs.png`
  - `results/sweep_runtime_vs_regs.png`
  - `results/sweep_throughput_vs_regs.png`
- 汇总表：`results/sweep_summary.csv`

说明：趋势图默认使用 `high_reg` 行，横轴是 `regs_per_thread`，更直接反映“寄存器压力 -> occupancy/性能”的关系。
