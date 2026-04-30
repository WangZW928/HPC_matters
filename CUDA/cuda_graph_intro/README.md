# CUDA Graph 概念入门项目

这个项目用一个最小实验解释 CUDA Graph 是什么。

## 1. CUDA Graph 是什么

CUDA Graph 可以把“一串固定的 GPU 工作流（kernel/memcpy 等）”先录制成图（graph），
再反复执行这个图（graph replay）。

直觉上它解决的是：

- 每次都从 CPU 发多次 launch 的开销
- 小 kernel 很多时，launch overhead 占比偏大

所以常见收益场景是：

- 计算步骤固定
- 重复执行很多轮
- 单步 kernel 比较短小

## 2. 本项目做了什么

每一轮都执行 3 个 kernel：

1. `add_bias`
2. `scale`
3. `relu`

然后比较两种模式：

- `normal`：每轮都逐次 launch 这 3 个 kernel
- `graph`：先 capture 一次，再每轮 `cudaGraphLaunch` replay

最终输出每轮平均耗时（ms）并比较 speedup。

## 3. 项目结构

```text
.
├── CMakeLists.txt
├── src/
│   └── graph_bench.cu
├── scripts/
│   └── plot_results.py
├── requirements.txt
├── results/
└── README.md
```

## 4. 构建与运行

```bash
cmake -S . -B build
cmake --build build -j
./build/graph_bench
```

可选参数：

```bash
./build/graph_bench <output_csv> <repeats> <warmup> <n>
# 示例
./build/graph_bench results/graph_benchmark.csv 3000 300 1048576
```

## 5. 可视化

```bash
python -m pip install -r requirements.txt
python scripts/plot_results.py --input results/graph_benchmark.csv --outdir results
```

输出：

- `results/graph_vs_normal.png`
- `results/summary.txt`

## 6. 你该怎么理解结果

如果 `graph` 更快，通常意味着你当前 workload 有明显 launch 开销可优化。

如果差异不大，常见原因是：

- kernel 本身计算很重，launch 开销占比小
- 重复次数不够高
- 图里节点太少或工作流不够固定

## 7. 一句话记忆

CUDA Graph 不是让单个 kernel 变“算得更快”，而是让一串固定 GPU 工作的“调度开销更低”。
