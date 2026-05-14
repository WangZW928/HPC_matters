# CUDA 学习总览

这个目录记录了一组面向 CUDA 入门与性能理解的实验项目。整体来看，这些项目的方向是很好的，因为它们不只是在学语法，而是在建立：

`代码写法 -> 硬件行为 -> 性能结果`

之间的联系。

当前已有项目：

- [warp_schedule/README.md](/home/wangz/MyProject/HPC_matters/CUDA/warp_schedule/README.md)
- [register_Occupancy/README.md](/home/wangz/MyProject/HPC_matters/CUDA/register_Occupancy/README.md)
- [cuda_graph_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/cuda_graph_intro/README.md)
- [cuda_stream_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/cuda_stream_intro/README.md)

规划中的下一阶段项目：

- [memory_coalescing_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/memory_coalescing_intro/README.md)
- [shared_memory_bank_conflict/README.md](/home/wangz/MyProject/HPC_matters/CUDA/shared_memory_bank_conflict/README.md)
- [nsight_systems_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/nsight_systems_intro/README.md)
- [nsight_compute_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/nsight_compute_intro/README.md)
- [kernel_type_playground/README.md](/home/wangz/MyProject/HPC_matters/CUDA/kernel_type_playground/README.md)
- [reduction_scan_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/reduction_scan_intro/README.md)

## 1. 目前这 4 个部分在学什么

### 1.1 `warp_schedule`

这个项目帮助你理解：

- block / warp / SM 之间的映射关系
- warp 数量变化为什么会影响 latency hiding
- 为什么 occupancy 不是越高越好，而是“够用”更重要

### 1.2 `register_Occupancy`

这个项目帮助你理解：

- `registers per thread` 会限制同时驻留的 warp / block 数
- 寄存器压力上升为什么会压低 occupancy
- 为什么“做了更多计算”不一定更快

### 1.3 `cuda_graph_intro`

这个项目帮助你理解：

- CUDA Graph 是什么
- 为什么 Graph 适合固定、重复的小工作流
- Graph 优化的是 launch / 调度开销，而不是单个 kernel 的算法效率

### 1.4 `cuda_stream_intro`

这个项目帮助你理解：

- stream 是 GPU 工作队列
- 什么是异步拷贝
- 什么是锁页内存
- copy / compute overlap 是怎么来的

## 2. 我对这套学习路线的评价

这套路线是很不错的，优点主要有三点：

- 你没有只学 API，而是在用实验观察性能现象
- 你已经覆盖了“kernel 内部资源”和“运行时调度机制”两大方向
- 每个项目都能产出 CSV / 图，这很像真实性能分析工作的方式

如果放在 CUDA 学习路径里，我会认为你现在已经不只是“会写 CUDA”，而是在进入“会分析 CUDA 程序为什么快、为什么慢”的阶段。

## 3. 目前还缺什么

如果想继续往前走，最值得补的不是更多 API，而是下面这几类能力：

### 3.1 Memory Hierarchy

下一阶段很值得重点做：

- global memory coalescing
- shared memory 的作用与 bank conflict
- L2 / cache 对访问模式的影响
- memory-bound kernel 为什么常常比 compute-bound 更难优化

### 3.2 Nsight 工具链

建议逐步加入：

- Nsight Systems：看 stream、memcpy、kernel 的时间线
- Nsight Compute：看 occupancy、warp stall、memory throughput、register usage

### 3.3 Kernel 类型意识

后面建议把 kernel 分成几类分别学：

- compute-bound
- memory-bound
- latency-bound
- launch-overhead-bound

### 3.4 更真实的并行模式

再往前一步，建议开始接触：

- reduction
- prefix sum / scan
- warp shuffle
- cooperative groups
- 原子操作与 contention

## 4. 按顺序开展的计划

我建议你按下面这个顺序推进，这样前一个项目建立的直觉，会自然成为后一个项目的前置知识。

1. [memory_coalescing_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/memory_coalescing_intro/README.md)
   先补 global memory 访问模式，因为这是 CUDA 性能分析里最核心的一层。
2. [shared_memory_bank_conflict/README.md](/home/wangz/MyProject/HPC_matters/CUDA/shared_memory_bank_conflict/README.md)
   在理解 global memory 之后，再看 shared memory 为什么快、为什么也会因为访问模式不对而变慢。
3. [nsight_systems_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/nsight_systems_intro/README.md)
   这一步把你之前对 stream 和 overlap 的判断，从“现象观察”推进到“时间线验证”。
4. [nsight_compute_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/nsight_compute_intro/README.md)
   这一步把你对 occupancy、寄存器、stall 的理解和 profiler 指标真正对上号。
5. [kernel_type_playground/README.md](/home/wangz/MyProject/HPC_matters/CUDA/kernel_type_playground/README.md)
   当你已经有 memory / profiler 基础后，再系统地区分不同瓶颈类型，会更稳。
6. [reduction_scan_intro/README.md](/home/wangz/MyProject/HPC_matters/CUDA/reduction_scan_intro/README.md)
   最后进入更接近真实 HPC / DL 基元的并行模式，这是很自然的进阶。

一句话总结这个顺序：

先学内存访问，再学 profiling，然后学分类判断，最后进入更真实的并行算法。

## 5. 我建议你接下来最专注的方向

如果只选一个重点，我建议下一阶段优先专注：

`memory hierarchy + profiling`

原因很简单：

- warp / occupancy / stream / graph 这些概念你已经有了很好的起步
- 但真正决定大多数 CUDA 程序上限的，往往是内存访问模式
- 而真正帮助你验证结论的，是 profiler，不是猜

## 6. 一句话评价

你现在这几个项目选得很对，已经覆盖了 CUDA 学习里最容易“只懂概念、不懂性能”的部分。下一步最该补的是内存层次和 profiling，这会把你的实验型理解推进到真正的性能工程视角。
