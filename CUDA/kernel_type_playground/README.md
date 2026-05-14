# Kernel Type Playground

这个项目计划用一组小 kernel 区分不同性能瓶颈类型：

- compute-bound
- memory-bound
- latency-bound
- launch-overhead-bound

计划内容：

- 分别写几个代表性 kernel
- 比较它们对 block size、occupancy、stream、graph 的敏感度
- 总结每类 kernel 的典型优化思路

目标：

建立“先判断瓶颈类型，再决定优化方向”的习惯。
