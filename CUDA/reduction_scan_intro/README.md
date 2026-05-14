# Reduction And Scan Intro

这个项目计划说明：

- 什么是 reduction
- 什么是 prefix sum / scan
- 为什么这些操作是 CUDA 学习里的经典主题

计划内容：

- 先实现一个基础 reduction
- 再逐步引入 shared memory、warp shuffle、同步优化
- 如果进展顺利，再补一个简单 scan 示例

目标：

从“实验型 kernel”走向更接近真实 HPC / DL 基元的并行模式。
