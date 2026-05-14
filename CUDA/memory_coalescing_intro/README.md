# Memory Coalescing Intro

这个项目计划用最小实验说明：

- 什么是 global memory coalescing
- 为什么连续访问通常比跨步访问更快
- warp 内访问模式为什么会直接影响带宽利用率

计划内容：

- 写一个连续访问版本 kernel
- 写一个 stride 访问版本 kernel
- 比较两者时间和吞吐量
- 用图展示 stride 增大后性能如何变化

目标：

建立“访问模式 -> 带宽效率 -> kernel 性能”的第一层直觉。
