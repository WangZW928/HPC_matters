# Nsight Systems Intro

这个项目计划说明：

- Nsight Systems 是看什么的
- 为什么它适合看 stream、memcpy、kernel 的时间线
- 怎样用它验证 copy/compute overlap 是否真的发生

计划内容：

- 选一个已有 stream 项目作为观测对象
- 记录默认执行与多 stream 执行时间线
- 对照时间线解释哪些部分发生了重叠

目标：

把“我猜发生了并发”升级成“我从时间线看到它确实发生了并发”。
