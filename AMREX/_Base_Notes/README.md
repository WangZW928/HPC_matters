# Base Notes Index

这个目录用来记录 `Src/Base` 下面一些核心基础模块的阅读笔记。

## 推荐阅读顺序

1. [`ArrayND.md`](./ArrayND.md)
2. [`Array4.md`](./Array4.md)
3. [`AMREX_GPU_HOST_DEVICE.md`](./AMREX_GPU_HOST_DEVICE.md)
4. [`ParallelFor_box_lambda.md`](./ParallelFor_box_lambda.md)
5. [`ParallelFor_to_launch_global.md`](./ParallelFor_to_launch_global.md)
6. [`Reduction_Notes.md`](./Reduction_Notes.md)

## 现有笔记

- [`ArrayND.md`](./ArrayND.md)
  `ArrayND` 的模板结构、数据成员、stride、索引方式。
- [`Array4.md`](./Array4.md)
  `Array4` 的定义、使用方式、内存布局、和真实内存的映射关系。
- [`AMREX_GPU_HOST_DEVICE.md`](./AMREX_GPU_HOST_DEVICE.md)
  `AMREX_GPU_HOST_DEVICE` 宏的定义、展开方式和简单例子。
- [`ParallelFor_box_lambda.md`](./ParallelFor_box_lambda.md)
  `ParallelFor(box, lambda)` 的重载链和 GPU 线性化入口。
- [`ParallelFor_to_launch_global.md`](./ParallelFor_to_launch_global.md)
  从 `ParallelFor(box, lambda)` 到 `launch_global` 的 GPU 调用链。
- [`Reduction_Notes.md`](./Reduction_Notes.md)
  AMReX 规约接口的基本调用逻辑：`Reduce/Reducer`、`ParReduce`、`ParallelAllReduce/ParallelReduce` 的关系和源码位置。

## 后续可继续补的主题

- `BaseFab / FArrayBox / MultiFab` 关系
- `MFIter` 的驱动逻辑
- `Box / BoxArray / DistributionMapping` 入门
- `Arena / The_Arena / The_Pinned_Arena`
- `Gpu::stream` 和 `AsyncArray`
