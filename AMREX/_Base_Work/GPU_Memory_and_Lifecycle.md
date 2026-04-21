# GPU 内存与生命周期

这份文档聚焦移植里最容易晚期暴雷的一层：内存模型、生命周期与异步语义。

## 1. 这层在 AMReX 里负责什么

- 分配策略：device / managed / pinned / host。
- 统一分配器：allocator 抽象和容器接入。
- 生命周期延长：异步 launch 场景下对象有效期管理。
- 异步数据路径：`AsyncArray`、临时 buffer、copy/sync。
- Arena 体系：上层对象最终依赖的内存后端。

## 2. 关键文件地图

### 2.1 核心内存抽象

- [`AMReX_GpuMemory.H`](../AMReX_GpuMemory.H)
- [`AMReX_GpuAllocators.H`](../AMReX_GpuAllocators.H)
- [`AMReX_Arena.H`](../AMReX_Arena.H)
- [`AMReX_Arena.cpp`](../AMReX_Arena.cpp)

### 2.2 生命周期与异步对象

- [`AMReX_GpuElixir.H`](../AMReX_GpuElixir.H)
- [`AMReX_GpuElixir.cpp`](../AMReX_GpuElixir.cpp)
- [`AMReX_GpuAsyncArray.H`](../AMReX_GpuAsyncArray.H)
- [`AMReX_GpuAsyncArray.cpp`](../AMReX_GpuAsyncArray.cpp)
- [`AMReX_GpuBuffer.H`](../AMReX_GpuBuffer.H)

### 2.3 运行时配套

- [`AMReX_GpuDevice.H`](../AMReX_GpuDevice.H)
- [`AMReX_GpuDevice.cpp`](../AMReX_GpuDevice.cpp)
- [`AMReX_GpuControl.H`](../AMReX_GpuControl.H)

## 3. 移植时必须尽早确认的语义

1. 是否支持异步分配/释放与异步 copy。
2. 是否有 pinned host memory 对应机制。
3. managed/unified memory 是否存在等价语义。
4. stream/queue 语义是否能保证对象生命周期安全。
5. Arena 层是否可复用，还是需要替换 allocator backend。

## 4. Bring-up 推荐顺序

1. 先通 `Arena + device alloc/free`。
2. 再通 H2D/D2H copy（含 async + sync）。
3. 再通 `AsyncArray` 最小用例。
4. 再验证 `Elixir` 生命周期保护是否生效。
5. 最后再扩 managed/pinned 与性能优化。

## 5. 最小验证用例（建议）

- 用例 A: 分配 device buffer -> kernel 写入 -> 拷回 host 验证。
- 用例 B: async copy + stream sync，验证顺序语义。
- 用例 C: 临时对象在异步 kernel 期间不提前释放（Elixir/等价机制）。
- 用例 D: Arena 分配路径覆盖 `Array4`/`Fab` 典型大小。

## 6. 常见移植坑

- 释放发生在异步任务完成前，导致随机错。
- stream 语义与 AMReX 默认 stream 假设不一致。
- allocator 对齐策略不匹配，导致 vectorized kernel 或 atomic 异常。
- managed memory 语义缺失但上层误以为可用。

## 7. 交付导向检查表

- [ ] device alloc/free 基本可用
- [ ] H2D / D2H copy（sync + async）可用
- [ ] `AsyncArray` 可用且生命周期正确
- [ ] `Elixir` 或等价机制可保护异步对象
- [ ] Arena 路径可支撑 `Array4` / `FabArray` 常见场景

