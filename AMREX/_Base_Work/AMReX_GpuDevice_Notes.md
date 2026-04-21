# AMReX_GpuDevice.H + AMReX_GpuDevice.cpp 移植笔记

对应源码：

- [`AMReX_GpuDevice.H`](../AMReX_GpuDevice.H)
- [`AMReX_GpuDevice.cpp`](../AMReX_GpuDevice.cpp)

## 1. 文件职责

该组文件是 GPU runtime 控制面（control plane）核心，主要负责：

- 设备发现与选择。
- stream/queue 管理。
- launch 参数辅助。
- 同步与错误检查协作。

## 2. 必查项

- 设备枚举模型能否映射 AMReX 预期接口。
- 默认 stream 与自定义 stream 语义是否一致。
- stream 生命周期与线程模型是否稳定。
- 与 `GpuControl` / `GpuLaunch` 的接口契合是否完整。

## 3. 高风险问题

- stream 语义不一致导致时序错误（定位成本高）。
- 多设备/多流场景下状态污染。
- 错误码回传不完整导致静默失败。

## 4. 最小验证建议

1. 单流串行 launch + sync 正确性。
2. 多流并发 launch + 显式同步一致性。
3. 设备切换场景（若支持多设备）状态隔离性。
