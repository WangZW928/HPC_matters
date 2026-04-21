# Minimal Reduction Project

这个小项目和 `ParallelFor` 保持相近量级，用来突出 AMReX 本地规约的核心学习点。

## 学习目标

这个例子只做一件事：

- 对一组二维向量 `v = (x, y)`
- 一次遍历同时规约出：
  - `sum(|v|)`
  - `max(|v|)`

这里的重点不是数学本身，而是规约调用形状：

```cpp
Reducer reducer;
auto result = reducer.eval(n, [=] BASE_GPU_DEVICE (int i) noexcept -> ReduceTuple {
    Vec2 v = data[i];
    double mag = sqrt(v.x * v.x + v.y * v.y);
    return {mag, mag};
});
```

这对应 AMReX 里“每个点返回一个 tuple，然后内部做规约合并”的设计哲学。

## `sum(|v|)` 和 `max(|v|)` 到底是什么意思

这里不是直接对向量分量 `x` 和 `y` 做规约，而是先对每个向量计算模长：

```text
|v| = sqrt(x^2 + y^2)
```

然后对这些模长做两种不同的汇总：

- `sum(|v|)`
  把所有向量模长加起来
- `max(|v|)`
  取所有向量模长中的最大值

也就是说，程序实际做的是：

```text
先算 |v0|, |v1|, |v2|, ...
然后得到
sum = |v0| + |v1| + |v2| + ...
max = max(|v0|, |v1|, |v2|, ...)
```

## 为什么 `max` 对应另一种规约操作

因为“规约操作”的本质不是“返回什么值”，而是“怎么把很多局部值合并成一个最终值”。

在这个例子里，每个点返回的局部值其实都是同一个东西：

```cpp
mag
```

但是：

- 如果合并规则是 `+`，那最终得到的就是 `sum`
- 如果合并规则是 `取最大值`，那最终得到的就是 `max`

所以：

- `sum` 对应一种 reduce operation
- `max` 对应另一种 reduce operation

它们的区别不在于局部值不同，而在于**全局汇总规则不同**。

从 AMReX 的角度看，这正对应：

- `ReduceOpSum`
- `ReduceOpMax`

所以这个例子里虽然写的是：

```cpp
return {mag, mag};
```

但这两个 `mag` 会分别走两套不同的合并逻辑：

- tuple 第一个分量按 `sum` 合并
- tuple 第二个分量按 `max` 合并

这就是“一次遍历做多个规约”的核心思想。

## 文件说明

- `Backend.H`
  CPU/GPU 后端宏和同步接口。
- `Buffer.H`
  最小内存容器，负责 host/device 数据。
- `Reduce.H`
  这个项目最核心的文件。里面放了：
  - `Vec2`
  - `ReduceTuple`
  - `ReduceData`
  - `Reducer`
- `main.cpp`
  最小 demo。
- `CMakeLists.txt`
  构建脚本。

## 设计重点

这个例子故意保持轻量，但尽量贴近 AMReX 思想：

- 外层先有数据容器 `Buffer<Vec2>`
- 用户传入一个返回 tuple 的 lambda
- `Reducer` 负责执行本地规约
- `ReduceData` 负责承载 partial result 和最终结果

所以它不是在模仿 `ParallelFor`，而是在模仿：

- `ReduceData`
- `Reducer`
- “一次遍历做多个规约”的思想

## 结果含义

最终程序会输出：

- `sum(|v|)`
- `max(|v|)`

这样你可以同时看到：

- 一个 `sum` 规约
- 一个 `max` 规约
- 它们如何在一次 traversal 里一起完成

## 构建

### CPU

```bash
cmake -S . -B build
cmake --build build -j
./build/reduction_cpu
```

### GPU

如果系统里有 CUDA 编译器，CMake 会自动生成 `reduction_gpu`：

```bash
cmake -S . -B build
cmake --build build -j
./build/reduction_gpu
```
