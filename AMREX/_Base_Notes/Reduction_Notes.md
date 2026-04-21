# AMReX Reduction Notes

这份笔记重点梳理 **AMReX 的单卡 / 单进程本地规约**。这是初步学习规约时最应该先抓住的主线。

先给一个最重要的结论：

**当前阶段先把 AMReX 规约理解成“返回值版的 ParallelFor”最合适。**

也就是说：

- `ParallelFor` 是每个点执行一次，然后把结果写到外部数组
- `Reduce` 是每个点执行一次，但不是写数组，而是返回一个局部值
- AMReX 在内部把这些局部值按 `sum/min/max/...` 规则合并成最终结果

所以最先要理解的是：

```text
每个点返回什么
-> 这些返回值怎么在本地被合并
-> 最后怎么拿到本地规约结果
```

而不是先去想 MPI。

## 1. 先只看“本地规约”这条主线

从学习角度，AMReX 的规约可以分成三层：

1. 本地规约，不做 MPI 通信
   - `amrex::Reduce::Sum/Min/Max/MinMax/...`
   - `amrex::Reducer<...>`
2. 针对 `FabArray/MultiFab` 的本地并行规约
   - `amrex::ParReduce(...)`
3. MPI 级别的跨 rank 规约
   - `amrex::ParallelAllReduce::Sum/Min/Max`
   - `amrex::ParallelReduce::Sum/Min/Max`

但对于初步学习，**最重要的是前两层**。

也就是：

- 一个点上怎么产生局部值
- AMReX 怎么在单卡 / 单进程内部把这些值合并

MPI 那层先放到最后看就够了。

## 2. 本地规约主入口在哪里

最核心的源码文件是：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H)

这里面最值得先看的对象是：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`ReduceOpSum / ReduceOpMin / ReduceOpMax / ReduceOpLogicalAnd / ReduceOpLogicalOr`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`ReduceOps<...>`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`ReduceData<...>`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reducer<Ops, Ts>`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reduce::Sum / Min / Max / MinMax / AnyOf`

建议你把它们先理解成：

- `ReduceOpXxx`
  定义“如何合并”
- `ReduceOps<...>`
  定义“一组规约操作怎么一起跑”
- `ReduceData<...>`
  定义“规约结果存在哪”
- `Reducer<...>`
  定义“用户怎么调”

## 3. 最简单的本地规约：`Reduce::Sum / Min / Max`

如果规约对象只是一个一维范围 `0..n-1`，最简单的接口就是：

```cpp
auto s = amrex::Reduce::Sum<Real>(n,
    [=] AMREX_GPU_DEVICE (Long i) -> Real {
        return f(i);
    });
```

这类接口在：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reduce::Sum`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reduce::Min`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reduce::Max`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`Reduce::MinMax`

它们的特点是：

- 只做本地规约
- 不涉及 `MultiFab`
- 不涉及 MPI
- 很适合先理解“规约的本质”

你可以把它想成：

```text
for i in [0,n):
    value_i = f(i)
final = reduce(value_0, value_1, ..., value_{n-1})
```

只是这个 `reduce(...)` 会在 CPU / GPU 后端下由 AMReX 内部完成。

## 4. 规约和 `ParallelFor` 的关系

这是最值得先建立的直觉。

### `ParallelFor`

```cpp
ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
    arr(i,j,k) = g(i,j,k);
});
```

特点：

- 每个点执行一次
- 对外部数组有副作用写入
- 不关心“所有点最后合成一个结果”

### `Reducer` / `Reduce`

```cpp
reducer.eval(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Result {
    return { arr(i,j,k) };
});
```

特点：

- 每个点执行一次
- 每个点返回一个局部值
- AMReX 负责把所有局部值合并成一个最终结果

所以对学习来说，规约最核心的一句话就是：

**规约 = “返回值版 ParallelFor”。**

## 5. `ReduceOp` 到底定义了什么

`ReduceOpSum/Min/Max` 这些对象在源码里定义了三件事：

1. `init(t)`
   规约变量怎么初始化
2. `local_update(d, s)`
   串行 / 线程局部层面怎么合并新值
3. `parallel_update(d, s)`
   GPU block 内怎么做并行合并

源码位置：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`struct ReduceOpSum`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`struct ReduceOpMin`
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`struct ReduceOpMax`

例如：

- `ReduceOpSum`
  - 初值是 `0`
  - 合并规则是 `d += s`
- `ReduceOpMin`
  - 初值是 `+inf`
  - 合并规则是 `d = min(d, s)`
- `ReduceOpMax`
  - 初值是 `-inf`
  - 合并规则是 `d = max(d, s)`

所以如果只看“思想”，规约操作本质上就是：

```text
单位元 + 合并规则
```

## 6. `ReduceData` 是干什么的

`ReduceData<Ts...>` 是规约过程中真正承载结果的对象。

可以把它理解成：

- 用来放局部规约结果
- 用来放 block 级规约结果
- 最后负责把这些结果整理成最终 tuple

GPU 路径下，它大致会做：

- 分配 host pinned memory 存最终结果
- 分配 device memory 存 block partial results
- 在 `value()` 时把 partial results 再归并

源码位置：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`class ReduceData`（GPU 版）
- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`class ReduceData`（非 GPU 版）

所以 `ReduceData` 可以粗略理解成：

**规约版的“结果缓冲区 + 最终取值器”。**

## 7. `ReduceOps` 是怎么连到迭代空间上的

`ReduceOps<...>` 真正负责把规约操作应用到不同迭代空间。

它支持的 `eval(...)` 形状包括：

- `eval(box, reduce_data, f)`
- `eval(box, ncomp, reduce_data, f)`
- `eval(mf, nghost, reduce_data, f)`
- `eval(mf, nghost, ncomp, reduce_data, f)`
- `eval(n, reduce_data, f)`

源码位置：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`class ReduceOps`

这点很重要，因为它说明：

**规约框架本身并不绑死在某一种迭代空间上。**

同样一套 `ReduceOp + ReduceData`，可以用于：

- 1D index range
- `Box`
- `Box + component`
- `MultiFab/FabArray`

## 8. 初学时最值得掌握的接口：`Reducer`

对你现在这个阶段，我觉得最应该重点看的是 `Reducer`。

源码位置：

- [`AMReX_Reduce.H`](../AMReX_Reduce.H):`class Reducer`

它是最能体现 AMReX 本地规约思想的用户接口。

最常见形状：

```cpp
amrex::Reducer<amrex::ReduceOpSum, Real> reducer;
using Result = typename decltype(reducer)::Result_t;

reducer.eval(box,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Result {
        return { arr(i,j,k) };
    });

auto r = reducer.getResult();
Real s = amrex::get<0>(r);
```

它的内部工作流是：

1. `Reducer` 持有 `ReduceOps`
2. `Reducer` 持有 `ReduceData`
3. `eval(...)` 负责遍历迭代空间
4. lambda 为每个点返回一个 tuple
5. 内部做本地规约
6. `getResult()` 返回最终本地结果

这就是 AMReX 本地规约的主干逻辑。

## 9. 为什么 `Reducer` 比 `Reduce::Sum` 更值得学

`Reduce::Sum/Min/Max` 很简单，但它更像是：

- 针对 1D range 的快捷接口

而 `Reducer` 更接近 AMReX 真正常见的规约使用方式，因为它能自然覆盖：

- `Box`
- `Box + ncomp`
- `MultiFab`
- 多个规约一起做

所以如果你要读源码、建立统一心智模型，优先级我建议是：

1. 先理解 `Reduce::Sum` 的思想
2. 再重点理解 `Reducer`
3. 最后再看 `ParReduce`

## 10. `ParReduce` 是什么

如果规约对象是 `FabArray / MultiFab`，最常见的便捷接口是：

```cpp
auto const& ma = mf.const_arrays();
Real s = amrex::ParReduce(amrex::TypeList<amrex::ReduceOpSum>{},
                          amrex::TypeList<Real>{},
                          mf, IntVect(0),
[=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
    -> amrex::GpuTuple<Real>
{
    return { ma[box_no](i,j,k) };
});
```

源码在：

- [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)

它的本质其实很简单：

1. 创建 `ReduceOps<...>`
2. 创建 `ReduceData<...>`
3. 调 `reduce_op.eval(fa, nghost, reduce_data, f)`
4. 返回结果

也就是说：

**`ParReduce` 不是新的规约体系，它只是针对 `FabArray/MultiFab` 的便捷包装。**

所以从学习顺序上，`ParReduce` 不需要单独神化。
它本质上还是 `ReduceOps + ReduceData` 那套东西。

## 11. 如果你只想记一条“本地规约主线”

就记这个：

```text
lambda 为每个点返回局部值
-> ReduceOp 定义怎么合并
-> ReduceOps 在给定迭代空间里执行规约
-> ReduceData 存中间结果和最终结果
-> Reducer / ParReduce 把这套机制包装成用户接口
```

这就是 AMReX 单卡 / 单进程本地规约最核心的调用逻辑。

## 12. 学源码时推荐顺序

如果你现在只想聚焦本地规约，我建议顺序这样看：

1. [`AMReX_Reduce.H`](../AMReX_Reduce.H)
   先看 `ReduceOpSum/Min/Max`
2. [`AMReX_Reduce.H`](../AMReX_Reduce.H)
   再看 `ReduceData`
3. [`AMReX_Reduce.H`](../AMReX_Reduce.H)
   再看 `ReduceOps`
4. [`AMReX_Reduce.H`](../AMReX_Reduce.H)
   最后看 `Reducer`
5. [`AMReX_ParReduce.H`](../AMReX_ParReduce.H)
   看 `ParReduce` 如何包装前面的东西

## 13. MPI 规约这里只做补充理解

等你把本地规约搞清楚以后，再补这层：

- [`AMReX_ParallelReduce.H`](../AMReX_ParallelReduce.H):`namespace ParallelAllReduce`
- [`AMReX_ParallelReduce.H`](../AMReX_ParallelReduce.H):`namespace ParallelReduce`

它们解决的是：

- 跨 MPI rank 做 `Allreduce / Reduce`

而不是：

- 单卡 / 单进程内部怎么做规约

所以从学习角度，不要把这层和 `Reducer / ParReduce` 混在一起。

## 14. 一个最容易混淆的点

很多人第一次读 AMReX 规约时会把：

- `Reduce/Reducer/ParReduce`
- `ParallelReduce/ParallelAllReduce`

混成一层。

其实更好的理解是：

- 前者：本地规约框架
- 后者：MPI 汇总接口

所以当前阶段你真正该重点抓住的是前者。
