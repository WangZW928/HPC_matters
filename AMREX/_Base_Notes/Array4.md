# AMReX `amrex::Array4` 学习笔记

## 1. 它是什么

`amrex::Array4<T>` 是 AMReX 里非常核心的一个“轻量级四维数据访问器”。

- 它本身通常**不拥有内存**
- 它只是把一段已经存在的连续内存解释成 `(i,j,k,n)` 这样的访问形式
- 其中前三维通常表示空间索引，最后一维表示分量 `component`
- 它被设计成可以很方便地按值传进 GPU kernel / lambda

在源码里，`Array4` 不是独立类，而是一个别名：

```cpp
using Array4 = ArrayND<T, 4, true>;
```

这表示：

- 底层模板是 `ArrayND`
- 维度数 `N = 4`
- 最后一维 `true` 表示“最后一维是 component，而不是普通空间维”

## 2. 核心定义位置

最重要的源码位置如下。

### 主定义

- [`AMReX_Array4.H`](../AMReX_Array4.H):1306
  `using Array4 = ArrayND<T, 4, true>;`

### 关键构造函数

- [`AMReX_Array4.H`](../AMReX_Array4.H):375
  用 `Dim3 begin/end + ncomp` 构造 `Array4`
- [`AMReX_Array4.H`](../AMReX_Array4.H):423
  组件切片构造：`Array4(rhs, start_comp)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):447
  组件切片构造：`Array4(rhs, start_comp, num_comp)`

### 索引访问

- [`AMReX_Array4.H`](../AMReX_Array4.H):917
  `operator()(int i, int j, int k)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):951
  `operator()(int i, int j, int k, int n)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1088
  `ptr(int i, int j, int k, int n)`

### 步长与布局

- [`AMReX_Array4.H`](../AMReX_Array4.H):1266
  `set_stride()`

### 辅助函数

- [`AMReX_Array4.H`](../AMReX_Array4.H):1317
  `lbound(Array4<T> const&)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1331
  `ubound(Array4<T> const&)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1345
  `length(Array4<T> const&)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1382
  `PolymorphicArray4`

## 3. `Array4` 内部保存了什么

从 [`AMReX_Array4.H`](../AMReX_Array4.H) 可见，`ArrayND`/`Array4` 主要保存这些信息：

- `T* p`
  指向数据首地址
- `stride`
  每一维跨越的步长信息
- `begin`
  每一维的起始下标，`begin` 是**包含式**
- `end`
  每一维的结束下标，`end` 是**排除式**

对 `Array4<T>` 来说，内部逻辑上可理解为：

- `begin = (ilo, jlo, klo, 0)`
- `end   = (ihi+1, jhi+1, khi+1, ncomp)`

也就是说：

- 空间维度是 box 的闭区间下标，但在内部存成半开区间 `[begin, end)`
- component 永远从 `0` 开始

## 4. 它怎么构造出来

你平时很少手写 `Array4` 构造，更常见的是由 `BaseFab` / `FArrayBox` / `MultiFab` 帮你生成。

### `BaseFab` 工厂函数

- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):92
  `makeArray4(T* p, Box const& bx, int ncomp)`

它本质上做的是：

```cpp
return Array4<T>{p, amrex::begin(bx), amrex::end(bx), ncomp};
```

### `BaseFab` 常用入口

- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):382
  `Array4<T const> array() const`
- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):400
  `Array4<T> array()`
- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):418
  `Array4<T const> const_array() const`

### `FabArray` / `MultiFab` 常用入口

- [`AMReX_FabArray.H`](../AMReX_FabArray.H):566
  `array(const MFIter& mfi) const`
- [`AMReX_FabArray.H`](../AMReX_FabArray.H):572
  `array(const MFIter& mfi)`
- [`AMReX_FabArray.H`](../AMReX_FabArray.H):590
  `const_array(const MFIter& mfi) const`

因此最常见的写法就是：

```cpp
for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    auto const& arr = mf.array(mfi);
    auto const& box = mfi.validbox();
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        arr(i,j,k,0) = 1.0;
    });
}
```

## 5. 典型使用方式

### 5.1 读写单个分量

```cpp
Array4<Real> const& a = fab.array();
a(i,j,k,0) = 3.14;
```

### 5.2 省略 component

```cpp
Array4<Real> const& a = fab.array();
a(i,j,k) = 3.14;
```

这等价于访问 `component = 0`。源码位置在：

- [`AMReX_Array4.H`](../AMReX_Array4.H):917

### 5.3 只读访问

```cpp
Array4<Real const> const& a = fab.const_array();
Real x = a(i,j,k,2);
```

### 5.4 组件切片

```cpp
auto a_all = fab.array();
Array4<Real> a_sub(a_all, 2, 3);  // 从第2个分量开始，取3个分量
```

切片后：

- 指针 `p` 会偏移到新的起始 component
- 新视图中的 component 编号又从 `0` 开始

这一点非常重要，因为 `a_sub(i,j,k,0)` 对应原数组的 `a_all(i,j,k,2)`。

## 6. 内存是怎么存的

`Array4` 的注释明确写了：它遵循 **Fortran-style column-major ordering**。

直观理解就是：

- `i` 方向变化最快
- 然后是 `j`
- 然后是 `k`
- 最后才是 `n`（component）

也就是同一个 component 下，一个 box 内的所有空间点先连续排完，再接下一个 component。

### 步长是如何设置的

[`AMReX_Array4.H`](../AMReX_Array4.H):1266 的 `set_stride()` 逻辑是：

```cpp
Long current_stride = 1;
for each dimension d = 0 .. N-2:
    current_stride *= (end[d] - begin[d]);
    stride[d] = current_stride;
```

对 `Array4` 来说可写成：

- `stride[0] = nx`
- `stride[1] = nx * ny`
- `stride[2] = nx * ny * nz`

其中：

- `nx = end[0] - begin[0]`
- `ny = end[1] - begin[1]`
- `nz = end[2] - begin[2]`

### 实际索引公式

[`AMReX_Array4.H`](../AMReX_Array4.H):951 中 4 维访问的核心公式是：

```cpp
p[(i-begin[0]) +
  (j-begin[1]) * stride[0] +
  (k-begin[2]) * stride[1] +
  n * stride[2]]
```

所以内存偏移量是：

```text
offset = (i-ilo)
       + (j-jlo) * nx
       + (k-klo) * nx*ny
       + n       * nx*ny*nz
```

这正是 AMReX 里非常常见的 “cell-major in space, component last” 布局。

## 7. 和 `MultiFab` / `FArrayBox` 的关系

一个很重要的认识是：

- `FArrayBox` / `BaseFab` 才是“拥有数据的对象”
- `Array4` 更像是“访问视图 view”

通常关系是：

1. `MultiFab` 管很多个 `FArrayBox`
2. `MFIter` 遍历这些 box
3. `mf.array(mfi)` 或 `mf.const_array(mfi)` 返回当前 box 的 `Array4`
4. 然后在 `ParallelFor` 里用 `arr(i,j,k,n)` 访问

这也是为什么你会在 AMReX 源码中频繁看到这种模式。

[`AMReX_Reduce.H`](../AMReX_Reduce.H):1589-1591 的注释就给了一个典型例子：

```cpp
Array4<Real const> const& ar =  mf.const_array(mfi);
Array4<int  const> const& ai = imf.const_array(mfi);
reducer.eval(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Result_t
```

## 8. CPU / GPU 视角下为什么它很好用

`Array4` 是轻量、非 owning、可按值捕获的，所以很适合：

- CPU 循环
- `amrex::ParallelFor`
- GPU kernel lambda

另外还有 GPU 读辅助：

- [`AMReX_GpuUtility.H`](../AMReX_GpuUtility.H):28
  `LDG(Array4<T> const&, i, j, k)`
- [`AMReX_GpuUtility.H`](../AMReX_GpuUtility.H):39
  `LDG(Array4<T> const&, i, j, k, n)`

这说明 `Array4` 不只是“语法糖”，它也是 GPU 访问路径中的基础数据接口。

## 9. 初学时最容易混淆的点

### 9.1 它不是容器

`Array4` 不负责分配和释放内存。不要把它理解成 `std::vector` 或真正的 4D 数组对象。

### 9.2 `operator()(i,j,k)` 默认访问的是 `n = 0`

如果你的 `MultiFab` 有多个分量，省略 `n` 只会访问第 0 个分量。

### 9.3 `end` 是排除式

内部保存的是半开区间 `[begin, end)`，这和 `Box` 对外常见的闭区间语义不同，阅读源码时要注意换算。

### 9.4 低维问题里“缺的维度”通常被补成 0

AMReX 文档注释里专门说明了：

- 2D 情况常见访问是 `(i,j,0,n)`
- 1D 情况常见访问是 `(i,0,0,n)`

所以很多 kernel 仍然统一写成 `(i,j,k)` 风格。

### 9.5 component 是最后一维，不是第一维

这和某些 C/C++ 数组习惯不同，但和 Fortran / BaseFab 的历史设计一致。

## 10. 一个简单脑图

可以把它想成：

```text
BaseFab / FArrayBox / MultiFab
        |
        | 提供底层连续内存
        v
      Array4
        |
        | 提供 (i,j,k,n) 访问
        v
   ParallelFor / GPU lambda / 算法内核
```

## 11. 建议你接下来继续看的文件

如果你准备继续深挖，推荐按这个顺序看：

1. [`AMReX_Array4.H`](../AMReX_Array4.H)
   先彻底吃透 `Array4` 的索引和切片
2. [`AMReX_BaseFab.H`](../AMReX_BaseFab.H)
   看 owning 数据对象如何生成 `Array4`
3. [`AMReX_FabArray.H`](../AMReX_FabArray.H)
   看 `MultiFab`/`FabArray` 怎样在 `MFIter` 中提供 `Array4`
4. [`AMReX_MFIter.H`](../AMReX_MFIter.H)
   看 box 遍历机制
5. `AMReX_GpuLaunch*.H` 和 [`AMReX_GpuUtility.H`](../AMReX_GpuUtility.H)
   看 `Array4` 如何进入 GPU 执行路径

## 12. 一句话总结

`amrex::Array4<T>` 可以理解成：

“把 `BaseFab/MultiFab` 的一段连续内存，包装成适合 CPU/GPU 内核使用的 `(i,j,k,n)` 四维非 owning 访问视图，并采用 `i` 最快、`component` 最慢的 Fortran 风格内存布局。”

## 13. `Array4` 和真实内存的映射关系

这是理解 `Array4` 最关键的一点。

### 13.1 `Array4` 很轻量，但它不是数据本体

可以把 `Array4` 看成一个“索引访问壳子”或者“view”。

它自己通常只保存：

- 一个原始指针 `p`
- 边界 `begin/end`
- 步长 `stride`

因此它本身非常轻量，适合：

- 按值传递
- 被 lambda 捕获
- 在 GPU kernel 里直接使用

但它**不负责**：

- 分配内存
- 释放内存
- 决定数据在 CPU 还是 GPU 上
- 改变底层数据布局

这些事情通常由 `BaseFab`、`FArrayBox`、`MultiFab` 以及它们背后的 `Arena` / memory allocator 来负责。

### 13.2 `Array4` 不改变底层数据存储

所以你说“`Array4` 的数据存储并不影响 CPU/GPU memory 的真实存储”，这个理解是对的。

更准确地说：

- 底层真实内存先已经存在
- `Array4` 只是拿到这段内存的首地址，再附带边界和步长信息
- 然后把它解释成 `(i,j,k,n)` 这样的四维访问方式

也就是说，`Array4` 是“解释内存”，不是“创造内存”。

### 13.3 它是怎么 map 到真实地址的

关键思路是：

1. 底层对象先提供一个连续内存首地址 `p`
2. `Array4` 记录 box 的起止索引
3. `Array4` 根据 box 尺寸计算 stride
4. 当你写 `a(i,j,k,n)` 时，它把四维索引换算成一维偏移
5. 最终访问的是 `p[offset]`

源码实现见：

- [`AMReX_Array4.H`](../AMReX_Array4.H):951
  `operator()(int i, int j, int k, int n)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1266
  `set_stride()`

四维索引到一维偏移的大意是：

```text
offset = (i-ilo)
       + (j-jlo) * nx
       + (k-klo) * nx*ny
       + n       * nx*ny*nz
```

然后真实访问就是：

```text
address = p + offset
value   = *(p + offset)
```

所以它不是在查某种复杂的树状结构，也不是去做额外映射表，而是非常直接地做“步长换算 + 指针偏移”。

### 13.4 `Array4` 和真实 owning 对象的关系

通常调用链可以理解成：

```text
MultiFab / FArrayBox / BaseFab
        |
        | 持有真实内存
        v
   data pointer (dptr)
        |
        | 生成 Array4
        v
   Array4(p, begin, end, ncomp)
        |
        | 通过 stride 做地址换算
        v
   arr(i,j,k,n) -> p[offset]
```

比如在 `BaseFab` 里：

- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):400
  `array()` 返回 `Array4<T>`
- [`AMReX_BaseFab.H`](../AMReX_BaseFab.H):418
  `const_array()` 返回 `Array4<T const>`

本质上都是把 `BaseFab` 自己维护的底层数据指针包装成 `Array4` view。

### 13.5 CPU 和 GPU 的区别到底体现在哪

区别不在 `Array4` 这个类型本身，而在 `p` 指向的那块内存是否对当前执行位置可访问。

也就是说：

- 如果 `p` 指向主机可访问内存，那么在 CPU 上访问就是普通内存读写
- 如果 `p` 指向设备可访问内存，那么在 GPU kernel 里访问就是设备内存读写
- 如果使用的是 unified / managed memory，那么同一个 view 可能在 CPU/GPU 两边都可用

所以 `Array4` 更像统一访问接口，而不是存储后端。

### 13.6 一个直观例子

假设一个 box 大小是：

- `nx = 4`
- `ny = 3`
- `nz = 2`
- `ncomp = 5`

那么整块数据会按下面顺序连续存储：

```text
先存所有 (i,j,k,0)
再存所有 (i,j,k,1)
再存所有 (i,j,k,2)
...
```

如果你访问：

```cpp
a(2,1,0,3)
```

`Array4` 不会去“找一个四维对象里的节点”，而是直接算出它在连续数组中的偏移，然后访问那一格。

### 13.7 一句话抓重点

`Array4` 的本质不是“存数据”，而是“拿着一个真实数据指针，用 AMReX 规定的 stride 规则，把 `(i,j,k,n)` 翻译成底层线性内存地址”。

