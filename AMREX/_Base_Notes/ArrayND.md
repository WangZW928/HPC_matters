# AMReX `amrex::ArrayND` 学习笔记

## 1. 它是什么

`amrex::ArrayND` 是 AMReX 中更底层、更通用的多维数组访问视图模板。

你可以把它理解成：

- 一个**非 owning** 的多维数据访问器
- 用来把一段连续内存解释成 `N` 维索引访问形式
- 能同时服务于普通 `N` 维数组，以及“最后一维是 component”的数组

`Array4` 本质上就是 `ArrayND` 的一个特化别名，所以学习 `ArrayND` 就是在学习 `Array4` 背后的通用设计。

主定义在：

- [`AMReX_Array4.H`](../AMReX_Array4.H):281
  `template<typename T, int N, bool last_dim_component = false> struct ArrayND`

## 2. 模板参数是什么意思

`ArrayND` 有三个模板参数：

```cpp
template<typename T, int N, bool last_dim_component = false>
struct ArrayND
```

它们分别表示：

- `T`
  元素类型，比如 `Real`、`int`、`const Real`
- `N`
  数组维数
- `last_dim_component`
  最后一维是否被解释为 component 维，而不是普通空间维

### 两种典型模式

#### 模式 1：普通 N 维数组

```cpp
ArrayND<Real, 3, false>
```

表示一个普通 3 维数组，比如 `(i,j,k)`。

#### 模式 2：最后一维是 component

```cpp
ArrayND<Real, 4, true>
```

表示前三维是空间维，最后一维是 component，也就是 `(i,j,k,n)`。

这正是 `Array4`：

- [`AMReX_Array4.H`](../AMReX_Array4.H):1306
  `using Array4 = ArrayND<T, 4, true>;`

## 3. 它内部保存了什么

`ArrayND` 本身非常轻量，主要保存四类信息：

- `T* p`
  数据首地址
- `stride`
  步长信息
- `begin`
  每一维起始下标，包含式
- `end`
  每一维结束下标，排除式

对应源码：

- [`AMReX_Array4.H`](../AMReX_Array4.H):291
  `T* p`
- [`AMReX_Array4.H`](../AMReX_Array4.H):294
  `begin`
- [`AMReX_Array4.H`](../AMReX_Array4.H):295
  `end`

所以 `ArrayND` 不是容器，不负责分配内存，而只是“描述如何访问一块现有连续内存”。

## 4. 它和 `Array4` 的关系

这是最重要的一层关系：

```text
ArrayND 是通用模板
Array4  是 ArrayND<T,4,true> 的别名
```

也就是说：

- `ArrayND` 负责提供通用多维索引机制
- `Array4` 只是 AMReX 最常用的一个专门命名版本
- `Array4` 额外有一组更方便的 `(i,j,k)` / `(i,j,k,n)` 专用接口

因此你可以把 `Array4` 看成“最常见应用场景下的 `ArrayND` 快捷版本”。

## 5. 构造方式

### 5.1 普通 `BoxND<N>` 构造

当 `last_dim_component == false` 时，可以直接用 `BoxND<N>` 构造：

- [`AMReX_Array4.H`](../AMReX_Array4.H):330
  `ArrayND (T* a_p, BoxND<N> const& box)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):358
  `ArrayND (T* a_p, IntVectND<N> const& a_begin, IntVectND<N> const& a_end)`

这表示一个真正的 `N` 维数组区域。

### 5.2 空间维 + component 数构造

当 `last_dim_component == true` 时，可以传空间 box 和 `ncomp`：

- [`AMReX_Array4.H`](../AMReX_Array4.H):344
  `ArrayND (T* a_p, BoxND<M> const& box, int ncomp)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):389
  `ArrayND (T* a_p, IntVectND<M> const& a_begin, IntVectND<M> const& a_end, int ncomp)`

这时最后一维会被自动设成 component 维。

### 5.3 `Array4` 专用 `Dim3` 构造

- [`AMReX_Array4.H`](../AMReX_Array4.H):375
  `ArrayND (T* a_p, Dim3 const& a_begin, Dim3 const& a_end, int a_ncomp)`

这就是 `Array4` 最常见的来源。

### 5.4 const 转换构造

- [`AMReX_Array4.H`](../AMReX_Array4.H):316
  可以从 `ArrayND<T,...>` 隐式转成 `ArrayND<const T,...>`

这也是你常见 `array()` / `const_array()` 模式的基础。

### 5.5 component 切片构造

- [`AMReX_Array4.H`](../AMReX_Array4.H):423
  `ArrayND(rhs, start_comp)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):447
  `ArrayND(rhs, start_comp, num_comp)`

这说明当最后一维是 component 时，可以只取部分分量，形成新的 view。

## 6. 索引访问是怎么工作的

### 6.1 通用 `operator()`

对一般 `ArrayND`，最核心的访问接口是：

- [`AMReX_Array4.H`](../AMReX_Array4.H):493
  `operator()(idx... i)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):517
  `operator()(IntVectND<M> const& iv)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):539
  `operator()(IntVectND<M> const& iv, int n)`

它们本质上都做同一件事：

```text
多维索引 -> 线性 offset -> p[offset]
```

### 6.2 指针访问 `ptr()`

除了返回引用，也可以直接返回元素地址：

- [`AMReX_Array4.H`](../AMReX_Array4.H):557
  `ptr(idx... i)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):578
  `ptr(IntVectND<M> const& iv)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):598
  `ptr(IntVectND<M> const& iv, int n)`

所以 `ArrayND` 的核心是：

- 要么给你元素引用
- 要么给你元素指针

## 7. offset 是怎么计算的

`ArrayND` 真正重要的底层逻辑在 `get_offset()`：

- [`AMReX_Array4.H`](../AMReX_Array4.H):777
  `get_offset(IntVectND<M> const& iv)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):809
  `get_offset(IntVectND<M> const& iv, int n)`

### 7.1 普通 N 维情况

可以抽象理解成：

```text
offset = (i0 - begin[0])
       + (i1 - begin[1]) * stride[0]
       + (i2 - begin[2]) * stride[1]
       + ...
```

### 7.2 最后一维是 component 的情况

如果 `last_dim_component = true`，那么 component 偏移会额外加上：

```text
n * stride[N-2]
```

对 `Array4` 来说，这就变成了熟悉的：

```text
offset = (i-ilo)
       + (j-jlo) * nx
       + (k-klo) * nx*ny
       + n       * nx*ny*nz
```

## 8. stride 是怎么来的

步长由 `set_stride()` 自动计算：

- [`AMReX_Array4.H`](../AMReX_Array4.H):1266
  `set_stride()`

逻辑是：

```cpp
Long current_stride = 1;
for d = 0 .. N-2:
    current_stride *= (end[d] - begin[d]);
    stride[d] = current_stride;
```

这说明 `ArrayND` 使用的是一种 Fortran 风格布局：

- 第 0 维变化最快
- 第 1 维次之
- 依次类推
- 如果最后一维是 component，那么 component 变化最慢

## 9. 有哪些辅助接口

### 9.1 `dataPtr()`

- [`AMReX_Array4.H`](../AMReX_Array4.H):614
  返回原始数据指针

### 9.2 `nComp()`

- [`AMReX_Array4.H`](../AMReX_Array4.H):623
  如果最后一维是 component，则返回 component 数；否则返回 1

### 9.3 `size()`

- [`AMReX_Array4.H`](../AMReX_Array4.H):638
  返回当前 view 覆盖的总元素数

### 9.4 `get_stride<d>()`

- [`AMReX_Array4.H`](../AMReX_Array4.H):660
  返回第 `d` 维对应的 stride

### 9.5 `contains()`

- [`AMReX_Array4.H`](../AMReX_Array4.H):679
  `contains(idx...)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):699
  `contains(IntVectND<M> const& iv)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):725
  `contains(IntVectND<M> const& iv, int n)`

这些接口可以用来做边界判断。

## 10. 边界检查与调试

在 debug 或启用边界检查时，`ArrayND` 会做索引合法性检查：

- [`AMReX_Array4.H`](../AMReX_Array4.H):825
  `index_assert(IntVectND<N> const& iv)`
- [`AMReX_Array4.H`](../AMReX_Array4.H):865
  少一维时默认最后一维处理
- [`AMReX_Array4.H`](../AMReX_Array4.H):883
  带 component 的 `index_assert`

这说明 AMReX 在 release 模式下尽量保持轻量，而在 debug 模式下可以帮助你尽早发现越界。

## 11. deduction guide 是什么意思

源码末尾还有 deduction guide：

- [`AMReX_Array4.H`](../AMReX_Array4.H):1281
  `ArrayND (T*, BoxND<N> const&) -> ArrayND<T, N, false>`
- [`AMReX_Array4.H`](../AMReX_Array4.H):1286
  `ArrayND (T*, BoxND<N> const&, int) -> ArrayND<T, N+1, true>`

这表示编译器有时可以根据构造参数自动推导出模板参数。

比如：

- 传 `BoxND<N>`，推导成普通 `N` 维数组
- 传 `BoxND<N> + ncomp`，推导成“空间 N 维 + component 1 维”

这也是 `ArrayND` 设计得比较现代的一点。

## 12. 为什么说它是 GPU 友好的

`ArrayND` 设计上就非常适合异构执行：

- 结构轻量
- trivially copyable 风格
- 按值传入 lambda 成本低
- 主要由指针、边界、stride 组成
- 成员函数大多带 `AMREX_GPU_HOST_DEVICE`

所以它既可以在 CPU 上访问，也可以在 GPU kernel 中访问，前提只是底层指针 `p` 对当前执行位置可访问。

## 13. `ArrayND` 和 `Array4` 的学习顺序建议

如果你已经理解了 `Array4`，回头看 `ArrayND` 时建议抓住这三层：

1. `ArrayND` 是通用模板
2. `Array4` 是最常见的专门别名
3. 真正底层逻辑是：`pointer + bounds + stride + offset`

只要这三层清楚了，后面再读：

- `BaseFab`
- `FArrayBox`
- `MultiFab`
- `FabArray`
- `ParallelFor`

就会顺很多。

## 14. 一句话总结

`amrex::ArrayND` 的本质是：

“一个不拥有数据的通用 N 维数组视图模板，它通过 `begin/end/stride` 把多维索引翻译成线性内存地址；而 `Array4` 只是它在 AMReX 中最常见的 `(空间维 + component)` 特化形式。”

