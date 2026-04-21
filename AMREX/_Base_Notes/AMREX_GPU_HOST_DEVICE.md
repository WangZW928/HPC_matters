# AMReX `AMREX_GPU_HOST_DEVICE` 简单说明

## 1. 它是什么

`AMREX_GPU_HOST_DEVICE` 是 AMReX 提供的一个宏，用来修饰函数或成员函数，表示：

- 这个函数可以在 CPU 端调用
- 这个函数也可以在 GPU device 端调用

你可以把它理解成 AMReX 对 CUDA/HIP 等后端的一层统一封装。

## 2. 它的定义

定义文件：

- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H)

关键定义位置：

- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H):20
  当启用 GPU 且不是 SYCL 时：

```cpp
#define AMREX_GPU_HOST_DEVICE __host__ __device__
```

- [`AMReX_GpuQualifiers.H`](../AMReX_GpuQualifiers.H):31
  如果不是 GPU 编译环境：

```cpp
#define AMREX_GPU_HOST_DEVICE
```

也就是说：

- 在 GPU 编译环境下，它会展开成类似 `__host__ __device__`
- 在纯 CPU 编译时，它通常就是空的

## 3. 什么时候用

当一个函数需要同时支持：

- 普通 CPU 代码调用
- GPU kernel / GPU lambda 调用

就可以加上 `AMREX_GPU_HOST_DEVICE`。

这在 AMReX 里非常常见，因为很多基础数据结构和小函数既要在 host 上用，也要在 device 上用。

常见场景：

- `Array4` / `ArrayND` 的索引函数
- 小型数学函数
- `IntVect`、`Dim3` 等轻量工具函数

## 4. 怎么使用

最常见写法就是把它放在函数声明前面：

```cpp
AMREX_GPU_HOST_DEVICE
int add (int a, int b) noexcept
{
    return a + b;
}
```

也常和 `AMREX_FORCE_INLINE` 一起出现：

```cpp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int add (int a, int b) noexcept
{
    return a + b;
}
```

对于成员函数也一样：

```cpp
struct Foo {
    AMREX_GPU_HOST_DEVICE
    int value () const noexcept { return 42; }
};
```

## 5. 一个 AMReX 里的真实例子

在 `Array4` 里，索引访问函数就是这样写的：

- [`AMReX_Array4.H`](../AMReX_Array4.H):917

```cpp
[[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
U& operator() (int i, int j, int k) const noexcept
```

这意味着 `arr(i,j,k)` 这个接口：

- 可以在 CPU 代码里调用
- 也可以在 GPU lambda / kernel 中调用

## 6. 一个简单例子

### 6.1 定义一个 host/device 通用函数

```cpp
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real square (amrex::Real x) noexcept
{
    return x * x;
}
```

### 6.2 在 `ParallelFor` 里使用

```cpp
amrex::Array4<amrex::Real> const& arr = mf.array(mfi);
amrex::Box const& bx = mfi.validbox();

amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
{
    arr(i,j,k,0) = square(arr(i,j,k,0));
});
```

这里：

- `square()` 带了 `AMREX_GPU_HOST_DEVICE`
- 所以它既能在 host 上编译，也能在 device lambda 里调用

如果没有这个宏，这种函数在 GPU 代码里往往不能直接调用。

## 7. 使用时要注意什么

带 `AMREX_GPU_HOST_DEVICE` 的函数通常要尽量保持“轻量”和“可设备端编译”。

也就是说，函数体里一般不要随便使用：

- `std::cout`
- 复杂的主机专用库调用
- 只允许 host 端使用的接口
- 某些异常/RTTI 相关特性

因为这个函数需要同时通过 host 和 device 两边的编译。

## 8. 一句话总结

`AMREX_GPU_HOST_DEVICE` 的作用就是：

“把一个函数声明成 CPU 和 GPU 两边都可以调用的函数。”

