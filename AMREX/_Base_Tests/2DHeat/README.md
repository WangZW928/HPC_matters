# 2D Poisson Example (Jacobi, no reduction)

这个目录现在是一个最小 2D 泊松方程教学例子：

```text
-Delta(u) = rhs
```

当前版本使用固定 Jacobi 迭代步数，不做残差规约，重点是先跑通：

- `parallelFor2D(nx, ny, lambda)`
- stencil 访存
- 双缓冲 `u_old/u_new`
- CPU/GPU 统一调用接口
- C++ 直接输出结果图

## 方程与离散

- 边界条件：齐次 Neumann，`du/dn = 0`
- 右端项：`rhs = cos(2*pi*x) * cos(2*pi*y)`（零均值）
- Jacobi 迭代公式：

```text
u_new(i,j) = ((uE+uW)/dx^2 + (uN+uS)/dy^2 + rhs(i,j)) / (2*(1/dx^2 + 1/dy^2))
```

边界通过 ghost cell 镜像填充实现（齐次 Neumann）。

## Neumann 相容条件

Neumann-Poisson 的连续相容条件是：

```text
∫_Omega rhs dOmega = ∫_{∂Omega} (du/dn) ds
```

在当前齐次 Neumann (`du/dn = 0`) 情况下，需要：

```text
∫_Omega rhs dOmega = 0
```

离散对应就是 `sum(rhs)` 需要接近 0。
当前示例选用 `cos(2*pi*x) * cos(2*pi*y)`，满足零均值，因而满足相容条件。

另外，纯 Neumann 问题存在常数零空间，所以代码里固定了一个参考点 `u(0,0)=0` 来消除不唯一性。

## Jacobi 的矩阵表达（简版）

离散后可写为：

```text
A u = b
A = D + L + U
```

Jacobi 迭代：

```text
u^(k+1) = D^{-1} ( b - (L+U) u^(k) )
```

等价地：

```text
u^(k+1) = (I - D^{-1}A) u^(k) + D^{-1} b
```

## 文件结构

- `main.cpp`：Poisson 主程序（固定步数 Jacobi）
- `ParallelFor2D.H`：CPU/OpenMP 与 GPU kernel launch 统一接口
- `Array2DView.H`：二维视图
- `Buffer.H`：host/device 内存抽象
- `Print2D.H`：CSV/PPM 输出

## 输出文件

程序结束后会写出：

- `poisson_final.csv`：数值结果
- `poisson_final.ppm`：C++ 直接导出的伪彩色图

## 构建与运行

### CPU

```bash
cmake -S . -B build
cmake --build build -j
./build/heat2d_cpu
```

### GPU

如果系统里有 CUDA 编译器，CMake 会自动生成 `heat2d_gpu`：

```bash
cmake -S . -B build
cmake --build build -j
./build/heat2d_gpu
```
