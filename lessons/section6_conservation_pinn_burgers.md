# 第六节课课件：面向 Burgers 方程的保守型 PINN

> 课程定位：从第五节的“线性平流方程守恒约束”正式迁移到论文主线：**Burgers 方程上的 Conservation-PINN**。

---

# 0. 本节课目标

本节课结束后，学生应该能够：

1. 说清楚线性平流方程与 Burgers 方程的关系；
2. 理解 Burgers 方程为什么是流体与非线性波动问题中的经典模型；
3. 写出 Burgers 方程的 PINN 残差；
4. 理解为什么周期边界下 Burgers 方程仍然具有质量守恒；
5. 把第五节的 `conservation_loss` 迁移到 Burgers 方程；
6. 设计 Baseline PINN 与 Conservation-PINN 的对比实验。

---

# 1. 从第五节课回顾：我们已经做了什么？

第五节课我们研究的是线性平流方程：

$$
u_t + c u_x = 0
$$

它可以写成守恒律形式：

$$
u_t + F(u)_x = 0, \qquad F(u)=cu
$$

在周期边界下：

$$
M(t)=\int_0^1 u(x,t)\,dx
$$

保持不变。

第五节课的核心思想是：

> 普通 PINN 只约束局部 PDE 残差，不一定严格保持全局守恒量。  
> Conservation-PINN 把守恒量误差也写进损失函数，使网络更重视物理结构。

---

# 2. 为什么第六节要讲 Burgers 方程？

Burgers 方程是流体力学和非线性 PDE 中非常经典的入门模型。它虽然比 Navier-Stokes 简单很多，但已经包含了几个重要特征：

- 非线性对流；
- 波形传播；
- 波形变陡；
- 粘性扩散；
- 守恒律结构；
- 长时间预测中的稳定性问题。

因此它非常适合作为本科生论文里的主算例。

一句话理解：

> 热传导方程主要展示扩散；线性平流方程主要展示平移；Burgers 方程开始展示“非线性流动”。

---

# 3. Burgers 方程的两种常见形式

## 3.1 无粘 Burgers 方程

无粘 Burgers 方程写成：

$$
u_t + u u_x = 0
$$

也可以写成守恒律形式：

$$
u_t + \left(\frac{u^2}{2}\right)_x = 0
$$

因为：

$$
\left(\frac{u^2}{2}\right)_x = u u_x
$$

所以两种写法等价。

其中通量函数为：

$$
F(u)=\frac{u^2}{2}
$$

## 3.2 粘性 Burgers 方程

为了教学和训练稳定性，本节课建议先使用粘性 Burgers 方程：

$$
u_t + u u_x = \nu u_{xx}
$$

也可写为残差形式：

$$
u_t + u u_x - \nu u_{xx}=0
$$

其中：

- $u(x,t)$：待求解变量；
- $u u_x$：非线性对流项；
- $\nu u_{xx}$：粘性扩散项；
- $\nu$：粘性系数。

课堂中可以取：

$$
\nu = \frac{0.01}{\pi}
$$

这是很多 PINN 入门例子里常用的设置。

---

# 4. Burgers 方程的物理直觉

## 4.1 线性平流：所有点以同一速度走

线性平流方程：

$$
u_t + c u_x = 0
$$

速度 $c$ 是常数。无论某处 $u$ 大还是小，传播速度都一样。

因此波形整体平移，形状基本不变。

## 4.2 Burgers：传播速度由自己决定

Burgers 方程：

$$
u_t + u u_x = 0
$$

可以理解为：局部传播速度就是 $u$ 本身。

如果某处 $u$ 大，它传播得更快；如果某处 $u$ 小，它传播得更慢。于是波形会逐渐变陡。

这就是 Burgers 方程比线性平流更接近真实流动问题的原因。

---

# 5. Burgers 方程的守恒结构

## 5.1 无粘形式下的质量守恒

从守恒律形式出发：

$$
u_t + F(u)_x = 0
$$

对空间区间 $[0,1]$ 积分：

$$
\frac{d}{dt}\int_0^1 u(x,t)\,dx
+
F(u(1,t))-F(u(0,t))=0
$$

若周期边界成立：

$$
u(0,t)=u(1,t)
$$

则：

$$
F(u(0,t))=F(u(1,t))
$$

因此：

$$
\frac{d}{dt}\int_0^1 u(x,t)\,dx=0
$$

即：

$$
M(t)=M(0)
$$

## 5.2 粘性 Burgers 下为什么质量仍守恒？

粘性 Burgers 方程：

$$
u_t + \left(\frac{u^2}{2}\right)_x = \nu u_{xx}
$$

对 $[0,1]$ 积分：

$$
\frac{d}{dt}\int_0^1 u\,dx
+
\left[\frac{u^2}{2}\right]_0^1
=
\nu [u_x]_0^1
$$

若采用周期边界，并且导数也周期一致：

$$
u(0,t)=u(1,t),\qquad u_x(0,t)=u_x(1,t)
$$

则两侧边界项都抵消：

$$
\frac{d}{dt}\int_0^1 u\,dx=0
$$

所以粘性 Burgers 方程在周期边界下仍然保持总质量守恒。

---

# 6. PINN 如何求解 Burgers 方程？

## 6.1 网络表示

仍然用神经网络表示未知解：

$$
u_\theta(x,t)\approx u(x,t)
$$

输入：

$$
(x,t)
$$

输出：

$$
u_\theta(x,t)
$$

## 6.2 自动微分

用 PyTorch 自动微分得到：

$$
u_t,\quad u_x,\quad u_{xx}
$$

## 6.3 PDE 残差

对于粘性 Burgers 方程：

$$
f_\theta(x,t)
=
(u_\theta)_t
+
u_\theta (u_\theta)_x
-
\nu (u_\theta)_{xx}
$$

训练时希望：

$$
f_\theta(x,t)\approx 0
$$

对应代码核心为：

```python
def pde_residual_burgers(model, x, t, nu):
    u = model(x, t)
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    f = u_t + u * u_x - nu * u_xx
    return f
```

---

# 7. 损失函数设计

## 7.1 普通 PINN

普通 PINN 总损失为：

$$
\mathcal L_{base}
=
\lambda_{ic}\mathcal L_{ic}
+
\lambda_{bc}\mathcal L_{bc}
+
\lambda_f\mathcal L_f
$$

其中：

$$
\mathcal L_{ic}
=
\frac{1}{N_{ic}}\sum_i
|u_\theta(x_i,0)-u_0(x_i)|^2
$$

$$
\mathcal L_{bc}
=
\frac{1}{N_{bc}}\sum_i
|u_\theta(0,t_i)-u_\theta(1,t_i)|^2
+
\frac{1}{N_{bc}}\sum_i
|(u_\theta)_x(0,t_i)-(u_\theta)_x(1,t_i)|^2
$$

$$
\mathcal L_f
=
\frac{1}{N_f}\sum_j |f_\theta(x_j,t_j)|^2
$$

## 7.2 Conservation-PINN

引入守恒量：

$$
M_\theta(t)=\int_0^1 u_\theta(x,t)\,dx
$$

理论守恒量：

$$
M_0=\int_0^1 u_0(x)\,dx
$$

守恒损失：

$$
\mathcal L_{cons}
=
\frac{1}{N_t}\sum_n
|M_\theta(t_n)-M_0|^2
$$

改进后的总损失：

$$
\mathcal L
=
\lambda_{ic}\mathcal L_{ic}
+
\lambda_{bc}\mathcal L_{bc}
+
\lambda_f\mathcal L_f
+
\lambda_{cons}\mathcal L_{cons}
$$

---

# 8. 本节课建议采用的初始条件和边界条件

为了避免一开始就进入激波问题，建议使用光滑周期初值：

$$
u(x,0)=1+0.5\sin(2\pi x)
$$

周期边界：

$$
u(0,t)=u(1,t)
$$

导数周期一致：

$$
u_x(0,t)=u_x(1,t)
$$

理论质量：

$$
M_0=\int_0^1 \left(1+0.5\sin(2\pi x)\right)dx=1
$$

这和第五节线性平流的设置保持一致，方便学生比较。

---

# 9. 和第五节代码相比，需要改哪里？

第五节线性平流的残差是：

```python
f = u_t + c * u_x
```

第六节 Burgers 残差改为：

```python
f = u_t + u * u_x - nu * u_xx
```

也就是说：

1. 需要多算一个二阶导 `u_xx`；
2. 需要加入非线性项 `u * u_x`；
3. 需要设置粘性系数 `nu`；
4. 守恒量损失 `conservation_loss` 可以基本复用；
5. 周期边界损失也可以继续复用。

---

# 10. 实验设计

## 10.1 对比模型

| 实验组 | 方法 | 目的 |
|---|---|---|
| A | Baseline PINN | 普通 PINN 基线 |
| B | Conservation-PINN | 检查守恒约束是否降低质量漂移 |
| C | 不同 $\lambda_{cons}$ | 研究守恒权重敏感性 |

## 10.2 建议参数

| 参数 | 建议值 | 说明 |
|---|---:|---|
| $N_{ic}$ | 128 | 初值点数量 |
| $N_{bc}$ | 128 | 边界点数量 |
| $N_f$ | 3000-8000 | Burgers 比线性平流更难，可适当增加 |
| $N_{cons,t}$ | 50 | 守恒约束时间切片 |
| $N_{cons,x}$ | 200 | 质量积分网格 |
| hidden_dim | 64 或 128 | 网络宽度 |
| num_hidden | 4 或 5 | 网络深度 |
| learning rate | 1e-3 | Adam 初始学习率 |
| $\lambda_{cons}$ | 0.1, 1, 10 | 做敏感性实验 |

---

# 11. 评价指标

## 11.1 预测误差

如果有参考解 $u_{ref}$，可用：

$$
E_{L2}
=
\frac{\|u_\theta-u_{ref}\|_2}{\|u_{ref}\|_2}
$$

如果没有解析解，可以用高精度 FDM/FVM 解作为参考解。

## 11.2 守恒量漂移

$$
E_{cons}(t)=|M_\theta(t)-M_0|
$$

可以统计：

- mean conservation error；
- max conservation error；
- final-time conservation error。

## 11.3 长时间外推稳定性

训练区间：

$$
t\in[0,1]
$$

外推区间：

$$
t\in[1,2]\quad \text{或}\quad t\in[1,3]
$$

观察：

- 波形是否明显漂移；
- 幅值是否衰减过快或爆炸；
- 守恒量是否持续偏离；
- Conservation-PINN 是否比 Baseline 更稳。

---

# 12. 论文图表规划

第六节课之后，论文主实验应逐渐形成以下图表：

## Figure 1：方法示意图

展示：

```text
IC / BC / PDE residual / Conservation residual
                ↓
          Total loss
                ↓
          Neural network u_theta(x,t)
```

## Figure 2：Burgers 解场对比

建议包含：

- 参考解；
- Baseline PINN 预测解；
- Conservation-PINN 预测解；
- 两种方法的误差图。

## Figure 3：守恒量漂移曲线

横轴：$t$  
纵轴：$|M_\theta(t)-M_0|$

比较：

- Baseline PINN；
- Conservation-PINN。

## Figure 4：长时间外推时间切片

选择：

$$
t=0,\ 0.5,\ 1.0,\ 1.5,\ 2.0
$$

比较预测曲线和参考曲线。

## Table 1：定量指标汇总

| 方法 | L2 error | mean cons error | max cons error | final cons error |
|---|---:|---:|---:|---:|
| Baseline PINN | | | | |
| Conservation-PINN | | | | |

---

# 13. 课堂中要特别提醒学生的问题

## 13.1 守恒量更好，不代表点态误差一定更小

Conservation loss 主要约束全局积分量。它可能显著降低质量漂移，但不一定让每个点的误差都下降。

因此评价时不能只看一个指标。

## 13.2 Burgers 比线性平流更难训练

原因：

- 非线性项 $u u_x$ 增加训练难度；
- 粘性系数较小时，解会出现更尖锐结构；
- 长时间预测中误差更容易累积；
- 损失权重更敏感。

## 13.3 如果训练失败，优先检查这些点

1. `u_xx` 是否正确计算；
2. `requires_grad=True` 是否设置；
3. `lambda_cons` 是否过大；
4. `N_f` 是否太少；
5. 学习率是否过高；
6. 边界条件是否真正周期一致。

---

# 14. 60 分钟课堂安排

## 0-8 分钟：回顾第五节

- 线性平流方程；
- 守恒量 $M(t)$；
- Conservation loss；
- 长时间预测对比。

## 8-22 分钟：Burgers 方程数学背景

- 无粘 Burgers；
- 粘性 Burgers；
- 守恒律形式；
- 非线性通量 $F(u)=u^2/2$。

## 22-35 分钟：PINN 残差修改

重点讲：

```python
f = u_t + u * u_x - nu * u_xx
```

并解释每一项的物理含义。

## 35-48 分钟：Conservation-PINN 实验设计

- Baseline vs Conservation；
- $\lambda_{cons}$ 敏感性；
- 训练区间和外推区间；
- 守恒量误差曲线。

## 48-56 分钟：论文图表规划

让学生知道每张图是为了回答什么问题。

## 56-60 分钟：布置课后任务

明确下一步要改代码、跑实验、整理结果。

---

# 15. 课后任务

## 配套实验代码

本节配套代码为：

```text
section6_conservation_pinn_burgers.py
```

该文件使用 `# %%` 分隔教学单元，可在 VS Code 中像 Notebook 一样逐单元运行，也可在终端完整运行：

```bash
cd lessons
python -m pip install -r requirements.txt

# 小规模冒烟测试，确认环境和流程正确
python section6_conservation_pinn_burgers.py --quick

# 正式对比实验，每个模型训练 3000 轮
python section6_conservation_pinn_burgers.py --epochs 3000 --lambda-cons 1
```

实验结果保存在 `results/section6_burgers/`，包括训练曲线、解场与误差图、质量漂移曲线、长时间外推切片、定量指标 CSV 和模型权重。

做守恒权重敏感性实验时，可分别运行：

```bash
python section6_conservation_pinn_burgers.py --epochs 3000 --lambda-cons 0.1
python section6_conservation_pinn_burgers.py --epochs 3000 --lambda-cons 1
python section6_conservation_pinn_burgers.py --epochs 3000 --lambda-cons 10
```

不同权重的结果会自动保存到不同的 `results/section6_burgers_lambda_*/` 目录，避免相互覆盖。

## 必做

1. 复制第五节 Notebook，命名为 `section6_conservation_pinn_burgers.ipynb`。
2. 把 PDE 残差从线性平流改为 Burgers：

```python
f = u_t + u * u_x - nu * u_xx
```

3. 保留周期边界损失和守恒量损失。
4. 跑 Baseline PINN 与 Conservation-PINN。
5. 输出以下图：
   - 训练 loss 曲线；
   - 解场误差图；
   - 守恒量漂移曲线；
   - 长时间外推切片图。

## 选做

1. 比较 $\lambda_{cons}=0.1,1,10$；
2. 比较不同粘性系数 $\nu$；
3. 尝试 Adam + LBFGS 两阶段训练；
4. 用 FDM 生成参考解，而不是只依赖解析形式。

---

# 16. 一句话总结

第五节课让学生知道“守恒量可以写进 loss”。  
第六节课要让学生完成关键迁移：

> 从线性守恒律迁移到 Burgers 非线性守恒律，开始形成论文主实验。
