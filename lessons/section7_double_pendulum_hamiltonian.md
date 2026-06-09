# 第七节课课件：二维平面双摆的 Hamilton 方程与正则变换

> 课程定位：从前两节的 PDE 守恒约束转向有限维 Hamilton 系统，理解“同一个动力系统可以在不同正则坐标中求解”。

---

# 0. 本节课目标

本节课结束后，学生应该能够：

1. 从双摆的广义坐标、动量理解 Hamilton 状态变量；
2. 写出 Hamilton 正则方程；
3. 用自动微分计算 Hamiltonian 对状态变量的导数；
4. 直接在原始正则坐标 $(q,p)$ 中求解双摆；
5. 构造一个简单正则变换，并在新坐标 $(Q,P)$ 中求解；
6. 将新坐标的结果映射回原坐标，验证两种表示描述同一条物理轨迹；
7. 用 Hamiltonian 漂移评价数值积分质量。

---

# 1. 先回答：是不是有两个方法？

可以设计成两个对照方法：

| 方法 | 求解变量 | Hamiltonian |
|---|---|---|
| A：直接求解 | 原始正则变量 $(q_1,q_2,p_1,p_2)$ | $H(q,p)$ |
| B：正则变换后求解 | 新正则变量 $(Q_1,Q_2,P_1,P_2)$ | $K(Q,P)=H(q(Q,P),p(Q,P))$ |

但要准确理解：

> 这不是两个不同的物理模型，也不是正则变换后方程一定更容易。它们是同一个 Hamilton 系统的两种正则坐标表示。

如果正则变换、方程和数值积分都正确，把方法 B 的结果映射回原变量后，应与方法 A 基本一致。两者之间的差异主要来自时间离散误差。

---

# 2. 最简单的二维双摆模型

考虑无阻尼、无外力的平面双摆：

- 两个质点质量为 $m_1,m_2$；
- 两根无质量刚杆长度为 $l_1,l_2$；
- 重力加速度为 $g$；
- $q_1,q_2$ 均为杆相对竖直向下方向的**绝对角度**。

质点位置为：

$$
x_1=l_1\sin q_1,\qquad y_1=-l_1\cos q_1
$$

$$
x_2=l_1\sin q_1+l_2\sin q_2
$$

$$
y_2=-l_1\cos q_1-l_2\cos q_2
$$

---

# 3. 从 Lagrangian 到 Hamiltonian

令：

$$
\Delta=q_1-q_2
$$

动能为：

$$
T=
\frac12(m_1+m_2)l_1^2\dot q_1^2
+\frac12m_2l_2^2\dot q_2^2
+m_2l_1l_2\cos\Delta\,\dot q_1\dot q_2
$$

势能为：

$$
V=-(m_1+m_2)gl_1\cos q_1-m_2gl_2\cos q_2
$$

Lagrangian 为：

$$
L=T-V
$$

共轭动量定义为：

$$
p_i=\frac{\partial L}{\partial \dot q_i}
$$

即：

$$
p_1=(m_1+m_2)l_1^2\dot q_1
+m_2l_1l_2\cos\Delta\,\dot q_2
$$

$$
p_2=m_2l_2^2\dot q_2
+m_2l_1l_2\cos\Delta\,\dot q_1
$$

Hamiltonian 为：

$$
H(q,p)=
\frac{
m_2l_2^2p_1^2+(m_1+m_2)l_1^2p_2^2
-2m_2l_1l_2\cos\Delta\,p_1p_2
}{
2m_2l_1^2l_2^2\left(m_1+m_2\sin^2\Delta\right)
}
-(m_1+m_2)gl_1\cos q_1-m_2gl_2\cos q_2
$$

因为系统无阻尼且 Hamiltonian 不显含时间，理论上：

$$
\frac{dH}{dt}=0
$$

---

# 4. 方法 A：直接求解 Hamilton 正则方程

Hamilton 正则方程为：

$$
\dot q_i=\frac{\partial H}{\partial p_i},
\qquad
\dot p_i=-\frac{\partial H}{\partial q_i}
$$

写成向量形式：

$$
\dot z=J\nabla_z H,\qquad
z=(q_1,q_2,p_1,p_2)^T
$$

其中：

$$
J=
\begin{bmatrix}
0&I\\
-I&0
\end{bmatrix}
$$

实验中不手工展开复杂导数，而是使用 PyTorch 自动微分计算：

```python
grad_h = torch.autograd.grad(hamiltonian(z), z)[0]
dzdt = torch.cat([grad_h[2:], -grad_h[:2]])
```

---

# 5. 方法 B：在正则变换后求解

## 5.1 为什么不能只换角度？

若只定义相对角：

$$
Q_1=q_1,\qquad Q_2=q_2-q_1
$$

却仍把原来的 $p_1,p_2$ 当作新动量，通常不能保证变换是正则的。

正则变换必须同时正确变换坐标和共轭动量，使 Hamilton 方程的形式保持不变。

## 5.2 使用生成函数构造正则变换

选择第二类生成函数：

$$
F_2(q,P)=q_1P_1+(q_2-q_1)P_2
$$

根据：

$$
Q_i=\frac{\partial F_2}{\partial P_i},
\qquad
p_i=\frac{\partial F_2}{\partial q_i}
$$

得到：

$$
Q_1=q_1,\qquad Q_2=q_2-q_1
$$

$$
p_1=P_1-P_2,\qquad p_2=P_2
$$

因此逆变换为：

$$
q_1=Q_1,\qquad q_2=Q_1+Q_2
$$

$$
p_1=P_1-P_2,\qquad p_2=P_2
$$

新 Hamiltonian 为：

$$
K(Q,P)=H\left(Q_1,Q_1+Q_2,P_1-P_2,P_2\right)
$$

然后求解：

$$
\dot Q_i=\frac{\partial K}{\partial P_i},
\qquad
\dot P_i=-\frac{\partial K}{\partial Q_i}
$$

---

# 6. 两种方法应该比较什么？

## 6.1 映射回原变量后的轨迹误差

把方法 B 的结果映射回 $(q,p)$，计算：

$$
E_z(t)=\|z_A(t)-z_B(t)\|_2
$$

若代码正确并且时间步足够小，两者应非常接近。

## 6.2 Hamiltonian 漂移

定义：

$$
E_H(t)=|H(t)-H(0)|
$$

普通 RK4 并不是辛积分方法，所以长时间计算中 Hamiltonian 不会严格守恒，但减小时间步后漂移应下降。

## 6.3 双摆物理轨迹

绘制第二个质点 $(x_2,y_2)$ 的轨迹，以及若干时刻的双摆构型，可以直观检查两种坐标表示是否一致。

---

# 7. 配套实验代码

本节配套代码为：

```text
section7_double_pendulum_hamiltonian.py
```

运行方式：

```bash
cd lessons
python -m pip install -r requirements.txt

# 小规模快速检查
python section7_double_pendulum_hamiltonian.py --quick

# 正式实验
python section7_double_pendulum_hamiltonian.py --t-end 10 --dt 0.005
```

结果保存在 `results/section7_double_pendulum/`，包括：

- 两种方法的角度时间历程；
- 双摆末端轨迹；
- Hamiltonian 漂移；
- 映射回原变量后的状态误差；
- 定量指标 CSV。

---

# 8. 课堂实验步骤

1. 写出双摆 Hamiltonian；
2. 用自动微分实现 Hamilton 正则方程；
3. 用 RK4 直接积分原始变量；
4. 写出生成函数和正则变换；
5. 在新变量中积分，并映射回原变量；
6. 对比两条轨迹、Hamiltonian 漂移和状态误差；
7. 将时间步减半，观察误差是否下降。

---

# 9. 60 分钟课堂安排

## 0-10 分钟：从守恒律过渡到 Hamilton 系统

- PDE 中的质量守恒；
- ODE Hamilton 系统中的能量与辛结构；
- 状态变量为什么是 $(q,p)$ 而不是只用 $q$。

## 10-25 分钟：双摆 Hamiltonian

- 坐标、动量和 Hamiltonian；
- Hamilton 正则方程；
- 自动微分计算右端项。

## 25-40 分钟：正则变换

- 相对角度的直觉；
- 为什么动量也必须变换；
- 用生成函数构造正则变换。

## 40-53 分钟：运行对照实验

- 原变量直接求解；
- 新变量求解并映射回来；
- 检查两种结果是否一致。

## 53-60 分钟：讨论与作业

- RK4 为什么不严格保持 Hamiltonian；
- 正则变换和普通变量替换的区别；
- 下一步如何引出辛积分方法。

---

# 10. 课后任务

## 必做

1. 跑通两种正则坐标下的求解；
2. 将 `dt` 从 `0.01` 改为 `0.005` 和 `0.0025`；
3. 比较最大状态误差与最大 Hamiltonian 漂移；
4. 解释为什么两种方法映射回原变量后应该一致；
5. 解释为什么只变换角度、不变换动量不是完整的正则变换。

## 选做

1. 改变初始角度，观察规则运动与复杂运动；
2. 延长仿真时间，观察双摆对初值的敏感性；
3. 将 RK4 替换为辛积分方法；
4. 设计 Hamiltonian Neural Network，把 Hamiltonian 结构写入神经网络。

---

# 11. 一句话总结

> 正则变换不会改变物理系统，它改变的是描述 Hamilton 系统的正则坐标；正确变换后，Hamilton 方程的形式仍然保持不变。
