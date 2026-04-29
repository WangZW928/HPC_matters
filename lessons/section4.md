# 第四节课课件：残差神经网络与 Res-PINN（1D 热传导方程）

> 课程定位：在第三节最小 PINN 的基础上，不改变 PDE、初值条件、边界条件和采样口径，只升级网络结构，理解为什么“更深的网络未必更好训”，以及为什么残差连接可能更稳。

---

# 0. 本节课目标（45~60 分钟）

1. 学生能区分两种“残差”：
   - PDE 残差
   - ResNet 残差连接
2. 学生能写出残差块的基本数学形式。
3. 学生能理解：残差连接为什么常常让深层网络更容易优化。
4. 学生能把第三节的 PINN 改成 Res-PINN，并做公平对比实验。
5. 学生能用 `loss + L2 relative error + 可视化` 三类指标判断“是否真的更好”。

---

# 1. 先把问题范围说清楚

本节课**不改变 case 设置**，仍然使用第三节完全相同的 1D 热传导方程：

\[
\frac{\partial u}{\partial t}=\alpha\frac{\partial^2u}{\partial x^2},\quad x\in[0,1],\ t\in[0,1],\ \alpha=0.1
\]

初始条件仍为：

\[
u(x,0)=\sin(\pi x)
\]

边界条件仍为：

\[
u(0,t)=0,\quad u(1,t)=0
\]

解析解仍为：

\[
u(x,t)=e^{-\alpha\pi^2 t}\sin(\pi x)
\]

也就是说，本节课唯一要问的是：

> 当 PDE、IC、BC、采样点、loss 形式都不变时，仅仅把网络从普通 MLP 改成残差网络，训练表现会不会更稳？

---

# 2. 先纠正一个非常容易混淆的点

在 PINN 里，第三节已经出现过“残差”这个词：

\[
f_\theta(x,t)=u_t-\alpha u_{xx}
\]

这里的残差是 **PDE residual**，意思是：

> 网络预测是否满足方程。

而本节课讲的残差神经网络中的“残差”，指的是 **skip connection / shortcut connection**：

\[
h^{(\ell+1)} = h^{(\ell)} + F^{(\ell)}\bigl(h^{(\ell)}\bigr)
\]

这里的残差是：

> 让这一层不必直接学习一个全新的映射，而是学习“相对输入的增量修正”。

一句话区分：

- PINN 的 PDE 残差：约束**物理方程**
- ResNet 的残差连接：改善**网络优化**

---

# 3. 为什么普通深层 MLP 可能更难训？

第三节中我们用的是普通前馈网络：

\[
h^{(0)} = [x,t]
\]
\[
h^{(\ell+1)}=\sigma\left(W^{(\ell)}h^{(\ell)}+b^{(\ell)}\right)
\]
\[
u_\theta(x,t)=W^{(L)}h^{(L)}+b^{(L)}
\]

当层数变深时，常见问题有：

1. 梯度传播变弱或变乱；
2. 参数更新越来越敏感；
3. 训练初期 loss 下降慢；
4. 看起来“表达能力更强”，但优化器不一定找得到更好解。

对于 PINN，这个问题往往更明显，因为 PINN 不只做前向预测，还要计算：

\[
u_x,\quad u_t,\quad u_{xx}
\]

也就是说，网络不仅要“会拟合”，还要“对输入的导数表现稳定”。  
网络一旦太深、太难训，PDE loss 往往也更容易卡住。

---

# 4. 残差连接的核心数学形式

## 4.1 普通层

普通层可以写成：

\[
h^{(\ell+1)} = G^{(\ell)}\bigl(h^{(\ell)}\bigr)
\]

其中 \(G^{(\ell)}\) 往往是：

\[
G^{(\ell)}(h)=\sigma(Wh+b)
\]

## 4.2 残差层

残差层改成：

\[
h^{(\ell+1)} = h^{(\ell)} + F^{(\ell)}\bigl(h^{(\ell)}\bigr)
\]

其中 \(F^{(\ell)}\) 是一个较小的非线性变换，例如：

\[
F^{(\ell)}(h)=W_2\,\sigma(W_1 h+b_1)+b_2
\]

这表示：

> 输出 = 输入 + 修正项

如果修正项 \(F^{(\ell)}(h)\) 暂时学不到太多内容，网络至少还能保留输入主干，而不是把信息完全“重新洗一遍”。

---

# 5. 为什么残差网络常常更容易优化？

这是本节最重要的理论解释。

## 5.1 从“恒等映射更容易保留”理解

有时理想映射本来就接近恒等映射。  
普通层要直接学：

\[
H(h)
\]

而残差层改成学：

\[
F(h)=H(h)-h
\]

于是原问题变成：

\[
H(h)=h+F(h)
\]

如果目标映射本来就和输入差得不远，那么让网络学一个“小修正”通常比“从头学整个映射”更容易。

## 5.2 从梯度传播角度理解

设损失函数为 \(\mathcal L\)，残差层满足：

\[
h^{(\ell+1)} = h^{(\ell)} + F^{(\ell)}(h^{(\ell)})
\]

则链式法则给出：

\[
\frac{\partial \mathcal L}{\partial h^{(\ell)}} =
\frac{\partial \mathcal L}{\partial h^{(\ell+1)}}
\left(I+\frac{\partial F^{(\ell)}}{\partial h^{(\ell)}}\right)
\]

更准确地写成右乘形式可理解为：

\[
\nabla_{h^{(\ell)}} \mathcal L=
\nabla_{h^{(\ell+1)}} \mathcal L
\cdot
\left(I+\frac{\partial F^{(\ell)}}{\partial h^{(\ell)}}\right)
\]

这里最关键的是出现了单位映射 \(I\)。  
它意味着即使 \(F^{(\ell)}\) 的梯度部分不理想，梯度仍然有一条“直接通路”可以往前传。

相比之下，普通网络层层相乘更像：

\[
\nabla_{h^{(\ell)}} \mathcal L=
\nabla_{h^{(L)}} \mathcal L
\prod_{k=\ell}^{L-1}
\frac{\partial h^{(k+1)}}{\partial h^{(k)}}
\]

多层连乘后，容易出现：

- 某些方向越来越小
- 某些方向越来越大
- 优化过程变得不稳定

## 5.3 放到 PINN 语境下的理解

对于 PINN，我们不仅关心：

\[
u_\theta(x,t)
\]

还关心它的导数：

\[
u_t,\ u_x,\ u_{xx}
\]

残差连接不能保证“导数一定更准”，但它常常能让深层网络更容易进入一个可训练区间，从而使：

- 总 loss 更平滑地下去；
- PDE loss 更不容易长期停滞；
- 更深网络的收益更有机会体现出来。

所以严谨说法应该是：

> 残差连接的主要价值是改善优化难度，而不是数学上保证误差一定更小。

---

# 6. 第三节 PINN 如何扩展为 Res-PINN？

第三节的 PINN 结构是：

\[
(x,t)\longrightarrow \text{MLP}\longrightarrow u_\theta(x,t)
\]

现在改成：

\[
(x,t)\longrightarrow \text{Input Layer}
\longrightarrow \text{Residual Blocks}
\longrightarrow \text{Output Layer}
\longrightarrow u_\theta(x,t)
\]

一个最简单的残差块可写为：

\[
z^{(\ell)}=\sigma\left(W_1^{(\ell)}h^{(\ell)}+b_1^{(\ell)}\right)
\]
\[
F^{(\ell)}(h^{(\ell)})=W_2^{(\ell)}z^{(\ell)}+b_2^{(\ell)}
\]
\[
h^{(\ell+1)}=h^{(\ell)}+F^{(\ell)}(h^{(\ell)})
\]

如果隐藏维度一致，就可以直接相加。  
这也是课堂实现最省事、最清晰的一种方式。

---

# 7. Res-PINN 的损失函数不变

这一点非常重要。  
我们**不改物理约束，不改 loss 定义**，仍然用：

\[
\mathcal L=
\lambda_{ic}\mathcal L_{ic}+
\lambda_{bc}\mathcal L_{bc}+
\lambda_f\mathcal L_f
\]

其中：

\[
\mathcal L_{ic}=\frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}}
\left|u_\theta(x_i,0)-\sin(\pi x_i)\right|^2
\]

\[
\mathcal L_{bc}=
\frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}}
\left|u_\theta(0,t_i)-0\right|^2
+
\frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}}
\left|u_\theta(1,t_i)-0\right|^2
\]

\[
\mathcal L_f=
\frac{1}{N_f}\sum_{j=1}^{N_f}
\left|u_t(x_f^j,t_f^j)-\alpha u_{xx}(x_f^j,t_f^j)\right|^2
\]

因此，本节课的实验是一个标准的“控制变量”：

- PDE 不变
- IC/BC 不变
- 采样点不变
- loss 不变
- 优化器不变
- 主要变化：网络结构

---

# 8. 第四节最推荐的实验设计

建议至少做三组：

## 实验 A：普通 PINN（浅层）

- 目的：作为第三节基线
- 预期：能正常收敛

## 实验 B：普通 PINN（深层）

- 目的：故意把网络加深，观察优化难度
- 预期：loss 可能下降变慢，或者误差未必继续变好

## 实验 C：Res-PINN（同等深度）

- 目的：在同样深度下比较残差连接是否更稳
- 预期：训练曲线更平滑，深层结构更容易训

课堂上可以明确告诉学生：

> 我们不是为了“证明 ResNet 永远赢”，而是为了训练科研判断力：面对一个改动，如何做公平实验，如何解释结果。

---

# 9. 课堂上应该看哪些结果？

不要只看总 loss。  
至少看下面四类结果：

## 9.1 总 loss 曲线

看整体是否下降更快、更平稳。

## 9.2 三项子损失

- \(\mathcal L_{ic}\)
- \(\mathcal L_{bc}\)
- \(\mathcal L_f\)

看是否只是某一项改善，而其他项被牺牲。

## 9.3 误差指标

例如：

\[
L_2\ \text{relative error}=
\frac{\|u_{\text{pred}}-u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}
\]

这比只看 loss 更接近“解算质量”。

## 9.4 可视化

- 预测解热力图
- 真解热力图
- 绝对误差热力图
- 时间切片曲线

特别要关注：

- 边界附近是否更稳
- 后期衰减趋势是否更准
- 误差是否更集中或更分散

---

# 10. 一个老师可直接讲的“理论 + 实验”串联版本

你可以这样讲：

> 第三节我们已经知道，PINN 的核心是把 PDE、初值和边界条件写进 loss。  
> 但当网络加深后，模型虽然更有表达能力，却可能更难优化。  
> 残差网络的思路不是改变 PDE，也不是改变 loss，而是在网络内部增加一条 shortcut，让每一层只学习“修正量”。  
> 这样做常常能让梯度更容易传播，训练更稳定。  
> 所以第四节我们做的不是“换一个更花哨的模型”，而是在同一个热传导方程上做一个控制变量实验，检验残差连接对 PINN 训练是否真的有帮助。

---

# 11. 课堂代码实现提示

## 11.1 最小残差块

```python
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = self.act(residual + out)
        return out
```

## 11.2 Res-PINN 主体

```python
class ResPINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.act = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        h = self.act(self.input_layer(xt))
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)
```

说明：

- 输入输出接口与第三节保持一致；
- 自动微分部分不用改；
- PDE residual 写法不用改；
- 训练循环基本不用改。

---

# 12. 本节课最值得学生记住的 4 句话

1. PINN 的 PDE 残差，和 ResNet 的残差连接，不是同一个“残差”。
2. 残差连接主要改善的是优化难度，而不是直接修改物理模型。
3. 在深层网络里，残差连接常常让梯度传播更顺畅。
4. 判断 Res-PINN 是否有效，不能只看总 loss，要看误差指标和可视化。

---

# 13. 课后作业建议

## 必做

1. 跑通 `section4` 对应 Notebook。
2. 比较普通 PINN 与 Res-PINN 的训练曲线。
3. 记录两者的 `L2 relative error`。
4. 用自己的话解释：为什么残差连接可能让深层网络更容易训。

## 选做

1. 比较不同残差块数量的影响。
2. 尝试普通深层 MLP 与 Res-PINN 的同深度公平对比。
3. 记录三项子损失的变化，分析“谁改善了、谁没有改善”。

---

# 14. 配套文件

- 本课讲义：`section4.md`
- 本课代码（Jupyter）：`section4_respinn_heat1d.ipynb`

---

# 15. 一句话收束

第四节课的重点不是“残差网络一定更强”，而是：

> 当 PINN 变深后，残差连接是一种非常值得尝试的优化手段；我们要学会用数学解释它、也学会用实验检验它。
