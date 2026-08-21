# VFS 算法详解

> **范围校正（2026-08-20）**：本文原有 level-set、两相物性、表面张力、波浪与风机段落可作为“历史 VFS 手册背景”的数学参考，但当前 `vwis2.0/` 源码树没有相函数、rho/mu 字段、界面推进器或相应调用链，不能视为本版本实现。本仓库可由代码证实的是固定曲线网格上的单相不可压缩投影、LES、sharp-interface IBM 和可选 FSI。引用本文公式时须标注“手册背景”或“源码已确认”。

本文基于 `VFS-manual.txt` 第 2 章的数值算法描述，并结合 `_Docs/00_整体架构.md` 到 `_Docs/05_可移植性与GPU优化分析.md` 对当前代码实现的整理，系统说明 VFS-Wind 的主要数学模型与离散算法。手册中的公式较为紧凑，本文从守恒律、坐标变换和物理源项出发补全推导，并说明各项在代码中的含义。

## 1. 控制方程

### 1.1 从不可压缩质量守恒到曲线坐标连续性方程

对不可压缩两相流，虽然密度在空气和水之间变化，但每一相内部采用不可压缩速度约束。局部体积守恒为：

$$
\nabla \cdot \mathbf{u}=0,
$$

其中 $\mathbf{u}=(u_1,u_2,u_3)$ 是笛卡尔速度。令物理坐标为 $\mathbf{x}=(x_1,x_2,x_3)$，计算坐标为 $\boldsymbol{\xi}=(\xi^1,\xi^2,\xi^3)$。坐标映射写作：

$$
\mathbf{x}=\mathbf{x}(\xi^1,\xi^2,\xi^3).
$$

为避免源码中的 `Aj` 与通常文献记号混淆，以下定义

$$
J_\xi=\det\!\left(\frac{\partial\boldsymbol\xi}{\partial\mathbf x}\right),
\qquad \mathcal J=\det\!\left(\frac{\partial\mathbf x}{\partial\boldsymbol\xi}\right)=\frac1{J_\xi}.
$$

legacy `Aj` 存 $J_\xi$，不是物理体积；`CurvGrid::FormMetrics` 先算
$\mathcal J$ 再取倒数。因而

$$
dV=\mathcal J\,d\xi^1d\xi^2d\xi^3=\frac{1}{J_\xi}d\xi^1d\xi^2d\xi^3.
$$

对任意控制体 $\Omega$，体积守恒积分形式为：

$$
\int_{\partial \Omega} \mathbf{u}\cdot \mathbf{n}\,dS=0.
$$

曲线网格中，一个 $\xi^i=\text{const}$ 面的有向面积余因子为

$$
\mathbf a^i=\mathcal J\nabla\xi^i=\frac{1}{J_\xi}\nabla\xi^i.
$$

必须区分逆变速度与守恒面通量：

$$
q^i=\mathbf u\cdot\nabla\xi^i=\xi^i_lu_l,
\qquad
\widehat U^i=\mathbf u\cdot\mathbf a^i=\frac{q^i}{J_\xi}.
$$

其中 $\xi^i_l=\partial\xi^i/\partial x_l$。将笛卡尔散度变换到计算空间：

$$
\nabla\cdot\mathbf u
=J_\xi\frac{\partial\widehat U^i}{\partial\xi^i}
=J_\xi\frac{\partial}{\partial\xi^i}\left(\frac{q^i}{J_\xi}\right).
$$

有限体积离散直接对 $\widehat U^i$ 作相邻面差。手册常把
$\widehat U^i$ 简写成 $U^i$，于是 Eq. 2.1 写作：

$$
J_\xi\frac{\partial U^i}{\partial \xi^i}=0.
\tag{2.1}
$$

这里简写的 $U^i$ 是 $\widehat U^i$，不是 $q^i$。源码证据是：
`FormMetrics` 用两条局部边的叉积生成 `Csi/Eta/Zet`（面积余因子），
`Contra2Cart` 解 $U^i=\mathbf u\cdot\mathbf a^i$，而
`CalculateDivergence` 计算 `Aj × 面通量差`。因此 `Ucont` 已吸收面面积/Jacobian
因子；把它当作普通 face-normal velocity 会造成量纲错误。

物理解释：连续性方程不直接约束笛卡尔速度点值，而约束每个曲线网格单元的净体积流量。这样做使压力投影天然保持有限体积意义上的质量守恒，也使曲线网格和复杂地形边界能够共用同一通量框架。

### 1.2 从动量守恒到曲线坐标动量方程

不可压缩两相流的笛卡尔动量方程可从控制体动量守恒出发：

$$
\frac{\partial (\rho u_l)}{\partial t}
+\frac{\partial(\rho u_j u_l)}{\partial x_j}
=
-\frac{\partial p}{\partial x_l}
+\frac{\partial \sigma^{v}_{lj}}{\partial x_j}
-\frac{\partial \tau_{lj}}{\partial x_j}
+f^{\sigma}_l
+f^g_l,
$$

其中 $\sigma^v_{lj}$ 是分子粘性应力，$\tau_{lj}$ 是 LES 过滤后产生的亚格子应力，$f^\sigma_l$ 是表面张力体力，$f^g_l$ 是重力项。对不可压缩流，速度散度为零，且 VFS 以单位质量形式推进速度，因此除以局部密度 $\rho(\phi)$：

$$
\frac{\partial u_l}{\partial t}
+u_j\frac{\partial u_l}{\partial x_j}
=
-\frac{1}{\rho}\frac{\partial p}{\partial x_l}
+\frac{1}{\rho}\frac{\partial \sigma^v_{lj}}{\partial x_j}
-\frac{1}{\rho}\frac{\partial \tau_{lj}}{\partial x_j}
+\frac{1}{\rho}f^\sigma_l
+\frac{1}{\rho}f^g_l.
$$

VFS 的主未知量不是 $u_l$，而是曲线方向的守恒通量
$\widehat U^i=a^i_lu_l$。逆变速度 $q^i=\xi^i_lu_l$ 只用于连续坐标变换；二者不能混写：

$$
\widehat U^i=\frac{\xi^i_l}{J_\xi}u_l=a^i_lu_l.
$$

若网格固定，面积余因子 $a^i_l$ 不随时间变，于是：

$$
\frac{\partial \widehat U^i}{\partial t}
=a^i_l\frac{\partial u_l}{\partial t}.
$$

下面只给出连续变换的守恒核心；legacy `RHSSolver/Integrator` 对各项还使用
面/中心 metric 与离散缩放，不能仅凭手册紧凑记号补出一个已验证的离散 Eq. 2.2。

#### 1.2.1 对流项

笛卡尔保守对流通量为：

$$
\frac{\partial (u_j u_l)}{\partial x_j}.
$$

对不可压缩流有：

$$
\frac{\partial (u_j u_l)}{\partial x_j}
=u_j\frac{\partial u_l}{\partial x_j}
+u_l\frac{\partial u_j}{\partial x_j}
=u_j\frac{\partial u_l}{\partial x_j}.
$$

坐标变换链式法则给出：

$$
\frac{\partial u_l}{\partial x_j}
=\frac{\partial \xi^m}{\partial x_j}
\frac{\partial u_l}{\partial \xi^m}
=\xi^m_j\frac{\partial u_l}{\partial \xi^m}.
$$

将速度沿计算坐标投影，$u_j\xi^m_j=q^m$，得到非保守形式：

$$
u_j\frac{\partial u_l}{\partial x_j}
=q^m\frac{\partial u_l}{\partial \xi^m}.
$$

不可压缩时，对流项的计算空间守恒恒等式是：

$$
\frac{\partial(u_ju_l)}{\partial x_j}
=J_\xi\frac{\partial}{\partial\xi^m}\left(\widehat U^m u_l\right).
$$

物理解释：$\widehat U^m$ 表示穿过 $\xi^m$ 面的体积通量，$u_l$ 是被输运的笛卡尔动量分量。VFS 先在笛卡尔分量上构造对流，再用曲线度量投影回 `Ucont`；额外的 $J_\xi$ 或 $\mathcal J$ 因子必须按具体离散位置核对，不能把 $q^m$ 与 $\widehat U^m$ 互换。

#### 1.2.2 粘性扩散项

牛顿流体不可压缩粘性应力为：

$$
\sigma^v_{lj}
=\mu(\phi)
\left(
\frac{\partial u_l}{\partial x_j}
+\frac{\partial u_j}{\partial x_l}
\right).
$$

若采用标准 Laplacian 简化，且粘度常数，则：

$$
\frac{\partial \sigma^v_{lj}}{\partial x_j}
=\mu \frac{\partial^2 u_l}{\partial x_j\partial x_j}.
$$

两相流中 $\mu(\phi)$ 随界面平滑变化，所以更稳妥的守恒扩散形式是：

$$
\frac{\partial}{\partial x_j}
\left(
\mu(\phi)\frac{\partial u_l}{\partial x_j}
\right).
$$

将梯度变换到计算空间：

$$
\frac{\partial u_l}{\partial x_j}
=\xi^k_j\frac{\partial u_l}{\partial \xi^k}.
$$

散度变换为：

$$
\frac{\partial q_j}{\partial x_j}
=J\frac{\partial}{\partial \xi^m}
\left(\frac{\xi^m_j q_j}{J}\right).
$$

令 $q_j=\mu \xi^k_j\partial u_l/\partial \xi^k$，则：

$$
\frac{\partial}{\partial x_j}
\left(
\mu\frac{\partial u_l}{\partial x_j}
\right)
=
J\frac{\partial}{\partial \xi^m}
\left(
\frac{\mu \xi^m_j\xi^k_j}{J}
\frac{\partial u_l}{\partial \xi^k}
\right).
$$

再乘以投影度量 $\xi^i_l$、除以 $\rho Re$，得到手册 Eq. 2.2 中的扩散结构：

$$
\mathcal{D}^i
=
\frac{1}{\rho(\phi)Re}
\frac{\partial}{\partial \xi^m}
\left(
\frac{\mu(\phi)\xi^m_l\xi^k_l}{J}
\frac{\partial u_l}{\partial \xi^k}
\right),
$$

其中指标在手册排版中较紧凑，核心含义是 metric tensor $g^{mk}=\xi^m_l\xi^k_l$ 把物理空间 Laplacian 转换为计算空间中的各向异性扩散算子。

物理解释：曲线网格上的“相邻计算点距离”并不等于物理距离，扩散必须通过度量张量修正方向和尺度。网格拉伸越强，$g^{mk}$ 的非对角项越重要，Poisson 和粘性模板也会从 7 点变成含交叉项的 19 点结构。

#### 1.2.3 压力梯度项

笛卡尔压力力为：

$$
-\frac{1}{\rho}\frac{\partial p}{\partial x_l}.
$$

用链式法则：

$$
\frac{\partial p}{\partial x_l}
=\xi^m_l\frac{\partial p}{\partial \xi^m}.
$$

投影到逆变方向后，压力项可写为：

$$
\mathcal{P}^i
=-\frac{1}{\rho(\phi)}
\xi^i_l\xi^m_l
\frac{\partial p}{\partial \xi^m}.
$$

手册写为散度相容形式：

$$
\mathcal{P}^i
=
-\frac{1}{\rho(\phi)}
\frac{\partial}{\partial \xi^j}
\left(
\frac{\xi^j_l p}{J}
\right),
$$

其离散目的不是把压力当作普通标量通量，而是让压力梯度与后续 Poisson 投影使用同一套 metric/Jacobian，保证投影后满足 Eq. 2.1。

物理解释：压力不是热力学状态方程给出，而是不可压缩约束的 Lagrange 乘子。压力梯度的离散一致性决定了投影法能否消除中间速度散度。

#### 1.2.4 LES 亚格子项

LES 对 Navier-Stokes 方程做空间过滤：

$$
\overline{u_i u_j}
=\bar{u}_i\bar{u}_j
+\left(\overline{u_i u_j}-\bar{u}_i\bar{u}_j\right).
$$

定义亚格子应力：

$$
\tau_{ij}
=\overline{u_i u_j}-\bar{u}_i\bar{u}_j.
$$

过滤后的动量方程中，对流项分解为可解析尺度对流和未解析尺度动量通量，未解析部分以散度形式进入：

$$
-\frac{1}{\rho}\frac{\partial \tau_{lj}}{\partial x_j}.
$$

坐标变换后，VFS 手册 Eq. 2.2 中写作：

$$
\mathcal{S}^i
=-\frac{1}{\rho(\phi)}
\frac{\partial \tau_{lj}}{\partial \xi^j},
$$

实际实现会结合曲线坐标速度梯度、中心度量和 `Nu_t` 构造 SGS 粘性通量。物理上该项代表网格尺度以下涡旋对已解析流场的动量抽取和回馈，通常表现为附加湍流粘性。

#### 1.2.5 表面张力项

自由面为 level set 零等值面：

$$
\Gamma=\{\mathbf{x}\mid \phi(\mathbf{x},t)=0\}.
$$

界面单位法向取：

$$
\mathbf{n}=\frac{\nabla \phi}{|\nabla \phi|}.
$$

曲率为：

$$
\kappa=\nabla\cdot\mathbf{n}.
$$

连续表面力模型把界面上的表面张力 $\sigma\kappa\mathbf{n}\delta_\Gamma$ 转化为窄带体力。由于 smoothed Heaviside $h(\phi)$ 的梯度近似界面 delta：

$$
\nabla h(\phi)=\delta_\epsilon(\phi)\nabla \phi
\approx \mathbf{n}\delta_\Gamma,
$$

所以无量纲表面张力体力为：

$$
\mathbf{f}^{\sigma}
=-\frac{\kappa}{We^2}\nabla h(\phi).
$$

第 $j$ 个笛卡尔分量为：

$$
f^\sigma_j
=-\frac{\kappa}{We^2}
\frac{\partial h(\phi)}{\partial x_j}.
$$

除以局部密度后进入动量方程：

$$
\mathcal{T}_j
=-\frac{\kappa}{\rho(\phi)We^2}
\frac{\partial h(\phi)}{\partial x_j}.
$$

物理解释：曲率为正的界面会产生指向曲率中心的恢复力，抑制短波界面扰动。$We$ 越小，表面张力相对惯性越强。

#### 1.2.6 重力项

以特征速度 $U$ 和长度 $L$ 无量纲化时，重力加速度项 $\mathbf{g}$ 与惯性尺度 $U^2/L$ 的比值为：

$$
\frac{g}{U^2/L}=\frac{gL}{U^2}=\frac{1}{Fr^2}.
$$

若竖直方向取第 2 个坐标方向，重力源项可写为：

$$
\mathcal{G}_i=\frac{\delta_{i2}}{Fr^2},
$$

其中 $\delta_{i2}$ 是 Kronecker delta。符号取决于坐标轴正方向和压力是否包含静水压；手册 Eq. 2.2 使用 $+\delta_{i2}/Fr^2$ 的约定。

#### 1.2.7 Eq. 2.2 的综合形式

将上述各项合并，VFS 求解的曲线坐标动量方程可概括为：

$$
\frac{1}{J}\frac{\partial U^i}{\partial t}
=
-\frac{\xi^i_l}{J}
\frac{\partial}{\partial \xi^j}(U^j u_l)
+\frac{1}{\rho(\phi)Re}
\frac{\partial}{\partial \xi^j}
\left(
\frac{\mu(\phi)\xi^j_l\xi^k_l}{J}
\frac{\partial u_l}{\partial \xi^k}
\right)
-\frac{1}{\rho(\phi)}
\nabla_{\xi}p
-\frac{1}{\rho(\phi)}\nabla_{\xi}\cdot \boldsymbol{\tau}
-\frac{\kappa}{\rho(\phi)We^2}\nabla h(\phi)
+\frac{\delta_{i2}}{Fr^2}.
\tag{2.2}
$$

手册中的排版包含具体指标形式，本文保留其物理和坐标变换结构。当前代码实现中，`RHSSolver` 负责对流、粘性、压力梯度和 LES 项，`PoissonSolver` 通过压力修正 enforcing Eq. 2.1，`UData::Contra2Cart` 在 `Ucont` 与 `Ucat` 间转换。

### 1.3 Level set、物性平滑与 Heaviside 函数

VFS 使用 signed distance level set 函数描述两相界面：

$$
\phi(\mathbf{x},t)
\begin{cases}
>0, & \text{水相},\\
=0, & \text{空气/水界面},\\
<0, & \text{空气相}.
\end{cases}
$$

若 $\phi$ 是严格 signed distance，则：

$$
|\nabla \phi|=1.
$$

两相密度和粘度在界面厚度 $2\epsilon$ 内平滑过渡。Eq. 2.4 和 Eq. 2.5 为：

$$
\rho(\phi)=\rho_{air}+(\rho_{water}-\rho_{air})h(\phi),
\tag{2.4}
$$

$$
\mu(\phi)=\mu_{air}+(\mu_{water}-\mu_{air})h(\phi).
\tag{2.5}
$$

构造 smoothed Heaviside 的原则是：

1. 远离界面时保持分段常数，避免污染各相内部物性。
2. 在 $[-\epsilon,\epsilon]$ 内连续可导，使密度、粘度和表面张力体力不产生数值尖峰。
3. 导数 $\delta_\epsilon(\phi)=h'(\phi)$ 可作为平滑 delta 函数。

手册 Eq. 2.6 为：

$$
h(\phi)=
\begin{cases}
0, & \phi<-\epsilon,\\
\frac{1}{2}+\frac{\phi}{2\epsilon}
+\frac{1}{2\pi}\sin\left(\frac{\pi\phi}{\epsilon}\right),
& -\epsilon\le \phi\le \epsilon,\\
1, & \epsilon<\phi.
\end{cases}
\tag{2.6}
$$

中间段的导数为：

$$
h'(\phi)
=\frac{1}{2\epsilon}
+\frac{1}{2\epsilon}\cos\left(\frac{\pi\phi}{\epsilon}\right)
=\frac{1}{2\epsilon}
\left[
1+\cos\left(\frac{\pi\phi}{\epsilon}\right)
\right].
$$

在 $\phi=\pm\epsilon$ 处，$h'(\phi)=0$，与外侧常数段平滑连接；在 $\phi=0$ 处，$h'(0)=1/\epsilon$，表示界面中心处 delta 近似最强。

Level set 随流体运动满足纯对流方程：

$$
\frac{\partial \phi}{\partial t}
+\mathbf{u}\cdot\nabla \phi=0.
$$

变换到曲线坐标为手册 Eq. 2.7：

$$
\frac{1}{J}\frac{\partial \phi}{\partial t}
+U^j\frac{\partial \phi}{\partial \xi^j}=0.
\tag{2.7}
$$

对流后 $\phi$ 往往不再满足 $|\nabla\phi|=1$，所以代码需要重初始化。重初始化的目标不是移动零等值面，而是恢复 signed distance 质量，使物性过渡带厚度和表面张力计算保持稳定。

### 1.4 无量纲化与 Re、Fr、We

选择特征长度 $L$、速度 $U$、水相密度 $\rho_w$、水相粘度 $\mu_w$。定义无量纲变量：

$$
\mathbf{x}=L\mathbf{x}^*, \qquad
t=\frac{L}{U}t^*, \qquad
\mathbf{u}=U\mathbf{u}^*, \qquad
p=\rho_w U^2 p^*.
$$

惯性项尺度为：

$$
\rho_w\frac{U^2}{L}.
$$

粘性项尺度为：

$$
\mu_w\frac{U}{L^2}.
$$

两者比值给出 Reynolds 数：

$$
Re=\frac{\rho_w U L}{\mu_w}.
$$

重力项尺度为 $\rho_w g$，与惯性尺度比值为：

$$
\frac{\rho_w g}{\rho_w U^2/L}
=\frac{gL}{U^2}
=\frac{1}{Fr^2},
\qquad
Fr=\frac{U}{\sqrt{gL}}.
$$

表面张力力密度尺度可估为 $\sigma/L^2$，与惯性尺度比值为：

$$
\frac{\sigma/L^2}{\rho_w U^2/L}
=\frac{\sigma}{\rho_w U^2 L}
=\frac{1}{We},
$$

手册定义为：

$$
We=\frac{U\sqrt{\rho_w L}}{\sqrt{\sigma}},
$$

因此 $We^2=\rho_w U^2L/\sigma$，表面张力项写成 $1/We^2$。手册 Eq. 2.3 为：

$$
Re=\frac{UL\rho_{water}}{\mu_{water}},
\qquad
Fr=\frac{U}{\sqrt{gL}},
\qquad
We=U\sqrt{\frac{\rho_{water}L}{\sigma}}.
\tag{2.3}
$$

物理解释：

- $Re$ 衡量惯性与粘性之比，越大越容易出现湍流和薄边界层。
- $Fr$ 衡量惯性与重力波效应之比，控制自由面波速和浮力主导程度。
- $We$ 衡量惯性与表面张力之比，越大表面张力越弱，界面更容易被惯性拉伸破碎。

### 1.5 分步投影时间推进

VFS 使用 fractional step 方法。先忽略新时刻不可压缩约束，求中间通量 $U^{i,*}$：

$$
\frac{1}{J}\frac{U^{i,*}-U^{i,n}}{\Delta t}
=P(p^n,\phi^n)
+\frac{1}{2}
\left[
F(U^*,u^*,\phi^{n+1})
+F(U^n,u^n,\phi^n)
\right],
\tag{2.8}
$$

其中 $F$ 不含压力修正项，$P$ 表示旧压力梯度。为了使修正后速度满足连续性，设：

$$
U^{i,n+1}=U^{i,*}-\Delta t\,\mathcal{G}^i(\Pi),
$$

其中 $\Pi$ 是压力修正。代入连续性：

$$
J\frac{\partial U^{i,n+1}}{\partial \xi^i}=0,
$$

得到：

$$
J\frac{\partial}{\partial \xi^i}
\left[
U^{i,*}-\Delta t\,\mathcal{G}^i(\Pi)
\right]=0.
$$

整理为 Poisson 方程：

$$
-J\frac{\partial}{\partial \xi^i}
\left[
\frac{1}{\rho(\phi)}
\frac{\xi^i_l}{J}
\frac{\partial}{\partial \xi^j}
\left(
\frac{\xi^j_l \Pi}{J}
\right)
\right]
=
\frac{1}{\Delta t}
J\frac{\partial U^{j,*}}{\partial \xi^j}.
\tag{2.9}
$$

求解后：

$$
p^{n+1}=p^n+\Pi,
\tag{2.10}
$$

$$
U^{i,n+1}
=U^{i,*}
-J\Delta t
\frac{1}{\rho(\phi)}
\frac{\xi^i_l}{J}
\frac{\partial}{\partial \xi^j}
\left(
\frac{\xi^j_l\Pi}{J}
\right).
\tag{2.11}
$$

代码层面，动量预测由 PETSc SNES 非线性求解，压力 Poisson 由 HYPRE GMRES/PCG + BoomerAMG 求解，投影后再把 `Ucont` 转换为 `Ucat`。

## 2. CURVIB 方法

### 2.1 Sharp-interface IB 的基本思想

传统贴体网格要求流体网格边界与固体表面重合。复杂运动物体会导致网格生成、重构和质量控制非常困难。浸入边界方法改用固定背景网格，把固体表面网格叠加到流体网格上。

VFS 的 CURVIB 是 sharp-interface IB：固体边界不被扩散成厚体力带，而是在流固界面附近通过插值重构速度边界条件。它与连续力型 IBM 的核心差别是：

$$
\text{连续力型：在若干网格宽度内分布体力}
\quad
\text{sharp-interface：在近界面网格点重构边界状态}.
$$

CURVIB 的特别之处是背景网格可以是广义曲线网格。简单边界，例如地形、河床或规则外边界，可以由曲线网格贴合；复杂运动物体，例如叶片、圆柱、浮体，则由独立三角面 IBM 几何处理。

### 2.2 网格节点分类

结构表面由非结构三角面网格给出。对每个背景网格节点或单元中心，判断其相对固体的位置：

1. **固体节点**：位于物体内部，从流体计算域 blank out。代码中 `Nvert` 通常用大于流体阈值的值标记，例如 `Nvert=4`。
2. **IB 节点**：位于流体内，但紧邻固体表面，需要重构速度边界条件。代码文档中常见 `Nvert=2`。
3. **流体节点**：远离固体边界，直接求解控制方程。

几何判定通常包括：

$$
\text{point} \xrightarrow{\text{ray-triangle intersection}}
\text{inside/outside}.
$$

射线法的基础是拓扑奇偶性：从点发出一条射线，若与闭合三角面相交次数为奇数，则点在内部；偶数则在外部。为降低成本，VFS 为三角面建立空间桶，只在候选桶中做三角相交检测。

薄体或网格分辨率不足时，单元中心内外判定可能漏掉穿越单元的薄几何，因此代码还提供 thin-body 相关检测，通过单元边与三角面的穿越关系修正 `Nvert`。

### 2.3 IB 节点速度重构

对一个 IB 节点 $P$，寻找最近的固体表面点 $B$，沿壁面法向向流体侧取 image point $I$。设 $s_b=|PB|$，$s_c=|BI|$。边界无滑移条件给定表面速度 $\mathbf{u}_B$，流体侧 image point 速度 $\mathbf{u}_I$ 由周围流体点插值得到。

线性重构假定法向速度剖面在 $B$ 到 $I$ 之间线性：

$$
\mathbf{u}(s)=\mathbf{u}_B
+\frac{s}{s_c}(\mathbf{u}_I-\mathbf{u}_B).
$$

IB 节点位于固体侧或近界面位置，根据几何定义可写成：

$$
\mathbf{u}_P
=\mathbf{u}_B
+\frac{s_P}{s_c}(\mathbf{u}_I-\mathbf{u}_B).
$$

若需要直接由 $P$、$B$、$I$ 的相对距离构造 ghost value，常见镜像形式为：

$$
\mathbf{u}_P
=\left(1+\frac{s_b}{s_c}\right)\mathbf{u}_B
-\frac{s_b}{s_c}\mathbf{u}_I.
$$

二次重构则再加入第二个流体侧点 $I_2$，构造：

$$
\mathbf{u}(s)=\mathbf{a}s^2+\mathbf{b}s+\mathbf{c},
$$

并用：

$$
\mathbf{u}(0)=\mathbf{u}_B,\qquad
\mathbf{u}(s_c)=\mathbf{u}_{I_1},\qquad
\mathbf{u}(s_2)=\mathbf{u}_{I_2}
$$

求出系数，再外推到 IB 节点。二次格式在低 Reynolds 数、边界层被解析时能给出更高精度；高 Reynolds 数粗网格 LES 中，近壁速度剖面不满足简单多项式，此时 VFS 使用壁函数模型替代线性/二次插值。

image point 速度由背景网格插值得到。若 $I$ 落在某个曲线网格单元内，可用三线性插值：

$$
\mathbf{u}_I
=\sum_{m=1}^{8} w_m \mathbf{u}_m,
\qquad
\sum_{m=1}^{8}w_m=1.
$$

代码文档显示当前实现会为 IB 点保存插值点索引、三角形重心权重、表面速度和距离比例，`IBMInterpolationAdvanced` 据此重构 `Ucat`，再由边界工具修正 `Ucont` 面通量。

### 2.4 曲线坐标中的 IB 处理

CURVIB 在物理空间中做几何关系判断，因为固体三角面、法向和距离都具有笛卡尔几何含义；但流体方程在计算空间中离散。因此它需要两个一致性条件：

1. **几何搜索使用物理坐标**：最近三角面、法向、交点、image point 都以 $\mathbf{x}$ 计算。
2. **边界条件回写到曲线变量**：重构得到 $\mathbf{u}$ 后，需要通过度量转换为面通量 $U^i$，保证压力投影和连续性仍使用曲线网格有限体积通量。

这解释了 VFS 中 `Ucat` 和 `Ucont` 双速度系统的必要性：`Ucat` 适合物理模型、壁面模型和力积分；`Ucont` 适合守恒、Poisson RHS 和通量修正。

### 2.5 Level set 在固体界面的重构

手册指出，距离函数 $\phi$ 也需要在固体-流体界面重构。为了避免 level set 穿透固体边界，VFS 在流体节点与 IB 节点之间的单元面设置：

$$
\nabla \phi \cdot \mathbf{n}_{face}=0.
$$

其物理含义是固体壁面对 level set 函数施加零法向梯度，即自由面 signed distance 不通过固体边界产生额外通量。这相当于对 level set advection/reinitialization 使用反射或 Neumann 型边界处理。

### 2.6 运动边界处理

运动固体的表面速度来自结构运动。刚体点 $\mathbf{X}$ 的速度为：

$$
\mathbf{u}_B
=\mathbf{V}_c+\boldsymbol{\omega}\times(\mathbf{X}-\mathbf{X}_c),
$$

其中 $\mathbf{V}_c$ 是质心平动速度，$\boldsymbol{\omega}$ 是角速度，$\mathbf{X}_c$ 是质心或旋转中心。每个时间步或 FSI 子迭代中：

1. 根据结构求解器更新位移和角度。
2. 移动 IBM 三角面节点。
3. 更新三角面法向、面积和表面速度。
4. 清空或更新 `Nvert`。
5. 重新执行 IB 搜索和插值信息构造。
6. 用新的 $\mathbf{u}_B$ 重构 IB 节点速度。

如果采用强耦合 FSI，步骤 1 到 6 会在同一个真实时间步内多次重复，使流体力和结构位移相互收敛。

## 3. LES 模型

### 3.1 过滤方程与 SGS 应力来源

LES 的出发点是空间过滤：

$$
\bar{f}(\mathbf{x})
=\int_{\Omega}G_\Delta(\mathbf{x}-\mathbf{r})f(\mathbf{r})\,d\mathbf{r},
$$

其中 $G_\Delta$ 是滤波核，$\Delta$ 是滤波宽度，通常与网格尺度相关。对不可压缩 Navier-Stokes 过滤后：

$$
\frac{\partial \bar{u}_i}{\partial t}
+\frac{\partial \overline{u_i u_j}}{\partial x_j}
=-\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i}
+\nu\frac{\partial^2\bar{u}_i}{\partial x_j\partial x_j}.
$$

加入并减去 $\bar{u}_i\bar{u}_j$：

$$
\frac{\partial \bar{u}_i}{\partial t}
+\frac{\partial \bar{u}_i\bar{u}_j}{\partial x_j}
=-\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i}
+\nu\nabla^2\bar{u}_i
-\frac{\partial}{\partial x_j}
\left(
\overline{u_i u_j}-\bar{u}_i\bar{u}_j
\right).
$$

定义 SGS 应力：

$$
\tau_{ij}
=\overline{u_i u_j}-\bar{u}_i\bar{u}_j.
$$

未解析尺度对 resolved field 的作用即 $-\partial \tau_{ij}/\partial x_j$。

### 3.2 Smagorinsky 涡粘模型

SGS 应力分为各向同性和偏应力：

$$
\tau_{ij}
=\frac{1}{3}\tau_{kk}\delta_{ij}
+\left(\tau_{ij}-\frac{1}{3}\tau_{kk}\delta_{ij}\right).
$$

各向同性部分可并入压力：

$$
p^*=p+\frac{1}{3}\rho\tau_{kk}.
$$

因此只需闭合偏应力。Smagorinsky 假设未解析涡旋对大尺度动量的作用类似粘性耗散：

$$
\tau_{ij}-\frac{1}{3}\tau_{kk}\delta_{ij}
=-2\mu_t\bar{S}_{ij}.
\tag{2.25}
$$

其中 resolved strain-rate tensor 为：

$$
\bar{S}_{ij}
=\frac{1}{2}
\left(
\frac{\partial \bar{u}_i}{\partial x_j}
+\frac{\partial \bar{u}_j}{\partial x_i}
\right),
$$

其模长：

$$
|\bar{S}|=(2\bar{S}_{ij}\bar{S}_{ij})^{1/2}.
$$

混合长度假设给出：

$$
\nu_t=(C_s\Delta)^2|\bar{S}|.
$$

手册 Eq. 2.26 写成：

$$
\mu_t=C_s\Delta^2|\bar{S}|.
\tag{2.26}
$$

结合现有代码文档，当前实现中的 `Cs` 更接近已经平方或动态合并后的系数，因此代码公式通常解释为：

$$
\nu_t=C_s^{code}\Delta^2|\bar{S}|.
$$

曲线网格中滤波宽度常取单元体积立方根：

$$
\Delta=\left(\Delta V\right)^{1/3}
=\left(\frac{1}{J}\right)^{1/3}.
$$

物理解释：$\Delta$ 表示最小可解析涡尺度，$|\bar{S}|$ 表示局部大尺度剪切强度。剪切越强、网格越粗，SGS 涡粘越大。

### 3.3 动态 Smagorinsky

动态模型用 Germano 恒等式从流场自适应计算系数。引入测试滤波 $\widehat{\cdot}$，其尺度 $\hat{\Delta}>\Delta$。定义：

$$
L_{ij}
=\widehat{\bar{u}_i\bar{u}_j}
-\widehat{\bar{u}}_i\widehat{\bar{u}}_j.
$$

$L_{ij}$ 表示网格滤波尺度到测试滤波尺度之间可观测的应力。模型应力在两个尺度上分别为：

$$
\tau_{ij}^{model}
-\frac{1}{3}\tau_{kk}^{model}\delta_{ij}
=-2C_s\Delta^2|\bar{S}|\bar{S}_{ij},
$$

$$
T_{ij}^{model}
-\frac{1}{3}T_{kk}^{model}\delta_{ij}
=-2C_s\hat{\Delta}^2|\widehat{\bar{S}}|\widehat{\bar{S}}_{ij}.
$$

Germano 关系写作：

$$
L_{ij}=T_{ij}-\widehat{\tau}_{ij}.
$$

代入模型后得到：

$$
L_{ij}^{dev}=C_s M_{ij},
$$

其中：

$$
M_{ij}
=-2\hat{\Delta}^2|\widehat{\bar{S}}|\widehat{\bar{S}}_{ij}
+2\widehat{\Delta^2|\bar{S}|\bar{S}_{ij}}.
$$

用最小二乘求 $C_s$：

$$
C_s
=\frac{L_{ij}M_{ij}}{M_{ij}M_{ij}}.
$$

当前代码文档中实际采用：

$$
C_s^{code}=0.5\frac{LM}{MM+\epsilon_{les}},
$$

并裁剪到 $[0,max\_cs]$。可选均匀方向平均用于减少动态系数的局部噪声。近壁和 IBM 区域会限制或置零 $\nu_t$，避免壁面附近过度耗散。

## 4. FSI 算法

### 4.1 6 自由度刚体动力学

刚体广义坐标向量为：

$$
\mathbf{Y}
=
\begin{bmatrix}
x_c & y_c & z_c & \theta_x & \theta_y & \theta_z
\end{bmatrix}^T.
$$

从牛顿第二定律出发，平动满足：

$$
m\frac{d^2\mathbf{x}_c}{dt^2}
=\mathbf{F}_f+\mathbf{F}_e-\mathbf{C}_t\frac{d\mathbf{x}_c}{dt}-\mathbf{K}_t\mathbf{x}_c.
$$

转动满足 Euler 刚体方程。若在主惯性轴下且忽略陀螺耦合或采用小角度形式，可写为：

$$
\mathbf{I}\frac{d^2\boldsymbol{\theta}}{dt^2}
=\mathbf{M}_f+\mathbf{M}_e-\mathbf{C}_r\frac{d\boldsymbol{\theta}}{dt}-\mathbf{K}_r\boldsymbol{\theta}.
$$

合并为手册 Eq. 2.12：

$$
M\frac{\partial^2Y_i}{\partial t^2}
+C\frac{\partial Y_i}{\partial t}
+KY_i
=F_i^f+F_i^e,
\qquad i=1,\ldots,6.
\tag{2.12}
$$

其中 $M$ 对平动是质量，对转动是转动惯量；$C$ 是阻尼；$K$ 是刚度；$F_i^f$ 和 $F_i^e$ 分别是流体和外力/外矩。

### 4.2 流体力和力矩积分

固体表面上的 Cauchy 应力为：

$$
\boldsymbol{\sigma}
=-p\mathbf{I}+\boldsymbol{\tau}^v.
$$

单位法向 $\mathbf{n}$ 指向流体时，流体作用在结构上的面力为：

$$
d\mathbf{F}
=\boldsymbol{\sigma}\cdot\mathbf{n}\,d\Gamma
=(-p\mathbf{n}+\boldsymbol{\tau}^v\cdot\mathbf{n})d\Gamma.
$$

积分得到手册 Eq. 2.13：

$$
\mathbf{F}_f
=\int_{\Gamma}-p\mathbf{n}\,d\Gamma
+\int_{\Gamma}\tau_{ij}n_j\,d\Gamma.
\tag{2.13}
$$

对参考点 $\mathbf{x}_r$ 的力矩为：

$$
d\mathbf{M}
=\mathbf{r}\times d\mathbf{F},
\qquad
\mathbf{r}=\mathbf{x}-\mathbf{x}_r.
$$

指标形式使用 Levi-Civita 符号 $\epsilon_{ijk}$：

$$
M_i
=\int_{\Gamma}
-\epsilon_{ijk}r_j p n_k\,d\Gamma
+\int_{\Gamma}
\epsilon_{ijk}r_j \tau_{kl}n_l\,d\Gamma.
\tag{2.14}
$$

代码通过 IBM 表面三角元或近界面插值信息近似这些积分，并输出力系数、力矩系数和功率。

### 4.3 分区耦合、弱耦合和强耦合

VFS 使用 partitioned FSI：流体求解器和结构求解器各自保持独立，通过界面力和界面运动交换信息。

弱耦合 LC-FSI 的一步流程为：

1. 用当前结构位置求流场。
2. 积分得到 $\mathbf{F}_f,\mathbf{M}_f$。
3. 推进结构方程得到新位置和速度。
4. 移动 IBM 几何，进入下一时间步。

弱耦合成本低，但当流体附加质量与结构质量同量级时可能不稳定。

强耦合 SC-FSI 在同一时间步内迭代：

$$
\mathbf{Y}^{(m)}
\xrightarrow{\text{IBM geometry}}
\mathbf{u}^{(m)},p^{(m)}
\xrightarrow{\text{force integration}}
\mathbf{F}^{(m)}
\xrightarrow{\text{structure solve}}
\mathbf{Y}^{(m+1)}.
$$

收敛准则通常比较结构位移、速度或界面力残差：

$$
\frac{\|\mathbf{Y}^{(m+1)}-\mathbf{Y}^{(m)}\|}
{\|\mathbf{Y}^{(m+1)}\|+\epsilon}
<tol.
$$

Aitken 松弛用于加速固定点迭代。若结构更新写为：

$$
\mathbf{Y}^{(m+1)}
=\mathbf{Y}^{(m)}+\omega^{(m)}\mathbf{r}^{(m)},
$$

其中 $\mathbf{r}^{(m)}$ 是未松弛更新残差，则 Aitken 根据连续两次残差差值调整：

$$
\omega^{(m)}
=-\omega^{(m-1)}
\frac{(\mathbf{r}^{(m-1)})^T
(\mathbf{r}^{(m)}-\mathbf{r}^{(m-1)})}
\|\mathbf{r}^{(m)}-\mathbf{r}^{(m-1)}\|^2}.
$$

物理解释：强耦合在每个时间步内寻找流体载荷和结构位移的相容状态，适合浮体、VIV 和强 added-mass 问题。

## 5. 波浪生成

### 5.1 内部源区造波

VFS 使用内部自由面 forcing 方法。在计算域内部设置源区，向动量方程右端加入局部体力，使自由面产生目标波列。生成的波向两侧传播，边界附近再用 sponge layer 衰减，减少反射。

目标单色波面为手册 Eq. 2.27：

$$
\eta(x,y,t)
=A\cos(k_xx+k_yy-\omega t+\theta).
\tag{2.27}
$$

线性波理论中，有限水深色散关系为：

$$
\omega^2=g|\mathbf{k}|\tanh(|\mathbf{k}|h),
$$

其中 $h$ 是水深，$|\mathbf{k}|=\sqrt{k_x^2+k_y^2}$。深水极限 $\tanh(|\mathbf{k}|h)\to1$，得到 $\omega^2=g|\mathbf{k}|$。

为了只在自由面附近、源区附近施加力，源项构造为两个平滑 delta 的乘积：

$$
S_i(x,y,t)
=n_i(\phi)P_0
\delta(x;\epsilon_x)
\delta(\phi;\epsilon_\phi)
\sin(\omega t-k_yy-\theta).
\tag{2.28}
$$

其中 $n_i(\phi)$ 是界面法向分量，$\delta(\phi;\epsilon_\phi)$ 把力限制在自由面窄带，$\delta(x;\epsilon_x)$ 把力限制在源区宽度内。

平滑 delta 函数 Eq. 2.30 为：

$$
\delta(\alpha;\beta)
=
\begin{cases}
\frac{1}{2\beta}
\left[
1+\cos\left(\frac{\pi\alpha}{\beta}\right)
\right],
&-\beta<\alpha<\beta,\\
0,&\text{otherwise}.
\end{cases}
\tag{2.30}
$$

该函数满足：

$$
\int_{-\beta}^{\beta}\delta(\alpha;\beta)\,d\alpha
=1,
$$

因为：

$$
\int_{-\beta}^{\beta}
\frac{1}{2\beta}
\left[
1+\cos\left(\frac{\pi\alpha}{\beta}\right)
\right]d\alpha
=1+\frac{1}{2\beta}
\left[
\frac{\beta}{\pi}\sin\left(\frac{\pi\alpha}{\beta}\right)
\right]_{-\beta}^{\beta}
=1.
$$

源强系数为 Eq. 2.29：

$$
P_0
=\frac{Ag^2}{\omega^2\epsilon_x}
f(\epsilon_x,k_x)
\frac{2\rho_w}{\rho_a+\rho_w}
k_x
-\sqrt{k_x^2+k_y^2}.
\tag{2.29}
$$

手册排版中最后两个波数项较紧凑，其含义是根据目标波幅、频率、源区宽度、密度比和方向波数校准体力强度，使线性响应的自由面振幅为 $A$。辅助函数 Eq. 2.31 为：

$$
f(\epsilon_x,k_x)
=
\frac{\pi^2}
{k_x(\pi^2-\epsilon_x^2k_x^2)}
\sin(k_x\epsilon_x).
\tag{2.31}
$$

物理解释：源区越窄，体力越集中，数值上更接近边界造波但更容易产生高频噪声；源区越宽，波形更平滑但占用更长计算域。

### 5.2 多频波和外部波场

宽谱波可写为多方向、多频率叠加：

$$
\eta(x,z,t)
=\sum_{k_z}\sum_{k_x}
a_{k_z,k_x}
\cos(k_z z+k_x x-\omega_{k_z,k_x}t+\theta_{k_z,k_x}).
$$

手册第 4.2.5 节说明外部 `WAVE_infoXXXXXX.dat` 提供各分量的幅值和相位。数值实现上，只需对每个分量计算源项并线性叠加：

$$
\mathbf{S}
=\sum_m \mathbf{S}^{(m)}.
$$

这使 VFS 能使用理论谱、测量波场或前置大域波浪模拟作为入射条件。

### 5.3 Sponge layer

为防止波在开边界反射，VFS 在边界附近加入阻尼源项 Eq. 2.32：

$$
S_i
=-\left[\mu C_0u_i+\rho C_1u_i|u_i|\right]
\frac{
\exp\left[\left(\frac{x_s-x}{x_s}\right)^{n_s}\right]-1
}
{\exp(1)-1}.
\tag{2.32}
$$

该式包含线性阻尼和二次阻尼。线性项类似粘性阻尼，二次项在大速度时更强。指数权重让阻尼从 sponge 起点平滑增强到边界，避免阻尼区入口产生人工反射。

## 6. 风机参数化模型

### 6.1 Actuator disk

Actuator disk 用一个圆盘表示转子，不解析叶片几何。动量理论认为转子从流体中抽取轴向动量，表现为圆盘上的压力跃迁或体力 sink。

圆盘面积：

$$
A_d=\frac{\pi D^2}{4}.
$$

若推力为 $F_T$，单位面积体力为 Eq. 2.15：

$$
F_{AD}=-\frac{F_T}{A_d}
=-\frac{F_T}{\pi D^2/4}.
\tag{2.15}
$$

负号表示力与来流方向相反，即从流场抽取动量。推力由 Eq. 2.16 给出：

$$
F_T
=\frac{1}{2}\rho C_T A_d U_\infty^2.
\tag{2.16}
$$

一维 actuator disk 理论中，诱导因子 $a$ 定义为圆盘处速度降低比例：

$$
u_d=(1-a)U_\infty.
$$

因此 Eq. 2.17：

$$
U_\infty=\frac{u_d}{1-a}.
\tag{2.17}
$$

推力系数为：

$$
C_T=4a(1-a).
$$

圆盘平均速度由三角面面积加权：

$$
u_d
=\frac{1}{A_d}
\sum_{X=1}^{N_t}u(X)A(X)
=\frac{4}{\pi D^2}
\sum_{X=1}^{N_t}u(X)A(X).
\tag{2.18}
$$

由于转子网格与流体网格不重合，需要离散 delta 插值：

$$
u(X)
=\sum_{x=1}^{N_D}
u(x)\delta_h(x-X)V(x).
\tag{2.19}
$$

其中 $\delta_h$ 是紧支撑离散 delta，$V(x)$ 是流体单元体积。力再从圆盘网格分布回流体网格：

$$
f_{AD}(x)
=\sum_{X=1}^{N_D}
F_{AD}(X)\delta_h(x-X)A(X).
\tag{2.20}
$$

物理解释：actuator disk 捕捉整体推力和尾流动量亏损，但不能解析叶尖涡、叶片局部攻角和旋转非均匀载荷。

### 6.2 Actuator line

Actuator line 把每片叶片表示为沿半径方向的线段集合。每个线段根据局部相对速度、弦长和翼型表计算升阻力。

局部相对速度来自轴向速度和切向相对速度：

$$
\mathbf{V}_{rel}
=(u_z,\;u_\theta-\Omega r),
\tag{2.23}
$$

模长为：

$$
V_{rel}=\sqrt{u_z^2+(u_\theta-\Omega r)^2}.
$$

攻角由相对速度方向、叶片扭角和桨距角决定：

$$
\alpha=\tan^{-1}
\left(
\frac{u_z}{\Omega r-u_\theta}
\right)
-\beta_{twist}(r)-\beta_{pitch}.
$$

查翼型表得到 $C_L(\alpha,Re_c)$ 和 $C_D(\alpha,Re_c)$，升阻力为 Eq. 2.21 和 Eq. 2.22：

$$
L=\frac{1}{2}\rho C_L C V_{rel}^2,
\tag{2.21}
$$

$$
D=\frac{1}{2}\rho C_D C V_{rel}^2.
\tag{2.22}
$$

其中 $C$ 是局部弦长。把局部升阻力从叶片坐标投影到笛卡尔坐标，得到 $\mathbf{F}(X)$。再用离散 delta 分布到流体：

$$
f_{AL}(x)
=\sum_{X=1}^{N_L}
F(X)\delta_h(x-X)A(X).
\tag{2.24}
$$

物理解释：actuator line 比 disk 更细，能产生旋转叶片载荷、叶尖涡和非均匀尾流，但仍不解析真实叶片边界层。它的精度依赖翼型数据、delta 宽度、叶片线段分辨率和局部速度采样方式。

## 7. 算法模块之间的关系

一次 VFS 时间步可按以下数学依赖理解：

1. `LESModel` 根据 $\mathbf{u}^n$ 计算 $\nu_t$ 和 SGS 应力闭合。
2. `RHSSolver` 在曲线网格上构造对流、扩散、SGS、体力和旧压力梯度。
3. `Integrator` 用 SNES 求中间通量 $U^*$。
4. `PoissonSolver` 解压力修正 $\Pi$，投影得到散度为零的 $U^{n+1}$。
5. `BcsUtility` 和 `ImmersedBoundary` 施加物理边界与 sharp-interface IB 条件。
6. （手册背景，非本树实现）若另有 level set 模块，才求解 $\phi$ 对流和重初始化并更新相物性/界面源项。
7. 若启用 FSI，积分流体力并更新结构位置，移动 IBM 网格后可进入强耦合子迭代。
8. 若启用风机模型，actuator disk/line 源项加入动量方程右端，模拟转子抽取动量。

这个设计的核心是“结构化曲线网格上的守恒投影法 + sharp-interface IBM + 可选多物理模块”。因此任何重构或迁移都必须保持三件事的一致性：曲线度量、通量散度/压力梯度伴随关系、以及 IB/FSI 改变几何后对 `Nvert` 和边界重构的同步更新。
