---
title: Approximating KL Divergence
tags: math
abbrlink: 75ed94ba
date: 2025-03-22 23:02:47
---
# <center> Approximating KL Divergence </center>

主要介绍使用MC方法来近似KL散度的技巧。

参考[link](http://joschu.net/blog/kl-approx.html)

### KL公式：

$$KL[q,p]=\sum_xq(x)\log{\frac{q(x)}{p(x)}}=\mathbb{E}_{x\sim q}\log\frac{q(x)}{p(x)}$$

前置假设：
1. 我们知道概率密度计算，但是没法做遍历x做求和或者积分。
2. 已知$x_1,x_2,... \sim q$，即从真实分布中采样的样本。
3. 一般在机器里，我们的模型可以表示$p$的函数。


### K1

一个straightforward的做法是直接使用$k_1 =\log\frac{q(x)}{p(x)}=-\log r$，这里定义$r=\frac{p(x)}{q(x)}$，我们可以用MC方法抽样算$k_1$来近似。

1. 但是有high variance, 它甚至可能是负的。
2. 无偏的

### K2

$\frac{1}{2}(\log\frac{q(x)}{p(x)})^2=\frac{1}{2}(\log r)^2$

1. 都是正的
2. 有低的variance
3. 低bias


Estimator $k_2 = \frac{1}{2} (\log r)^2$ 的期望实际上是一个 **f-divergence**。f-divergence 是一类衡量概率分布差异的函数族，定义为：

$$D_f(p, q) = \mathbb{E}_{x \sim q} \left[ f\left( \frac{p(x)}{q(x)} \right) \right]$$

其中 $f$ 是一个凸函数。KL 散度、χ² 散度、Total Variation 等常见的概率距离都属于 f-divergence。在分布 $p \approx q$ 的情形下，所有可微的 f-divergence 在数学上都近似于一个统一的二次形式：

$$D_f(p_0, p_\theta) = \frac{f''(1)}{2} \theta^T F \theta + \mathcal{O}(\|\theta\|^3)$$

其中 $F$ 是 Fisher 信息矩阵。这意味着不同的 f-divergence 在局部行为上高度相似。以 $k_2$ 为例，它对应的 $f(x) = \frac{1}{2} (\log x)^2$，而 KL[q‖p] 对应的是 $f(x) = -\log x$，它们在 $x = 1$ 处的二阶导数都是 1，因此在 $p \approx q$ 时提供几乎相同的距离估计。因此，$k_2$ 不仅具有明确的理论解释，而且在实际中是一个偏差小、方差低的 KL 散度近似器。

所以在两个分布p，q接近时可以用k2来近似KL散度，增加了稳定性。


### K3

$$k_3 = (r-1) -\log r$$

我们希望构造一个对 KL 散度（例如 $\mathrm{KL}[q \| p]$）的估计器，既无偏又具有较低方差。一个常见的技巧是使用 **control variate**：即在原始估计量上加一个期望为零、但与其负相关的项，以减少方差。

KL[q‖p] 的标准估计器是：
$$k_1 = -\log r, \quad \text{其中 } r = \frac{p(x)}{q(x)}$$
为了降低其方差，可以加上 $\lambda(r - 1)$，因为：
$$\mathbb{E}_{x \sim q}[r - 1] = \mathbb{E}_q\left[ \frac{p(x)}{q(x)} - 1 \right] = 1 - 1 = 0$$
于是构造出一类无偏估计器：
$$-\log r + \lambda(r - 1)$$
通过最小化方差可以解出最优的 $\lambda$，但这个解依赖于 $p(x)$ 和 $q(x)$，通常难以解析求得。

为了解决这个问题，可以采用一个更简单又合理的选择：$\lambda = 1$。由于 $\log x \le x - 1$（因为对数函数是 concave 的），所以这个选择保证估计器始终为正：

$$k_3 = (r - 1) - \log r$$

这个形式正好就是前面提到的 KL[q‖p] 的 Bregman 形式，几何意义上表示 $r = \frac{p(x)}{q(x)}$ 下，$\log x$ 与其在 $x = 1$ 处切线之间的垂直距离。它不仅是一个无偏估计器，而且更稳定、易计算，是强化学习等场景中常用的 KL 近似方式之一。


#### 从数学角度推广上述感性想法

我们可以推广 Bregman 散度的思想，构造出对任意 f-divergence 的**始终为正的估计器**。给定凸函数 $f(x)$，f-divergence 定义为：

$$D_f(p, q) = \mathbb{E}_{x \sim q}\left[ f\left( \frac{p(x)}{q(x)} \right) \right]$$

由于凸函数始终位于其切线之上，我们可以用以下表达式作为 f-divergence 的估计器：(r-1在期望下是0所以是无偏的)

$$f(r) - f'(1)(r - 1), \quad \text{其中 } r = \frac{p(x)}{q(x)}$$

这个估计器**永远非负**，其几何含义是：$f(r)$ 与它在 $r = 1$ 处的切线之间的垂直距离。

具体到 KL 散度：

- 对于 $\mathrm{KL}[p \| q]$，对应的 $f(x) = x \log x$，有 $f'(1) = 1$，估计器为：
  $$r \log r - (r - 1)$$

- 对于 $\mathrm{KL}[q \| p]$，对应的 $f(x) = -\log x$，有 $f'(1) = -1$，估计器为：
  $$(r - 1) - \log r$$

这两个表达式不仅无偏，且始终为正，是在机器学习和强化学习中非常实用的 KL 散度估计方法。