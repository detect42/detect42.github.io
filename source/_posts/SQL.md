---
title: SQL
tags: RL
Category:
  - RL
abbrlink: 4d712855
date: 2024-10-17 23:10:53
---


# <center> SQL </center>

Soft Q-leaning引入了熵作为正则项，可以说提出了一种新的评价体系，加强了策略的探索性。

同时很意外的与Energy-based model有一定的联系。

## 值函数改变

以前的Q只是对当前策略未来收益的预测，现在加入熵项：

$$Q^{\star}_{soft} = r_t + \mathbb{E}_{(s_{t+1},…)\sim\rho_{\pi^*_{MaxEnt}}}\left[ \sum_{l=1}^\infin\gamma^l\left(r_{t+l}+\alpha\mathcal{H}\left(\pi^*_{MaxEnt}\left(\cdot|s_{t+l}\right)\right)\right) \right]$$

$$\mathcal{H}(\pi(\cdot|s_t)) = \mathbb{E}_{a\sim\pi(\cdot|s_t)}[-\log\pi(a|s_t)]$$

对上式做等价变形：

$$Q^\pi(s,a) = \mathbb{E}\Bigg[r(s,a) + \sum_{t=2}^\infin \gamma^{t-1} \Big(r(s_t, a_t) -\alpha \log\pi(a_t|s_t)\Big)\Big|s_1 = s, a_1 = a\Bigg]$$

现在Q值函数会对当前策略的未来回报以及策略的熵进行评估。

## Policy improvement

现在考虑given Q，如何找到最优策略。

以往PG做法求梯度啥的是可以，但是这里我们可以直接用Q值解出符合当前Q值的最优策略。

现在已知Q(s,a)，我们求$\pi^\star(\cdot|s)$
$$ \max_{\pi} E_{a\sim\pi(\cdot|s)}[Q(s,a)] + \alpha\mathcal{H}(\pi(\cdot|s))  \\ =\alpha \sum_a \pi(a|s)\left( \frac{1}{\alpha} Q(s,a) - \log\pi(a|s)\right) $$

$$ \min_\pi \alpha \sum_a \pi(a|s) \left[ \log \frac{\pi(a|s)}{\frac{1}{\alpha}\exp\left(\frac{1}{Z_s}Q(s,a) \right)} - \log Z_s\right] \\ Z_S = \int_\mathcal{A}\exp(\frac{1}{\alpha}Q(s,a))da$$

因为$Z_s$和策略无关，我们可以无视它。

$$\min_\pi KL(\pi | \frac{1}{Z_S}\exp(\frac{1}{\alpha}Q(s,a))) \\ \pi^\star(s,a) = \frac{\exp(\frac{1}{\alpha}Q(s,a))}{Z_s}$$

可以看到，我们直接通过Q值，解出了满足当前Q值的最优策略解。

## V值函数

基于我们对Q值的新定义，我们可以得到V值函数：

$$V^\pi(s) = \mathbb{E}_{a\sim\pi(\cdot|s)}[Q(s,a) - \alpha\log\pi(a|s)]$$

即按照策略加权的下一步的Q函数，加上自己这一步的熵。

在Q函数确定，$\pi^\star$也跟着确定下，我们可以直接由Q求出V值函数。

$$V(s) = \int_\mathcal{B} \frac{\exp(\frac{1}{\alpha}Q(s,a))}{Z_s} \cdot [Q(s,a) - (Q(s,a) - \alpha\log Z_s)] \\ = \alpha \log Z_s \\ =\alpha \log \int_\mathcal{A} \exp \left( \frac{1}{\alpha}Q(s,a)\right)da$$

形式上就是对Q的Softmax。

## Policy evaluation

用bellman error来更新Q函数：


$$Given (s,a,r,s^\prime) \\ Q(s,a) \leftarrow r(s,a) + V(s') ... (1)\\ = r(s,a) + \alpha \log \int_\mathcal{A}\exp\left(\frac{1}{\alpha}Q(s',a)\right)da ...(2)$$

实际上那个积分是intractable的，我们只能用采样去近似，同时非常惊讶地发现采样分布就tm是Energy-based model distribution。采样方面有一些较成熟高效的做法，比如郎之万采样。


### Important insights

值得注意的是上述(1)式子是纯血的policy evaluation，而(2)式子则在此基础上融合了policy improvement。

这意味着(2)式子只用一个更新公式包含了两者，我们不在需要像PG算法那样交替进行policy evaluation和policy improvement。