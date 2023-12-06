---
title: 演化算法HW3-复杂度&期望运行时间理论分析
tags: HW
abbrlink: ebc186c0
date: 2023-12-06 23:35:12
---

# <center> Homework 3 </center>

### <center> 221502011 王崧睿 detect0530@gmail.com </center>

# 1. 证明No Free Lunch(NFL)定理

整体思路采用归纳法，首先，让我们证明：

$$\sum_fP(d_1^y|f,m=1,a) = \sum_f\mu(d_1^y,f(d_1^x))$$

$\mu$函数代表指示函数，成立时为1，反之为0。

注意这里的$d_x^x$是由$a$（算法）决定的。

因为f函数有个sum up，故上述式子的可以化简为$|Y|^{|X|-1}$，与$a$无关！

接下来，按照归纳法，我们假设$\sum_fP(d_m^y|f,m,a)$和a无关，然后去证明$\sum_fP(d_{m+1}^y|f,m+1,a)$也和a无关。

首先变化一下式子：

$$P(d_{m+1}^y|f,m+1,a) = P(d_{m+1}^y(m+1)|d_m,f,m+1,a) \cdot  P(d_m^y|f,m+1,a) $$

同时注意到，新的$d_{m+1}^y(m+1)$，依赖与新的$x$值、f函数。

继续化简：

$$\sum_fP(d_{m+1}^y|f,m+1,a)=\sum_fP(d_{m+1}^y|f,x)\cdot P(x|d_m^y,f,+1,a)\cdot P(d_m^y|f,m+1,a)$$

$$=\sum_{f,x}\mu(d_{m+1}^y(m+1),f(x))\cdot P(x|d_m^y,f,m+1,a)\cdot P(d_m^y|f,m+1,a)$$

接下来，在让我们考虑x的的依赖，只要我们知道了a，然后知道了$d_m^x$和$d_m^y$就可以算出x。

所以进一步变化：

$$\sum_fP(d_{m+1}^y|f,m+1,a)=\sum_{f,d_m^x}\mu(d_{m+1}^y(m+1),f(a(d_m)))\cdot P(d_m|f,m,a)$$

又实用一开始的函数计数技巧，因为已经走了m步，所以function上有$m$个点已经确定了，故$\sum_{f(x\notin d_m^x)}\mu(d_{m+1}^y(m+1),f(a(d_m)))$有$|Y|^{|X|-m-1}$的贡献。

所以最后：

$$\sum_fP(d_{m+1}|f,+1,a)=|Y|^{|X|-m-1}\cdot \sum_fP(d_m^y|f,m,a)$$

由我们的归纳假设知道$\sum_fP(d_m^y|f,m,a)$是与a无关的，所以有$\sum_fP(d_{m+1}|f,+1,a)$也是与a无关的，归纳完成。

所以：
$$\sum_f\sum_{d_m^y}\phi(d_m^y)P(d_m^y|f,m,A)=\sum_{d_m^y}\phi(d_m^y)\sum_fP(d_m^y|f,m,A)$$

有上述归纳法，知这个式子与$A$无关。

证毕。


# 2. 分析LeadingOnes问题

分层就按简单的leadingones个数来分。

$$P(\phi(i)\to \phi(i+1)) > \frac{1}{n} \cdot (1-\frac{1}{n})^i$$

$$E(T)=\sum_{i=1}^{n-1}P(\phi(i))*\frac{1}{P(\phi(1) \to \phi(i+1))}<\cdot \sum n*(1-\frac{1}{n})^n = en^2=O(n^2) $$

即期望运行时间是$O(n^2)$。

# 3. 分析OneMax问题

我们定义距离函数$V(x)=n-countone(x)$

这个定义满足距离函数的要求（最优解为距离为0等）

$$E(V(\xi_t)-V(\xi_(t+1))|\xi_t=x) = E(countone(\xi_{t+1})-countone(\xi_{t})) > \frac{n-countone(\xi_t)}{n}\cdot (1-\frac{1}{n})^{n-1} = V(\xi_t=x)\cdot \frac{1}{n} \cdot (1-\frac{1}{n})^{n-1}$$


所以根据乘法漂移公式：

Upper bound: $\sum_{x\in X}\pi_0(x)\cdot \frac{1+ln(V(x)/V_{min})}{\delta}$

我们知道$V_{min}=1$，$\pi_0(x) <=  1$，$\delta=\frac{1}{n}\cdot (\frac{n-1}{n})^n$

带入并做不等式变化：

$$Upperbound < \frac{1+ln(n)}{\delta} = (1+ln(n)) \cdot n \cdot  e = O(nlogn)$$

即期望运行时间是$O(nlogn)$。

# 4 分析COCZ问题

首先，分析一下这个问题，第一个维度是1的个数，第二个维度是前一半1的个数加上后一半0的个数。

同时一个Pareto front的形式可以表示为：

1. 前一半都是1
2. 后一半1的个数从0到n/2，各取一个就可以了。

所以完整的Pareto front大小是$n/2$。

同时，按照SEMO算法，因为第一维度是1的个数，所以任意时刻子群大小小于等于n。

运行时间的计算分为两个阶段：

- 找到全1串

- 找到所有的Pareto front


### 1. 首先 找到全1串：

假设只是$1+1\  EA \ onebitmutation$，

$\sum_{i=1}^{n-1} P(\phi(x)=i)\frac{1}{\frac{n-i}{n}} < \sum n\cdot \frac{1}{n-i} = nlogn$

现在只是多了一个从不超过n的子代里选择，故每次至多多了一个$\frac{1}{n}$的选择，所以找到全1的期望时间不超过$n^2logn$

### 2. 找到所有的Pareto front

现在后半段0的数量为0，我们只要找到后半段0的数量为1到$n/2$就找全了。

而0的数量从i变到i+1的概率为$\frac{n/2-i}{n}$，然后假设这种情况只发生在子代解里取到0最多的情况，也就是至少有$\frac{1}{n}$的概率会拓展一个Pareto front。

$$E(T)<\sum_{i=0}^{n/2-1}\frac{1}{\frac{1}{n}\cdot \frac{n/2-i}{n}} = n^2 log(n/2) = O(n^2logn)$$


综上所述，所以SEMO算法找到COCZ问题的Pareto front的期望时间是$O(n^2logn) + O(n^2log(n/2)) = O(n^2logn)$。




