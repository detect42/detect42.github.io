---
title: Score-function blog 速记
tags: diffusion
Category:
  - diffusion
abbrlink: bd741ffa
date: 2025-02-17 15:10:53
---
## Score-function blog 速记

为什么score function是log p的梯度，某种意义上我们一般的定义是
<img src="score-match/image.png" alt="" width="70%" height="70%">
以往我们都是参数化f，这里log刚好去掉指数。
同时log后让对p的大幅度变化不敏感。

<img src="score-match/image-1.png" alt="" width="70%" height="70%">

加上高斯卷积导致这种效果：
<img src="score-match/image-2.png" alt="" width="70%" height="70%">
高斯模糊效果，和直接线性添加高斯噪音不同
<img src="score-match/image-3.png" alt="" width="70%" height="70%">
退火版朗之万采样法


By generalizing the number of noise scales to infinity , we obtain not only higher quality samples, but also, among others, exact log-likelihood computation, and controllable generation for inverse problem solving.
<img src="score-match/image-4.png" alt="" width="70%" height="70%">

也许也可以看出score function的重要性

<img src="score-match/image-5.png" alt="" width="70%" height="70%">
<img src="score-match/image-6.png" alt="" width="70%" height="70%">
<img src="score-match/image-7.png" alt="" width="70%" height="70%">

在 SDE 和 ODE 过程中，任意时刻  t  的数据分布是相同的。
同时ODE时，一个采样和一个噪音是双射的

