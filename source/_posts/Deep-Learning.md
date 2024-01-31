---
title: Deep_Learning 简介
tags: ML
categories: 
- DL
- Lee's notes
abbrlink: ebc18e1f
date: 2023-11-21 14:55:42
---

## 1. Difference between Deep network and Shollow network

![Alt text](Deep_Learning简介/image-11.png)

deep network 显著地减少了参数的数量，一方面避免了参数过多而过拟合的风险，一方面也减少了计算量，提高了训练速度。

![Alt text](Deep_Learning简介/image-12.png)

深度学习可以在参数量小的同时，loss值同样小。做到了鱼与熊掌的兼得。

![Alt text](Deep_Learning简介/image-13.png)

## 两难困境

1. 模型太大了，理想解恨好，但是训练出来效果不好
2. 模型小了，理想和现实接近但是都不好

![Alt text](Deep_Learning简介/image.png)

## 为什么需要Hidden Layer

### 正常的函数逼近
![Alt text](Deep_Learning简介/image-1.png)
制造很多上图中的sigmoid函数，然后叠加在一起，就可以逼近任何函数了

![Alt text](Deep_Learning简介/image-2.png)

于是通过神经网络的连接，通过设置很多神经元的参数：weight and bios，再通过sigmoid激活函数后组合起来，就可以逼近任何函数了

![Alt text](Deep_Learning简介/image-3.png)


当然更多时候，我们用relu激活函数代替sigmoid函数，因为relu函数更简单，计算量更小，而且效果更好。

![Alt text](Deep_Learning简介/image-4.png)

### 为什么需要多层神经网络

Why we want "Deep" network， while not "fat" network? Just because it sounds cool?


![Alt text](Deep_Learning简介/image-6.png)

![Alt text](Deep_Learning简介/image-7.png)

 - Yes, one hidden layer can erpresent any function

- However, using deep structure is more effective.

实际上，deep network需要更少的参数，不容易overfitting。

![Alt text](Deep_Learning简介/image-8.png)

`实际上，比如拟合一个函数，使用deep的参数远小于shalow的参数，也就是说如果使用shallow达到和deep一样的效果，参数会非常多，很容易overfitting，为避免overfitting反而需要更多的数据。`

所以深度学习适合function复杂且有规律的condition！

![Alt text](Deep_Learning简介/image-9.png)

![Alt text](Deep_Learning简介/image-10.png)