---
title: Meta Learning
tags: 机器学习
categories:
  - DL
  - Lee's notes
abbrlink: ae52508d
date: 2025-01-22 11:21:48
---

## <center> Meta-Learning </center>


### 关于参数

<img src="meta-learning/image.png" alt="" width="70%" height="70%">

参数也是meta-learning的一个分支。

---

<img src="meta-learning/image-1.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-2.png" alt="" width="70%" height="70%">


---

<img src="meta-learning/image-3.png" alt="" width="70%" height="70%">

我们现在目标是学习F本身，包括网络架构，初始参数，学习率啥的。每一个东西都是meta learning的一个分支。

比如一个学习二元分类的meta learning：

<img src="meta-learning/image-4.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-5.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-6.png" alt="" width="70%" height="70%">

然后就是强行train，如果不能微分，就上RL或者EA。

<img src="meta-learning/image-7.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-8.png" alt="" width="70%" height="70%">


- ML v.s. Meta-Learning

difference:

<img src="meta-learning/image-9.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-10.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-12.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-11.png" alt="" width="70%" height="70%">


similarity:

<img src="meta-learning/image-13.png" alt="" width="70%" height="70%">


----

- MAML找一个初始化参数

<img src="meta-learning/image-14.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-15.png" alt="" width="70%" height="70%">

好的原因：

<img src="meta-learning/image-16.png" alt="" width="70%" height="70%">

- 还可以学习optimizer
<img src="meta-learning/image-17.png" alt="" width="70%" height="70%">

- Network Architecture Search

<img src="meta-learning/image-18.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-19.png" alt="" width="70%" height="70%">


- Data Augmentation

<img src="meta-learning/image-20.png" alt="" width="70%" height="70%">

- Sample Reweighting
<img src="meta-learning/image-21.png" alt="" width="70%" height="70%">

---


### 应用

<img src="meta-learning/image-22.png" alt="" width="70%" height="70%">


### 补充学习

<img src="meta-learning/image-23.png" alt="" width="70%" height="70%">

- self-supervised learning

<img src="meta-learning/image-24.png" alt="" width="70%" height="70%">


- knowledge distillation

<img src="meta-learning/image-25.png" alt="" width="70%" height="70%">

有文献指出，成绩好的teacher不见得是好的teacher。

<img src="meta-learning/image-26.png" alt="" width="70%" height="70%">

引入meta learning，可以让teacher学习如何去teach。

<img src="meta-learning/image-27.png" alt="" width="70%" height="70%">

- Domain Adaptation

在有label data的domain上很容易进行meta learning。

这里特别说一下在domain generalization上的应用。（即对一个未知的target domain，进行预测）

<img src="meta-learning/image-28.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-29.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-30.png" alt="" width="70%" height="70%">

注意这里train的结果，是学习一个初始化参数。

<img src="meta-learning/image-31.png" alt="" width="70%" height="70%">

- Lifelong Learning
<img src="meta-learning/image-33.png" alt="" width="70%" height="70%">
<img src="meta-learning/image-32.png" alt="" width="70%" height="70%">

传统方法，设计constraint，让模型不要忘记之前的知识。

<img src="meta-learning/image-34.png" alt="" width="70%" height="70%">

我们也可以尝试用meta learning找一个比较好的leanring algorithm，可以避免catasrophic forgetting。

<img src="meta-learning/image-35.png" alt="" width="70%" height="70%">

