---
title: SVM
tags: 机器学习
categories:
  - DL
  - Lee's notes
abbrlink: '75378e04'
date: 2025-01-25 11:09:48
---

## <center> Support Vector Machine (SVM) </center>

<img src="SVM/image.png" alt="" width="70%" height="70%">
<img src="SVM/image-1.png" alt="" width="70%" height="70%">
<img src="SVM/image-2.png" alt="" width="70%" height="70%">

- hinge loss function:

max操作促使f(x)大于1，且不需要超过1太多，对比cross entropy： hinge loss及格就好：
（也是ideal function的upper bound）

<img src="SVM/image-3.png" alt="" width="70%" height="70%">

SVM和logistic regression的区别就是定义的loss function不同，SVM的loss function是hinge loss function，logistic regression的loss function是cross entropy loss function。


<img src="SVM/image-4.png" alt="" width="70%" height="70%">

脑洞一下，linear SVM可以用gradient descent来求解。

<img src="SVM/image-5.png" alt="" width="70%" height="70%">

---

常见的做法：

<img src="SVM/image-6.png" alt="" width="70%" height="70%">


---

- SVM特点

1. w参数实际是x的线性组合，所以SVM是线性分类器。这一点可以用lagrange multiplier或者梯度下降来证明。
2. $a_n$是sparse的，大部分是0，也就是说SVM结果只和少数几个样本有关。

<img src="SVM/image-7.png" alt="" width="70%" height="70%">



---

经过等价变化，和最后一步推广，产生kernel trick。
<img src="SVM/image-8.png" alt="" width="70%" height="70%">
<img src="SVM/image-9.png" alt="" width="70%" height="70%">
<img src="SVM/image-10.png" alt="" width="70%" height="70%">
<img src="SVM/image-11.png" alt="" width="70%" height="70%">
<img src="SVM/image-12.png" alt="" width="70%" height="70%">
当使用RBF kernel，如果不使用kernel而是朴素的特征变换，需要的特征维度是无穷的，而kernel trick可以避免这个问题。

<img src="SVM/image-13.png" alt="" width="70%" height="70%">

**Kernel function是一个投影到高维的inner product，所以往往就是something like similarity**

<img src="SVM/image-14.png" alt="" width="70%" height="70%">


---

与deep learning的比较：

这个很形象也很重要：

<img src="SVM/image-15.png" alt="" width="70%" height="70%">