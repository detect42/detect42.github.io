---
title: Network Compression
tags: 机器学习
categories:
  - DL
  - Lee's notes
abbrlink: 5cb6a14b
date: 2025-01-21 11:13:38
---
## <center> Network Compression </center>

网络压缩是指通过减少模型的大小和计算量来提高模型的性能。这个技术在移动端和嵌入式设备上（即resource-constrained的时候，比如手表，无人机）非常有用，因为这些设备的计算资源有限。网络压缩的方法有很多，比如剪枝、量化、知识蒸馏等。

<img src="network-compression/image.png" alt="" width="70%" height="70%">

### Network Pruning

去评估参数和神经元。

<img src="network-compression/image-1.png" alt="" width="70%" height="70%">

<img src="network-compression/image-2.png" alt="" width="70%" height="70%">

一个很有意思的发现：

大模型里面小模型中奖了可以出结果，所以从大模型pruning出来的小模型是幸运的可以train的起来的参数。（与之对比，重新初始化结构相同的pruning出来的小模型是不行的）
<img src="network-compression/image-4.png" alt="" width="70%" height="70%">
<img src="network-compression/image-3.png" alt="" width="70%" height="70%">

### Knowledge Distillation

为什么不直接train一个小的network？因为效果不好。



<img src="network-compression/image-5.png" alt="" width="70%" height="70%">

一个直觉的说法，是teacher可以提供额外的咨询。

比如老师告诉说，1和7比较接近，而不是0%，100%。（比较平滑？）

**同时注意到，knowledge distillation在model ensemble上的运用，实际中很难同时做1000个模型output的ensemble，但是我们可以提前train的一个对于ensemble结果的蒸馏模型。**

<img src="network-compression/image-6.png" alt="" width="70%" height="70%">


### Parameter Quantization

用一个小点的bit来表示参数。

<img src="network-compression/image-7.png" alt="" width="70%" height="70%">

- Binary Neural Network

<img src="network-compression/image-8.png" alt="" width="70%" height="70%">

一个参数值要么是1，要么是-1。

### Architecture Design

这是传统的CNN：
<img src="network-compression/image-10.png" alt="" width="70%" height="70%">
这是改后的Depthwise Separable Convolution：
<img src="network-compression/image-9.png" alt="" width="70%" height="70%">

比较：

<img src="network-compression/image-11.png" alt="" width="70%" height="70%">

原理：

<img src="network-compression/image-12.png" alt="" width="70%" height="70%">
<img src="network-compression/image-13.png" alt="" width="70%" height="70%">


### Dynamic Computation

让network自由地调整自己的计算量。

<img src="network-compression/image-14.png" alt="" width="70%" height="70%">
<img src="network-compression/image-15.png" alt="" width="70%" height="70%">




