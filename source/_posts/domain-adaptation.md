---
title: Domain Adaptation
tags: 机器学习
categories:
  - DL
  - Lee's notes
abbrlink: 6e7ac1bd
date: 2025-01-19 17:56:38
---

## <center> Domain Adaptation </center>

当我们有一个模型在一个domain上训练好了，我们想要将这个模型应用到另一个domain上，这时候就需要domain adaptation。


简单来说，就是训练集和测试集的分布不一样，我们需要让模型适应新的分布。其实和transfer learning很像，但是transfer learning更加广泛，不仅仅是domain adaptation。

<img src="domain-adaptation/image.png" alt="" width="70%" height="70%">

### Transfer Learning

按照有无label，可以分为几种case：

<img src="domain-adaptation/image-15.png" alt="" width="70%" height="70%">

- Fine-tune

直接fine-tune，加上参数的L2正则化。
<img src="domain-adaptation/image-1.png" alt="" width="70%" height="70%">

或者只对一个layer进行fine-tune。
<img src="domain-adaptation/image-2.png" alt="" width="70%" height="70%">

但是哪些layer会被fine-tune呢？这个不同任务差异很大。
<img src="domain-adaptation/image-3.png" alt="" width="70%" height="70%">

- Multi-task learning

<img src="domain-adaptation/image-4.png" alt="" width="70%" height="70%">

一个成功的例子是多语言的speech recognition。

<img src="domain-adaptation/image-5.png" alt="" width="70%" height="70%">

- Domain-adversarial training


和GAN有点像，希望把domain的信息去掉，只保留task-specific的信息。
<img src="domain-adaptation/image-8.png" alt="" width="70%" height="70%">
<img src="domain-adaptation/image-9.png" alt="" width="70%" height="70%">
<img src="domain-adaptation/image-7.png" alt="" width="70%" height="70%">
<img src="domain-adaptation/image-6.png" alt="" width="70%" height="70%">

- Zero-shot learning

<img src="domain-adaptation/image-10.png" alt="" width="70%" height="70%">

先提取attribute，然后查表，看哪一个最接近。

<img src="domain-adaptation/image-11.png" alt="" width="70%" height="70%">

甚至可以做attribute embedding:

<img src="domain-adaptation/image-12.png" alt="" width="70%" height="70%">

然后直接在embedding space上找最接近的。

两个zero-shot例子：
<img src="domain-adaptation/image-13.png" alt="" width="70%" height="70%">
<img src="domain-adaptation/image-14.png" alt="" width="70%" height="70%">

- self-taught learning

先学一个好用的feature extractor，然后在target domain上用这个feature extractor去learn。

---

让我们回归到domain adaptation。

- domain shift

有很多种：

<img src="domain-adaptation/image-16.png" alt="" width="70%" height="70%">

我们按照对target domain的了解程度分类：

- little but labeled

<img src="domain-adaptation/image-17.png" alt="" width="70%" height="70%">

- large amount of unlabeled data

<img src="domain-adaptation/image-18.png" alt="" width="70%" height="70%">

为了利用共同的feature，我们尝试找这个feature extractor

<img src="domain-adaptation/image-19.png" alt="" width="70%" height="70%">

<img src="domain-adaptation/image-20.png" alt="" width="70%" height="70%">

<img src="domain-adaptation/image-21.png" alt="" width="70%" height="70%">


同时我们希望对于target domain的内容的predictor输出离boundary越远越好。
<img src="domain-adaptation/image-23.png" alt="" width="70%" height="70%">
<img src="domain-adaptation/image-22.png" alt="" width="70%" height="70%">



