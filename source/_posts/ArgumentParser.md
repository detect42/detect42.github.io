---
title: ArgumentParser
tags: python
abbrlink: 726db7c5
date: 2024-03-18 17:09:10
---

<center> <h1> ArgumentParser </h1> </center>

更具体细节的讲解可以参考[知乎_Link](https://zhuanlan.zhihu.com/p/388930050?utm_id=0)

## 创建parser

```py
import argparse
parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
```

创建parser对象，可以传入一个字符串作为描述信息。

## 添加参数

```py
parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
```

- `--max_train_steps`：参数名，前面加`--`表示是可选参数
- `type=int`：参数类型
- `default=int(3e6)`：默认值
- `help=" Maximum number of training steps"`：帮助信息, 用于`-h`时显示

## 解析命令行参数

```py
args = parser.parse_args()
```

得到args对象,包含了所以之前的参数,之后都过传递args对象来调用定好的参数.