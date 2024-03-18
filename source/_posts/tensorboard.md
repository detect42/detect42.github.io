---
title: tensorboard
tags: python
abbrlink: 1acdf70
date: 2024-03-18 19:43:47
---

<center> <h1> Tensorboard </h1> </center>


详细的介绍和常见报错可以参考

- [TensorBoard最全使用教程：看这篇就够了](https://blog.csdn.net/qq_41656402/article/details/131123121)
- [Tensorboard 无法打开网页 /不显示数据 / 命令行报错等解决办法集合](https://blog.csdn.net/White_lies/article/details/124141102)
## 导入库

tensorboard是tensorflow的可视化工具，可以用来查看模型的训练过程，模型结构，模型参数等。

但是现在pytorch也支持tensorboard了，在处理loss,acc等数据的时候，可以使用tensorboard来可视化。

## 使用方法

```python
# 生成一个writer对象，并指定生成图片的目录，其中logs为指定生成事件的目录
writer = SummaryWriter("logs")
for i in range(100):
    # 以y=x为标题，i为横坐标值，i同时也为纵坐标值，相当于绘制了一个y=x(0<=x<100)的直线
    writer.add_scalar("y=x",i,i)
writer.close()
```

运行代码之后会生成一个指定名字的目录，并且会包含一个事件在里面，这个事件要用tensorboard打开。

## 一个例子

```python
# Build a tensorboard
writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
```

注意,我们需要一个清晰的分类目录,这样避免了重名,也方便查找。

```py
writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
```

![alt text](Tensorboard/image.png)

![alt text](Tensorboard/image-1.png)

## 两个例子

```python
# tensorboard
summary_writer = SummaryWriter(log_dir = "DQN_cosine-warmup", comment= "good_makeatari")
```

```python
if i % print_interval == 0:
    print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
    summary_writer.add_scalar("Temporal Difference Loss", loss, i)
    summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
    summary_writer.add_scalar("Epsilon", epsilon, i)

summary_writer.close()
```

![alt text](Tensorboard/image-2.png)

保存在同一个大文件下不同的小文件夹再统一开启tensorboard可以看到不同的训练过程和训练数据。

