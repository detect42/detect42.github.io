---
title: JAX base
tags: python
abbrlink: ffeffb62
date: 2025-05-06 00:55:07
---
# <center> JAX </center>

学习资料：

1. https://www.zhihu.com/column/c_1729967479470804992 从入门到jax神经网络手写数字识别 对应的[github仓库](https://github.com/oneday88/HelloJax/tree/main)

2. https://space.bilibili.com/478929155/search?keyword=jax b站上的视频讲解，挺好懂的。

```python
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

# 原始函数
def f(x):
    return jnp.sin(x) + x**2

# 1. 求导
f_grad = grad(f)

# 2. 向量化
f_grad_batch = vmap(f_grad)

# 3. 编译加速
f_grad_batch_jit = jit(f_grad_batch)

# 创建一批数据
x_batch = jnp.linspace(0, 10, 10000000)  # 10000个样本

# 调用
y_batch = f_grad_batch_jit(x_batch)

print(y_batch.shape)  # (10000,)
print(y_batch[:5])    # 打印前5个结果
```

**JAX的vmap，jit，grad等特性需要一个重要的限制：pure function。**

纯函数 = 没有随机性，没有副作用，计算是可预测的 → JAX 可以放心变换和优化它

有这么几点限制：

1. 纯函数内部不要读或写函数外的变量。
2. 随机数 种子也必须传入
3. 在函数内部动态构造对象（如 optimizer、model）是不行的

<img src="JAX-base/Pasted Graphic 57.png" alt="" width="70%" height="70%">
<img src="JAX-base/Pasted Graphic 58.png" alt="" width="70%" height="70%">

如果我们不小心写出了non-pure function，然后进行transformation怎么办？你肯定指望JAX抛出一个异常，可惜的是，JAX内部并没有检查函数是否pure的机制，对于non-pure，transformation的行为属于undefined，有点像C语言中的野指针，此时函数的执行结果不可预测。

另外的一些雷点：

1. 切片和if 在jit里不能有传入的变量 因为jit时是在做tracer 没有具体值
<img src="JAX-base/Pasted Graphic 52.png" alt="" width="70%" height="70%">
<img src="JAX-base/Pasted Graphic 53.png" alt="" width="70%" height="70%">
2. 条件分支
<img src="JAX-base/Pasted Graphic 54.png" alt="" width="70%" height="70%">
3. vmap 指定保存的维度out_axes=0(default)
<img src="JAX-base/Pasted Graphic 55.png" alt="" width="70%" height="70%">



### vamp的三个例子

```python
import jax
import jax.numpy as jnp
import time


#! jit放这里和放下面 区别就是放这里 第一次编译快一些 放下面二次后打包整体更快，但是第一次略慢
def l1_distance(x, y):
    assert x.ndim == y.ndim == 1  # only works on 1D inputs
    return jnp.sum(jnp.abs(x - y))
xs = jax.random.normal(jax.random.key(0), (20000, 3))

# 无 jit 的版本
start = time.time()
_ = jax.vmap(jax.vmap(l1_distance, (0, None)), (None, 0))(xs, xs)
print("No JIT:", time.time() - start)


# jit 外包一层
@jax.jit
def fast_pairwise(xs):
    return jax.vmap(jax.vmap(l1_distance, (0, None)), (None, 0))(xs, xs)


# 第一次 jit 慢
start = time.time()
_ = fast_pairwise(xs)
print("With JIT (1st):", time.time() - start)

# 第二次快
start = time.time()
_ = fast_pairwise(xs)
print("With JIT (2nd):", time.time() - start)


#! example 2 cal (A,B) and (C,D)
import jax
import jax.numpy as jnp


def f(x, y):
    return jnp.sum(x) + jnp.sum(y)


# A=2, B=3 → x.shape = (2, 3)
# C=4, D=3 → y.shape = (4, 3)
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
y = jnp.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0]])  # shape (4, 3)

# f(x[i], y[j]) → shape (2, 4)
out = jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))(x, y)
print(out)


#! example3 simulate matrix multiplication


# 单个元素 c_ij = dot(A[i], B[:,j])
def dot_fn(x, y):
    return jnp.dot(x, y)


# 构造矩阵
A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
B = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])  # shape (3, 2)
# A.shape = (2, 3)
# B.shape = (3, 2)
matmul_vmap = jax.vmap(jax.vmap(dot_fn, in_axes=(None, 1)), in_axes=(0, None))  # axis=1：列方向
C = matmul_vmap(A, B)
print("A @ B using vmap:\n", C)
print("Ground truth (jnp.dot):\n", A @ B)

```

结果
<img src="JAX-base/image.png" alt="" width="70%" height="70%">

### grad同时保存值，和指定求导的argnums

<img src="JAX-base/image-1.png" alt="" width="70%" height="70%">

### vmap用in__axes和out_axes来指定输入和输出的维度

<img src="JAX-base/image-2.png" alt="" width="70%" height="70%">

### 设备控制&&自带上下文管理器
<img src="JAX-base/image-3.png" alt="" width="70%" height="70%">

### 对于有累积variable的for-loop

和reduce像：
<img src="JAX-base/image-4.png" alt="" width="70%" height="70%">

<img src="JAX-base/image-5.png" alt="" width="70%" height="70%">
<img src="JAX-base/image-6.png" alt="" width="70%" height="70%">

### 使用 cond 进行条件判断

<img src="JAX-base/image-7.png" alt="" width="70%" height="70%">

这里如果是对一批值进行条件赋值，用jnp.where最好（虽然不是惰性的）

**否则需要使用lax.cond来进行可以trace的条件判断**


### 随机数生成器RNG
<img src="JAX-base/image-8.png" alt="" width="70%" height="70%">


### 有时候不得不在jax里调用其它包的函数
使用pure_callback来调用非JAX函数

<img src="JAX-base/image-9.png" alt="" width="70%" height="70%">
```python
# 引入纯Python函数
def estimate_entropy(actions, num_components=3):
    # actions: (batch, sample, dim)
    import numpy as np
    from sklearn.mixture import GaussianMixture

    ...
    return final_entropy  # 一个 float 标量
#像cpp一样把接口定义好即可
entropy = jax.pure_callback(
    estimate_entropy, jax.ShapeDtypeStruct((), jnp.float32), actions  # 非JAX函数  # 输出 shape 描述  # 输入
)
```
