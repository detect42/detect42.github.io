---
title: Python Grammar
tags: python
abbrlink: 22c2c40d
date: 2024-03-18 16:45:07
---
# <center> Python

```python
D = collections.defaultdict(list)
```
- 创建一个默认映射到list的字典。（帮助避免了很多意外情况）

---

```python
op,l,r,y,z = map(int,input().split())
t = ''.join(input().split(" "))
```

 - 很方便的处理一行的输出，split()默认跳过空白字符。
 - 将输入拼成一个字符串，方便处理。

---

```python
 a.sort(key=lambda x:x.r)
```

- 对a本生进行排序，这里有key函数：按照key=（比较函数），进行比较。比较函数，可以很方便的用lambda来表示。

- 如果想反过来排序，可以用reverse=True参数。

---

```python
print("{0} - > {1}".format(i.l,i.r),end="\n")
```

- 格式化输出，这里的0和1是format函数的参数，可以用来指定输出的位置。同时python print自带换行,如果不想换行可以自定义end参数。


---

```python
import bisect
j = bisect.bisect_right(a,3,key=lambda x:x.r)
```

- 在有序数组中二分查找,right表示严格大于,注意key参数指定的比较大小,所以x.r和3一定是可以比较的类

---


```py
fav = [False]*26
f = [[0]*n for _ in range(n)]
```

- 开出指定大小的二维数组,且带有初始值.

---

```py
fav[ord(c)-ord('A')] = True
```

利用ord转成ascii码,然后减去'A'的ascii码,得到了一个0-25的数字,可以用来索引数组.

---

## python函数传参规则
在 Python 中，调用函数时参数的传递有特定的规则。特别是关于位置参数和关键字参数的顺序，Python 对此有严格的要求。错误信息 `SyntaxError: positional argument follows keyword argument` 表示在函数调用时，位置参数出现在了关键字参数之后，这是不允许的。

### 位置参数和关键字参数

1. **位置参数**：
   - 位置参数是在函数调用时按照函数定义中参数的顺序传递的值。它们不需要使用参数名，因为它们的位置决定了哪个是哪个。

2. **关键字参数**：
   - 关键字参数通过“关键字=值”的形式在函数调用中指定。使用关键字参数时，参数的顺序可以与函数定义中的顺序不同，因为参数值是通过关键字（即参数名）映射的。

### 规则

在 Python 的函数调用中，关键字参数必须位于所有位置参数之后。这样做的目的是为了避免歧义和提高代码的可读性。如果位置参数出现在关键字参数之后，解释器将无法正确理解每个参数的值应该赋予哪个形式参数。

### 示例

假设有一个函数定义如下：

```python
def my_function(a, b, c):
    print("a:", a)
    print("b:", b)
    print("c:", c)
```

正确的调用方法可以是：

```python
my_function(1, 2, c=3)  # 所有位置参数出现在关键字参数之前
```

错误的调用会导致 `SyntaxError`：

```python
my_function(1, b=2, 3)  # SyntaxError: positional argument follows keyword argument
```

在上述错误的例子中，位置参数 `3` 被放在了关键字参数 `b=2` 之后，这违反了参数传递的规则。

### 解决方案

要避免这类错误，确保在任何关键字参数之后不再出现位置参数。如有必要，可以将所有参数都转换为关键字参数，这样顺序就不会影响函数调用了。

### 总结

当你在编写和调用函数时，应始终保持参数的顺序，使所有位置参数都位于关键字参数之前。这样做不仅符合 Python 的语法规则，还可以使你的代码更加清晰和易于理解。如果遇到 `SyntaxError: positional argument follows keyword argument`，检查并调整参数的顺序即可解决问题。

# <center>Numpy

```py
array = np.zeros((args.batch_size, args.state_dim))
```

- 创建一个numpy数组,大小由我指定.

---

```py
s = torch.tensor(self.s, dtype=torch.float)
```

- 从numpy数组转换为torch张量. 并且制定dtype的类型.

-----

```py
self.std = np.zeros_like(x)
```

- 创建一个和x一样大小的数组,并且初始化为0. 便于快速shape和val的初始化.

-----


```py
#x is list
x = np.array(x)
y = x.copy()
```

- 将list转换为numpy数组,并且深复制一份.

----


```py
if x.all() == 0:
    print("x is all zero")
```
numpy有.all(),.any()方法,可以用来判断数组是否全为0,或者是否有0.

进一步的可以有:
```py
# 创建一个示例数组
array = np.array([3, 4, 5, 6])
# 检测数组中的所有元素是否都大于 2
result = (array > 2).all()
print(result)  # 如果数组中所有元素都大于2，将打印 True，否则打印 False
```

---

### np.nonzero（包含indices使用技巧）

`np.nonzero()` 是一个非常有用的 NumPy 函数，它返回所有非零元素的索引。在多维数组的情况下，`np.nonzero()` 会返回一个元组，元组中的每个数组代表了对应维度的索引。

### 示例：使用多维数组

假设我们有一个 2x3 的二维数组，我们可以使用 `np.nonzero()` 来找到所有非零元素的位置。下面是具体的代码示例和解释：

```python
import numpy as np

# 创建一个 2x3 的二维数组
arr = np.array([[0, 3, 0],
                [4, 0, 5]])

# 使用 np.nonzero() 获取非零元素的索引
indices = np.nonzero(arr)

# 打印非零元素的索引
print(indices)
```

### 输出和解释

对于上述代码，输出将是：

```
(array([0, 1, 1]), array([1, 0, 2]))
```

这里的输出解释如下：

- 第一个数组 `array([0, 1, 1])` 表示非零元素在行上的索引。
- 第二个数组 `array([1, 0, 2])` 表示非零元素在列上的索引。

这意味着：

- 数组中位置 `(0, 1)` 的元素是非零的，对应元素 `3`。
- 数组中位置 `(1, 0)` 的元素是非零的，对应元素 `4`。
- 数组中位置 `(1, 2)` 的元素是非零的，对应元素 `5`。

### 访问非零元素

如果你想直接访问这些非零元素，可以使用这些索引：

```python
# 使用索引访问非零元素
nonzero_elements = arr[indices]
print(nonzero_elements)
```

输出将是包含所有非零元素的一维数组：

```
[3 4 5]
```

### 小结

通过这个例子，你可以看到 `np.nonzero()` 如何有效地提供多维数组中所有非零元素的位置。这在处理科学数据或进行特征提取时特别有用，特别是在稀疏数据环境中。

---
## np.random 拓展

NumPy 的 `np.random` 模块提供了多种生成随机数的功能，能够覆盖从简单的均匀分布到复杂的正态（高斯）分布等。下面我将介绍几种常用的随机数生成函数及其用途：

### 1. 0 到 1 之间的均匀分布随机数

**函数：`numpy.random.rand()`**

此函数生成在[0, 1)区间内的均匀分布的浮点数。这意味着生成的数值大于等于0且小于1。

```python
import numpy as np

# 生成一个0到1之间的随机数
random_number = np.random.rand()
print(random_number)

# 生成一个5x5的0到1之间的随机数矩阵
random_matrix = np.random.rand(5, 5)
print(random_matrix)
```

### 2. 0 到 n 之间的整数随机数

**函数：`numpy.random.randint()`**

此函数用于生成一个指定范围内的随机整数。参数包括低（inclusive）和高（exclusive）边界，还可以指定输出数组的形状。

```python
# 生成一个0到10之间的随机整数
random_int = np.random.randint(0, 10)
print(random_int)

# 生成一个形状为(2, 3)的随机整数数组，范围从0到10
random_int_array = np.random.randint(0, 10, size=(2, 3))
print(random_int_array)
```

##  3. 高斯（正态）分布随机数

**函数：`numpy.random.randn()` 和 `numpy.random.normal()`**

- `numpy.random.randn()` 生成标准正态分布的随机数（均值为0，标准差为1）。
- `numpy.random.normal()` 可以生成具有指定均值和标准差的正态分布的随机数。

```python
# 生成一个标准正态分布的随机数
standard_normal = np.random.randn()
print(standard_normal)

# 生成一个均值为0，标准差为0.5的正态分布随机数
normal = np.random.normal(loc=0, scale=0.5)
print(normal)

# 生成一个形状为(3, 4)的，均值为0，标准差为0.5的正态分布随机数矩阵
normal_array = np.random.normal(0, 0.5, (3, 4))
print(normal_array)
```

### 4. numpy.random.uniform()
np.random.uniform() 函数用于生成指定区间内的均匀分布的随机数。你可以指定区间的最小值和最大值。
```py
# 生成一个在10到20之间的随机浮点数
random_uniform = np.random.uniform(10, 20)
print(random_uniform)

# 生成一个形状为(2, 3)的，范围在-1到0之间的均匀分布随机数数组
random_uniform_array = np.random.uniform(-1, 0, (2, 3))
print(random_uniform_array)

```

### 总结

这些是 NumPy 中常用的几种随机数生成方法，通过它们可以满足绝大多数对随机数的需求，无论是简单的均匀分布、整数随机选择还是复杂的正态分布模拟。每种方法都具有高度的灵活性，可以通过调整参数来满足不同的统计需求和数组形状要求。

---
## nan, inf, 浮点数比较

在NumPy中，`np.nan`和`np.inf`是用来表示特殊的浮点数值的。`np.nan`表示"非数"（Not a Number），而`np.inf`表示正无穷大。此外，`-np.inf`表示负无穷大。这些特殊值在处理涉及未定义或超出数值范围的运算时非常有用。

### np.nan (Not a Number)

`np.nan`用于表示未定义或不可表示的值，如`0/0`的结果。它是IEEE浮点数规范的一部分，并在NumPy中广泛用于表示缺失数据或异常值。在NumPy中，任何涉及`np.nan`的运算通常都会返回`np.nan`，除了一些特定的函数，比如`np.nanmean`，这些函数可以忽略`nan`值进行计算。

### np.inf 和 -np.inf

`np.inf`表示正无穷大，通常用于表示超出浮点数最大范围的值，比如`1/0`的结果。`-np.inf`表示负无穷大，用于表示超出浮点数最小范围的值，如`-1/0`的结果。在NumPy中，无穷大的值可以参与计算，其行为符合数学上的预期（例如任何正数乘以`np.inf`等于`np.inf`）。

### 比较涉及浮点数的值

浮点数因为其表示方式，常常无法精确表示某些值，例如常见的`0.1 * 3`不会精确等于`0.3`，尽管数学上它们是相等的。这是由于浮点数的精度问题导致的。

#### 如何判断两个浮点数是否相等？

使用NumPy时，可以使用`np.isclose()`或`np.allclose()`函数来比较两个浮点数或浮点数组是否近似相等，考虑到计算中的精度误差。

```python
import numpy as np

a = 0.1 * 3
b = 0.3

# 使用 np.isclose 来判断两个数是否足够接近
print(np.isclose(a, b))  # 输出 True 如果 a 和 b 足够接近

# 您也可以指定相对和绝对容差参数
print(np.isclose(a, b, rtol=1e-05, atol=1e-08))  # 默认参数
```

这个方法提供了一个实用的方式来处理浮点数的比较，通过设定容忍的误差范围，使得在实际应用中可以判断两个数在数值上是否"相等"。

### 总结

在使用NumPy处理数据时，了解`np.nan`和`np.inf`的行为非常重要，特别是在数据清洗和预处理阶段。此外，了解如何正确比较浮点数也是科学计算中一个关键的问题。这些工具和函数可以帮助你更有效地进行数据分析和数值计算。

---

## 矩阵乘法

- 使用 `*` 或 `np.multiply` 来进行元素对元素的乘法。
- 使用 `dot` 或 `@` 来执行矩阵乘法，适用于一维数组时，计算内积。
- 使用 `np.matmul` 或 `@` 来执行矩阵乘法，不支持一维数组的标量乘积，对高维数组有特定的广播规则。

选择哪种运算取决于您的具体需求，尤其是在处理矩阵运算和更高维度数据时。


---

## np.random.choice

`np.random.choice()` 是 NumPy 库中一个非常有用的函数，它允许从给定的一维数组或者范围内随机抽取元素，同时也可以指定抽取元素的概率分布和输出样本的数量。这个函数在模拟、抽样调查、随机实验等多种场景中非常实用。

### 函数基本用法

基本语法：
```python
numpy.random.choice(a, size=None, replace=True, p=None)
```

- **a**：如果是一个一维数组或者列表，随机数将从这些元素中抽取。如果是一个整数，抽取的元素将是从 `np.arange(a)` 生成的。
- **size**：输出样本的数量。默认为 `None`，这意味着输出一个值。可以是一个整数或元组（用于生成多维数组）。
- **replace**：布尔值，表示是否可以重复选择同一元素。默认为 `True`，即允许重复抽取。如果为 `False`，则抽取的样本中不会有重复元素，此时 `size` 不能超过 `a` 的长度。
- **p**：一维数组或列表，指定 `a` 中每个元素对应的抽取概率。数组的长度必须与 `a` 相匹配，并且概率之和应为1。

### 示例

#### 从给定数组中随机选择

```python
import numpy as np

# 定义一个数组
data = np.array([10, 20, 30, 40, 50])

# 随机选择一个元素
sample = np.random.choice(data)
print("Randomly selected item:", sample)

# 不放回地抽取3个不同的元素
sample_no_replace = np.random.choice(data, size=3, replace=False)
print("Sample without replacement:", sample_no_replace)

# 使用自定义概率分布抽取
sample_probs = np.random.choice(data, size=3, p=[0.1, 0.1, 0.1, 0.3, 0.4])
print("Sample with custom probabilities:", sample_probs)
```

#### 从数字范围内随机选择

```python
# 从0到9中随机选择5个数字，允许重复
sample_from_range = np.random.choice(10, size=5)
print("Random sample from range 0-9:", sample_from_range)
```

### 使用场景

1. **模拟实验**：在进行科学实验和模拟研究时，`np.random.choice()` 可以用来随机选择试验条件或样本。
2. **数据采样**：在机器学习和统计分析中，从大数据集中随机抽取样本进行训练和测试。
3. **概率研究**：通过指定不同的概率分布，研究和模拟概率事件的结果。

`np.random.choice()` 提供了一种灵活的方式来实现和模拟从给定数据集或数值范围内的随机抽样过程，其多功能性使它成为数据科学和统计学中一个不可或缺的工具。

---

## in place operation

在 NumPy 中使用 `np.add(A, B, out=B)` 和直接使用 `B = A + B` 来计算两个数组的和虽然都能得到相同的数学结果，但这两种方法在内存使用和计算效率方面存在差异。下面是这两种方法的主要区别：

### 1. `np.add(A, B, out=B)`
使用 `np.add()` 函数并指定 `out` 参数可以在现有数组上就地进行操作，这意味着结果直接存储在 `out` 指定的数组中。在这种情况下，`out=B` 表示结果将被存储在数组 `B` 中，覆盖原有数据。这种方法的主要优点是减少内存分配，因为不需要创建新的数组来存储结果：

```python
import numpy as np

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

np.add(A, B, out=B)
print(B)  # 输出将显示更新后的 B，即 [5, 7, 9]
```

这里 `B` 被更新为 `A` 和 `B` 的和。这种方式特别适合处理大数组，因为它可以减少内存消耗和可能的内存分配延迟。

### 2. `B = A + B`
这种方式比较直观，是将 `A` 和 `B` 相加的结果赋值给 `B`。这实际上涉及到先计算 `A + B` 的和，然后创建一个新的数组存储这个结果，最后将这个新数组的引用赋值给变量 `B`。这意味着原始的 `B` 数组被新的数组引用替代了，原来的数组如果没有其他引用将被垃圾回收：

```python
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

B = A + B
print(B)  # 输出同样是 [5, 7, 9]
```

尽管结果相同，但这种方法会使用更多的内存（至少在计算过程中），因为需要临时存储 `A + B` 的结果，然后再复制给 `B`。对于小数组，这种差异可能不明显，但对于处理非常大的数组时，额外的内存分配和释放可能会导致性能下降。

### 总结
- **`np.add(A, B, out=B)`** 是一种就地操作，可以帮助节省内存，特别适用于数据量大的数组计算。
- **`B = A + B`** 更直观，适用于快速编程和小规模数据处理，但可能涉及更多的内存分配。

选择哪种方法取决于特定应用的需求，包括对性能的关注、对内存使用的优化，以及代码的可读性。

---

# <center> Torch

一个典型的神经网络forward过程:

```py
 def __init__(self,args):
    super(Actor_Beta,self).__init__()
    self.fc1 = nn.Linear(args.state_dim,args.hidden_width)
    self.fc2 = nn.Linear(args.hidden_width,args.hidden_width)
    self.alpha_layer = nn.Linear(args.hidden_width,args.action_dim)
    self.beta_layer = nn.Linear(args.hidden_width,args.action_dim)
    self.activate_func = [nn.ReLU(),nn.Tanh()][args.use_tanh] # choose specific activation function

    if args.use_orthogonal_init:
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self,s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1
        beta = F.softplus(self.beta_layer(s)) + 1
        return alpha,beta

    def get_dist(self,s):
        alpha,beta = self.forward(s)
        dist = Beta(alpha,beta)
        return dist
```

- 这里定义了一系列全连接层和激活函数,并且通过神经网络拟合了两个参数,通过参数返回一个分布.

----

```py
self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
```

- 定义了一个Adam优化器,并且指定了学习率和eps.
- self.actor.parameters()指定了所以可训练的参数.


----

```py
def evaluate(self,s):
    s = torch.unsqueeze(torch.FloatTensor(s),0)
    a = self.actor.mean(s).detach().numpy().flatten()
    return a
```

- 将s转换成浮点张量,同时即使是单个数据点，也需要模拟成批量大小为1的数据,用来满足模型输入要求.
-这个 `evaluate` 函数定义了一个评估过程，主要用于获取给定状态 `s` 下的行动者（actor）模型输出的行动 `a`。该函数通过几个步骤处理输入状态并最终返回一个行动值。下面是对这些步骤的详细解释：


- **模型预测**：
   - `a = self.actor.mean(s).detach().numpy().flatten()`：这行代码执行了几个操作来获取行动值 `a`。
     - `.detach()`：然后，使用 `.detach()` 将预测结果从当前计算图中分离，这样做是为了避免在这一步对梯度进行追踪，因为我们只是在评估而非训练模型。
     - `.numpy()`：接着，使用 `.numpy()` 将结果转换为 NumPy 数组，以便进行进一步的处理或分析。
     - `.flatten()`：最后，使用 `.flatten()` 将数组展平，这意味着如果结果是多维的，它将被转换为一维数组。

这里重点再谈谈.flatten()和squeeze()的区别:

是的，在这个特定的情况下，你可以使用 `.squeeze()` 方法替代 `.flatten()` 来去除单维度条目。`.squeeze()` 方法用于从数组的形状中移除单维度条目，不改变数组中元素的总数。如果你的目标是将形状为 `(1, n)` 的数组转换成 `(n,)`，`.squeeze()` 完全可以胜任，因为它正是用来移除那些大小为1的维度。

### 使用 `.squeeze()` 的效果

假设原始的 NumPy 数组形状是 `(1, n)`，使用 `.squeeze()` 后，数组将变为形状 `(n,)`。这与 `.flatten()` 得到的结果相同，但 `.squeeze()` 更加直观地表达了你的意图，即移除单一维度而非展平整个数组。

### `.flatten()` 与 `.squeeze()` 的区别

- **`.flatten()`**：无论数组的原始形状如何，都会返回一个完全展平的一维数组。
- **`.squeeze()`**：只移除数组中的单一维度（大小为1的维度），不改变其他维度的大小。

因此，如果你确定需要移除的是单一维度，特别是在处理批量数据（批量大小为1）的场景中，`.squeeze()` 是一个更合适的选择。它不仅可以达到预期的效果，也使代码的意图更加明确。

**总的来说**，在这种情况下，`.squeeze()` 可以替代 `.flatten()` 来移除数组中的单一维度，使得代码的意图更清晰，同时保持了结果的一致性。

可以看到,GPT告诉我们这里因为我们是评估阶段,一次只会送批量为1的数据去预测,所以返回的也是批量为1的线性动作空间,直观上用squeeze()更加合适. flatten()是将多维数组展平成一维数组,在此刚好也能用罢了.

---

```py
.squezze()
```

`.squeeze()` 方法在使用时可以不指定参数，也可以指定一个参数来指定需要移除的维度索引。这取决于你的具体需求。

1. **不指定参数**：当你不指定参数时，`.squeeze()` 会移除数组中所有单维度条目。这意味着，如果数组中有任何大小为1的维度，它们都会被移除，结果数组的形状将排除这些维度。

   ```python
   import numpy as np

   a = np.array([[1, 2, 3]])  # 形状为 (1, 3)
   b = a.squeeze()  # b 的形状将变为 (3,)
   ```

2. **指定参数**：你可以指定一个维度索引作为参数给 `.squeeze()`，这样就只会尝试移除该指定维度。如果指定的维度不是单维度（即其大小不为1），那么数组将不会改变。

   ```python
   import numpy as np

   a = np.array([[[1, 2, 3]]])  # 形状为 (1, 1, 3)
   b = a.squeeze(0)  # 尝试移除第0维，b 的形状变为 (1, 3)
   ```

   如果尝试移除的维度大小不为1，使用 `.squeeze()` 时指定该维度将不会引发错误，数组也不会改变。这与 `.reshape()` 或其他可能引发异常的方法不同。

**总的来说**，`.squeeze()` 方法是灵活的，既可以不带参数使用，也可以通过指定参数来移除特定的单一维度。不指定参数时，它会移除所有单维度条目，而指定参数时，则只针对特定维度操作，这为处理不同的数据形状提供了便利。

----

```py
with torch.no_grad():
```

- 梯度追踪：PyTorch 会继续追踪操作以计算梯度，这在训练时是必需的，但在模型评估或执行推理时是不必要的。继续追踪梯度会导致不必要的计算资源消耗，增加内存使用，并可能导致在不需要梯度的场景中意外修改权重。

---

```py
for delta,d in zip(reversed(deltas.flatten().numpy()),reversed(done.flatten().numpy())):
```

- 函数在这里被用于同时遍历两个反转后的数组。它会创建一个迭代器，该迭代器生成一对元素的元组，其中每对元素分别来自两个数组的相同位置。因为两个数组都已被反转，所以实际上这个遍历是从原始数组的末尾开始，向前遍历到数组的开头。
- 如果deltas和done不等长,这样的迭代会在短的数组到头后停止.

---

```py
for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)),self.mini_batch_size,False):
```

这段代码,在随机采样上有高的泛用性,继续交给GPT4解释:

这行代码在 PyTorch 中使用了 `BatchSampler` 和 `SubsetRandomSampler` 杂交的方式来迭代指定批量大小的索引。这是在处理批量数据时，特别是在机器学习和深度学习训练过程中常见的做法。下面详细解释这个过程：

1. **`range(self.batch_size)`**:
   - 这部分生成一个从0到`self.batch_size - 1`的整数序列。`self.batch_size` 表示整个数据集中单次迭代（或称为"批次"）的样本数。

2. **`SubsetRandomSampler(range(self.batch_size))`**:
   - `SubsetRandomSampler` 接收一个索引列表作为输入，并从这个列表中随机采样，没有替换（即每个样本只能被采样一次）。在这里，它用于从上述范围（即整个批次）中随机选择索引，这种随机性增加了模型训练的泛化能力。

3. **`BatchSampler(SubsetRandomSampler(...), self.mini_batch_size, False)`**:
   - `BatchSampler` 从其接收到的采样器（这里是 `SubsetRandomSampler`）中进一步采样，将数据集分成指定大小的小批量（mini-batches）。`self.mini_batch_size` 表示这些小批量中的样本数。最后一个参数 `False` 表示不允许最后一批小批量的大小小于 `self.mini_batch_size`（如果为 `True`，则允许最后一批小于 `self.mini_batch_size`）。

4. **`for index in BatchSampler(...)`**:
   - 这个循环遍历 `BatchSampler` 生成的每个小批量的索引。每次迭代提供的 `index` 是一个包含 `self.mini_batch_size` 大小的小批量索引的列表，可以用这些索引来从数据集中抽取相应的样本进行训练或其他处理。

**总结**：这段代码通过结合 `SubsetRandomSampler` 和 `BatchSampler`，实现了对整个批次数据随机采样并分成指定大小的小批量的过程。这种方法常用于机器学习训练中，有助于提升模型的泛化性能并降低训练过程中的内存消耗。

----

```py
dist_entropy = dist_now.entropy().sum(1,keepdim=True)
```

- 求和操作通常会减少维度,我们用keepdim=True来保持维度.

----


```py
torch.clamp
```

`torch.clamp` 是 PyTorch 中的一个函数，用于将输入张量中的所有元素限制在指定的范围内。如果一个元素的值低于给定的最小值，它会被设置为这个最小值；如果一个元素的值高于给定的最大值，它会被设置为这个最大值。对于在这个范围内的元素，它们的值不会改变。这个操作在深度学习中常用于梯度裁剪、激活函数实现等场景，以防止数值溢出或确保数值稳定性。

函数的基本使用格式如下：

```python
torch.clamp(input, min, max, *, out=None) -> Tensor
```

- `input`：输入张量。
- `min`：范围的下限。所有小于 `min` 的元素都将被设置为 `min`。
- `max`：范围的上限。所有大于 `max` 的元素都将被设置为 `max`。
- `out`：输出张量。如果指定，结果将被写入这个张量中。

**示例：**

```python
import torch

# 创建一个张量
x = torch.tensor([0.5, 1.5, -0.5, -1.5, 2.5])

# 将张量中的元素限制在[0, 1]范围内
clamped_x = torch.clamp(x, min=0, max=1)

print(clamped_x)
```

输出结果将是：

```py
tensor([0.5, 1.0, 0.0, 0.0, 1.0])
```

这个例子中，`-0.5` 和 `-1.5` 被设置为了 `0`（因为它们小于下限0），`2.5` 被设置为了 `1`（因为它大于上限1），而 `0.5` 和 `1.5` 被分别设置为了自己和上限 `1`。

---

```py
for p in self.optimizer_actor.param_groups:
    p['lr'] = lr_a_now
```

动态调整指定model的学习率, 也有很强的泛用性.

这段代码遍历 `self.optimizer_actor` 中的所有参数组（`param_groups`），并将每个参数组的学习率（`'lr'`）设置为当前的学习率值 `lr_a_now`。在 PyTorch 中，优化器（如 Adam、SGD 等）可以有多个参数组，每个组可以有自己的学习率和其他优化设置。这种设计允许对模型的不同部分应用不同的学习率和更新策略，提供了更高的灵活性。

**解释**

- **`self.optimizer_actor`** 是一个 PyTorch 优化器实例，负责更新某个模型（在这个上下文中是“行动者”模型）的参数。这个优化器在初始化时，会被分配一个或多个参数组。

- **`param_groups`** 是优化器实例中的一个属性，包含了所有的参数组。每个参数组是一个字典，包含了参数（如权重）的引用和这组参数的优化设置（如学习率 `lr`）。

- **循环 `for p in self.optimizer_actor.param_groups:`** 遍历所有的参数组。

- **`p['lr'] = lr_a_now`** 在每次循环中，将当前参数组的学习率设置为 `lr_a_now`。这允许动态地调整学习率，是实现学习率衰减策略或根据训练进度调整学习率的常用手段。

**应用场景**

这种做法通常用在需要根据训练进度动态调整学习率的场景中，例如实现学习率衰减（learning rate decay）、预热学习率（learning rate warmup）或其他复杂的学习率调度策略时。动态调整学习率可以帮助模型更好地收敛，避免在训练早期的大幅度参数更新或在训练后期的过度拟合。


### 直接在GPU上创建新张量

```py
import torch
# 假设我们的系统有至少一个GPU
device = torch.device('cuda:0')  # 这里的'cuda:0'指的是第一个GPU
# 直接在GPU上创建一个大小为3x3的空浮点张量
empty_tensor = torch.empty(3, 3, device=device)
# 直接在GPU上创建一个大小为4x4的随机浮点张量
random_tensor = torch.rand(4, 4, device=device)
# 直接在GPU上创建一个大小为2x2的张量，所有元素初始化为0
zeros_tensor = torch.zeros(2, 2, device=device)
# 直接在GPU上创建一个大小为2x2的张量，所有元素初始化为1
ones_tensor = torch.ones(2, 2, device=device)
```


### .detach()方法的作用与特性

在PyTorch中，`.detach()`方法是一个非常重要的功能，尤其是在处理计算图和梯度计算时。了解`.detach()`的作用和特性对于有效地控制梯度流和优化内存使用非常关键。

#### 作用

1. **分离梯度**: `.detach()`方法创建一个新的张量，该张量与原始张量共享数据但不共享历史。换句话说，使用`.detach()`方法返回的张量不会在其操作上记录梯度，不参与梯度传播，因此对其的任何操作都不会影响到原始张量的梯度。

2. **避免梯度计算**: 它常用于防止PyTorch自动计算梯度或保存计算历史，这在某些情况下非常有用，比如在评估模型时不想影响到模型参数的梯度，或者当你只需要张量的数据而不需要其梯度时。

#### 特性

- **共享数据**: `.detach()`返回的新张量与原始张量共享内存（即数据），这意味着如果更改了其中一个张量的数据，另一个张量的数据也会相应改变。这有助于减少内存使用，因为不需要复制数据。

- **不参与自动梯度计算**: 对`.detach()`返回的张量进行的操作不会被纳入到计算图中，因此这些操作不会影响梯度。这使得`.detach()`非常适合用于停止某些变量的梯度跟踪。

- **用途广泛**: `.detach()`方法在模型训练、评估、特定操作的梯度屏蔽等多种场合中都非常有用。例如，它可以用于冻结部分模型参数的梯度，或者在计算某些指标时暂时停止跟踪梯度。

#### 使用场景示例

假设你在训练一个模型，并希望使用模型的一部分输出作为后续计算的输入，但你不希望这些后续计算影响到模型参数的梯度。这时，就可以使用`.detach()`方法：

```python
import torch

# 假设x是模型的输入，model是你的模型
x = torch.randn(1, 3, requires_grad=True)
output = model(x)

# 假设我们只需要output的数据，而不希望其参与接下来的梯度计算
output_detached = output.detach()

# 现在，你可以对output_detached进行操作，而这些操作不会影响到原始output变量的梯度
```

在上述代码中，`output_detached`和`output`共享数据，但对`output_detached`的任何操作都不会影响到`output`的梯度，因为`output_detached`已经从计算图中分离出来了。

总之，`.detach()`是处理PyTorch张量时控制梯度流动的一个强大工具，可以帮助开发者更精确地管理内存使用和梯度计算。

实际上，虽然`.detach()`创建的张量`output_detached`与原始张量`output`共享数据，但对`output_detached`进行的操作不会直接影响`output`的值。这是因为直接修改`output_detached`的值（例如，通过索引赋值）通常是不允许的，因为它们是从计算图中分离出来的，旨在用于读取或作为不需要梯度计算的操作的输入。

共享数据的意思是，如果原始张量`output`的值因为某些操作而改变，那么`output_detached`的值也会随之改变，因为它们底层指向的是同一块内存区域。但是，通常我们不直接修改这些张量的值，而是通过进行计算或应用函数来生成新的张量。对`output_detached`进行的任何计算都不会影响到`output`，因为`output_detached`已经从计算图中被分离出来了，它的操作不会回溯到原始张量。

这里有一个简单的例子来说明这个概念：

```python
import torch

# 创建一个需要梯度的张量
x = torch.randn(3, requires_grad=True)

# 通过一些操作得到新的张量
y = x * 2

# 分离y得到y_detached，它与y共享数据，但不会影响到y的梯度计算
y_detached = y.detach()

# 尝试修改y_detached的值
# 这样的操作是不被允许的，因为直接修改张量的值通常会被PyTorch阻止
# y_detached[0] = 1000  # 这会抛出错误

# 对y_detached进行操作，生成新的张量z
z = y_detached + 1

# 打印y和z，你会发现y的值没有因为对y_detached的操作而改变
print("y:", y)
print("z:", z)
```

在这个例子中，`y_detached`被用来生成了一个新的张量`z`，但这对`y`本身没有任何影响。这就是`.detach()`方法的一个关键特性：它允许我们在保持数据共享的同时，避免对原始张量的梯度计算产生影响。

----

### torch 根据概率抽样

```python
probs = self.actor(state)
action_dist = torch.distributions.Categorical(probs)
action = action_dist.sample()
```


---


### 获得网络全部参数self.actor.parameters()

注意这里坑点，假设你有一个简单的神经网络，其中包括两个全连接层，每层的权重和偏置都需要计算梯度。例如：

第一层权重形状为 [100, 50]，偏置形状为 [50]。
第二层权重形状为 [50, 10]，偏置形状为 [10]。
计算这些参数的梯度后，你将得到四个梯度张量，它们的形状分别是 [100, 50]、[50]、[50, 10]、[10]。如果你想将这些梯度合并为一个向量进行进一步处理（比如更新参数），你需要先将每个张量展平。grad.view(-1) 正是用于这个目的：将任意形状的张量重塑为一维向量。

---

### 张量中提取指定索引元素

```py
log_probs = torch.log(actor(states).gather(1, actions))
```

.gather(dim, index) 是一个PyTorch操作，用于根据index指定的索引在指定的维度dim上选取元素。
在这个上下文中，actions 应该是一个形状为 [batch_size, 1] 的张量，包含每个状态对应的选择动作的索引。
gather(1, actions) 从 actor(states) 的输出中沿着第二维（dim=1，动作维度）收集动作概率。这意味着从每行（每个状态的动作概率分布）中提取由 actions 指定的动作的概率。

---

### 显式向量化处理参数

在 PyTorch 中，`torch.nn.utils.convert_parameters.vector_to_parameters` 函数用于将一个一维向量转换成一个与网络参数形状相匹配的参数集。这通常用在优化算法中，尤其是在那些需要显式地处理参数作为向量的场景，如某些高级优化技术或者参数更新策略中。

- 详细语法解析

函数 `torch.nn.utils.convert_parameters.vector_to_parameters(vector, parameters)` 接受两个参数：

1. **vector**: 这是一个一维向量，其中包含了应该被加载到模型中的所有参数值。这个向量通常是通过展平模型的所有参数获得的（例如使用 `torch.nn.utils.parameters_to_vector` 函数）。

2. **parameters**: 这是一个参数迭代器，通常可以通过模型的 `.parameters()` 方法获得。它应该包括所有将要被更新的参数。

- 函数作用

这个函数的作用是将一维向量中的参数值赋值回具有特定形状的模型参数中。这通常在以下情况中非常有用：

- **参数恢复**：在进行一些操作（如优化步骤）后，可能需要将一维向量形式的参数恢复到模型的参数形状中。
- **优化算法**：在某些优化算法中，尤其是那些需要计算参数的Hessian向量积的算法中，参数可能会被临时转换成向量形式进行计算，之后需要转换回原始形状以进行模型更新。


示例代码：

假设你有一个简单的神经网络模型，你需要将其参数展平后的向量恢复到模型的参数中：

```python
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, convert_parameters

# 定义一个简单的网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化网络
model = SimpleNet()

# 获取模型的所有参数
original_parameters = list(model.parameters())

# 将参数转换为一个向量
params_vector = parameters_to_vector(original_parameters)

# 模拟一些变化到这个向量
new_params_vector = params_vector + 0.5

# 将更新后的向量参数恢复到模型的参数中
convert_parameters.vector_to_parameters(new_params_vector, model.parameters())

# 检查模型参数是否更新
for param in model.parameters():
    print(param)
```

在这个例子中，`vector_to_parameters` 函数用于将修改后的参数向量`new_params_vector`重新赋值给模型的参数。这是在复杂的自定义优化流程中常见的一个步骤。

---

### 一个元素的张量到标准 Python 类型的转换

在你的 take_action 函数中，使用 [action.item()] 来返回动作是为了处理几个实际的问题，并且确保代码的通用性和效率。这里是具体原因的详细解释：

张量到标准 Python 类型的转换：
action.item(): 这个方法是用来从一个只有一个元素的张量中提取其值并转换成一个标准的Python数值（如float或int）。action_dist.sample() 会返回一个张量，哪怕这个张量只包含一个数值（这在大多数单一动作决策场景中都是这样）。使用 .item() 可以有效地从张量中提取出那个单一的动作值。

---

### 在分部下计算log_prob概率

```py
action_dists = torch.distributions.Normal(mu, std)
log_probs = action_dists.log_prob(actions)
# 这里的actions是一个轨迹中的动作序列
```

功能说明:
- action_dists: 这是一个概率分布对象，通常是由策略网络（例如前面讨论的 PolicyNetContinuous）生成的。在连续动作空间中，这个分布通常是正态（高斯）分布，由动作的均值（mu）和标准差（std）参数化。
- log_prob(actions): 这是一个方法，用于计算给定动作 actions 在 action_dists 定义的概率分布下的对数概率。对数概率是很多强化学习算法中的关键组成部分，尤其是那些基于概率的策略梯度方法。

now-pc-updata