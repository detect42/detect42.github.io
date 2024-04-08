---
title: Python Grammar
tags: python
abbrlink: 22c2c40d
date: 2024-03-18 16:45:07
---
## Python

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

## Numpy

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



## Torch

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