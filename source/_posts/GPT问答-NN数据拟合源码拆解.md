---
title: GPT问答_NN数据拟合源码拆解
tags: [GPT,神经网络,机器学习]
categories: 
- DL
- code
abbrlink: a43615c7
date: 2023-10-19 11:26:03
---

## Sample code for PyTorch

```python
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

这是一个函数，看起来用于设置随机数生成器的种子以实现可重复性。以下是每一句的解释：

1. `def same_seed(seed):`
   - 这是一个Python函数的定义，名为`same_seed`，它接受一个参数`seed`。

2. `'''Fixes random number generator seeds for reproducibility.'''`
   - 这是函数的文档字符串，提供了对函数目的的简要描述。

3. `torch.backends.cudnn.deterministic = True`
   - 这一行设置了PyTorch的CuDNN后端以确定性模式。这意味着CuDNN将使用固定的算法，以确保每次运行代码时生成的结果都相同，以实现可重复性。

4. `torch.backends.cudnn.benchmark = False`
   - 这一行禁用了CuDNN的性能基准。通常，CuDNN会根据硬件性能自动选择最佳算法，但禁用它可以确保每次都使用相同的算法，以实现可重复性。

5. `np.random.seed(seed)`
   - 这一行设置了NumPy库中随机数生成器的种子，确保NumPy的随机操作也是可重复的。

6. `torch.manual_seed(seed)`
   - 这一行设置了PyTorch的随机数生成器的种子，以确保PyTorch中的随机操作也是可重复的。

7. `if torch.cuda.is_available():`
   - 这一行检查是否有可用的CUDA（GPU）设备。如果系统支持GPU，并且GPU可用，它将执行以下代码块。

8. `torch.cuda.manual_seed_all(seed)`
   - 这一行设置所有可用的CUDA设备上的随机数生成器的种子，以确保在使用GPU时也能实现可重复性。

这个函数的作用是在深度学习任务中设置各种随机数生成器的种子，以确保在每次运行相同的代码时，结果都是可重复的，这对于实验和调试非常有用。

---------

```python
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
```

这是一个函数，用于将提供的训练数据集分割成训练集和验证集。以下是每一句的解释：

1. `def train_valid_split(data_set, valid_ratio, seed):`
   - 这是一个Python函数的定义，名为`train_valid_split`，它接受三个参数：`data_set`（数据集），`valid_ratio`（验证集比例），和`seed`（随机数种子）。

2. `'''Split provided training data into training set and validation set'''`
   - 这是函数的文档字符串，提供了对函数目的的简要描述。

3. `valid_set_size = int(valid_ratio * len(data_set))`
   - 这一行计算了验证集的大小。`valid_ratio` 是一个介于0和1之间的比例，表示验证集在总数据集中的比例。`len(data_set)` 给出了总数据集的大小。

4. `train_set_size = len(data_set) - valid_set_size`
   - 这一行计算了训练集的大小，即总数据集大小减去验证集的大小。

5. `train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))`
   - 这一行使用`random_split`函数将数据集分割成训练集和验证集。`random_split`函数接受三个参数：数据集本身 (`data_set`)，一个包含两个整数的列表，分别表示训练集和验证集的大小 (`[train_set_size, valid_set_size]`)，以及一个随机数生成器 (`torch.Generator().manual_seed(seed)`)，以确保拆分是可重复的。函数返回训练集 (`train_set`) 和验证集 (`valid_set`)。

6. `return np.array(train_set), np.array(valid_set)`
   - 最后，函数返回训练集和验证集，这些集合被转换为NumPy数组（`np.array`），以便后续使用。

这个函数的作用是将给定的训练数据集按照指定的验证集比例和种子随机分成训练集和验证集，以用于深度学习模型的训练和验证。

### 关于随机数生成器

您可以使用其他随机数生成器替代`torch.Generator().manual_seed(seed)`，但具体要使用哪个生成器取决于您的需求和代码中使用的深度学习框架。

在PyTorch中，`torch.Generator().manual_seed(seed)`是一种设置种子以确保随机性可重复的方式。如果您想使用其他生成器，通常可以考虑使用Python内置的`random`模块或NumPy的随机模块，例如`numpy.random.RandomState`。以下是一些示例：

1. 使用Python内置的`random`模块：
   ```python
   import random

   random.seed(seed)
   ```

2. 使用NumPy的随机模块：
   ```python
   import numpy as np

   rng = np.random.RandomState(seed)
   ```

这些生成器也可以用于设置随机种子以确保可重复性。但请注意，如果您使用的是PyTorch或其他深度学习框架的功能，通常最好使用该框架提供的生成器，以确保与框架的其他随机操作相协调。在这种情况下，使用`torch.Generator().manual_seed(seed)`可能是更好的选择。

-----

```python
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds
```

这是一个Python函数，用于使用PyTorch模型进行推断（预测）。下面是它的逐句解释：

1. `def predict(test_loader, model, device):`
   - 这是一个Python函数的定义，名为`predict`，它接受三个参数：`test_loader`，`model`和`device`。

2. `model.eval()`
   - `model.eval()` 是PyTorch中的方法，将神经网络模型切换到评估（evaluation）模式。在评估模式下，模型不会进行梯度计算，通常用于推断（inference）阶段，以加快计算速度并节省内存。

3. `preds = []`
   - 创建一个空列表 `preds` 以存储模型的预测结果。

4. `for x in tqdm(test_loader):`
   - 这是一个循环，遍历`test_loader`，`test_loader`通常是一个数据加载器（data loader），用于提供测试数据。`tqdm`用于在循环中创建一个进度条，以便跟踪处理进度。

5. `x = x.to(device)`
   - 将输入数据 `x` 移动到指定的计算设备（通常是GPU或CPU），由`device`参数指定。

6. `with torch.no_grad():`
   - `with torch.no_grad():` 是一个上下文管理器，用于告诉PyTorch不要跟踪计算梯度，因为在评估模式下，不需要梯度信息。

7. `pred = model(x)`
   - 使用模型 `model` 对输入数据 `x` 进行预测，生成预测结果 `pred`。

8. `preds.append(pred.detach().cpu())`
   - 将预测结果 `pred` 从计算设备上移除（`.detach()`）并移动到CPU上（`.cpu()`），然后将它添加到 `preds` 列表中。

9. `preds = torch.cat(preds, dim=0).numpy()`
   - 将列表中的所有预测结果 `preds` 连接成一个张量，连接维度为 `dim=0`，然后将结果转换为NumPy数组，最终返回这个NumPy数组。

10. `return preds`
    - 返回包含所有测试样本预测结果的NumPy数组。

这个函数的作用是使用给定的PyTorch模型对测试数据进行推断，返回一个包含所有预测结果的NumPy数组。在这个过程中，确保模型处于评估模式，不进行梯度计算，提高推断效率。

----

`with torch.no_grad():` 上下文管理器的作用是告诉PyTorch在这个上下文中不要跟踪计算梯度。这在推断（inference）阶段非常有用，具体原因如下：

1. **梯度计算开销：** 在训练模型时，我们需要计算每个操作的梯度，以便进行反向传播和参数更新。然而，在推断阶段，我们通常不需要这些梯度信息，因为我们只是用模型进行前向传播，计算预测结果。

2. **内存优化：** 禁用梯度跟踪可以显著减少内存占用。在训练期间，PyTorch会为每个操作保留梯度信息，这可能占用大量内存。在推断期间，我们可以通过使用`with torch.no_grad():` 来减少内存使用，特别是当处理大量数据时。

3. **加速计算：** 由于不进行梯度计算，禁用梯度跟踪还可以提高计算速度。这对于实时应用或需要高性能的情况非常有帮助。

因此，当你在推断模型时，使用`with torch.no_grad():` 可以提高效率，减少内存占用，并确保不会因不必要的梯度计算而导致性能下降。

----

`preds.append(pred.detach().cpu())` 的主要原因是将预测结果从GPU（或其他计算设备）移动到CPU是为了使其更易于处理和使用。这样做通常是有几个原因的：

1. **兼容性：** 在PyTorch中，如果你的模型在GPU上计算，那么GPU上的张量与CPU上的张量具有不同的类型。在某些情况下，你可能希望将结果移回CPU，以便与其他CPU上的数据进行处理和组合。

2. **NumPy互操作：** 很多时候，你可能需要将PyTorch的张量转换为NumPy数组以便进行进一步的数据处理或可视化。通常，NumPy更容易与CPU上的数据交互，所以将数据移回CPU使得这一过程更简单。

3. **内存管理：** GPU上的内存通常有限，将不再需要的数据移回CPU可以释放GPU内存，以便在后续计算中使用。这对于大型模型或大规模数据处理非常重要。

总之，将预测结果从GPU移回CPU通常是为了提高兼容性、便于后续处理，以及有效地管理内存。但需要根据具体的应用场景和需求来决定是否需要这个步骤。如果你只在GPU上进行后续计算，并且不需要与CPU上的数据交互，那么可以跳过这一步。


--------


```python
class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
```

这是一个自定义的PyTorch数据集类 `COVID19Dataset`，用于加载和处理COVID-19数据集。以下是该类的逐句解释：

1. `class COVID19Dataset(Dataset):`
   - 这是一个类定义，创建了一个名为 `COVID19Dataset` 的PyTorch数据集类，它继承自 `Dataset` 类。

2. `def __init__(self, x, y=None):`
   - 这是类的构造函数，它接受两个参数：`x` 和可选的 `y`，其中 `x` 是特征数据，`y` 是目标数据（如果提供了）。如果没有提供目标数据，你可以在构造数据集对象时将 `y` 参数设置为 `None`。

3. `if y is None: self.y = y`
   - 这里检查是否提供了目标数据 `y`。如果没有提供，`self.y` 被设置为 `None`，表示这是一个用于预测的数据集，而不是用于监督学习的数据集。

4. `self.y = torch.FloatTensor(y)`
   - 如果提供了目标数据 `y`，则将其转换为PyTorch张量并将其存储在 `self.y` 中。这假设目标数据是浮点数类型。

5. `self.x = torch.FloatTensor(x)`
   - 将特征数据 `x` 转换为PyTorch张量并将其存储在 `self.x` 中。同样，这假设特征数据是浮点数类型。

6. `def __getitem__(self, idx):`
   - 这是一个特殊方法，用于定义如何获取数据集中的单个样本。根据是否提供了目标数据，它返回不同的内容。如果没有提供目标数据，只返回特征数据；否则，返回特征数据和目标数据。

7. `def __len__(self):`
   - 这是另一个特殊方法，用于返回数据集的长度，即数据集中包含多少个样本。它返回特征数据 `self.x` 的长度。

这个自定义数据集类的主要作用是为PyTorch提供一种有效的方式来加载和处理COVID-19数据，适用于监督学习和预测任务，具体取决于是否提供了目标数据。这样的数据集类通常在训练和测试神经网络时非常有用。

-------

```py
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x
```

这是一个自定义的PyTorch神经网络模型类 `My_Model`，以下是该类的逐句解释：

1. `class My_Model(nn.Module):`
   - 这是一个类定义，创建了一个名为 `My_Model` 的PyTorch神经网络模型类，它继承自 `nn.Module`。

2. `def __init__(self, input_dim):`
   - 这是类的构造函数，它接受一个参数 `input_dim`，表示输入数据的维度。

3. `super(My_Model, self).__init__()`
   - 在构造函数中，首先调用父类 `nn.Module` 的构造函数以初始化模型。

4. `self.layers = nn.Sequential(...)`
   - 这里定义了神经网络的结构。`self.layers` 是一个由多个层组成的序列（Sequential）模块。模型的结构包括三个线性层（全连接层）和两激活函数层。具体结构如下：
     - 输入层：接受大小为 `input_dim` 的输入。
     - 第一个隐藏层：包含 16 个神经元，并使用 ReLU（Rectified Linear Unit）激活函数。
     - 第二个隐藏层：包含 8 个神经元，并使用 ReLU 激活函数。
     - 输出层：包含 1 个神经元，用于回归任务。

5. `def forward(self, x):`
   - 这是模型的前向传播方法，用于定义数据在模型中的传递过程。
   - `x = self.layers(x)` 执行了输入数据 `x` 经过定义的层序列 `self.layers` 的前向传播。
   - `x = x.squeeze(1)` 将输出张量中的维度 1 压缩，从 (B, 1) 到 (B)，其中 B 表示批量大小（Batch Size）。
   - 最终返回处理后的张量 `x` 作为模型的输出。

-----

这是一个很好的问题，涉及到Python面向对象编程和继承中的特定语法。在这里，`My_Model`是一个类，它继承自`nn.Module`，这是PyTorch中所有神经网络模型的基类。

`super(My_Model, self).__init__()`是一个调用父类`nn.Module`的`__init__`方法的特殊方式。让我们分解这个语法：

1. **`super()`**: 这是一个Python内置函数，用于调用父类（在这种情况下是`nn.Module`）的方法。这是在子类中调用父类的方法的推荐方式，特别是当有多重继承时。

2. **`My_Model, self`**: 这是`super()`函数的两个参数。
   - `My_Model`: 这是当前的子类。
   - `self`: 这是当前对象的实例。在Python中，类的方法总是接收其自身的实例作为第一个参数，通常命名为`self`。

3. **`.__init__()`**: 这是调用的方法名。在这里，我们正在调用`nn.Module`类的初始化方法`__init__`。

- 为什么要这样做？当您创建一个PyTorch模型并继承自`nn.Module`时，您需要确保`nn.Module`的初始化代码也被执行。这对于后续的一些PyTorch内部操作，如模型的参数注册等，是必要的。

- 至于为什么`self`有时在前，有时在后：在方法定义中，`self`总是作为第一个参数出现，表示对象本身。但在使用`super()`时，`self`通常作为第二个参数，这是Python的语法规定。

---

这个自定义模型类是一个典型的前馈神经网络（Feedforward Neural Network），适用于回归任务，它将输入数据通过一系列全连接层和激活函数层传递，最终产生一个输出。你可以根据具体任务的需要，修改模型的结构。

在PyTorch中，每次创建`nn.Linear`层时，你实际上是创建了一个新的线性层对象，而不是修改或覆盖以前的线性层。这是因为每个`nn.Linear`层都有自己的权重和偏置，它们是独立的。

所以，当你创建`nn.Linear(16, 8)`时，它实际上是一个新的线性层，而不会影响先前创建的`nn.Linear(input_dim, 16)`。这两个层是不同的对象，它们在模型中执行不同的转换。

这种分层的方法允许你创建具有不同权重和偏置的不同层，以便模型可以学习不同的特征表示。它是神经网络的关键之一，因为不同的层可以学习不同的特征并协同工作以解决问题。

------

在选择神经网络的架构时，特别是当考虑输入层到输出层之间的层大小时，没有固定的“一刀切”的规则。选择是否使用平滑的递减、指数递减或其他策略取决于特定的应用和问题的性质。但是，我们可以提供一些建议和观点来帮助决策：

1. **平滑递减**:
   - **优点**: 可能有助于逐渐提取特征，并避免在前几层中丢失大量信息。
   - **缺点**: 可能会增加模型的参数数量，导致更长的训练时间和可能的过拟合。

2. **指数递减**:
   - **优点**: 快速减少参数的数量，可以帮助模型更快地训练。
   - **缺点**: 如果递减得太快，可能会在网络的前部丢失大量信息。

3. **问题的性质**: 对于一些问题，特征的维度可以迅速减少，因为许多输入特征可能并不都是必要的。而对于其他问题，可能需要更平滑的递减，以捕获更复杂的特征表示。

4. **实验与验证**: 最佳的方法通常是尝试多种架构，并使用验证集来评估它们的性能。这可以帮助确定哪种策略（平滑递减、指数递减或其他）对于特定的问题最有效。

5. **避免过拟合**: 如果网络太大，有太多的参数，它可能会过拟合训练数据。可以使用正则化技术（如dropout、L2正则化等）来对抗这个问题。

6. **考虑计算资源**: 更大的网络意味着需要更多的计算资源来训练。根据可用的硬件，你可能需要权衡网络的大小。

总之，选择网络架构是一个需要实验和迭代的过程。理想的层大小和结构可能因问题而异。在实践中，通常需要尝试多种不同的配置，并基于验证性能进行选择。

-----

```python
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```

这是一个用于选择特征列的函数 `select_feat`，它用于执行回归任务。以下是这个函数的逐句解释：

1. `def select_feat(train_data, valid_data, test_data, select_all=True):`
   - 这是函数的定义，它接受四个参数：`train_data`，`valid_data`，`test_data`，以及一个布尔值参数 `select_all`，默认为 `True`。

2. `y_train, y_valid = train_data[:,-1], valid_data[:,-1]`
   - 这两行代码从 `train_data` 和 `valid_data` 中提取目标变量（标签），通常在回归任务中，这是模型要预测的值。

3. `raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data`
   - 这三行代码从输入数据中分别提取特征，通过切片 `[:,:-1]` 取掉最后一列，以便获得没有目标变量的特征数据。

4. `if select_all: feat_idx = list(range(raw_x_train.shape[1]))`
   - 这是一个条件语句，如果 `select_all` 为 `True`，则 `feat_idx` 将包含所有特征的索引。这意味着将使用所有可用的特征进行建模。

5. `else: feat_idx = [0,1,2,3,4]`
   - 如果 `select_all` 为 `False`，则 `feat_idx` 将包含一个手动选择的特征索引列表。这是一个示例，表示你可以手动选择使用的特征列，以便建模。

6. 最后，函数返回了已选择特征的训练集、验证集和测试集，以及相应的目标变量。所选的特征列由 `feat_idx` 决定，如果 `select_all` 为 `True`，则使用所有特征。

这个函数的目的是根据 `select_all` 参数，选择特征列，并返回相应的数据子集，以便用于回归任务的训练和测试。根据具体问题，你可以选择使用所有特征或手动选择感兴趣的特征列。


-----


```python
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```

接下来一步一步分析：

```pyhton
criterion = nn.MSELoss(reduction='mean')
```
这行代码用于定义损失函数，其中 `nn.MSELoss` 表示均方误差损失（Mean Squared Error Loss），通常用于回归任务。让我解释一下它的各个部分：

- `nn.MSELoss`: 这部分表示你正在使用PyTorch的均方误差损失函数。均方误差是回归任务中常用的损失函数之一，它用于衡量模型的预测值与实际目标值之间的差距。

- `reduction='mean'`: 这是损失函数的一个参数，它定义了如何计算损失的标量值。在这里，设置为 `'mean'` 表示计算所有样本的均方误差，然后求平均值，得到一个标量损失值。其他可能的选项包括 `'sum'`（计算总和）和 `'none'`（不进行降维，保持与输入张量相同的维度）。

所以，`nn.MSELoss(reduction='mean')` 的作用是定义了一个均方误差损失函数，它会计算模型的预测值与实际目标值之间的均方误差，并将其求平均得到一个标量损失值，用于衡量模型的性能。这是回归任务中常见的损失函数之一。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
```

这是PyTorch中定义优化器的语句。
`optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)`

- `optimizer`: 是一个变量，存储了优化器的实例。

- `torch.optim.SGD`: 这是PyTorch中随机梯度下降（SGD）的实现。

- `model.parameters()`: 这会返回模型中所有的可训练参数。这些参数是我们在训练过程中需要更新的。

- `lr=config['learning_rate']`: 这设置了学习率为配置字典中的`'learning_rate'`键对应的值。学习率是一个关键的超参数，它决定了在每次更新时参数应该改变的幅度。如果学习率太高，模型可能无法收敛；如果太低，则训练可能会非常慢。

- `momentum=0.9`: 动量是SGD优化器的一个变种，它在更新时考虑了前几次的梯度，以便更快地收敛并减少震荡。动量值通常设置在0.5到0.9之间。在此，它被设置为0.9，意味着当前梯度的90%和前一次的梯度的10%将被用来更新参数。

总之，这行代码创建了一个随机梯度下降优化器，该优化器用于更新模型的参数，它使用了动量来加速收敛。

```pyhton
writer = SummaryWriter() # Writer of tensoboard.
```

这行代码与其注释是关于TensorBoard的可视化工具的。

`writer = SummaryWriter()`

- `SummaryWriter()`: 这是PyTorch的`torch.utils.tensorboard`中的一个类，它允许用户记录数据，例如损失、准确率等，以便稍后在TensorBoard中进行可视化。

- `writer`: 是创建的`SummaryWriter`对象的实例。之后，你可以使用这个实例来添加标量、图像、文本等到TensorBoard日志中。

注释 `# Writer of tensorboard.` 提供了有关这一行代码的简短描述，即这是TensorBoard的写入器。

TensorBoard 是一个由TensorFlow提供的可视化工具，但也可以与PyTorch一起使用。它可以帮助用户可视化模型的训练进程、结构、梯度分布等。这在调试、优化和理解模型时非常有用。使用`SummaryWriter`，你可以很容易地将这些数据记录到日志中，并在TensorBoard中查看。

```py
train_pbar = tqdm(train_loader, position=0, leave=True)
```

这行代码涉及`tqdm`库，它是一个用于在Python中显示进度条的库。当你有一个长时间运行的循环或任务时，`tqdm`可以为你提供一个视觉上友好的进度条，以便你知道任务的进展情况。

让我们分析这行代码：

`train_pbar = tqdm(train_loader, position=0, leave=True)`

1. **`tqdm(train_loader)`**: 
   - `train_loader`: 是一个可迭代对象，通常是一个PyTorch数据加载器。这意味着我们将在进度条中追踪`train_loader`的迭代进度。
   - 当你迭代`train_pbar`（例如在一个for循环中）时，它实际上会迭代`train_loader`，同时更新和显示进度条。

2. **`position=0`**: 
   - 这定义了进度条在屏幕上的位置。当你有多个进度条或在多线程环境中使用`tqdm`时，`position`参数确保进度条在正确的位置显示。

3. **`leave=True`**: 
   - 当进度条完成时，这个参数决定进度条是否仍然保留在屏幕上。如果为`True`，进度条在完成后将保留在屏幕上；如果为`False`，进度条完成后将被清除。

总的来说，这行代码创建了一个用于显示`train_loader`迭代进度的进度条，并将其保存在`train_pbar`变量中。当你在后续的代码中迭代`train_pbar`时，进度条会自动更新。

```pyhton
optimizer.zero_grad()
```

如果不在每个batch开始时调用`optimizer.zero_grad()`清空参数的梯度，会导致问题。具体来说，PyTorch的设计是为了累积梯度，也就是说，每当`.backward()`被调用时，梯度都会累积到现有的梯度上，而不是替换它们。

这种累积的特性有以下后果：

1. **错误的参数更新**：如果你不清空梯度，那么新的梯度值将被添加到上一个batch的梯度值上。这意味着，模型的参数更新会基于多个batch的梯度累积，而不是仅仅基于当前batch的梯度。这可能会导致模型收敛得更慢或者完全不收敛。

2. **梯度爆炸**：累积的梯度可能会非常大，尤其是在深度网络或大数据集上。这可能会导致梯度爆炸，即梯度变得非常大，导致模型参数在训练过程中变得不稳定。

3. **增加的内存使用**：持续的梯度累积可能会导致更大的内存占用。

4. **不正确的模型评估**：如果你在评估模型时忘记清空梯度，并执行了`.backward()`（虽然在评估时通常不需要），那么梯度仍然会累积，这可能会影响到下一次的训练周期。

为了避免这些问题，建议在每个batch开始时调用`optimizer.zero_grad()`，确保在每次前向和反向传播时都是从零开始计算梯度。

PyTorch选择累积梯度而不是直接替换具有其特定的原因和用途，这种设计为某些特定的模型和技术提供了灵活性：

1. **支持更复杂的模型和损失结构**：有时，我们可能需要从多个源头计算梯度，并将它们聚合。例如，当我们有多个损失函数时，我们可能希望分别计算每个损失的梯度，并在之后将它们相加。累积梯度允许我们这样做。

2. **RNNs和BPTT**：对于循环神经网络（RNNs）和通过时间反向传播（BPTT），由于RNN的时间步长，我们可能希望在多个时间步上累积梯度，然后一次性更新。

3. **稀疏更新**：对于一些特定的优化技术，我们可能不想在每个批次结束时立即更新所有的参数。通过梯度累积，我们可以在多个批次上计算梯度，然后再进行一次更大的更新。

4. **节省计算资源**：在某些场景下，特别是当GPU内存有限时，使用更小的批次进行前向和反向传播，然后累积梯度，可以模拟更大批次的效果而不需要增加计算负担。

5. **支持多GPU训练**：当使用多个GPU进行数据并行处理时，每个GPU可以处理一个批次的子集并计算梯度。然后，这些梯度可以在所有GPUs之间聚合（累积）。

考虑到这些情况，PyTorch的设计者选择了更灵活的累积梯度方法。但这也意味着，在大多数常规训练场景中，开发者需要记住在每个批次开始时清零梯度，以避免不必要的累积。

----

```pyhton
writer.add_scalar('Loss/train', mean_train_loss, step)
```

这行代码使用`writer`（一个`SummaryWriter`的实例，通常用于TensorBoard日志记录）来添加一个标量值到日志中。这个标量通常用于记录和可视化模型的训练过程中的某些指标，如损失或准确率。我将为您详细解释这行代码：

`writer.add_scalar('Loss/train', mean_train_loss, step)`

1. `add_scalar`: 这是`SummaryWriter`类的一个方法，用于添加一个标量到TensorBoard日志中。

2. `'Loss/train'`: 这是要添加的标量的标签或名称。在TensorBoard中，你将看到一个叫做"Loss/train"的图表。

3. `mean_train_loss`: 这是要记录的实际数值。在此，它是当前训练周期内所有批次的平均损失。

4. `step`: 这表示当前的全局步骤或迭代。它是一个整数，通常用于表示自训练开始以来已经处理过的批次数量。在TensorBoard的图表中，它将作为x轴，允许你随着时间看到损失的变化。

总的来说，这行代码将当前训练周期的平均损失记录到TensorBoard的日志中，允许你在训练过程中可视化和跟踪损失的变化。

```python
loss_record.append(loss.detach().item())
```

 这行代码涉及了将当前批次的损失值添加到一个记录损失的列表中。我会为您逐步解析这行代码：

`loss_record.append(loss.detach().item())`

1. `loss_record`: 这是一个Python列表，用于在整个训练周期内累积每个批次的损失值。

2. `append()`: 这是Python列表的一个方法，用于在列表的末尾添加一个元素。

3. `loss`: 这是计算得到的损失Tensor，它还包含关于模型参数的梯度信息。

4. `detach()`: 这是一个PyTorch Tensor的方法。它返回一个新的Tensor，这个新Tensor与原始Tensor共享数据但不需要梯度计算（即它不会跟踪计算历史）。这在此上下文中是有用的，因为我们只关心损失的数值，不需要保留关于损失的任何计算历史。

5. `item()`: 这是一个PyTorch Tensor的方法。对于只包含一个元素的tensor，它将返回tensor中的这个值作为一个标准的Python数字。在这种情况下，它返回损失的数值。

所以，这行代码的主要目的是取得当前批次的损失值，并将其添加到`loss_record`列表中，以便后续可以计算整个训练周期的平均损失或进行其他统计分析。

```python
if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```

这段代码实现了检查模型的验证损失是否改进，并根据结果决定是否保存模型和是否提前终止训练（early stopping）。下面是逐句的解释：

1. `if mean_valid_loss < best_loss:`
   - 判断当前的平均验证损失是否比之前记录的最佳验证损失更低。

2. `best_loss = mean_valid_loss`
   - 如果当前的验证损失是更低的，那么更新`best_loss`为当前的`mean_valid_loss`。

3. `torch.save(model.state_dict(), config['save_path'])`
   - 保存当前模型的参数到指定的路径。这里假定你只想保存最佳的模型，即验证损失最低的模型。

4. `print('Saving model with loss {:.3f}...'.format(best_loss))`
   - 打印一条消息，告知用户模型已被保存，并显示最佳的验证损失。

5. `early_stop_count = 0`
   - 重置早停计数器。这意味着模型在这个周期是有所改进的。

6. `else:`
   - 如果模型没有改进。

7. `early_stop_count += 1`
   - 早停计数器增加1。这意味着模型在这个周期没有改进。

8. `if early_stop_count >= config['early_stop']:`
   - 判断早停计数器是否达到预设的阈值。如果连续多个周期模型都没有改进，那么这个条件将为真。

9. `print('\nModel is not improving, so we halt the training session.')`
   - 打印一条消息，告知用户模型已经连续多个周期没有改进。

10. `return`
    - 提前结束`trainer`函数的执行。这将终止训练过程。

简而言之，这段代码的目的是在每个训练周期后检查模型的性能。如果模型在验证集上表现更好，它会保存模型。如果模型连续多个周期没有改进，它会终止训练。这是一个常用的策略，称为早停，可以帮助避免模型过度拟合并减少不必要的训练时间。


------

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

这段代码主要是设置运行训练的配置参数。我将为您逐步解析这段代码：

1. `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
   - 这行代码决定模型和数据应该在哪个设备上运行。如果CUDA（GPU支持）可用，它将选择GPU（`'cuda'`），否则它将选择CPU（`'cpu'`）。

2. `config = {...}`
   - 定义一个名为`config`的字典，其中包含多个训练参数和设置。

3. `'seed': 5201314`
   - 设置一个随机数种子，确保实验的可重复性。

4. `'select_all': True`
   - 一个布尔标志，表示是否使用所有特征进行训练。

5. `'valid_ratio': 0.2`
   - 表示验证集的大小是训练集大小的20%。

6. `'n_epochs': 3000`
   - 指定训练的总周期数。

7. `'batch_size': 256`
   - 在每次迭代中，模型将看到的数据批次的大小。

8. `'learning_rate': 1e-5`
   - 设置学习率，它决定了在训练中参数更新的幅度。

9. `'early_stop': 400`
   - 早停策略的参数。如果模型在连续400个周期中没有改进，训练将终止。

10. `'save_path': './models/model.ckpt'`
   - 定义了模型保存的路径。

这个`config`字典为后续的训练步骤提供了一种方便的方式来访问和管理所有的配置参数。在实际项目中，使用这样的配置字典或配置文件是很常见的，因为它使代码更加整洁，并允许更容易地更改参数。

-----

```python
# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
```

这段代码涉及数据处理和准备阶段，确保数据能够正确地被模型所使用。下面我会为您逐句解释这段代码：

1. `# Set seed for reproducibility`
   - 这是一个注释，说明接下来的代码是为了确保实验的可重复性。

2. `same_seed(config['seed'])`
   - 调用一个名为`same_seed`的函数来设置随机数种子，以确保实验的可重复性。

3. `train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values`
   - 使用pandas库读取训练和测试数据，并将其转化为NumPy数组。

4. `train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])`
   - 使用`train_valid_split`函数将原始训练数据分为训练数据和验证数据。

5. 打印训练数据、验证数据和测试数据的大小。

6. `x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])`
   - 使用`select_feat`函数从原始数据中选择特定的特征。

7. 打印所选择的特征的数量。

8. 创建三个数据集：训练数据集、验证数据集和测试数据集。这里使用了名为`COVID19Dataset`的自定义数据集类。

9. 使用PyTorch的`DataLoader`创建数据加载器，它们可以按批次加载数据，并提供其他功能，如数据打乱和内存管理。

- `shuffle=True`：在训练和验证加载器中使用，表示在每个训练周期中都要打乱数据。
- `pin_memory=True`：当使用GPU时，这可以加速数据从CPU传输到GPU的过程。

总之，这段代码的目的是从CSV文件中加载数据、处理数据、选择特征，并为训练、验证和测试准备数据加载器。

```py
pd.read_csv('./covid.train.csv').values
```
这行代码使用pandas库来读取CSV文件，并将其转换为NumPy数组。我会为您详细解释：

1. `pd.read_csv('./covid.train.csv')`: 使用pandas的`read_csv`函数从指定的路径（`'./covid.train.csv'`）读取CSV文件。这将返回一个pandas的DataFrame对象，它是一个二维标签化的数据结构。

2. `.values`: 这是一个DataFrame的属性，它会返回DataFrame中的数据作为NumPy数组。这对于进一步的数值处理或与需要NumPy数组的库（如PyTorch、Scikit-learn等）的互操作性很有用。

所以，这行代码的结果是将`'./covid.train.csv'`文件中的数据读入一个NumPy数组。这个数组可以直接用于大多数数据处理和机器学习任务。

```py
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
```
这行代码使用PyTorch的`DataLoader`来创建一个数据加载器，这个加载器用于按批次加载测试数据。我会为您逐步解析这行代码：

`test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)`

1. `DataLoader`: 这是PyTorch提供的一个类，它允许您按批次加载数据，并提供其他功能，如数据打乱、多线程数据加载等。

2. `test_dataset`: 这是您要加载的数据集，通常是一个PyTorch `Dataset`实例。

3. `batch_size=config['batch_size']`: 这定义了每个批次的大小。`config['batch_size']`从先前定义的`config`字典中获取批次大小。

4. `shuffle=False`: 这意味着在加载数据时不应随机打乱数据。这在测试时是常见的，因为您通常只想按原始顺序处理和评估测试数据。

5. `pin_memory=True`: 这是一个优化，当您使用GPU时，它可以加速数据从CPU传输到GPU的过程。在内部，它会为数据预留一个固定的、锁定的（或“钉住的”）内存区域，这使得数据传输更加高效。

总之，这行代码创建了一个测试数据的加载器，这个加载器按照指定的批次大小加载数据，不会打乱数据，并优化了数据的内存管理，特别是在使用GPU时。

`batch_size`是训练神经网络时的一个关键超参数，它决定了每次参数更新所使用的样本数量。`batch_size`的选择对训练的效率、效果以及模型的性能都有影响。下面列举了`batch_size`选择大或小时的一些差异和影响：

1. **计算效率**:
   - **大**: 利用矩阵运算的并行性，特别是在GPU上，大批次可以提高计算的效率。
   - **小**: 更频繁地进行参数更新，可能会导致硬件资源（特别是GPU）的利用率不足。

2. **内存使用**:
   - **大**: 需要更多的内存来存储大批次的数据和中间计算结果。
   - **小**: 使用更少的内存，允许在内存较小的设备上进行训练。

3. **收敛速度**:
   - **大**: 每个周期需要更少的参数更新，可能导致收敛速度变慢。
   - **小**: 更频繁的参数更新可能使模型更快地收敛。

4. **训练稳定性**:
   - **大**: 噪音较小，梯度估计更为准确，训练路径平滑。
   - **小**: 梯度估计可能存在更多的噪音，可能导致训练路径不稳定。

5. **泛化能力**:
   - **大**: 有研究显示，超大批次可能导致模型的泛化能力下降。
   - **小**: 小批次或随机梯度下降（SGD，即`batch_size=1`）经常被视为具有某种正则化效果，可能提高模型的泛化能力。

6. **卡在局部最小值**:
   - **大**: 由于梯度估计较为准确，可能更容易卡在非最优的局部最小值。
   - **小**: 更多的噪音可能帮助模型跳出某些局部最小值。

7. **调整学习率**:
   - **大**: 使用大批次可能需要对学习率进行调整，例如增大学习率。
   - **小**: 较小的学习率可能更适合小批次。

总的来说，`batch_size`的选择需要考虑多个因素，并可能需要实验来确定最佳值。在实践中，中等大小的批次（如32、64或128）往往是一个好的起点。

1. **超大批次可能导致模型的泛化能力下降**:
   
   泛化能力是指模型在未见过的数据上的表现。理论上，当使用小批次时，由于每批数据都带有一定的噪音，模型在训练过程中会看到更多的“噪声”，这种噪声可以起到某种正则化的效果，有助于提高模型的泛化能力。另一方面，当使用大批次时，梯度估计更为准确，噪音更少，这可能导致模型过度拟合训练数据，从而降低其在测试数据上的性能。实际上，一些研究发现，使用超大批次可以迅速减少训练损失，但可能会导致测试损失下降得较慢或不如预期。

2. **大批次与学习率的关系**:
   
   学习率定义了每次参数更新的幅度。当使用大批次时，梯度估计基于更多的数据，因此它们通常更稳定和准确。因此，可以使用较大的学习率而不会导致训练不稳定。另外，由于大批次训练的收敛速度可能较慢（因为每个周期的更新次数减少了），增大学习率也可以帮助加速收敛。

   另一方面，当使用小批次时，梯度估计可能会带有更多的噪音，因此较小的学习率可以帮助确保训练的稳定性，尤其是在训练的早期阶段。

   这也解释了为什么许多自适应学习率算法，如Adam，能够在各种批次大小下都工作得很好，因为它们根据梯度的统计信息动态地调整学习率。

综上所述，选择合适的批次大小和学习率是训练深度学习模型时的关键考虑因素，它们之间的关系也是深度学习研究和实践中的一个重要话题。


```py
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')
```

这段代码涉及加载一个训练好的模型、对测试数据进行预测，然后保存预测结果到CSV文件。我会为您逐步解析这段代码：

1. `def save_pred(preds, file):`
   - 这是定义一个名为`save_pred`的函数，用于保存预测结果。它接受两个参数：`preds`是预测值的列表或数组，`file`是要保存的文件名。

2. 在函数内部：
   - 使用`with open(file, 'w') as fp:`，它打开指定的文件以供写入。
   - `csv.writer(fp)`创建一个CSV写入器对象。
   - 使用`writer.writerow()`来写入行到CSV文件。

3. `model = My_Model(input_dim=x_train.shape[1]).to(device)`
   - 创建一个模型实例并将其移动到指定的计算设备。

4. `model.load_state_dict(torch.load(config['save_path']))`
   - 使用PyTorch的`torch.load`函数从先前指定的路径加载模型的状态字典（即模型的参数）。
   - 然后使用`load_state_dict`方法将这些参数加载到模型中。

5. `preds = predict(test_loader, model, device)`
   - 调用一个名为`predict`的函数（虽然该函数在给定的代码片段中未定义，但我们可以假设它执行模型的前向传递并返回预测）。
   - 它使用提供的测试数据加载器、模型和设备进行预测。

6. `save_pred(preds, 'pred.csv')`
   - 使用先前定义的`save_pred`函数将预测结果保存到`pred.csv`文件中。

总的来说，这段代码描述了如何从磁盘加载训练好的模型，使用该模型对测试数据进行预测，然后将这些预测保存到CSV文件中。


```py
model.load_state_dict(torch.load(config['save_path']))
```
这行代码是PyTorch中加载预训练模型权重的常见方式。让我们逐部分解析这行代码：

1. **`torch.load(config['save_path'])`**:
   - 使用PyTorch的`torch.load`方法从指定的路径加载数据。这通常是一个包含模型参数的字典，该字典在保存模型时使用`torch.save`方法创建。
   - `config['save_path']`是模型权重文件的路径。它从之前定义的`config`字典中获取。

2. **`model.load_state_dict(...)`**:
   - `load_state_dict`是PyTorch模型（或`nn.Module`实例）的一个方法，它负责将一个状态字典加载到模型中。
   - 这个状态字典包含模型的权重。当我们调用`load_state_dict`方法时，它会为模型中的每个层设置适当的权重。

总之，`model.load_state_dict(torch.load(config['save_path']))`这行代码的目的是从指定路径加载模型权重，并将这些权重应用到模型`model`中。这允许你继续使用之前训练过的模型，而不必重新从头开始训练。


------

`Adam`和`SGD`都是用于优化深度学习模型的梯度下降算法，但它们的工作原理有所不同。下面是它们之间的主要区别，以及各自的优缺点：

### Adam (Adaptive Moment Estimation):

**工作原理**:
- 结合了`Momentum`和`RMSprop`的优点。
- 使用梯度的一阶矩（mean）和二阶矩（uncentered variance）来自适应地调整每个参数的学习率。

**优点**:
1. **自适应学习率**: Adam会为每个参数自动调整学习率，这使得它对初始学习率和超参数选择的敏感性较低。
2. **快速收敛**: 由于Momentum的效果，Adam往往收敛得相对较快。
3. **适合大数据和参数**: 对于大型数据集和模型参数，Adam通常表现得很好。

**缺点**:
1. **内存占用**: 由于Adam需要为每个参数存储一阶和二阶矩估计，因此它的内存使用量通常比SGD大。
2. **可能过拟合**: 在某些情况下，特别是数据集较小的情况下，Adam可能会过拟合。

### SGD (Stochastic Gradient Descent):

**工作原理**:
- 在每次迭代中使用一部分（或一个）数据样本的梯度来更新模型参数。

**优点**:
1. **简单和高效**: SGD的实现很简单，计算效率也很高。
2. **泛化**: 在某些情况下，尤其是当正则化技巧与SGD结合使用时，SGD可能会得到更好的泛化性能。

**缺点**:
1. **需要仔细选择学习率**: SGD的收敛通常对初始学习率和学习率调度策略非常敏感。
2. **慢速收敛**: 相比于使用自适应学习率技术的优化器，SGD可能需要更多的迭代次数才能收敛。
3. **可能遇到局部最小值或鞍点**: 在某些非凸优化问题中，SGD可能会在局部最小值或鞍点附近停滞。

### 总结:
- **Adam**是一个自适应学习率的优化器，通常在不需要手动微调超参数的情况下就可以快速收敛。对于快速原型设计和大型数据集/模型，它往往是一个很好的选择。
- **SGD**虽然简单，但在某些情况下可能会提供更好的泛化性能。对于深度学习竞赛和高级研究项目，仍然值得考虑使用和微调SGD。

在实际应用中，建议尝试多种优化器，看看哪种方法在特定任务上表现最好。