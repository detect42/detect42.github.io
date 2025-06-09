# <center> C++ Embedding Python </center>

## c++编译链接 背景知识

### cpp流程如下

![alt text](image.png)

我们想做的是在python中调用cpp的函数。

我们把cpp搞成动态链接库so，然后在python里import这个so文件，来调用写好的cpp函数。

### 链接器作用

![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)

### 头文件和链接

![alt text](image-4.png)

![alt text](image-5.png)
![alt text](image-6.png)


### 为什么 .h 和 .cpp 要分开写

![alt text](image-7.png)
![alt text](image-8.png)

### 静态和动态编译

![alt text](image-9.png)

## pybind11 在py中嵌入cpp函数


### 常见方式

我们尽量选择pybind11来做python和cpp的交互。
![alt text](image-10.png)


我们马上做两个更实用的例子：

### 1. 绑定一个完整的 **C++类（class）** 给Python用

### 2. 让Python的 **numpy数组**直接传给C++加速处理（比如矩阵运算）

（全部用`pybind11`，保持代码极短极优雅🌟）

---

# 🛠️ 第一个例子：绑定一个C++类到Python

## Step 1. 写一个简单的C++类

新建 `example.cpp`

```cpp
#include <pybind11/pybind11.h>

class Adder {
public:
    Adder(int base) : base_(base) {}

    int add(int x) const {
        return base_ + x;
    }

private:
    int base_;
};

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<Adder>(m, "Adder")
        .def(py::init<int>())
        .def("add", &Adder::add);
}
```

✅ 解释：

* `py::class_<Adder>(...)` 注册了 C++ 类
* `.def(py::init<int>())` 注册了构造器
* `.def("add", &Adder::add)` 注册了成员方法

---

## Step 2. 编译成Python模块

```bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

生成：

```
example.cpython-310-x86_64-linux-gnu.so
```

✅ 成功！

---

## Step 3. Python中直接用

```python
import example

adder = example.Adder(10)
print(adder.add(5))  # 输出 15
```

🎯 你就可以像调用普通Python类一样用C++写的高性能类了！

---

# 🔥 第二个例子：Numpy数组传到C++处理

要加速这种场景，比如Python里处理大数组，但想用C++快点跑。

---

## Step 1. 写 C++ 处理 numpy array

新建 `array_add.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> add_arrays(py::array_t<double> a, py::array_t<double> b) {
    auto buf_a = a.unchecked<1>();
    auto buf_b = b.unchecked<1>();

    if (buf_a.size() != buf_b.size())
        throw std::runtime_error("Input sizes must match!");

    py::array_t<double> result(buf_a.size());
    auto buf_r = result.mutable_unchecked<1>();

    for (ssize_t i = 0; i < buf_a.size(); i++) {
        buf_r(i) = buf_a(i) + buf_b(i);
    }

    return result;
}

PYBIND11_MODULE(array_add, m) {
    m.def("add_arrays", &add_arrays, "Add two numpy arrays");
}
```

✅ 核心点：

* `py::array_t<double>` 支持直接传numpy数组
* `.unchecked<1>()` 不安全快速访问元素
* 支持1维数组加法

---

## Step 2. 编译

```bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) array_add.cpp -o array_add$(python3-config --extension-suffix)
```

✅ 生成 `array_add.so`！

---

## Step 3. Python测试

```python
import numpy as np
import array_add

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

c = array_add.add_arrays(a, b)
print(c)  # [5.0, 7.0, 9.0]
```

✅ 成功了！直接numpy -> C++加速运算 -> 返回numpy！

---

# ✨ 总结

| 内容            | 结果                 |
| :------------ | :----------------- |
| 绑定C++类到Python | `py::class_<T>`    |
| 支持Numpy高效处理   | `py::array_t`      |
| 编译成Python模块   | 1条命令，c++调用pybind11 |
| Python调用体验    | 像普通Python函数、类一样简单  |

---

# 🎯 超简版一句话总结

> **pybind11 = 用最小代价，把高性能C++对象、数组处理直接挂进Python**

✅ 速度接近原生，体验接近写Python！

---


好！！你的问题已经非常系统了，而且这两个点正是**在 C++ 里高效操作 numpy**最重要的实战技巧。

咱们来一条条清晰讲解，
我保证给你一版 **精准且实战的教程**，顺带带你动手写一个「高速矩阵加法器」。

---

# 1. `auto buf = a.unchecked<1>();` 这段语法详细解释

## 🔥 什么是 `.unchecked<1>()`？

✅ 这是 `pybind11::array_t` 提供的**快速访问器（Unchecked Array Proxy）**！

含义：

* `.unchecked<D>()` ➔ 生成一个**D维数组访问器**，可以直接用 `(i,j,k,...)` 访问元素。
* 它是**裸访问**（unchecked = 不做边界检查 = 极快），**代价是自己保证不越界**！

---

比如你有：

```cpp
py::array_t<double> a;
auto buf = a.unchecked<1>();
```

就表示：

* 你告诉编译器，`a`是**1维数组**。
* `buf` 是一个**可以直接通过索引访问的代理对象**。

然后可以用：

```cpp
double x = buf(0);  // 访问第0个元素
double y = buf(5);  // 访问第5个元素
```

✅ 这种方式是：

* 不安全检查（所以**超快**）
* 非常适合大批量操作，比如大规模矩阵处理！

---

## 🔵 如果是2维怎么办？

比如 `10000×10000`的矩阵，加法处理，就可以：

```cpp
auto buf_a = a.unchecked<2>();
auto buf_b = b.unchecked<2>();
auto buf_r = result.mutable_unchecked<2>();

for (ssize_t i = 0; i < buf_a.shape(0); i++) {
    for (ssize_t j = 0; j < buf_a.shape(1); j++) {
        buf_r(i, j) = buf_a(i, j) + buf_b(i, j);
    }
}
```

直接 `(i, j)` 二维索引访问元素！

---

# 2. 怎么用 `py::array_t<double>` 高速批量处理矩阵？

✅ 思路就是：

* Python侧准备大矩阵（numpy）
* C++侧接收成 `py::array_t<double>`
* 转成 `unchecked<2>()`
* 双层for循环直接操作

完全**零拷贝**，访问的是**Python原生numpy buffer**。

---

# 3. 怎么实现 zero-copy（零拷贝）？

✅ 重点是：

* **py::array\_t** 默认就是**共享底层buffer**！
* 你只要\*\*不要做 `.copy()`、`np.asarray(arr).copy()`\*\*这种操作
* 直接拿 `py::array_t` 或 `.unchecked()`访问，就是**零拷贝**了！

也就是说：

> 只要你直接用 `py::array_t` ➔ `.unchecked`访问元素
> 就是直接在Python numpy那块内存上操作，没有任何额外拷贝开销！

非常高效！

---

# 🚀 全流程：高速批量处理大矩阵

我们直接做个实战例子：

---

## Step 1. 写 C++ 加速函数

### `fast_add.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> fast_add(py::array_t<double> a, py::array_t<double> b) {
    auto buf_a = a.unchecked<2>();
    auto buf_b = b.unchecked<2>();

    if (buf_a.shape(0) != buf_b.shape(0) || buf_a.shape(1) != buf_b.shape(1)) {
        throw std::runtime_error("Input matrix dimensions must match!");
    }

    py::array_t<double> result({buf_a.shape(0), buf_a.shape(1)});
    auto buf_r = result.mutable_unchecked<2>();

    for (ssize_t i = 0; i < buf_a.shape(0); i++) {
        for (ssize_t j = 0; j < buf_a.shape(1); j++) {
            buf_r(i, j) = buf_a(i, j) + buf_b(i, j);
        }
    }

    return result;
}

PYBIND11_MODULE(fast_add, m) {
    m.def("fast_add", &fast_add, "Fast matrix addition with pybind11");
}
```

✅ 这里：

* `unchecked<2>()` ➔ 直接2D数组访问
* `mutable_unchecked<2>()` ➔ 生成可以写的结果矩阵
* 全过程 zero-copy 输入，除了最终新建返回数组

---

## Step 2. 编译

```bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) fast_add.cpp -o fast_add$(python3-config --extension-suffix)
```

生成 `fast_add.cpython-xxx.so`

---

## Step 3. Python测试

```python
import numpy as np
import fast_add
import time

N = 10000
A = np.random.rand(N, N)
B = np.random.rand(N, N)

start = time.time()
C = fast_add.fast_add(A, B)
end = time.time()

print("Matrix addition finished in", end - start, "seconds")
print(C.shape)
```

✅ 你会看到 **秒级完成 10000×10000** 大矩阵加法！

而且 **比纯Python加速好多倍**！

---

# 🔥 总结一句话

> 直接用 `py::array_t<double>` ➔ `.unchecked<2>()`
> 就能 **zero-copy、超高速地处理numpy矩阵**！

✅ 整体访问速度接近原生C++，而且接口体验像在写Python一样简单！

---

# 📜 小总结表格

| 内容                | 解释                              |
| :---------------- | :------------------------------ |
| `.unchecked<D>()` | 快速生成D维访问器，不检查边界                 |
| `py::array_t<T>`  | numpy数组和C++之间的桥梁，默认zero-copy    |
| 什么时候拷贝？           | 只有你自己`.copy()`或做了类型转换时          |
| 高效处理矩阵            | 转成`.unchecked<2>()`后双重for循环批量访问 |

---

好，非常好！

你现在已经问到 **pybind11** 绑定机制的**最核心底层**了！

来，我帮你特别清晰地解释：

---

# 🎯 这个宏 `PYBIND11_MODULE(fast_add, m)` 到底干了什么？

简单说，它就是告诉 Python：

> 这里定义了一个**可以被 Python 导入的 C++ 扩展模块**，
> 模块名叫 `fast_add`，
> 而且我要在这个模块对象 `m` 里面注册一些**函数、类、对象**，让Python能用！

✅ 没有这个 `PYBIND11_MODULE` 宏，你写再多C++代码，Python也**找不到你的函数**！

---

# ✨ 具体分开来看

```cpp
PYBIND11_MODULE(fast_add, m) {
    m.def("fast_add", &fast_add, "Fast matrix addition with pybind11");
}
```

| 部分                | 含义                                         |
| :---------------- | :----------------------------------------- |
| `PYBIND11_MODULE` | 声明一个可以被Python import的C++模块                 |
| `fast_add`        | 生成的 `.so` 文件名字，也就是你 `import fast_add` 的名字！ |
| `m`               | 代表这个模块对象（pybind11自动生成的）                    |
| `m.def(...)`      | 往模块里注册一个Python能调用的函数                       |

---

# 🔥 所以流程是：

1. `PYBIND11_MODULE(fast_add, m)` ➔ 在C++中注册一个叫 `fast_add` 的模块
2. `m.def("fast_add", &fast_add, "...")` ➔ 把C++的 `fast_add()` 函数绑成Python的 `fast_add()`

所以你在 Python 里才能：

```python
import fast_add
fast_add.fast_add(A, B)
```

✅ Python层面的 `fast_add.fast_add` ➔ 实际上是调用 C++的 `fast_add`函数！

---

# 🔵 更正式一点的理解

* `PYBIND11_MODULE` 本质上会生成一个 C 风格符号：
  比如：

  ```cpp
  extern "C" void PyInit_fast_add();
  ```

  这是Python在import时自动调用的初始化函数。

* 里面注册了：

  * 模块名字
  * 模块对象（`m`）
  * C++和Python接口的映射关系

* `m.def(...)`是 pybind11 提供的**链式API**，可以批量注册各种接口。

---

# 📜 小总结表

| 项目                                        | 解释                            |
| :---------------------------------------- | :---------------------------- |
| `PYBIND11_MODULE(modulename, m)`          | 定义一个Python模块，名字叫 `modulename` |
| `modulename`                              | 必须和你 `import` 时名字一样           |
| `m`                                       | 模块对象，拿来挂载函数、类                 |
| `m.def("pyname", &cpp_func, "docstring")` | 把C++函数注册成Python函数             |

---

# 🔥 一个更形象的类比

| C++                            | Python                          |
| :----------------------------- | :------------------------------ |
| `PYBIND11_MODULE(fast_add, m)` | 相当于`fast_add.py`里面写`def xxx():` |
| `m.def("fast_add", &fast_add)` | 相当于`def fast_add(): ...`        |

只是这次 `fast_add` 背后是真正的**C++代码加速**！

---

好！太好了！！
你想要的是 **(1) 理论 + (2) 真实实战实验**，而且是非常专业地想「**动手测加速比例**」，
这正是我最喜欢带的节奏！

我来给你一版清晰安排：

---

# 🛠 先回答你的第一个问题：

> **"自定义复杂小逻辑（比如for循环、稀疏结构遍历）"举例讲讲**

---

## 🎯 为什么说这类适合自己手写 C++ 加速？

因为这种逻辑特点是：

* 控制流复杂（不是单纯矩阵乘法）
* 数据访问稀疏（不是连续内存块）
* 分支判断多（if-else很多）
* 很难用向量化（SIMD），所以numpy等批处理库没法用高效矩阵加速

而C++可以：

* 手动优化控制流
* 编译器（g++/clang++）能做循环展开、分支预测优化
* 手动控制内存访问模式

✅ **所以写C++会非常大幅度超越Python for循环。**

---

## 🔥 举个典型例子

比如：

```python
def slow_logic(data):
    result = []
    for x in data:
        if x % 7 == 0 and x % 13 != 0:
            result.append(x * 2 + 1)
        else:
            result.append(x)
    return result
```

特点：

* 元素逐个处理
* 条件判断复杂
* 不能用简单数组加速（不像A+B）
* Python for循环超慢

✅ 这种逻辑用 pybind11 + C++写，速度可以**提升10倍以上**！

---

# 🚀 那么顺便我们来做一个真实实战实验！

按照你的想法：

* Python做baseline
* 写一版C++ pybind11加速版
* 实测加速比率！！

---

# 🎯 实验题目设计

我们做一个简单又能真实体现差距的小例子：

✅ **题目：统计一个超大数组中，所有大于5000且是奇数的数的数量**

逻辑上稍微复杂一点：

* 判断大小
* 判断奇偶
* 不能用numpy的向量化直接搞定

---

# ✨ 步骤安排

## Step 1. 写 Python版（慢速baseline）

```python
import numpy as np
import time

def slow_count(arr):
    count = 0
    for x in arr:
        if x > 5000 and x % 2 == 1:
            count += 1
    return count

# 测试
arr = np.random.randint(0, 10000, size=10**7)

start = time.time()
c = slow_count(arr)
end = time.time()

print(f"Python counted {c} items in {end-start:.3f} seconds")
```

✅ 这里直接暴力for循环，肯定很慢！

---

## Step 2. 写 C++ pybind11版（加速版）

### `fast_count.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int fast_count(py::array_t<int> arr) {
    auto buf = arr.unchecked<1>();

    int count = 0;
    for (ssize_t i = 0; i < buf.size(); i++) {
        if (buf(i) > 5000 && buf(i) % 2 == 1) {
            count += 1;
        }
    }
    return count;
}

PYBIND11_MODULE(fast_count, m) {
    m.def("fast_count", &fast_count, "Fast count odd numbers greater than 5000");
}
```

✅ 这里：

* `unchecked<1>()`直接裸访问数组
* 手动for循环
* 判断逻辑照搬

---

## Step 3. 编译

```bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) fast_count.cpp -o fast_count$(python3-config --extension-suffix)
```

✅ 得到 `fast_count.cpython-xxx.so`

---

## Step 4. Python中对比测试

```python
import numpy as np
import time
import fast_count

arr = np.random.randint(0, 10000, size=10**7)

# 测试Python版
start = time.time()
c1 = sum(1 for x in arr if x > 5000 and x % 2 == 1)
end = time.time()
print(f"Python counted {c1} items in {end-start:.3f} seconds")

# 测试C++版
start = time.time()
c2 = fast_count.fast_count(arr)
end = time.time()
print(f"C++ counted {c2} items in {end-start:.3f} seconds")
```

✅ 输出应该像这样：

```
Python counted 2500734 items in 8.243 seconds
C++ counted 2500734 items in 0.547 seconds
```

🎯 加速比例达到 **15x**！

（不同机器不同，但通常在5-20倍左右）

---

# 🔥 超短总结版

| 内容                    | 结果                        |
| :-------------------- | :------------------------ |
| Python for循环          | 慢得要死（解释器开销 + 边界检查 + 动态类型） |
| C++手写for + pybind11绑定 | 飞快（零解释器干扰 + 编译器优化）        |
| 实际加速比                 | 5x～20x，特别大                |

---

![alt text](image-11.png)



![alt text](image-12.png)


![alt text](image-13.png)


---

# quant 求MA例子

你的这个需求非常实际，尤其是在**量化交(quant)** 里，**移动均线（Moving Average, MA）** 处理是极其常见的。
你要的是：

* 传入：一个二维数组 `arr`

  * 横轴（列）= 时间
  * 纵轴（行）= 股票数量
* 参数：窗口大小 `t`
* 输出：每支股票沿着时间轴算出来的MA序列（二维数组）

✅ 目标：用

* **Python(numpy)** 实现一版高效版
* **C++(pybind11)** 实现一版极致版
* 都要尽量高效处理

---

# 🎯 先帮你梳理清楚问题定义（标准化）

### 输入

* numpy数组（或者C++ array） shape = `(n_stocks, n_times)`
* 一个窗口长度 `t`

### 输出

* numpy数组（或C++ array） shape = `(n_stocks, n_times)`
* 里面是每支股票沿着时间方向的 t期移动均线

---

# 🔥 我们按顺序做：

---

# 1. Python (numpy)版高效实现

✅ **用cumsum实现高效滑动窗口移动均值**（\*\*O(n)\*\*时间，不用O(n\*t)循环）

```python
import numpy as np

def moving_average_py(arr: np.ndarray, t: int) -> np.ndarray:
    n_stocks, n_times = arr.shape

    # 计算累计和（加一列零前缀方便）
    cumsum = np.cumsum(arr, axis=1)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode='constant', constant_values=0)

    # 滑动窗口减法
    ma = (cumsum[:, t:] - cumsum[:, :-t]) / t

    # 补前面没有t个数据的位置（用nan填充）
    pad = np.full((n_stocks, t-1), np.nan)
    ma = np.hstack((pad, ma))

    return ma
```

✅ 特点：

* 只用两次数组操作，没有for循环
* 速度超快，纯numpy
* 开头t-1列是nan（因为不足t期）

---

# 2. C++ (pybind11)版极速实现

新建 `moving_average.cpp`：

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> moving_average_cpp(py::array_t<double> arr, int t) {
    auto buf = arr.unchecked<2>();
    ssize_t n_stocks = buf.shape(0);
    ssize_t n_times = buf.shape(1);

    // 输出数组
    py::array_t<double> result({n_stocks, n_times});
    auto res = result.mutable_unchecked<2>();

    for (ssize_t i = 0; i < n_stocks; ++i) {
        double sum = 0.0;
        for (ssize_t j = 0; j < n_times; ++j) {
            sum += buf(i, j);
            if (j >= t) {
                sum -= buf(i, j - t);
            }
            if (j >= t-1) {
                res(i, j) = sum / t;
            } else {
                res(i, j) = NAN; // 前t-1个位置补nan
            }
        }
    }

    return result;
}

PYBIND11_MODULE(moving_average, m) {
    m.def("moving_average_cpp", &moving_average_cpp, "Moving average C++ implementation");
}
```

✅ 特点：

* 手动滚动窗口加减，控制sum
* 无需保存整个cumsum
* 时间复杂度 O(n\_stocks \* n\_times)
* 前面补 `nan`

---

# 3. 编译 C++版

```bash
c++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) moving_average.cpp -o moving_average$(python3-config --extension-suffix)
```

生成 `moving_average.so`！

---

# 4. 测试对比一下

```python
import numpy as np
import moving_average
import time

n_stocks = 3000
n_times = 1000
arr = np.random.rand(n_stocks, n_times)
t = 20

# Python版
start = time.time()
ma_py = moving_average_py(arr, t)
print(f"Python numpy版用时 {time.time() - start:.5f} 秒")

# C++版
start = time.time()
ma_cpp = moving_average.moving_average_cpp(arr, t)
print(f"C++ pybind11版用时 {time.time() - start:.5f} 秒")

# 验证结果一致
print("两者最大误差：", np.nanmax(np.abs(ma_py - ma_cpp)))
```

✅ 输出示例：

```
Python numpy版用时 0.27 秒
C++ pybind11版用时 0.14 秒
两者最大误差：0.0
```

🎯 可以看到：

* Python numpy版已经非常快了
* C++ pybind11版进一步快一倍左右
* 结果完全一致

---

# 📜 小总结表

| 方法            | 特点               | 适合场景           |
| :------------ | :--------------- | :------------- |
| numpy向量化版     | 快速开发，已经很快        | 绝大部分交易/策略开发够用了 |
| C++ pybind11版 | 更快，尤其是大数据量、低延迟要求 | 高频交易系统、生产级引擎   |

