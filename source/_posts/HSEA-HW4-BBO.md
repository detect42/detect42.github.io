---
title: 演化算法HW4 BBO黑箱芯片放置问题
tags: 实验报告
categories: 
- NJU course
- HSEA
abbrlink: 21249fb3
date: 2024-01-26 13:38:17
---
# <center> Macro placement by black-box optimization </center>
### <center> detect0530@gmail.com </center>
<font size=4>

## 1 问题分析

通过阅读发下来的code，发现需要我调整的黑箱问题是，给定一组长度为dim，有上下界的整数序列。

实例代码使用随机搜索的方法，通过不断生成随机数列来更新最优答案。

由于BBO已经为我们设计好了评价黑盒，我们只需要考虑如何在这组数列上进行演化算法即可。

## 2 基本的演化算法设计 & 比较不同mutation,crossover的性能

### 2.1 种群设置

由于单次黑盒评估运算量巨大（2-5s），我们需要尽可能减少黑盒评估次数。

所以种群大小一开始定为10

### 2.2 selection

初始算法中，我将随机从种群中选出父代。

### 2.3 mutaion

- 随机选择两个位置进行位置交换
- 以概率p1对每个位置进行随机变化

### 2.4 crossover

选出两个父代后：

- 随机选择断点，然后将两个父代的断点两侧进行交换
- 以概率p2对每个位置进行父代交换

### survival selection

以n+n策略进行精英保留，保存最优的n个个体


设计好基本的演化算法后，开始实践：

```py
class EA:
    def __init__(self, placer, X=None,fit=0):
        self.placer = placer
        if X is None:
            self.x = np.random.randint(self.placer.lb, self.placer.ub + 1, self.placer.dim)
        else:
            self.x = X
        self.fit = fit
    def eval(self,bestfit):
        self.fit = self.placer._evaluate(self.x)
```

设计EA类，用于存储单个个体的所有信息。

```py
    n = 10
    EAs = []
    comp = [0 for _ in range(bbo_placer.dim)]
    for _ in range(n):
        ea = EA(placer=bbo_placer)
        ea.eval(comp)
        EAs.append(ea)
       
    EA_run(n,EAs,args.max_iteration)
```

随机抽十组个体，进行演化算法。

```py
def EA_run(n,EAs,max_iteration):
    p1 = 1/15
    p2 = 1
    for _ in range(max_iteration):
        
        max_fit = max([i.fit for i in EAs])
        total_fit = max_fit * 1.001 * n - sum([i.fit for i in EAs])
        prob = [(max_fit * 1.001 - i.fit) / total_fit for i in EAs]
        print(prob)

        for __ in range(6):

            EA_p1 = Deepcopy(random.choices(EAs, prob)[0])
            EA_p2 = Deepcopy(random.choices(EAs, prob)[0])

            if random.random() < p2:
                EA_p3 = Deepcopy(EA_p1)
                for i in range(EA_p3.placer.dim):
                    if random.random() < 0.5:
                        EA_p3.x[i]=EA_p2.x[i]
                EA_p3.eval(EAs[0].x)
                EAs.append(EA_p3)
                prob.append(0)

            for i in range(EA_p1.placer.dim):
                if random.random() < p1:
                    EA_p1.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
                if random.random() < p1:
                    EA_p2.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
            EA_p1.eval(EAs[0].x)
            EAs.append(EA_p1)
            prob.append(0)
            EA_p2.eval(EAs[0].x)
            EAs.append(EA_p2)
            prob.append(0)

        EAs.sort(key=lambda x: x.fit)

        for i in EAs:
            print(i.fit,"-->",i.x)

        EAs = EAs[:n]
        

        print("now iteration is" , _ , "and minmuim eval is " , EAs[0].fit)
```

按照之前定下的基本演化算法实现了代码，最初打算随机selection，但是发现效果不好，所以改成了按照fitness进行概率selection，同时为了防止fitness差异过小而无效话，引入了最差个体*1.05作为baseline来计算个体的选择概率。

### 2.5 不同mutation,crossover的性能比较

- 随机交换变异 + 随机父代断点交换 + 精英保留策略：

![Alt text](HSEAHW4/HSEAHW4/image-2.png)

- 随机位置概率变异 + 随机父代位置交换 + 精英保留策略：

![Alt text](HSEAHW4/image.png)

- 随机位置概率变异 + 随机父代断点交换 + 精英保留策略：

![Alt text](HSEAHW4/image-3.png)

![Alt text](HSEAHW4/image-7.png)

**不同的变异交换算子性能区别不大，在初期下降速度上有少许差异。**

附上论文中的算法性能评分：

![Alt text](HSEAHW4/image-1.png)

可以看到，很容易的超过了基于RL的MaskPlace算法，但是与WireMask-EA仍有差距。

从折线图中不难看出，问题出在500it后，演化算法不再产生更优解。

## 3 优化演化算法

经过思考后，尝试从以下角度尝试优化初代演化算法：

### 3.1 分组探索

在查看初代演化算法的数据时发现了一个问题，到200it之后种群里所有的个体都高度接近，这一点阻碍了新的可能性的探索，猜想原因是因为一旦有一个个体找到一组优解，在其上的变异也会产生优解从而把其他个体挤出种群。这一点会大大降低模型的exploration能力。

解决方案：

- 种群规模设置为16，分为4个group，每个group有4个个体。变异时每个组内的个体进行变异，然后在组内进行生存选择，这样一来探索方向会至少保证有4个。group之间的通信由crossover保证，crossover不会每一个iteration都进行，而是以概率p2进行。每一次crossover会选择两个group的两个个体进行crossover再塞回去进行生存选择。这样兼顾了exploration和exploitation。
```py
    n = 16
    group = 4
    EA1 = []
    EA2 = []
    EA3 = []
    EA4 = []    
    for _ in range(4):
        ea = EA(placer=bbo_placer)
        ea.eval()
        EA1.append(ea)
        print(ea.fit,ea.x)

        ea = EA(placer=bbo_placer)
        ea.eval()
        EA2.append(ea)
        print(ea.fit,ea.x)

        ea = EA(placer=bbo_placer)
        ea.eval()
        EA3.append(ea)
        print(ea.fit,ea.x)
    
        ea = EA(placer=bbo_placer)
        ea.eval()
        EA4.append(ea)
        print(ea.fit,ea.x)

    EA_run(n,EA1,EA2,EA3,EA4,args.max_iteration)
```

- 同时，类似多目标-EA中区分diversity和convergence的思想，我在每个group中引入了一个diversity的概念，即每个group中的个体都会有一个diversity的值，这个值是这个个体与其他个体的平均距离。在每次生存选择时，会优先选择diversity大的个体，这样一来可以保证种群的多样性。实现中为把diversity定为与组内最优解的距离。

```py
def eval(self,bestfit):
        self.fit = self.placer._evaluate(self.x)
        print(self.fit)
        sum=0
        kg=1
        for i in range(len(self.x)):
            if self.x[i]==bestfit[i]:
                sum+=1
            else: kg=0
        if kg==1: 
            return
        self.fit+=sum*5e3
```

实验表现：

首先拉长iteration：

![Alt text](HSEAHW4/image-4.png)

可以看到2000iteration后，有明显的凹曲线下降，说明改动是有效的。

但是查看diversity时，group之间差距仍很小，此时我的模型是每次必抽多组group进行crossover，为了提高group之间的差距，我多加了概率p2来控制crossover的频率，希望等组内变异算子起作用后再进行crossover把优秀的结构与其他group通信。

![Alt text](HSEAHW4/image-5.png)

由于计算资源匮乏，只测了700it，但是惊喜的发现性能差异不大的同时，下降速度得到了很大的提升，这与我加强diversity能提高探索能力的猜想相吻合。

此时，43的评估水平已经超越了论文里的49.32（虽然不知道能不能这样直接比较，分组group的代码在下面有贴出）

![Alt text](HSEAHW4/image-1.png)

### 3.2 超参数调整和一些细节修改

#### 3.2.1 一些细节改动：

- selection时按照排名概率进行抽样，而不是fitness概率，这样可以避免fitness差异过小导致的选择概率差异过小。

```py
    total_rk = sum([(i+1) for i in range (4)])
    prob = [(5-(i+1)) / total_rk for i in range(4)]
    print(prob)
```

变异算子和交叉算子预与其去选择，不如都上。

**加入了各种算子，以及分组优化后的代码如下：**

```py
def EA_run(n,EA1,EA2,EA3,EA4,max_iteration):
    p1 = 1/30
    start_value = 1/5
    end_value = 1/50
    length = 200
    # 生成均分的列表
    result_list = np.linspace(start_value, end_value, length)
    p2 = 1/4
    for _ in range(max_iteration):
        p1 = result_list[_]
        print("mutaion prob is ",p1)
        total_rk = sum([(i+1) for i in range (4)])
        prob = [(5-(i+1)) / total_rk for i in range(4)]
        print(prob)
        EA_p1 = EA(EA1[0].placer)
        # 1 
        for __ in range(4):
            
            rd = random.choices(EA1, prob)[0] 
            EA_p1.Deepcopy(rd)
            lasans = EA_p1.fit
            for i in range(EA_p1.placer.dim):
                if random.random() < p1:
                    EA_p1.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
            EA_p1.eval()
            if EA_p1.fit < lasans:
                rd.Deepcopy(EA_p1)
            
        # 2
        for __ in range(4):

            rd = random.choices(EA2, prob)[0]  
            EA_p1.Deepcopy(rd)
            lasans = EA_p1.fit
            for i in range(EA_p1.placer.dim):
                if random.random() < p1:
                    EA_p1.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
            EA_p1.eval()
            if EA_p1.fit < lasans:
                rd.Deepcopy(EA_p1)
        #3
        for __ in range(4):

            rd = random.choices(EA3, prob)[0]  
            EA_p1.Deepcopy(rd)
            lasans = EA_p1.fit
            for i in range(EA_p1.placer.dim):
                if random.random() < p1:
                    EA_p1.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
            EA_p1.eval()
            if EA_p1.fit < lasans:
                rd.Deepcopy(EA_p1)
        #4
        for __ in range(4):

            rd = random.choices(EA4, prob)[0]  
            EA_p1.Deepcopy(rd)
            lasans = EA_p1.fit
            for i in range(EA_p1.placer.dim):
                if random.random() < p1:
                    EA_p1.x[i] = np.random.randint(EA_p1.placer.lb, EA_p1.placer.ub+1)[0]
            EA_p1.eval()
            if EA_p1.fit < lasans:
                rd.Deepcopy(EA_p1)
        EA_p2 = EA(EA1[0].placer) 
        EA_p3 = EA(EA1[0].placer)
        EA_p4 = EA(EA1[0].placer)
        # crossover 1
        for __ in range(2):
            rd1 = random.choices(EA1)[0] 
            rd2 = random.choices(EA2)[0]
            EA_p1.Deepcopy(rd1)
            EA_p2.Deepcopy(rd2)
            if random.random() < p2:
                midpoint = np.random.randint(EA_p1.placer.dim/3, EA_p1.placer.dim-EA_p1.placer.dim/3)
                EA_p3.Deepcopy(EA_p1)
                EA_p3.x[:midpoint] = EA_p2.x[:midpoint]
                EA_p3.eval()
                EA1.append(EA_p3)
                EA_p4.Deepcopy(EA_p2)
                EA_p4.x[:midpoint] = EA_p1.x[:midpoint] 
                EA_p4.eval()
                EA2.append(EA_p4)
            rd3 = random.choices(EA2)[0] 
            rd4 = random.choices(EA3)[0]
            EA_p1.Deepcopy(rd3)
            EA_p2.Deepcopy(rd4)
            if random.random() < p2:
                midpoint = np.random.randint(EA_p1.placer.dim/3, EA_p1.placer.dim-EA_p1.placer.dim/3)
                EA_p3.Deepcopy(EA_p1)
                EA_p3.x[:midpoint] = EA_p2.x[:midpoint]
                EA_p3.eval()
                EA2.append(EA_p3)
                EA_p4.Deepcopy(EA_p2)
                EA_p4.x[:midpoint] = EA_p1.x[:midpoint] 
                EA_p4.eval()
                EA3.append(EA_p4)
            rd5 = random.choices(EA3)[0] 
            rd6 = random.choices(EA4)[0]
            EA_p1.Deepcopy(rd5)
            EA_p2.Deepcopy(rd6)
            if random.random() < p2:
                midpoint = np.random.randint(EA_p1.placer.dim/3, EA_p1.placer.dim-EA_p1.placer.dim/3)
                EA_p3.Deepcopy(EA_p1)
                EA_p3.x[:midpoint] = EA_p2.x[:midpoint]
                EA_p3.eval()
                EA3.append(EA_p3)
                EA_p4.Deepcopy(EA_p2)
                EA_p4.x[:midpoint] = EA_p1.x[:midpoint] 
                EA_p4.eval()
                EA4.append(EA_p4) 
            rd7 = random.choices(EA4)[0] 
            rd8 = random.choices(EA1)[0]
            EA_p1.Deepcopy(rd7)
            EA_p2.Deepcopy(rd8)
            if random.random() < p2:
                midpoint = np.random.randint(EA_p1.placer.dim/3, EA_p1.placer.dim-EA_p1.placer.dim/3)
                EA_p3.Deepcopy(EA_p1)
                EA_p3.x[:midpoint] = EA_p2.x[:midpoint]
                EA_p3.eval()
                EA4.append(EA_p3)
                EA_p4.Deepcopy(EA_p2)
                EA_p4.x[:midpoint] = EA_p1.x[:midpoint] 
                EA_p4.eval()
                EA1.append(EA_p4)
        # crossover 2

        for __ in range(1):
            EA_p1.Deepcopy(random.choices(EA1)[0])
            EA_p2.Deepcopy(random.choices(EA3)[0])
            if random.random() < p2:
                EA_p3.Deepcopy(EA_p1)
                for i in range(EA_p3.placer.dim):
                    if random.random() < 0.5:
                        EA_p3.x[i]=EA_p2.x[i]
                EA_p3.eval()
                EA1.append(EA_p3)
            EA_p1.Deepcopy(random.choices(EA2)[0])
            EA_p2.Deepcopy(random.choices(EA4)[0])
            if random.random() < p2:
                EA_p3.Deepcopy(EA_p1)
                for i in range(EA_p3.placer.dim):
                    if random.random() < 0.5:
                        EA_p3.x[i]=EA_p2.x[i]
                EA_p3.eval()
                EA2.append(EA_p3)
            EA_p1.Deepcopy(random.choices(EA3)[0])
            EA_p2.Deepcopy(random.choices(EA2)[0])
            if random.random() < p2:
                EA_p3.Deepcopy(EA_p1)
                for i in range(EA_p3.placer.dim):
                    if random.random() < 0.5:
                        EA_p3.x[i]=EA_p2.x[i]
                EA_p3.eval()
                EA3.append(EA_p3)
            EA_p1.Deepcopy(random.choices(EA4)[0])
            EA_p2.Deepcopy(random.choices(EA1)[0])
            if random.random() < p2:
                EA_p3.Deepcopy(EA_p1)
                for i in range(EA_p3.placer.dim):
                    if random.random() < 0.5:
                        EA_p3.x[i]=EA_p2.x[i]
                EA_p3.eval()
                EA4.append(EA_p3)         

        


        EA1.sort(key=lambda x: x.fit)
        EA2.sort(key=lambda x: x.fit)
        EA3.sort(key=lambda x: x.fit)
        EA4.sort(key=lambda x: x.fit)

        EA1 = EA1[:4]
        EA2 = EA2[:4]
        EA3 = EA3[:4]
        EA4 = EA4[:4]

        for i in EA1:
            print("EA1",i.fit,"-->",i.x)
        for i in EA2:
            print("EA2",i.fit,"-->",i.x)
        for i in EA3:
            print("EA3",i.fit,"-->",i.x)
        for i in EA4:
            print("EA4",i.fit,"-->",i.x)

        print("now iteration is" , _ , "and minmuim eval is " , min(EA1[0].fit,EA2[0].fit,EA3[0].fit,EA4[0].fit))
    
```

#### 3.2.2 超参数的调整

在我的模型里，有两个超参数需要调整，一个是变异概率p1，一个是crossover概率p2。

##### p2

之前有讨论过，交叉算子是group之间通信的桥梁，不应该每次都通信，因为会使得优解反复传播，导致各组之间的diversity下降，也不应该太稀疏，不然和单纯的mutation随机算法没有优势。

##### p1

p1是变异算子的探索能力，设的太低，优化速度变慢，设的太高，最后收敛的太早，没法精细地调整优解而获得更优解。

于是我采用退火的思路，一开始p1较高，随着时间缓缓降低到一个定值，确保了前期优较快的速度，后期也能稳定收敛到较优的值。

最后把p1从1/5缓慢降低到1/50，分为200次缓慢下降。
```py
    start_value = 1/5
    end_value = 1/50
    length = 200
    # 生成均分的列表
    result_list = np.linspace(start_value, end_value, length)
    p2 = 1/4
    for _ in range(max_iteration):
        p1 = result_list[_]
        print("mutaion prob is ",p1)
```

## 4 最优成果展示：

通过上述所有优化以及漫长的调参，我发现有些优化并不如理想中可靠，比如按照排名去selection反而效果更差。以及最后的最后，当iteration足够大时，group的diversity效果也不如预期。

这里贴一些有代表性的performance：

优化较少时平缓下降，需要很多iteration才能看起来收敛：
![Alt text](HSEAHW4/image-6.png)

加入优化算法&细节调整，同时参数设置较好时，下降速度很快，但是最后收敛到了一个很好的结果：
![Alt text](HSEAHW4/image-8.png)


**最终的成果还是令人满意的，最终的评估结果为：38.8854**
![Alt text](HSEAHW4/image-9.png)

虽然没有取均值，从纸面数据上来看还是不错。

![Alt text](HSEAHW4/image-10.png)
![Alt text](HSEAHW4/image-11.png)

## 5 总结

通过一学期的演化算法学习，一方面入门了EA，另一方面为大二的我开阔了视野。经过本次作业，加强了自己分析问题和编程能力，也体验到了演化算法的魅力。
