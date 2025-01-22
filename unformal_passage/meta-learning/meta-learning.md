## <center> Meta-Learning </center>


### 关于参数

![alt text](image.png)

参数也是meta-learning的一个分支。

---

![alt text](image-1.png)
![alt text](image-2.png)


---

![alt text](image-3.png)

我们现在目标是学习F本身，包括网络架构，初始参数，学习率啥的。每一个东西都是meta learning的一个分支。

比如一个学习二元分类的meta learning：

![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)

然后就是强行train，如果不能微分，就上RL或者EA。

![alt text](image-7.png)
![alt text](image-8.png)


- ML v.s. Meta-Learning

difference:

![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-12.png)
![alt text](image-11.png)


similarity:

![alt text](image-13.png)


----

- MAML找一个初始化参数

![alt text](image-14.png)
![alt text](image-15.png)

好的原因：

![alt text](image-16.png)

- 还可以学习optimizer
![alt text](image-17.png)

- Network Architecture Search

![alt text](image-18.png)
![alt text](image-19.png)


- Data Augmentation

![alt text](image-20.png)

- Sample Reweighting
![alt text](image-21.png)

---


### 应用

![alt text](image-22.png)


### 补充学习

![alt text](image-23.png)

- self-supervised learning

![alt text](image-24.png)


- knowledge distillation

![alt text](image-25.png)

有文献指出，成绩好的teacher不见得是好的teacher。

![alt text](image-26.png)

引入meta learning，可以让teacher学习如何去teach。

![alt text](image-27.png)

- Domain Adaptation

在有label data的domain上很容易进行meta learning。

这里特别说一下在domain generalization上的应用。（即对一个未知的target domain，进行预测）

![alt text](image-28.png)
![alt text](image-29.png)
![alt text](image-30.png)

注意这里train的结果，是学习一个初始化参数。

![alt text](image-31.png)

- Lifelong Learning
![alt text](image-33.png)
![alt text](image-32.png)

传统方法，设计constraint，让模型不要忘记之前的知识。

![alt text](image-34.png)

我们也可以尝试用meta learning找一个比较好的leanring algorithm，可以避免catasrophic forgetting。

![alt text](image-35.png)

