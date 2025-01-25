## <center> Support Vector Machine (SVM) </center>

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

- hinge loss function:

max操作促使f(x)大于1，且不需要超过1太多，对比cross entropy： hinge loss及格就好：
（也是ideal function的upper bound）

![alt text](image-3.png)

SVM和logistic regression的区别就是定义的loss function不同，SVM的loss function是hinge loss function，logistic regression的loss function是cross entropy loss function。


![alt text](image-4.png)

脑洞一下，linear SVM可以用gradient descent来求解。

![alt text](image-5.png)

---

常见的做法：

![alt text](image-6.png)


---

- SVM特点

1. w参数实际是x的线性组合，所以SVM是线性分类器。这一点可以用lagrange multiplier或者梯度下降来证明。
2. $a_n$是sparse的，大部分是0，也就是说SVM结果只和少数几个样本有关。

![alt text](image-7.png)



---

经过等价变化，和最后一步推广，产生kernel trick。
![alt text](image-8.png)
![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
当使用RBF kernel，如果不使用kernel而是朴素的特征变换，需要的特征维度是无穷的，而kernel trick可以避免这个问题。

![alt text](image-13.png)

**Kernel function是一个投影到高维的inner product，所以往往就是something like similarity**

![alt text](image-14.png)


---

与deep learning的比较：

这个很形象也很重要：

![alt text](image-15.png)