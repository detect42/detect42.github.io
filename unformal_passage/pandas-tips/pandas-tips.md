# <center> Pandas Tips </center>

1. series本来就是像dict，安全的获取元素方式
![alt text](<Pasted Graphic 4.png>)

2. series的运算最重要是自动更具index对齐
![alt text](<Pasted Graphic 5.png>)
![alt text](<Pasted Graphic 2.png>)

3. 当我们想要删除nan时
![alt text](df.dropna(axis=1).png)
![alt text](<2013-01-02 1.212112 -@.17328 @.119206 5.60.png>)

4. 当我们指定index和columns时，效果不一样，index是替换，columns是保留/选择
![alt text](<Pasted Graphic 7.png>)

5. 列的删除和添加
![alt text](<Pasted Graphic 8.png>)
![alt text](<Pasted Graphic 9.png>)

6. pandas浅拷贝和Copy on write

遇到list里面还有其他指针的可变对象时，需要deepcopy。
![alt text](<Pasted Graphic 10.png>)

常见的切片，取col操作都是COW的，真正共享底层内存需要直接对df.value进行操作。
![alt text](<Pasted Graphic 11.png>)

![alt text](<Pasted Graphic 12.png>)

![alt text](<Pasted Graphic 3.png>)

![alt text](<Pasted Graphic 4-1.png>)

7. 用df.assign()来添加列 配合lambda函数
![alt text](<Pasted Graphic 13.png>)

8. 挑选子集
![alt text](Operation.png)
注意直接df[]可以是col，也可以返回是row的切片组合df。
![alt text](<Pasted Graphic 1.png>)

9. df的运算
![alt text](<pandas 遇运算，先对齐；.png>)

10. describe()简单总结每一列的情况
![alt text](<Pasted Graphic.png>)

11. sort和query
![alt text](<Pasted Graphic 1-1.png>)
![alt text](<Pasted Graphic 2-1.png>)
![alt text](Excellent.png)

12. 单元素快速数据访问
![alt text](<Pasted Graphic 4-2.png>)
![alt text](<Pasted Graphic 5-1.png>)

13. 筛选过滤
![alt text](<Pasted Graphic 6.png>)
![alt text](<Pasted Graphic 7-1.png>)

要注意下面这种情况，不要连着用，不然解释器不知道你对第一个副本做还是对原始df做

使用df.loc明确说
![alt text](<Pasted Graphic 8-1.png>)

14. pandas的缺失元素
![alt text](<Pasted Graphic 9-1.png>)

15. 强制统一col的元素和顺序 目的是对齐
![alt text](你想统一为标准格式：.png)
![alt text](这在做多模型多批鼓据比对时非常常见，避免手动对齐。.png)

16. 运算

- 自动广播到每一列
  ![alt text](2013-01-01.png)
- agg 把每一行元素聚合
  ![alt text](-3.851445.png)
- transform 利用lambda函数式变化
  ![alt text](-3.851445-1.png)
![alt text](<Pasted Graphic 14.png>)

17. 生成df时，加一列是快的 ，加一行是慢的
提前存在list里面 然后直接生成df 不要iteratively去一行行add raw
![alt text](<Pasted Graphic 17.png>)
用字典生成时，[]是行，{}是列,小技巧。
- [{},{},] 一行行字典生成df
- {key1：[], key2:[],} 一列列字典生成df

18. merge合并两个表格
![alt text](<6.用多个 key 合并（多列连接）.png>)
![alt text](<Pasted Graphic 19.png>)
![alt text](<Pasted Graphic 20.png>)

19. df.groupby
![alt text](用法总结：df.groupby（键列“）［［目标列］］.agg（函数）.png)
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

20. MultiIndex
![alt text](<Pasted Graphic 22.png>)
![alt text](<Pasted Graphic 23.png>)

21. Category 分类变量
![alt text](<Pasted Graphic 24.png>)
![alt text](总结重点：Categorical能做什么？.png)