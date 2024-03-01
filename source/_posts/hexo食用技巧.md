---
title: hexo食用技巧
tags: hexo
categories: 
- tool
- hexo
abbrlink: b1320771
date: 2023-06-29 20:40:20
---

# 关于hexo的合理食用

## 1. 添加pdf
1. \post\pdf本地目录下本地加入xxx.pdf
2. md文件里加入
3. 也可以直接插入网址外部访问

```c++
 {% pdf pdf/xxx.pdf %}
```

## 插入图片

将图片保存在\post下，直接调用即可，可以改变图片长宽比例以及大小

```c++
<img src="xxx.png" width="10%" height="10%">
```

<img src="1.png" width="10%" height="10%">