---
title: 快速傅立叶变化FFT
tags: algorithm
catagories: 算法
abbrlink: 1c32d91d
date: 2024-01-11 11:16:41
---
FFT是一种可以在$O(n*log_n)$的优秀时间复杂度里，求出两个多项式乘积的优秀算法。

首先，我们如何表示一个$（n-1）$次多项式？

1. 用每一项的系数来表示

2. 用n个点的坐标来表示

所以算两个多项式乘积时，如果暴力计算系数，是$O(n^2)$的。

但是如果我们有这两个已知多项式的点的表达式，那么我们只需要$O(n)$的时间就可以得到带求多项式的点表达式。

不幸的是，为了得到两个多项式的n个点坐标，在横坐标随记下我们仍需要$O(n^2)$的时间来完成。

所以有没有什么办法可以优化找点的过程了。

以为名叫傅里叶的牛人发现了复数可以完美解决这个问题。

具体来说，我们再复数坐标戏中引入单位元（类比三角函数），因为复数有个性质，**两复数乘积的几何意义是模长相乘，辐角相加**。

因为我们取得点在单位圆上，于是同一个复数的$n$次乘积在几何意义上表现为辐角的成倍增长。

前方高能，我们开始搞事情了。

### 设带求多项式为F(x)

我们按照多项式次数的就分成两节。

即$F(x)=FL(x^2)+x*FR(x^2）$

其中FL(x)表示原来的多项式偶数系数提出来形成的长度为一半的多项式，FR（x）同理是奇数的系数提取出来的多项式。

同时定义$W_x^k$表示复数单位圆平分n段，逆时针第k条线表示的复数。

易得以下两式子：

1. $F(W_n^k)=FL(W^k_{n/2})+W^k_n*FR(W_{n/2}^k)$;

2. $F(W_n^{k+n/2})=FL(W^k_{n/2})-W^k_n*FR(W_{n/2}^k)$;

于是如果我们知道了FL,FR，我们可以$O(n)$得到$F(x)$的n组点值;

如果我们把FL，FR也看成多项式，这就变成了一个递归求解，即是

## 分治！

于是我们实现了$O(n*log_n)$的时间复杂度完成点值。

关于这一步的实现：

1. 我们可以自动把不足2的幂的项系数补位0

2. 因为我们是按奇偶分治，最后得到的系数顺序与具体值无关，于是我们预先处理出最后的顺序（找规律发现为二进制的翻转），然后递推上来即可。（可以让常数小上几倍）

-----

至于将点值转化回系数：

下面是结论时间：我们把点值倒回去做一遍上述过程，（其中单位值$W_n^1$变为$W_n^{-1}$）,得到的系数除以n即是答案。


updated 2020.07.23 

补充一点证明。

设

$$G[k]=\sum_{i=0}^{i<=n}(W_n^k)^iF[i]$$

$G[i]$表示对饮虚数的函数点值。

由单位根反演有：

$$F[k]*n=\sum_{i=0}^{i<=n}(W_n^{-k})^iG[i]$$

一点证明：

将上式带入下式有


$$\sum_{i=0}^{i<n}\sum_{j=0}^{j<n}W_n^{-ik}W_n^{ji}F[j]$$

$$\sum_{i=0}^{i<n}\sum_{j=0}^{j<n}W_n^{i(j-k)}F[j]$$

- $j==k$

值$F[k]*n$

- $j!=k$

因为
$$\sum_{j=0}^{j<n}W_n^j=0$$

对原式差分求和号后会发现都带有这样一个东西，所以贡献为0。

至此，我们完成了单位根的反演。

------------


------------


最后理一下算法流程：

1. 用fft算出两个多项式的点值

2. 将其相乘得到带求多项式的点值

3. 套用结论将点值做一次逆fft得到系数

---

code

```cpp
#include<bits/stdc++.h>
using namespace std;
const int M=3e6+5;
const double pi=acos(-1);
int n,m,tr[M];
struct cp{
	cp(double xx=0,double yy=0){
		x=xx,y=yy;
	}
   //这里简单说一下结构体语法（吐槽一波这都什么时候了连一篇正常一点的语法教程都没有，最后还是得靠自己摸索）
   //这里cp(double xx=0,double yy=0)
   //在定义函数时，对xx,yy没有操作，于是x=xx=0
   //在结构体前赋值cp(n,m)中，xx->0->n,最后的值是n,于是x=xx=n
	double x,y;
	cp operator + (cp const &b) const{
		return cp(x+b.x,y+b.y);
	}
	cp operator - (cp const &b) const{
		return cp(x-b.x,y-b.y);
	}
	cp operator * (cp const &b) const{
		return cp(x*b.x-y*b.y,x*b.y+y*b.x);
	}
}f[M],p[M];
//复数运算
void fast_fast_tle(cp *f,int flag){
	for(int i=0;i<n;i++) if(i<tr[i]) swap(f[i],f[tr[i]]);
    //调整底层顺序
	for(int pp=2;pp<=n;pp<<=1){//枚举长度
		int len=pp>>1;
		cp tg(cos(2*pi/pp),sin(2*pi/pp));//单位复数值
		if(flag==-1) tg.y*=-1;//逆fft，虚数项改为-1
		for(int k=0;k<n;k+=pp){
			cp buf(1,0);//算次数的底
			for(int l=k;l<k+len;l++){
				cp tt=buf*f[len+l];
				f[len+l]=f[l]-tt;//奇数项
				f[l]=f[l]+tt;//偶数项
				buf=buf*tg;//更新幂次项
			}
		}    
	}
}
int main(){
	cin>>n>>m;
	for(int i=0;i<=n;i++) scanf("%lf",&f[i].x);
	for(int i=0;i<=m;i++) scanf("%lf",&p[i].x);
	for(m+=n,n=1;n<=m;n<<=1);
	for(int i=0;i<n;i++) tr[i]=(tr[i>>1]>>1)|((i&1)?n>>1:0);
	fast_fast_tle(f,1);fast_fast_tle(p,1);
	for(int i=0;i<n;i++) f[i]=f[i]*p[i];
	fast_fast_tle(f,-1);
	for(int i=0;i<=m;i++) printf("%d ",(int)(f[i].x/n+0.5)); //注意精度问题
	return 0;
}
```


### 后记：

[NTT传送门](https://www.luogu.com.cn/blog/wsr/post-ntt)
