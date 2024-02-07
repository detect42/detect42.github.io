---
title: 无旋treap(fhq)
tags: algorithm
catagories: 算法
abbrlink: 52fe06d
date: 2024-01-11 11:14:42
---
之前写过一篇按权值的fhq，可以解决查询前驱后继，第k大等操作。

不过如果我们把fhq的特性转移到数列上，即类比线段树储存sum，标记之内，那么又是一个新毒瘤数据结构了。

补：（一些对FHQ的理解）

1. 一共只有n个节点，不像线段树有2n个左右，因为fhq节点的左右儿子都是直接的节点（而线段树需要辅助节点），同时如果是对序数fhq，中序遍历就是当前数组，而对权值fhq，中序遍历是从小到大的顺序。

2. split的时候，两个取址的参数是分别两个分裂子树的root，注意，他们是完全分割的两个东西，并没有直接联系，第一层分裂完后，下一个该归到同一个分裂fhq的作为它的左右儿子。

---
### 一些注意事项：

1. FHQ的正确性基于满足中序遍历和堆性质的树的形态是固定的。

2. pushdown操作如果要改变子树形态或者siz大小，需要小从把标记推完后，再进行操作。

------------

## [毒瘤板题传送门](https://www.luogu.com.cn/problem/P2042)

我们知道线段树可以实现区间和，区间更改，最大子段和等操作，但此题还要求加入一串数、删除一串数、翻转一串数的操作。

现在思考一下如何用fhq解决以上问题。

1. fhq本身由插入顺序决定相对位置(因为其本身是一棵二叉搜索树，只不过这里把权定位下标而已)，再由随记的rank保证时间复杂度

2. 由fhq的毒瘤特性，我们可以两次split取出一段区间（对应在fhq上是一棵子树），从而方便进行加入，删除，翻转等涉及到序列结构及相对顺序的操作。

3. 关于split一点小说明，我们提取的是两端区间的root元素，所以只要第一次发现大了或小了，就可以确定两个root，其余的递归更改的则是root的children及root的children的children.....

-----

剩下的就考验码力了
（p.s 垃圾回收真tm生动形象）


code

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
int getint(){
int summ=0,f=1;
char ch;
for(ch=getchar();!isdigit(ch)&&ch!='-';ch=getchar());
if(ch=='-')
{
f=-1;
ch=getchar();
}
for(;isdigit(ch);ch=getchar())
summ=(summ<<3)+(summ<<1)+ch-48;
return summ*f;
} 
const int M=1e6+5;
struct tree{
	int lc,rc,siz,sum,val,mx,lx,rx,op,rk;
	bool opt,rev;
}t[M<<1];
int bin[M<<3],bintop,n,m,tot,sta[M<<1],top;
char s[21];
int a[M<<2],root,pos,num;
inline int newnode(int x){
	int g=bintop?bin[bintop--]:++tot;
	t[g].siz=1,t[g].sum=t[g].val=t[g].mx=x;
	t[g].op=t[g].lc=t[g].rc=0;
	t[g].opt=t[g].rev=0;
	t[g].rx=t[g].lx=max((int)0,x),t[g].rk=rand();
	return g;
}
inline void pushup(int x){
	t[x].siz=t[t[x].lc].siz+t[t[x].rc].siz+1;
	t[x].sum=t[t[x].lc].sum+t[t[x].rc].sum+t[x].val;
	t[x].mx=max(t[x].val,t[t[x].lc].rx+t[t[x].rc].lx+t[x].val);
	if(t[x].lc) t[x].mx=max(t[x].mx,t[t[x].lc].mx);
	if(t[x].rc) t[x].mx=max(t[x].mx,t[t[x].rc].mx);
	t[x].lx=max(t[t[x].lc].lx,t[t[x].lc].sum+t[x].val+t[t[x].rc].lx);
	t[x].rx=max(t[t[x].rc].rx,t[t[x].rc].sum+t[x].val+t[t[x].lc].rx);
}
inline void rever(int x){
	swap(t[x].lc,t[x].rc);swap(t[x].lx,t[x].rx);t[x].rev^=1;
}
inline void cover(int x,int v){
	t[x].val=v;t[x].sum=t[x].siz*v;
	t[x].lx=t[x].rx=max((int)0,t[x].sum);
	t[x].mx=max(t[x].sum,t[x].val),t[x].op=v;t[x].opt=1;
}
inline void pushdown(int x){
	if(t[x].rev){
		if(t[x].lc) rever(t[x].lc);
		if(t[x].rc) rever(t[x].rc);
		t[x].rev^=1;
	}
	if(t[x].opt){
		if(t[x].lc) cover(t[x].lc,t[x].op);
		if(t[x].rc) cover(t[x].rc,t[x].op);
		t[x].opt=0,t[x].op=0;
	}
}
int build(int l,int r,int *a){
	if(l>r) return 0;
	int mid=l+r>>1;
	int g=newnode(a[mid]);
	t[g].lc=build(l,mid-1,a);
	t[g].rc=build(mid+1,r,a);
	pushup(g);
	return g;
}
int merge(int x,int y){
	pushdown(x),pushdown(y);
	if(!x||!y) return x+y;
	if(t[x].rk<t[y].rk){
		t[x].rc=merge(t[x].rc,y);pushup(x);return x;
	}
	else{
		t[y].lc=merge(x,t[y].lc);pushup(y);return y;
	}
}
void split(int u,int k,int &x,int &y){
	if(!u) x=y=0;
	else{
		pushdown(u);
		if(t[t[u].lc].siz>=k){
			y=u;split(t[u].lc,k,x,t[u].lc);
		} 
		else{
			x=u;split(t[u].rc,k-t[t[u].lc].siz-1,t[u].rc,y);
		}
		pushup(u); 
	}
}
void restore(int x){
	if(!x) return;
	bin[++bintop]=x,restore(t[x].lc),restore(t[x].rc);
}
int x,y,z; 
signed main(){
	srand((unsigned)(time(0)));
	cin>>n>>m;
	for(int i=1;i<=n;i++) a[i]=getint();
	root=build(1,n,a);
	while(m--){
		scanf("%s",s);
		if(s[0]=='G'){
			pos=getint();num=getint();
			if(num==0){
				cout<<0<<"\n";continue;
			}
			split(root,pos-1,x,y);split(y,num,z,y);
			cout<<t[z].sum<<"\n";
			root=merge(x,merge(z,y));
		}
		if(s[0]=='I'){
			pos=getint();num=getint();
			if(!num) continue;
			for(int i=1;i<=num;i++) a[i]=getint();
			int nroot=build(1,num,a);
			split(root,pos,x,y);
			root=merge(x,merge(nroot,y));
		}
		if(s[0]=='D'){
			pos=getint();num=getint();
			if(!num) continue;
			split(root,pos-1,x,y);split(y,num,z,y);
			root=merge(x,y);restore(z);
		}
		if(s[0]=='M'&&s[2]=='K'){
			pos=getint();num=getint();
			int v=getint();
			if(!num) continue;
			split(root,pos-1,x,y);split(y,num,z,y);
			cover(z,v);root=merge(x,merge(z,y));
		}
		if(s[0]=='M'&&s[2]=='X'){
			cout<<t[root].mx<<"\n";
		}
		if(s[0]=='R'){
			pos=getint();num=getint();
			if(!num) continue;
			split(root,pos-1,x,y);split(y,num,z,y);
			rever(z);root=merge(x,merge(z,y));
		}
	}
	return 0;
}
```


-----

## [例题2 ](https://www.luogu.com.cn/problem/P2596)

本题唯一的难点，是如何把找到编号对应的位置，因为我们知道要找的标号的节点，无法确定中序遍历是第几个。故此题我们要求出fa数组表示每个节点的父亲，只需要在split和merge时更改fa就行了。

然后在那个点不断沿着fa链上去，如果是右儿子，那么就ans就加上左子树size+1，因为中序遍历先走左儿子，如果是右儿子，并不会有贡献。

于是我们单独写个Find函数找到编号为s的节点是原序列第几个，然后就是split和merge的基本操作。

code

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int M=1e5;
int  n,m,id[M],a[M],root,r1,r2,r3,r4,cnt;
struct fhq{
	int l,r,fa,siz,rank,val;
}t[M];
int newnode(int x){
	t[++cnt].val=x;t[cnt].rank=rand();t[cnt].siz=1;id[x]=cnt;
	return cnt; 
}
inline void pushup(int x){
	t[x].siz=t[t[x].l].siz+t[t[x].r].siz+1;
} 
void split(int u,int k,int &x,int &y,int fax=0,int fay=0){
	if(!u){
		x=y=0;return;
	}
	if(t[t[u].l].siz>=k){
		t[u].fa=fay;y=u;split(t[u].l,k,x,t[u].l,fax,u);
	}
	else{
		t[u].fa=fax;x=u;split(t[u].r,k-t[t[u].l].siz-1,t[u].r,y,u,fay);
	}
	pushup(u);
}
int merge(int x,int y){
	if(!x||!y) return x+y;
	if(t[x].rank<t[y].rank){
		t[x].r=merge(t[x].r,y);
		t[t[x].r].fa=x;pushup(x);return x;
	}
	else{
		t[y].l=merge(x,t[y].l);
		t[t[y].l].fa=y;pushup(y);return y;
	}
}
inline void ins(int pos,int val){
	split(root,pos,r1,r2);
	root=merge(r1,merge(newnode(val),r2));
}
int Find(int cnt){
	int node=cnt,res=t[t[cnt].l].siz+1;
	while(node!=root&&cnt){
		if(t[t[cnt].fa].r==cnt) res+=t[t[t[cnt].fa].l].siz+1;
		cnt=t[cnt].fa;
	}
	return res;
}
signed main(){
	srand((unsigned)(time(0)));
	char s[10];
	cin>>n>>m;
	for(int i=1;i<=n;i++) scanf("%lld",&a[i]);
	for(int i=1;i<=n;i++) ins(i-1,a[i]);
    while(m--){
    	int x,y,z;
    	scanf("%s",s);scanf("%lld",&x);
    	if(s[0]=='T'){
    		int k=Find(id[x]);
    		split(root,k,r1,r3);
    		split(r1,k-1,r1,r2);
    		root=merge(r2,merge(r1,r3));
    	}
    	if(s[0]=='B'){
    		int k=Find(id[x]);
    		split(root,k,r1,r3);
    		split(r1,k-1,r1,r2);
    		root=merge(r1,merge(r3,r2));
    	}
    	if(s[0]=='I'){
    		scanf("%lld",&y);int k=Find(id[x]);
    		if(y){
    			if(y>0){
    				split(root,k+1,r3,r4);
    				split(r3,k,r2,r3);
    				split(r2,k-1,r1,r2);
    				root=merge(r1,merge(r3,merge(r2,r4)));
    			}
    			else{
    				split(root,k,r3,r4);
    				split(r3,k-1,r2,r3);
    				split(r2,k-2,r1,r2);
    				root=merge(r1,merge(r3,merge(r2,r4)));
    			}
    		}
    	} 
    	if(s[0]=='A'){
    		int k=Find(id[x]);
    		cout<<k-1<<"\n";
    	}
    	if(s[0]=='Q'){
    		split(root,x,r1,r2);
    		split(r1,x-1,r1,r3);
    		cout<<t[r3].val<<"\n";
    		/*int node=r1;
    		while(t[node].r) node=t[node].r;
    		cout<<t[node].val<<"\n";
    		root=merge(r1,r2);*/
    		root=merge(merge(r1,r3),r2);
    	}
    } 
	return 0;
}
```

---

## [例题3](https://www.luogu.com.cn/problem/P3765)

本题需要用平衡树维护线段树。当然还有更优秀的做法，不过树套树的做法更显然也更实用。

因为本题我们要随时查询一段区间出现超过一半的数，且带修。

首先我们想清楚，如果一段区间内有数超过了一半，那么我们把这段区间任意划分成两段，至少有一段超过一半。

于是我们依据这个结论可以用左右儿子出现的最大值更新父节点的值。

这个用线段树实现，但是如何快速判断一个数在一段区间内出现次数，我们对每个不同的数建一个fhq，于是split两次看所求区间的size是否大于一半即可。

code

```cpp
#include<bits/stdc++.h>
using namespace std;
int getint(){
int summ=0,f=1;
char ch;
for(ch=getchar();!isdigit(ch)&&ch!='-';ch=getchar());
if(ch=='-')
{
f=-1;
ch=getchar();
}
for(;isdigit(ch);ch=getchar())
summ=(summ<<3)+(summ<<1)+ch-48;
return summ*f;
}
void print(int x){
    if(x<0){
        putchar('-');
        x=-x;
    }
    if(x>9){
        print(x/10);
    }
    putchar(x%10+'0');
}
const int M=6e5+5;
int n,m,num[M],root[M],r1,r2,r3,r4;
struct node{
	int l,r,siz,rk;
}t[M]; 
inline void pushup(int k){
	t[k].siz=t[t[k].l].siz+1+t[t[k].r].siz;
}
void split(int u,int k,int &x,int &y){
	if(!u){
		x=y=0;return;
	}
	if(u<=k){
		x=u;split(t[u].r,k,t[u].r,y);
	}
	else{
		y=u;split(t[u].l,k,x,t[u].l);
	}
	pushup(u);
}
int merge(int x,int y){
	if(!x||!y) return x+y;
	if(t[x].rk<t[y].rk){
		t[x].r=merge(t[x].r,y);
		pushup(x);
		return x;
	}
	else{
		t[y].l=merge(x,t[y].l);
		pushup(y);
		return y;
	}
}
int mx[M*10];
inline int check(int u,int l,int r){
	int ans=0;
	split(root[u],l-1,r1,r2);
	split(r2,r,r2,r3);
	ans=t[r2].siz;
	root[u]=merge(merge(r1,r2),r3);
	return ans;
}
int qu(int k,int l,int r,int x,int y){
	if(l>=x&&r<=y) return mx[k];
	int mid=l+r>>1;
	if(mid>=y) return qu(k<<1,l,mid,x,y);
	if(mid<x) return qu(k<<1|1,mid+1,r,x,y);
	int c1=qu(k<<1,l,mid,x,y),c2=qu(k<<1|1,mid+1,r,x,y);
	int ll=max(l,x),rr=min(r,y);
	if(check(c1,ll,rr)>(rr-ll+1)/2) return c1;
	else if(check(c2,ll,rr)>(rr-ll+1)/2) return c2;
	else return -1;
}
void modify(int k,int l,int r,int e){
	if(l==r){
		mx[k]=num[l];return;
	}
	int mid=l+r>>1;
	if(mid>=e) modify(k<<1,l,mid,e);
	else modify(k<<1|1,mid+1,r,e);
	int c1=mx[k<<1],c2=mx[k<<1|1];
	if(check(c1,l,r)>(r-l+1)/2) mx[k]=c1;
	else if(check(c2,l,r)>(r-l+1)/2) mx[k]=c2;
	else mx[k]=-1; 
} 
void build(int k,int l,int r){
	if(l==r){
		mx[k]=num[l];return;
	}
	int mid=l+r>>1;
	build(k<<1,l,mid);build(k<<1|1,mid+1,r);
	int c1=mx[k<<1],c2=mx[k<<1|1];
	if(check(c1,l,r)>(r-l+1)/2) mx[k]=c1;
	else if(check(c2,l,r)>(r-l+1)/2) mx[k]=c2;
	else mx[k]=-1; 
}
signed main(){
	srand((unsigned)(time(0)));
	cin>>n>>m;
	for(int i=1,tmp;i<=n;i++){
		tmp=getint();
		t[i].rk=rand();
		if(!root[tmp]) root[tmp]=i;
		else root[tmp]=merge(root[tmp],i);
		num[i]=tmp;
	} 
	build(1,1,n);
	for(register int i=1;i<=m;i++){
		int l,r,s,k;
		l=getint();r=getint();s=getint();k=getint();
		int ans=qu(1,1,n,l,r);
		if(ans==-1) ans=s;
		for(register int j=1;j<=k;j++){
			int tmp=getint();
			if(num[tmp]!=ans){
				split(root[num[tmp]],tmp,r1,r2);
				split(r1,tmp-1,r1,r3);
				root[num[tmp]]=merge(r1,r2);
				split(root[ans],tmp,r1,r2);
				root[ans]=merge(merge(r1,tmp),r2);
				num[tmp]=ans;
				modify(1,1,n,tmp);
			}
		}
		printf("%d\n",ans);
	}
	cout<<mx[1]<<endl;
	return 0;
}
```

---

## [例题4](https://www.luogu.com.cn/problem/P4309)

我们考虑普通的dp转移方程，设dp[i]为以i结尾的最长不下降序列的长度，我们对于新加的点，需要在1->i-1中找出最大的dp[i]来更新dp(值为dp[i]+1)。

所以我们需要需要有两个操作，查询区间最大值，插入元素。用平衡树就行了。

注意一点：因为加数是按从小到大的顺序，所以新加的的数必定是原序列最大的数，并不会对新的pos之后的dp数组有影响，所以只用单点更新pos就行了（dp数组的下标不是序列顺序，是对应的数字，因为数字是唯一的，序列顺序会变化）

code

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int M=1e6;
int n,a[M],r1,r2,r3,dp[M];
struct node{
	int siz,mx,l,r,pos,rk;
}t[M]; 
inline void Pushup(int k){
	t[k].siz=t[t[k].l].siz+t[t[k].r].siz+1;
	t[k].mx=max(dp[t[k].pos],max(t[t[k].l].mx,t[t[k].r].mx));
}
int tot=0;
inline int Newpoint(int val,int pos){
	tot++;t[tot].mx=val;t[tot].rk=rand();t[tot].pos=pos;t[tot].siz=1;
	dp[pos]=val; return tot;
}
void Split(int u,int k,int &x,int &y){
	if(!u){
		x=y=0;return;
	}
	//x,y was divided
	if(t[t[u].l].siz>=k){
		y=u;Split(t[u].l,k,x,t[u].l);
	}
	else{
		x=u;Split(t[u].r,k-t[t[u].l].siz-1,t[u].r,y);
	}
	Pushup(u);
}
int Merge(int x,int y){
	if(!x||!y) return x|y;
	if(t[x].rk>=t[y].rk){
		t[y].l=Merge(x,t[y].l);
		Pushup(y);return y;
	}
	else{
		t[x].r=Merge(t[x].r,y);
		Pushup(x);return x;
	}
}
int root;
signed main(){
	srand((unsigned)(time(0)));
	cin>>n;
	for(int i=1,x;i<=n;i++) {
		scanf("%lld",&x);
		Split(root,x,r1,r2);
        //注意原来在新加入的点的位置的点会被自动判定为在右边（因为下标从0开始）
		root=Merge(Merge(r1,Newpoint(t[r1].mx+1,i)),r2);
		cout<<t[root].mx<<"\n";
	} 
	return 0;
}
```
