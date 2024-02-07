---
title: AC自动机
tags: algorithm
catagories: 算法
abbrlink: 5c28c80e
date: 2024-01-11 11:10:00
---
## Ac自动机

它由三个部分组成：

1. 建立字典树
2. 找出字典树上每个点的fail
3. 跑匹配串来搞事情

----

- ## 建立字典树

就是表面意思，我们把所有待匹配串来建一个字典树

- ## 寻找fail指针

这个和kmp类似，如果一段前缀和另外一段后缀重合，那么这段区间就不需要再来一个个匹配。

我们考虑用bfs来求fail指针（最大重合部分的指针）：

对于一个点u，现在目标求所有u的子节点的fail（注意顺带搞一下空节点，对后来的模式串匹配过程有帮助）

- 若存在u的连边v。

那么v的fail即为u的fail的v'儿子

- 若不存在u的连边v。

既然不存在这个点，那我们手动添加虚点，保证原来的字典树上所有点都有所有的儿子，这样将情况转回到1去。

- ## 匹配

我们沿着字典树跑一边匹配串

每跑一个点，都沿着fail把所有可以fail的跳一遍，这些点都是可到的。

然后接着一直跑到底。（注意如果跑出字典树的边界也无妨，因为虚点的连边又将其引回字典树上）

----

## [板题1 AC自动机（简单版）](https://www.luogu.com.cn/problem/P3808)

此题只要求出不出现，于是查询时不断打标记，保证同一个点不走两遍即可

code

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int M=1e6+5;
struct node{
	int son[27],fail,val;
}tr[M];
int n,m,tot;
inline int add(string s){
	int len=s.size();int now=0;
	for(int i=0;i<len;i++){
		int c=s[i]-'a';
		if(!tr[now].son[c]) tr[now].son[c]=++tot;
		now=tr[now].son[c];
	}
	tr[now].val++;
}
inline void prefail(){
	queue <int> q;
	int now=0;
	for(int i=0;i<26;i++) if(tr[now].son[i]) q.push(tr[now].son[i]);
	while(!q.empty()){
		int u=q.front();q.pop();
		for(int i=0;i<26;i++){
			if(tr[u].son[i]){
				tr[tr[u].son[i]].fail=tr[tr[u].fail].son[i];
				q.push(tr[u].son[i]);
			}
			else tr[u].son[i]=tr[tr[u].fail].son[i];
		}
	}
}
int Find(string s){
	int ans=0,u=0,len=s.size();
	for(int i=0;i<len;i++){
		u=tr[u].son[s[i]-'a'];int p=u;
		while(p!=0&&tr[p].val!=-1){
			ans+=tr[p].val;tr[p].val=-1;p=tr[p].fail;
		}
	}return ans;
}

signed main(){
	cin>>n;
	for(int i=1;i<=n;i++){
	    string s;
		cin>>s;add(s);
	}
	prefail();
	string s;
	cin>>s;cout<<Find(s)<<"\n";
	return 0;
}
```

## [板题2 AC自动机（二次加强版）](https://www.luogu.com.cn/problem/P5357)

我们要求出现次数，看起来只用在上一道题上改改就可以。

但是结果是：


### TLE


我们回到上一个题FIND函数，我们对于匹配串在trie树上走的每一个位置都getfail一直到根，这个过程复杂度超标了。

我们考虑如何操作才能使其变回线性。

差分

我们对匹配串每一个位置标记一下，再借build中的队列（其倒叙一定满足拓扑序且fail指针一定连向深度更浅的点），在树上差分即可。


code

```cpp
#include<bits/stdc++.h>
using namespace std;
const int M=2e6+4;
int n,m,ans[M],tot=0,re[M],sum[M];
struct node{
	int son[27],val,fail;
}tr[M];
inline void add(string s,int id){
	int len=s.size(),now=0;
	for(int i=0;i<len;i++){
		int c=s[i]-'a';
		if(!tr[now].son[c]) tr[now].son[c]=++tot;
		now=tr[now].son[c];
	}
	if(!tr[now].val) tr[now].val=id;re[id]=tr[now].val;
}
int q[M];
inline void prefail(){
	int now=0,h=1,t=0;
	for(int i=0;i<26;i++) if(tr[0].son[i]) q[++t]=tr[0].son[i];
	while(h<=t){
		int u=q[h++];
		for(int i=0;i<26;i++){
			if(tr[u].son[i]){
				tr[tr[u].son[i]].fail=tr[tr[u].fail].son[i];q[++t]=tr[u].son[i];
			}
			else tr[u].son[i]=tr[tr[u].fail].son[i];
		}
	}
}
inline void Find(string s){
	int len=s.size(),now=0;
	for(int i=0;i<len;i++){
		now=tr[now].son[s[i]-'a'];sum[now]++;
	}
	for(int i=tot;i>=1;i--){
		ans[tr[q[i]].val]=sum[q[i]];sum[tr[q[i]].fail]+=sum[q[i]];
	}
}
string ss[M];
signed main(){
cin>>n;{
		for(int i=1;i<=n;i++){
			cin>>ss[i];add(ss[i],i);
		}
		prefail();
		string s;cin>>s;
		Find(s);
		int mxsum=0;
		for(int i=1;i<=n;i++){
			mxsum=max(mxsum,ans[i]);cout<<ans[re[i]]<<"\n";
		}
	}
	return 0;
}
```

## [例题1 [JSOI2012]玄武密码](https://www.luogu.com.cn/problem/P5231)


此题只用求每个字符串出现在文本里的最长前缀。

我们把AC机建出来，把文本跑一遍记录位置同时标记，然后再对每一个串从头跑一遍看最长标记的点。


## [例题2 200606 T1 strings](https://www.luogu.com.cn/problem/T135693)

给定n个字符串和每个字符串的权值。

求构造出权值最大的字符串的价值。



我们对模式串建立AC自动机，那么每个节点的权值就是其回溯fail到根的所有点的权值和。我们可以在预处理fail时顺便加上。（bfs的fail，fail深度必定更浅，所以可以立刻加上）

然后我们考虑在ac机上dp：

设$dp(len,pos)$表示长度为len终点在pos的构造的最优价值。

那么，我们暴力枚举len和pos转移即可。

code

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int M=2000;
int n,m,dp[M][M],l,w[M],tot;
struct node{
	int son[27],val,fail;
}tr[M];
inline void add(string s,int id){
	int len=s.size(),now=0;
	for(int i=0;i<len;i++){
		int c=s[i]-'a';
		if(!tr[now].son[c]) tr[now].son[c]=++tot;
		now=tr[now].son[c];
	}
	tr[now].val+=w[id];
}
inline void prefail(){
	queue <int> q;
	for(int i=0;i<26;i++) if(tr[0].son[i]) q.push(tr[0].son[i]);
	while(!q.empty()){
		int u=q.front();q.pop();
		for(int i=0;i<26;i++){
			if(tr[u].son[i]){
				tr[tr[u].son[i]].fail=tr[tr[u].fail].son[i];q.push(tr[u].son[i]);
			}
			else tr[u].son[i]=tr[tr[u].fail].son[i];
		}tr[u].val+=tr[tr[u].fail].val;
	}
}
signed main(){
	cin>>n>>l;
	for(int i=1;i<=n;i++) scanf("%lld",&w[i]);
	for(int i=1;i<=n;i++){
		string s;cin>>s;
		add(s,i);
	}
	prefail();
	memset(dp,-1,sizeof(dp));dp[0][0]=0;
	for(int i=0;i<=l;i++)
	  for(int j=0;j<=tot;j++)
	     if(dp[i][j]!=-1){
	  		for(int k=0;k<26;k++){
	  			    int v=tr[j].son[k];
				    dp[i+1][v]=max(dp[i+1][v],dp[i][j]+tr[v].val);
				}
	  }
	int ans=0;
	for(int i=0;i<=tot;i++){
		ans=max(ans,dp[l][i]);
	}
	cout<<ans<<endl; 
	return 0;
}
```

## [例题3 200209 T1 串](http://192.168.110.252/Problem/1590)

题目描述

兔子们在玩字符串的游戏。首先，它们拿出了一个字符串集合 S，然后它们定义一个字符串为“好”的，当且仅当它可以被分成非空的两段，其中每一段都是字符串集合 S 中某个字符串的前缀。比如对于字符串集合 {"abc","bca"}，字符串 "abb"，"abab"是“好”的 ("abb"="ab"+"b",abab="ab"+"ab") ，而字符串 “bc”不是“好”的。

兔子们想知道，一共有多少不同的“好”的字符串。

$1≤N≤10000,1≤|S|≤30$

----

计数题有两种方法，总数减去不合法数 或者 不重不漏的计算。

我尝试两种都写出来。

### 解法1 不重不漏的计数

我们要在字典树上从根节点找两条路径拼接且答案字符串不能重复。

为了不重复计算，我们只计算每一种字符串的特殊形式。在这里我们定义为第一个串长最小。

对于一种合法答案，思考其在字典树上的走势。

1. 在一条链上

2. 在两条链上

- 先看情况2。

一定有一个分割点且在这个分割点跳了fail边。即对于所有不在原trie树上的边（faill跳过的边）的指向节点，都代表一个第二串的一段前缀，我们跳的是fail，注意对应的第一串的形式也许不是我们跳的出发点到根的串，而是把根到出发点的串减去从第一串跳fail到第二串中第二串和第一串后缀相同的前缀。

然后如果在这些第二串的初始节点沿着trie走就是添加字符，若按照fail转移就是缩短上述相同前后缀的长度。显然这样走出来的字符串都是答案。


好了，现在想想这样走出来的字符串会不会重复。

明确一点：经过由第一个串跳tail转移的下一个字符后（满足第二点中不在一条链上的限制），向后面无论是走trie还是跳fail，都可以找到一种方式匹配出一种 从 根 到第一串尾点 再到 当前点的转移字符边所构成的字符串。而显然的是，AC机上不会有两种不同的走法到达同一个点后对应的路径字符串相同。

思考上述方法会不会有额外的限制，按照trie树原边的转移显然不会有任何问题，但是考虑跳fail时，深度只会越变越浅，那么若终点深度小于第二串我们转移的长度，那么不可构造出一种可行解。（dp时特判掉这一种情况即可）
（再解释一下：出现不可行构造的条件是我们在fail边转移时，已经把从第一串以后加入的边的一截当做后缀都用来去转移了（注意此时一串二串不会有公共前后缀），**也就是说从第二串的起点起所有fail的边必须被完好保存至我们的终点**，所以当深度小于转移数时，说明出现了上述不可构造的情况。）

（再换一种说法来解释就是若有公共前后缀，那么一定合法，否则没有公共前后缀时，则不能消耗fail边去转移，又因为trie树的边会增加一的深度，**所以从第二串开头后的转移每一条边都不能少**，否则第一条从一串到二串的fail边一定会被来去转移而被损害。）

（我们还可以这样想，对于我们跑出来的串，若一串后有一条fail边没有出现在最后真实的一串二串中，那么一定不合法）

- 对于情况1

有解方案为终点的fail不指向根

----

所以我们再AC机上跑不同合法路劲方案数。

一些实现细节：

 1. 枚举长度时，不可能超过30，因为转移边全部都作为第二串的部分。
 
 code
 

```cpp
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int M=300005;
int n,m,mp[M][30],fail[M],tot,dp[31][M],ans=0,tr[M][30],dep[M];
inline void Insert(string s){
	int len=s.size(),now=0;
	for(int i=0;i<len;i++){
		int c=s[i]-'a';
		if(!mp[now][c]) mp[now][c]=++tot,tr[now][c]=1,dep[mp[now][c]]=dep[now]+1;
		now=mp[now][c];
	}
}
inline void prefail(){
	queue <int> q;
	for(int i=0;i<26;i++) if(mp[0][i]) q.push(mp[0][i]);
	while(!q.empty()){
		int u=q.front();q.pop();
		for(int i=0;i<26;i++){
			if(mp[u][i]) fail[mp[u][i]]=mp[fail[u]][i],q.push(mp[u][i]);
			else mp[u][i]=mp[fail[u]][i];
		}
	}
}
signed main(){
	cin>>n;
	for(int i=1;i<=n;i++){
		string s;cin>>s;
		Insert(s);
	}
	prefail();
	for(int i=1;i<=tot;i++) if(fail[i]) ans++;
	for(int i=1;i<=tot;i++){
		for(int j=0;j<26;j++){
			if(!tr[i][j]){
				dp[1][mp[i][j]]++;
			}
		}
	}
	for(int i=2;i<=30;i++)
	  for(int j=1;j<=tot;j++)
	    for(int k=0;k<26;k++){
	    	if(dep[mp[j][k]]>=i){
	    		dp[i][mp[j][k]]+=dp[i-1][j];
	    	}
	    } 
	for(int i=1;i<=30;i++)
	  for(int j=1;j<=tot;j++)
	    ans+=dp[i][j];
	cout<<ans<<endl;
	return 0;
}
```

