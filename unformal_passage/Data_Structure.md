## <center> Data Structure </center>

（mostly from oiwiki）

# B树

![Alt text](image.png)

### 查找

B 树中的节点包含有多个键。假设需要查找的是 k，那么从根节点开始，从上到下递归的遍历树。在每一层上，搜索的范围被减小到包含了搜索值的子树中。 子树值的范围被它的父节点的键确定。因为是从根节点开始的二分法查找，所以查找一个键的代码如下：

```cpp
BTreeNode *BTreeNode::search(int k) {
  // 找到第一个大于等于待查找键 k 的键
  int i = 0;
  while (i < n && k > keys[i]) i++;

  // 如果找到的第一个键等于 k , 返回节点指针
  if (keys[i] == k) return this;

  // 如果没有找到键 k 且当前节点为叶子节点则返回NULL
  if (leaf == true) return NULL;

  // 递归
  return C[i]->search(k);
}
```


### 插入

![Alt text](image-1.png)


```cpp
void BTree::insert(int k) {
  // 如果树为空树
  if (root == NULL) {
    // 为根节点分配空间
    root = new BTreeNode(t, true);
    root->keys[0] = k;  // 插入节点 k
    root->n = 1;        // 更新根节点的关键字的个数为 1
  } else {
    // 当根节点已满，则对B-树进行生长操作
    if (root->n == 2 * t - 1) {
      // 为新的根节点分配空间
      BTreeNode *s = new BTreeNode(t, false);

      // 将旧的根节点作为新的根节点的孩子
      s->C[0] = root;

      // 将旧的根节点分裂为两个，并将一个关键字上移到新的根节点
      s->splitChild(0, root);

      // 新的根节点有两个孩子节点
      // 确定哪一个孩子将拥有新插入的关键字
      int i = 0;
      if (s->keys[0] < k) i++;
      s->C[i]->insertNonFull(k);

      // 新的根节点更新为 s
      root = s;
    } else  // 根节点未满，调用 insertNonFull() 函数进行插入
      root->insertNonFull(k);
  }
}

// 将关键字 k 插入到一个未满的节点中
void BTreeNode::insertNonFull(int k) {
  // 初始化 i 为节点中的最后一个关键字的位置
  int i = n - 1;

  // 如果当前节点是叶子节点
  if (leaf == true) {
    // 下面的循环做两件事：
    // a) 找到新插入的关键字位置并插入
    // b) 移动所有大于关键字 k 的向后移动一个位置
    while (i >= 0 && keys[i] > k) {
      keys[i + 1] = keys[i];
      i--;
    }

    // 插入新的关键字，节点包含的关键字个数加 1
    keys[i + 1] = k;
    n = n + 1;
  } else {
    // 找到第一个大于关键字 k 的关键字 keys[i] 的孩子节点
    while (i >= 0 && keys[i] > k) i--;

    // 检查孩子节点是否已满
    if (C[i + 1]->n == 2 * t - 1) {
      // 如果已满，则进行分裂操作
      splitChild(i + 1, C[i + 1]);

      // 分裂后，C[i] 中间的关键字上移到父节点，
      // C[i] 分裂称为两个孩子节点
      // 找到新插入关键字应该插入的节点位置
      if (keys[i + 1] < k) i++;
    }
    C[i + 1]->insertNonFull(k);
  }
}

// 节点 y 已满，则分裂节点 y
void BTreeNode::splitChild(int i, BTreeNode *y) {
  // 创建一个新的节点存储 t - 1 个关键字
  BTreeNode *z = new BTreeNode(y->t, y->leaf);
  z->n = t - 1;

  // 将节点 y 的后 t -1 个关键字拷贝到 z 中
  for (int j = 0; j < t - 1; j++) z->keys[j] = y->keys[j + t];

  // 如果 y 不是叶子节点，拷贝 y 的后 t 个孩子节点到 z中
  if (y->leaf == false) {
    for (int j = 0; j < t; j++) z->C[j] = y->C[j + t];
  }

  // 将 y 所包含的关键字的个数设置为 t -1
  // 因为已满则为2t -1 ，节点 z 中包含 t - 1 个
  // 一个关键字需要上移
  // 所以 y 中包含的关键字变为 2t-1 - (t-1) -1
  y->n = t - 1;

  // 给当前节点的指针分配新的空间，
  // 因为有新的关键字加入，父节点将多一个孩子。
  for (int j = n; j >= i + 1; j--) C[j + 1] = C[j];

  // 当前节点的下一个孩子设置为z
  C[i + 1] = z;

  // 将所有父节点中比上移的关键字大的关键字后移
  // 找到上移节点的关键字的位置
  for (int j = n - 1; j >= i; j--) keys[j + 1] = keys[j];

  // 拷贝 y 的中间关键字到其父节点中
  keys[i] = y->keys[t - 1];

  // 当前节点包含的关键字个数加 1
  n = n + 1;
}
```

### 删除

B 树的删除操作相比于插入操作更为复杂，因为删除之后经常需要重新排列节点。

与 B 树的插入操作类似，必须确保删除操作不违背 B 树的特性。正如插入操作中每一个节点所包含的键的个数不能超过 2k-1 一样，删除操作要保证每一个节点包含的键的个数不少于 k-1 个（除根节点允许包含比 k-1 少的关键字的个数）。


**我们采用先删除再调整的思想，首先按照一般搜索树trick，如果删的不是叶子上的元素，就把后继元素提上来，然后木匾变成删除后继元素。完成删除后，采用“相邻兄弟&分界父节点合并”和“借相邻兄弟的元素”方法调整B树至合法状态。**

![Alt text](image-2.png)
![Alt text](image-3.png)
![Alt text](image-4.png)
![Alt text](image-5.png)

```cpp
B-Tree-Delete-Key(x, k) 
if not leaf[x] then 
    y ← Preceding-Child(x) 
    z ← Successor-Child(x) 
    if n[[y] > t − 1 then 
        k' ← Find-Predecessor-Key(k, x)]() 
        Move-Key(k', y, x) 
        Move-Key(k, x, z) 
        B-Tree-Delete-Key(k, z) 
    else if n[z] > t − 1 then 
        k' ← Find-Successor-Key(k, x) 
        Move-Key(k', z, x) 
        Move-Key(k, x, y) 
        B-Tree-Delete-Key(k, y) 
    else 
        Move-Key(k, x, y) 
        Merge-Nodes(y, z) 
        B-Tree-Delete-Key(k, y) 
    else (leaf node) 
    y ← Preceding-Child(x) 
    z ← Successor-Child(x) 
    w ← root(x) 
    v ← RootKey(x) 
        if n[x] > t − 1 then Remove-Key(k, x) 
        else if n[y] > t − 1 then 
            k' ← Find-Predecessor-Key(w, v) 
            Move-Key(k', y,w) 
            k' ← Find-Successor-Key(w, v) 
            Move-Key(k',w, x) 
            B-Tree-Delete-Key(k, x) 
        else if n[w] > t − 1 then 
            k' ← Find-Successor-Key(w, v) 
            Move-Key(k', z,w) 
            k' ← Find-Predecessor-Key(w, v) 
            Move-Key(k',w, x) 
            B-Tree-Delete-Key(k, x) 
        else 
            s ← Find-Sibling(w) 
            w' ← root(w) 
                if n[w'] = t − 1 then 
                    Merge-Nodes(w',w) 
                    Merge-Nodes(w, s) 
                    B-Tree-Delete-Key(k, x)
                else
                    Move-Key(v,w, x)
                    B-Tree-Delete-Key(k, x)
```



## B树的优势

![Alt text](image-6.png)

因为单点容量参数是我们可以主观确定，所以可以与磁盘单位适配，从而提高效率。

同时让逻辑上相邻的数据都能尽量存储在物理上也相邻的硬盘空间上，这意味着我们需要降低树的高度同时让单个节点存储的数据变多。

------

# Splay

![Alt text](image-7.png)

### Splay操作

将一个访问过的节点旋转至根节点。

![Alt text](image-8.png)

这是对p是根节点的操作，只有当x的层数是奇数时才会作为splay操作的最后一步执行。


接下来两种操作，都将x的深度减2。

根据x、父亲、爷爷的关系，分为两种情况：

![Alt text](image-9.png)

当连成一条线时，先转父亲&爷爷，然后再转下面x和父亲的边。

![Alt text](image-10.png)

当构成折线时，先转x和父亲，然后再转上面的父亲和爷爷的边。

### 时间复杂度证明

![Alt text](image-11.png)
![Alt text](image-12.png)

### 插入操作
![Alt text](image-13.png)

其实很简单，就是普通的插入，然后把插入的节点splay到根节点。


### 合并操作

![Alt text](image-14.png)

## 查询x排名

![Alt text](image-16.png)

注意一些访问操作结束后都需要splay。

### 删除操作

![Alt text](image-15.png)
