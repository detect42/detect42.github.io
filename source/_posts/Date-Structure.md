---
title: Date Structure
tags: DS
categories: 
- NJU course
- DS 
abbrlink: 1fe2fe25
date: 2023-09-09 17:40:24
---


## <center> Data Structure </center>

# 链表

我们让insert操作和delete操作都返回head，可以简化代码实现。

```c++
#include <iostream>
#include<bits/stdc++.h>
using namespace std;
// 定义双向链表节点的结构体
struct Node {
    int data;          // 存储数据
    Node* prev;        // 指向前一个节点的指针
    Node* next;        // 指向下一个节点的指针
    Node(int val) : data(val), prev(nullptr), next(nullptr) {} // 构造函数
};

// 创建一个空链表
Node* createLinkedList() {
    return nullptr;
}

// 在链表的末尾插入一个新节点
Node* insertAtEnd(Node* head, int data) {
    Node* newNode = new Node(data);

    if (head == nullptr) {
        return newNode; // 如果链表为空，新节点成为头节点
    }

    Node* current = head;
    while (current->next != nullptr) {
        current = current->next;
    }

    current->next = newNode;
    newNode->prev = current;
    return head;
}

// 在指定节点后插入一个新节点
Node* insertAfter(Node* head, Node* nodeToInsertAfter, int newData) {
    if (head == nullptr || nodeToInsertAfter == nullptr) {
        return head; // 空链表或要插入的节点为空
    }

    Node* newNode = new Node(newData);

    newNode->next = nodeToInsertAfter->next;
    if (nodeToInsertAfter->next != nullptr) {
        nodeToInsertAfter->next->prev = newNode;
    }

    nodeToInsertAfter->next = newNode;
    newNode->prev = nodeToInsertAfter;

    return head;
}

// 删除链表中的特定节点
Node* deleteNode(Node* head, Node* nodeToDelete) {
    if (head == nullptr || nodeToDelete == nullptr) {
        return head; // 空链表或要删除的节点为空
    }

    if (nodeToDelete->prev != nullptr) {
        nodeToDelete->prev->next = nodeToDelete->next;
    }

    if (nodeToDelete->next != nullptr) {
        nodeToDelete->next->prev = nodeToDelete->prev;
    }

    if (head == nodeToDelete) {
        head = nodeToDelete->next;
    }

    delete nodeToDelete;
    return head;
}

// 打印链表
void printLinkedList(Node* head) {
    Node* current = head;
    while (current != nullptr) {
        std::cout << current->data << " <-> ";
        current = current->next;
    }
    std::cout << "nullptr" << std::endl;
}

int main() {
    Node* head = createLinkedList();
    head = insertAtEnd(head, 1);
    Node* nodeToInsertAfter = insertAtEnd(head, 2);
    head = insertAtEnd(head, 3);

    std::cout << "原始链表: ";
    printLinkedList(head);

    int newData = 4;
    head = insertAfter(head, nodeToInsertAfter, newData);

    std::cout << "在节点 " << nodeToInsertAfter->data << " 后插入节点 " << newData << " 后的链表: ";
    printLinkedList(head);

    std::cout << "删除节点 " << nodeToInsertAfter->data << " 后的链表: ";
    head = deleteNode(head, nodeToInsertAfter);
    printLinkedList(head);

    return 0;
}
```


# AVL平衡树（无递归无栈version）

insert只用转一次，非常好处理，但是delete可能要转很多次，所以对于delete我们第一步转成尾递归，再展开为循环。

delete第一步找到结构上要被删除的点，然后再找到dep最大的不会因为删点导致高度下降的点，从这个点开始从上往下走，不断调整旋转树的结构以维护balance factor。

因为旋转方法可以根据两层内的bf直接确定，所以不需要递归回来的时候改，在从上往下的循环过程中就可以直接先转了再继续循环。


insert和delete的代码，由于没有递归和栈，略显臃肿。
```cpp
#include<bits/stdc++.h>
using namespace std;
template<class Type>
struct AVLNode{
    Type data;
    int height;
    AVLNode<Type> *left,*right;
    AVLNode():left(nullptr),right(nullptr),height(0){}
    AVLNode(const Type &data,AVLNode *left=nullptr,AVLNode *right=nullptr,int height=0):data(data),left(left),right(right),height(height){}
};
template<class Type> class AVLTree{
private:
    AVLNode<Type> *root;
    //void Release(AVLNode<Type> *Tree);
    inline int Height(AVLNode<Type> *Tree){return (Tree==nullptr)?0:Tree->height;}
    void SingleRotateLeft(AVLNode<Type> *&Tree);
    void SingleRotateRight(AVLNode<Type> *&Tree);
    void DoubleRotateLeft(AVLNode<Type> *&Tree);
    void DoubleRotateRight(AVLNode<Type> *&Tree);
    bool Insert(AVLNode<Type> *&Tree,const Type &x);
    bool Delete(AVLNode<Type> *Tree,const Type &x);

    //AVLNode<Type>* Search(AVLNode<Type> *Tree,Type &x);
    // void PreOrder(AVLNode<Type> *Tree,void );
public:
    static AVLNode<Type> *P,*FA,*Toroot;
    AVLTree():root(nullptr){}
    //   ~AVLTree(){Release(root);}
    int Insert(Type x){return Insert(root,x);}
    bool Delete(Type x){return Delete(root,x);}
    AVLNode<Type>* Root(){return root;}
    //  bool Search(Type &x){return !(Search(root,x)==nullptr);}
    //  friend ostream& operator<<(ostream &os,AVLTree<Type> &Tree){}
    // void PreOrder(void ){PreOrder(root);}
    int bf(AVLNode<Type> *Tree){return Height(Tree->right)-Height(Tree->left);}
    void Print(AVLNode<Type> *Tree);
    pair<AVLNode<Type>*,AVLNode<Type>*> Findnonshorter(AVLNode<Type> *Tree,Type x);
};
template<class Type>
AVLNode<Type> *AVLTree<Type>::P=nullptr;
template<class Type>
AVLNode<Type> *AVLTree<Type>::FA=nullptr;
template<class Type>
AVLNode<Type> *AVLTree<Type>::Toroot=nullptr;

template<class Type>
void AVLTree<Type>::Print(AVLNode<Type> *Tree) {
    if(Tree==nullptr)
        return;
    cout<<"(";
    if(Tree->left!=nullptr)
        Print(Tree->left);
    cout<<Tree->data<<","<<Height(Tree->right)-Height(Tree->left);//cout<<","<<Tree->height;
    if(Height(Tree->right)-Height(Tree->left)>1||Height(Tree->right)-Height(Tree->left)<-1)
        cout<<"!";
    if(Tree->right!=nullptr)
        Print(Tree->right);
    cout<<")";
}

template <class Type>
void AVLTree<Type>::SingleRotateLeft(AVLNode<Type> *&Tree) {
    AVLNode<Type> *t=Tree->left;
    Tree->left=t->right;
    t->right=Tree;

    Tree->height=max(Height(Tree->left),Height(Tree->right))+1;
    t->height=max(Height(t->left),Tree->height)+1;
    Tree=t;
}
template <class Type>
void AVLTree<Type>::SingleRotateRight(AVLNode<Type> *&Tree) {
    AVLNode<Type> *t=Tree->right;
    Tree->right=t->left;
    t->left=Tree;

    Tree->height=max(Height(Tree->left),Height(Tree->right))+1;
    t->height=max(Height(t->right),Tree->height)+1;
    Tree=t;
}
template <class Type>
void AVLTree<Type>:: DoubleRotateLeft(AVLNode<Type> *&Tree) {
    SingleRotateRight(Tree->left);
    SingleRotateLeft(Tree);
}
template <class Type>
void AVLTree<Type>::DoubleRotateRight(AVLNode<Type> *&Tree) {
    SingleRotateLeft(Tree->right);
    SingleRotateRight(Tree);
}

template<class Type>
bool AVLTree<Type>::Insert(AVLNode<Type>* &tree,const Type &x){

    if(tree==nullptr){
        tree=new AVLNode<Type>(x, nullptr, nullptr, 0);
        if(tree==nullptr){
            cout<<"Out of space"<<endl;
            exit(1);
        }
        tree->height=max(Height(tree->left),Height(tree->right))+1;
        return true;
    }
    AVLNode<Type> *node=nullptr,*fanode=nullptr;
    AVLNode<Type> *t=tree,*fa=nullptr;
    while(t!=nullptr){
        if(Height(t->right)-Height(t->left)!=0){
            fanode=fa;node=t;
        }
        if(x<t->data){
            fa=t;
            t=t->left;
        }
        else if(x>t->data){
            fa=t;
            t=t->right;
        }
        else{
            return false;
        }
    }

    if(node==nullptr) node=tree;
    t=node;fa=fanode;t->height--;
    while(t!=nullptr){
        if(x<t->data){
            t->height++;
            fa=t;
            t=t->left;
        }
        else if(x>t->data){
            t->height++;
            fa=t;
            t=t->right;
        }
        else{
            cout<<"Already exist"<<endl;
            return false;
        }
    }
    node->height=max(Height(node->left),Height(node->right))+1;
    if(x<fa->data) fa->left=new AVLNode<Type>(x,nullptr,nullptr,1);
    else fa->right=new AVLNode<Type>(x,nullptr,nullptr,1);

    if(bf(node)==2||bf(node)==-2){

        if(x<node->data){
            if(x<node->left->data){
                SingleRotateLeft(node);
            }
            else{
                DoubleRotateLeft(node);
            }
        }
        else{
            if(x>node->right->data){
                SingleRotateRight(node);
            }
            else{
                DoubleRotateRight(node);
            }
        }

    }
    if(fanode==nullptr)
        tree=node;
    else if(x<fanode->data)
        fanode->left=node;
    else
        fanode->right=node;

    if(fanode!=nullptr) fanode->height=max(Height(fanode->left),Height(fanode->right))+1;

    return true;
}


// nonshorternode + tobedelete

template<class Type>
pair<AVLNode<Type> *, AVLNode<Type> *> AVLTree<Type>::Findnonshorter(AVLNode<Type> *Tree, Type x) {

    AVLNode<Type> *t=Tree,*TobeDelete=nullptr,*nonshorter=nullptr,*child=nullptr;
    stack<AVLNode<Type>*> sta;int ok=0;
    while(t!=nullptr){

        sta.push(t);
        if(x<t->data){
            t=t->left;
        }
        else if(x>t->data){
            t=t->right;
        }
        else{
            P=t;ok=1;
            if(t->left==nullptr&&t->right==nullptr){
                TobeDelete=t;break;
            }
            else if(t->left==nullptr){
                TobeDelete=t;break;
            }
            else if(t->right==nullptr){
                TobeDelete=t;break;
            }
            else{
                AVLNode<Type> *t2=t->left;sta.push(t2);
                while(t2->right!=nullptr){
                    t2=t2->right;
                    sta.push(t2);
                }
                TobeDelete=t2;break;
            }
        }
    }
    if(ok==0) return make_pair(nullptr,nullptr);
    FA= nullptr;
    bool shorter=true;
    while(!sta.empty()){
        AVLNode<Type> *now=sta.top();sta.pop();
        if(now==TobeDelete){
            child=now;continue;
        }
        if(child==now->left){
            if(shorter&&((Height(now->left)-Height(now->right)==1)||(Height(now->left)-Height(now->right)==-1&&Height(now->right->left)-Height(now->right->right)!=0))){
                shorter=true;
            }
            else{
                shorter=false;
                if(!sta.empty()) FA=sta.top();
                nonshorter=now;
                break;
            }
        }
        else{
            if(shorter&&((Height(now->left)-Height(now->right)==-1)||(Height(now->left)-Height(now->right)==1&&Height(now->left->left)-Height(now->left->right)!=0))){
                shorter=true;
            }
            else{
                shorter=false;
                if(!sta.empty()) FA=sta.top();
                nonshorter=now;
                break;
            }
        }

        child=now;
    }
    if(nonshorter==nullptr) nonshorter=Tree;
    return make_pair(nonshorter,TobeDelete);

}

template<class Type>
bool AVLTree<Type>::Delete(AVLNode<Type>* now,const Type &x){
    pair<AVLNode<Type>*,AVLNode<Type>*> res=Findnonshorter(now,x);

    now=res.first;AVLNode<Type> *ToBeDelNode=res.second;
    if(ToBeDelNode==nullptr){
        return false;
    }

    Type val=ToBeDelNode->data;
    if(FA== nullptr) FA=now;
    AVLNode<Type> **pp=nullptr;
    if(FA->left==now) pp=&FA->left;
    else if(FA->right==now) pp=&FA->right;
    else pp=&root;

    while(1){
        if(now==ToBeDelNode){
            *pp=(now->right==nullptr)?now->left:now->right;
            break;
        }
        if(val<now->data){
            if(bf(now)<=0){
                now->height=max(Height(now->left)-1,Height(now->right))+1;
                *pp=now;

                pp=&now->left;
            }
            else if(bf(now)==1&&bf(now->right)>=0){
                now->left->height--;
                AVLNode<Type> *p=now,*q=now->right,*nLeftRight=q->left;
                *pp=q;
                q->left=p;
                p->right=nLeftRight;
                p->height=max(Height(p->left),Height(p->right))+1;
                q->height=max(Height(q->left),Height(q->right))+1;
                pp=&p->left;
            }
            else if(bf(now)==1&&bf(now->right)==-1){
                now->left->height--;
                AVLNode<Type> *p,*q,*r,*nLeftRight,*nRightLeft;
                p=now;q=now->right;r=q->left;nLeftRight=r->left;nRightLeft=r->right;

                *pp=r;
                r->left=p;
                r->right=q;
                p->right=nLeftRight;
                q->left=nRightLeft;
                p->height=max(Height(p->left),Height(p->right))+1;
                q->height=max(Height(q->left),Height(q->right))+1;
                r->height=max(Height(r->left),Height(r->right))+1;
                pp=&p->left;
            }
            else{
                cout<<"GG"<<endl;
            }
            now=*pp;
        }
        else if(val>now->data){
            if(bf(now)>=0){
                now->height=max(Height(now->left),Height(now->right)-1)+1;
                *pp=now;
                pp=&now->right;
            }
            else if(bf(now)==-1&&bf(now->left)<=0){

                now->right->height--;
                AVLNode<Type> *p=now,*q=now->left,*nRightLeft=q->right;
                *pp=q;
                q->right=p;
                p->left=nRightLeft;
                p->height=max(Height(p->left),Height(p->right))+1;
                q->height=max(Height(q->left),Height(q->right))+1;
                pp=&p->right;
            }
            else if(bf(now)==-1&&bf(now->left)==1){
                now->right->height--;
                AVLNode<Type> *p,*q,*r,*nLeftRight,*nRightLeft;
                p=now;q=now->left;r=q->right;nLeftRight=r->left;nRightLeft=r->right;

                *pp=r;
                r->right=p;
                r->left=q;
                p->left=nRightLeft;
                q->right=nLeftRight;
                p->height=max(Height(p->left),Height(p->right))+1;
                q->height=max(Height(q->left),Height(q->right))+1;
                r->height=max(Height(r->left),Height(r->right))+1;
                pp=&p->right;
            }
            else{
                cout<<"GG"<<endl;
            }
            now=*pp;

        }
        else{
            cout<<"Already exist"<<endl;
            return false;
        }



    }
    P->data=val;
    return true;
}

int main() {

    AVLTree<int> Tree;

    int n;cin>>n;
    for(int i=1;i<=n;i++){
        int op;
        int val;
        cin>>op>>val;
        if(op==1){
            Tree.Insert(val);
            Tree.Print(Tree.Root());cout<<"\n";
        }
        else{
            Tree.Delete(val);
            Tree.Print(Tree.Root());cout<<"\n";
        }
    }
    return 0;
}

```


（mostly from oiwiki）

# B树

![Alt text](DS/image.png)

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

![Alt text](DS/image-1.png)


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

![Alt text](DS/image-2.png)
![Alt text](DS/image-3.png)
![Alt text](DS/image-4.png)
![Alt text](DS/image-5.png)

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



### B树的优势

![Alt text](DS/image-6.png)

因为单点容量参数是我们可以主观确定，所以可以与磁盘单位适配，从而提高效率。

同时让逻辑上相邻的数据都能尽量存储在物理上也相邻的硬盘空间上，这意味着我们需要降低树的高度同时让单个节点存储的数据变多。

------

# Splay

![Alt text](DS/image-7.png)

### Splay操作

将一个访问过的节点旋转至根节点。

![Alt text](DS/image-8.png)

这是对p是根节点的操作，只有当x的层数是奇数时才会作为splay操作的最后一步执行。


接下来两种操作，都将x的深度减2。

根据x、父亲、爷爷的关系，分为两种情况：

![Alt text](DS/image-9.png)

当连成一条线时，先转父亲&爷爷，然后再转下面x和父亲的边。

![Alt text](DS/image-10.png)

当构成折线时，先转x和父亲，然后再转上面的父亲和爷爷的边。

### 时间复杂度证明

![Alt text](DS/image-11.png)
![Alt text](DS/image-12.png)

### 插入操作
![Alt text](DS/image-13.png)

其实很简单，就是普通的插入，然后把插入的节点splay到根节点。


### 合并操作

![Alt text](DS/image-14.png)

### 查询x排名

![Alt text](DS/image-16.png)

注意一些访问操作结束后都需要splay。

### 删除操作

![Alt text](DS/image-15.png)
