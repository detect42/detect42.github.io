---
title: Date_Structure
tags: DS
abbrlink: 1fe2fe25
date: 2023-09-09 17:40:24
---

## 链表

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


## AVL平衡树（无递归无栈version）

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