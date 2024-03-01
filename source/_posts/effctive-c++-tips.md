---
title: effctive c++ tips
abbrlink: a20e8ec8
date: 2023-03-23 20:08:14
tags: c++
categories: 
- c++
- tips
---

## 1. 除非有好的理由允许构造函数隐式转换，否则申明构造函数为explicit

----

## 2. default构造函数，copy构造函数，copy assignment操作符
新对象被定义一定有个构造函数被调用
```c++
Class w1;//default 
Class w2(w1);//copy构造
Class w1=w1;//copy赋值操作符
Class w3=w2;//copy构造
```

---

## 3. 类中声明常量
为确保此常量只有一个实例，用static
```c++ 
class{
    static const int Num=5;
}
```

同时也可以使用enum
```c++
class{
    enum{Num=5};
}
```
enum特征：
1. enum和define一样不允许指针指向自己，绝不会导致不必要的内存分配
2. 实用主义，很多代码用了它
---
## 4. const食用技巧

- ### 1. const对象只能调用const成员函数、不能调用非const成员函数；非const对象可以调用const成员函数

### 引发原因： 由调用成员函数时隐式传入的当前对象的this指针引起。

> 1、 非```const```成员函数中的隐式参数：classA* this


> 2、 ```const```成员函数中的隐式参数：```const classA* this```

### 根本原因：


> 1、 ```const```对象的指针为```const classA* this```，因此传入非const成员函数时编译器报错（类型不匹配，无法从const 指针转换为非const指针）；但传入const成员函数则类型匹配。

> 2、非```const```对象的指针为```classA* this```，可以调用const成员函数，因为const修饰符保证了不会修改该对象。

所以对于const对象可能调用的函数，可以直接加个const函数重载，c++支持const函数重载。

- ### 2. 类中使用mutable定义，这样在const成员函数中可以修改其值

- ### 3. const类不能作为按地址形式传入non-const函数中

- ### 4. 对operator * 使用const，防止如```a*b=c```的意外引用

---

## 5. Make sure that objects are initialzed before they're used
>建议永远使用```member initialization list```。（就算没有任何额外赋值，也建议在初始化成员列表里手动调用default或对内置数据类型附初值）

>实际上，构造函数在运行前会对所有类成员调用default构造一遍，这样花费了不必要的时间开销。况且，如const常量一定需要初始值，而不能被赋值。

-----

## 6. 构造/析构/赋值运算

- ### 1. 只要你写了一个构造函数，default版构造函数就不会创建
- ### 2. 当类中有const成员，引用成员(string&),或者base class的赋值操作符是private，那么编译器拒绝生成默认operator=。 

- ### 3. 不想编译器自动生成函数，可以```=delete```

- ### 4. 心得：只有当class内至少含有一个virtual函数，才为它声明virtual析构函数（virtual函数会开一个虚函数表占用内存）,同时因为STL标准库里全是non-virtual，所以一般不将其作为base class继承。

- ### 5. 在虚析构函数继承时需要对虚析构函数提供实现.
> 因为当我们不定义虚构函数的时候，编译器会默认生成一个什么都不做的析构函数，但是注意了默认生成的析构函数就是普通函数不是虚函数！！！（因为虚函数会带来额外开销，c++追求的是速度），所以指望不上编译器自动生成虚构函数，而析构时又一定会调用，所以需要我们手动实现虚析构函数的实现。


----

## 7.在构造函数和析构函数间不调用virtual函数。
> 因为这类调用从不下降至derived class。（需要的话可以用static函数从derived class传参向base class的构造函数，并在base里用non-virtual函数接收并按照接收参数的不同做出不同response。

----

## 8. 让operator=返回一个reference to *this
```c++
Widget& operator =(const Widget& rhs){
    //******
    return *this;
}
```
一方面可以实现连等，另一方面这样的协议被STL共同遵守。

-----

## 9. 以对象为资源管理(RALL(resource acquisition is initializaion))

- ### 1. 获得资源后立刻放进管理对象（如使用auto_ptr）
- ### 2. 管理对象(managing object)运用析构函数确保资源被释放 

以防止开的空间在delete前因各种原因中端而导致内存泄漏
auto_ptr一般用在heap_based资源

---


## 10. RALL对象的copy行为

- ### 1. 方法1：抑制copying（用private写copy）
- ### 2. 方法2：施行引用计数法(tr1::shared_ptr,可以以一个函数对象为第二参数，在计数为0时调用其)
----
## 11. 资源管理类中提供对原始资源的访问

我们用RALL类将对象装在类中防止内存泄漏发生，但是一些api要求原始的数据进行传参，这时就需要提供对原始资源的访问

> 对于智能指针```pINv.get()```即可调用其储存的原始资源

> 对于其他某些RALL类要么提供显式转换函数(如上)，要么提供隐式转换函数(```operator type()```)
----
## 12. 成对使用new和delete时要采用相同的形式
-----
## 13. 以独立语句将newed对象置于智能指针中
> 这样在new对象创立到将对象存进RALL中没有其他语句干扰，防止难以察觉的资源泄露（可能其他语句会中断，导致只创立了new对象没放到RALL类里面）
------
## 14. 让接口更容易被使用
> - 接口的一致性，与内置类型的行为兼容
> - “阻止误用”的办法包括建立新类型、限制类型上的操作、束缚对象值，以及消除客户的资源管理责任。
> - ```c++ tr1::shared_ptr```支持定制型删除器。
 

-----

## 15. 类型转化

>            T1 ---> T2
> - 在T1中写一个类型转换函数（operator T2）?????????????????
> - 或在T2中写一个non-explicit-one-aegument的构造函数
---
 ## 16. 尽可能用pass-by-reference-to-const替换pass-by-value 

> 第一，效率更好。第二，解决继承时的切割(slicing)问题
> 以上规则不适用于内置数据类型和STL
>
 ---

 ## 17. 返回值reference和object之间的抉择

 > 绝不要返回pointer或reference指向一个local stack对象，或返回reference指向一个local-allocated对象。
---
 ## 18. 将成员变量设计为private

 > 切记将成员变量设计成private，好处：1. 访问数据的一致性 2. 可以细微划分访问时的控制权限 3. 提供class作者充分的实现弹性。

> protected 并不比public 更具备封装性

---
## 19. prefer拿non-member non-friend函数替换member函数。


可以增加封装性，包裹弹性，和机能拓展性。 
> 越少的函数能直接调用私有数据，封装性越好

> 将多个功能函数（如clear功能、cookie功能）放在多个头文件但隶属于同一个命名空间，可以轻松拓展这些功能函数。（namespace可以跨越多个源码文件）
---
## 20. non-member函数

如果你需要为某个函数的所有参数（包括被this指针所指的那个隐喻参数）进行类型转换，那么这个函数必须是non-member。

最常见的是operator重载运算，因为this指针无法进行隐式类型转换，所以有必要变成non-member函数。但是是不是要friend是不一定的。不能只因函数不该是member，就让它自动成为friend。

---
## 21. swap的实现
 -   当std::swap对你的类型效率不高时，提供一个swap成员函数，并确定这个函数不抛出异常。
 - 如果你提供一个member swap，也提供一个non-member swap用来调用前者。对于classes（而非templates），也要特化std::swap>
```c++
class Widget{
    public:
    void swap(Widget& other){
        using std::swap;
        swap(Ptr,other.Ptr);
    }
};
namespace std{
    template<>
    void swap<Widget>(Widget& a,Widget& b){
        a.swap(b);
    }
}
```
- 调用swap时加上using std::swap，然后调用的时候不带任何“命名空间资格修饰”。

- swap调用顺序： 1：class专属的swap 2：std::swap内的专属特化版 3：默认版std::swap。
  
---

## 22. 尽可能延后变量定义式的出现时间

不只因该延后到非得使用该变量为止，甚至应该尝试延后这份定义知道能够给它初值实参为止。

不仅可以避免构造非必要对象，还可以避免无意义的default构造行为，更深一层说，以“具有明显意义之初值”将变量初始化还可以附带说明变量的目的。

-----

## 23. c++转型操作

- ###  宁可用c++style转型，不使用旧式转型，前者很容易被分辨出来，而且作用更加细化

> const_cast 用来将对象的常量性移除
> 
> dynamic_cast 用来“安全向下转型”，也就是用来决定某对象是否归属继承体系中的
某个类型。
>
> reinterpret_cast 低级转型，e.g. 将pointer to int转为int
> 
> static_cast 强迫隐式转型

- ###  使用旧式转型的唯一时机：调用一个explicit构造函数将一个对象传递给一个函数时。

- ###  单个对象（如一个derived类）可以拥有一个以上的地址，所以不要以为“转型什么都没做，只是告诉编译器把某种类型视为另一种类型“）

- ###  如果可以，尽量避免转型。（如指针动态链接，或virtual函数） 
------
## 24. 避免返回handles(reference,指针，迭代器)指向对象内部成分

- 可增加封装性，帮助const成员函数行为像一个const。
- 降低出现“虚吊号牌”(dangling handles)的可能性。（对象先寄了，导致成员赋值的内部指针虚吊。）

---

## 25. 关于inline

- 好处，调用其不用承受函数的额外开支
- inline将“对此函数的每一个调用”都一函数本体来替换之。但是，可能会造成代码膨胀，降低高速缓存的集中率。
- 注意：将函数定义在class定义式中会隐式变成inline
- virtual和inline不兼容，因为virtual意味着等待，运行阶段才能确定。而inline意味着执行前替换。
- 构造和析构函数往往不适合inline，哪怕是空的构造函数，编译器也会产生一定分量的代码，这样的inline往往会使代买膨胀。
- inline内容一旦改，意味着所有用到f的内容都要重新编译，如果不用inline，重新链接即可。
- 将大多数inline限制在小型的，被频繁调用的函数身上。

----

## 26.减小声明式的依赖性

- 能使用object reference或者object pointer，就不要使用object。（因为如果定义某类型的object，会需要其定义式。

- 如果可以，尽量以class声明式替换class的定义式。
  
  >注意，当你声明一个函数而用到某个class时，你并不需要其class的定义式，即使该函数以by value方式传递该class的参数(或者返回值)。
```c++
class date;
date Today();
void ClearAppointments(date d);
//合法的
```

----

## 27. public继承塑模出is-a关系

适用于base的每一件事也一定适用于derived身上，因为每一个derived也是一个base。

e.g. bird类里fly函数。但是Penguin is a bird，但是不会fly。所以bird类里不能放fly函数。

---

## 28. 避免遮盖继承而来的名称

- derived class内的名称会遮盖base class内的名称。在public继承下从来没有人希望如此。

- 为了让被遮盖的名称重见天日，可以使用```using声明式```或者```转交函数(forwarding function)```。

---

## 29. 接口继承与实现继承

- 对于base class为真的任何事一定对其derived class也为真。因此每个函数可以用于class上，也一定可以用于derived class上。
- 声明一个pure virtual函数的目的是为了让derived classes只继承函数接口。（不干涉derived怎么实现它）
- 声明impure derived函数的目的，是让derived class继承该函数的接口和**缺省实现**。（default版直接用域名解析调用）
- pure virtual函数也可以被实现，用来当缺省实现。
- non-virtual 函数的目的是为了令derived class继承函数的接口及一份强制性的实现。

## 30. 考虑用设计模式替换virtual函数

介绍两种设计模式，NVI(non-virtial interface)，主张将所有virtual设计为private。Strategy设计模式，用函数指针或者古典strategy，另外开个类。

因为是关于设计模式的介绍，详细可以看他人的 **[博客分享](https://blog.csdn.net/CltCj/article/details/128432338)**。

---

## 31. 绝不重新定义继承而来的non-virtual函数

- non-virtual函数为静态绑定，也就是说pointer-to-base永远调用的是base版本，即使其指向一个derived类。一个简单的例子是为什么多态中析构函数都是virtual：虚析构函数为了避免内存泄露,基类的析构函数一般都是虚函数。 如果基类析构函数不是虚函数:基类指针指向子类对象,delete基类指针,调用基类析构函数,不会调用子类析构函数,造成内存泄露。 

- non-virtual函数的不变性高于其特异性。（public继承下，一切base的non-virtual都适用于所有的derived classes）
---

## 32. 绝不重新定义继承而来的缺省参数值

理由很简单，virtual函数时动态绑定(dynamically bound)，而缺省参数值却是静态绑定。

所以如果使用多态时，重新定义了继承而来的缺省参数值，在动态绑定中，其默认参数却永远是base的，导致难以发现的错位。（继承中唯一该覆写的东西是virtual函数）

（如果想要统一默认参数，可以使用30条中的NVI设计模式。）

---
## 33. 复合关系

- 在应用域(application domain)，复合意味着has-a。
- 在实现域(implementation domain)，复合意味着 is-implemented-in-terms-of（根据某物实现出）。e.g. 用std::list 实现SET。

---
## 34. private继承

public继承在必要时可（为了让函数调用成功），可以将derived类转成base，而private继承则不行。

private base 继承而来的所有东西都会在derived类中变成private属性。

- **private继承意味着只有实现部分被继承，接口部分应略去。如果D以private形式继承B，意思是D对象根据B对象实现而得，再没有其他意涵了。**

复合(composition)比private继承 （都是is-implemented-in-terms-of）

- 复合优点：阻止derived类重新定义接口、可以解耦
```c++
class Widget{
    private: 
    class WidgetTimer: public Timer{
        virtual void onTick() const;
        //...
    };
    WidgetTimer timer;
    //...
};
```
- private继承优点：当derived函数需要访问protected based class的成员、或需要重新定义继承而来的virtual函数。

---

## 35. 多重继承

- 多重继承比单一继承更复杂。它可能导致新的歧义性，以及对virtual继承的需要。
- virtual继承会增加大小，速度，初始化复杂度成本。
- 多重继承的确有正当用途。比如：```"public继承某个inteface class","private继承某个协助实现的class"```的两两组合。
---

## 36.隐式接口和编译器多态

- class和template都支持接口和多态
- 对class而言接口是显示的，以函数签名为中心。多态则是通过virtual函数发生于运行期。
- 对template参数而言，接口是隐式的，奠基于有效表达式。多态则是通过template具现化和函数重载解析发生于编译期。