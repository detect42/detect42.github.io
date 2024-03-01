---
title: 人工智能导论HW2黑白棋实验报告
tags: 实验报告
categories: 
- NJU course
- IntroAI
abbrlink: 97fe6ca1
date: 2023-10-18 00:26:24
---
# <center> 报告题目：黑白棋游戏&博弈算法 </center>

### <center>  detect0530@gmail.com </center>

## 1 引言

过去曾有关注过博弈论的相关算法，比如那什均衡、博弈树。但是因为效率的担心，往往忽略了最为传统但又花样百出的搜索博弈。这次作业，我将尝试用搜索博弈的思维来考虑黑白棋游戏。

## 2 实验内容

### 2.1 Task1 介绍minimax的实现

```java
public MiniMaxDecider(boolean maximize, int depth) {
    this.maximize = maximize;
    this.depth = depth;
    computedStates = new HashMap<State, Float>();
}
```

这里的maximize表示当前的决策者是最大化还是最小化，depth表示搜索的深度，computedStates表示已经计算过的状态，用HashMap进行存储。

```java
	public Action decide(State state) {
		// Choose randomly between equally good options
		float value = maximize ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
		List<Action> bestActions = new ArrayList<Action>();
		// Iterate!
		int flag = maximize ? 1 : -1;
		for (Action action : state.getActions()) {
			try {
				// Algorithm!
				State newState = action.applyTo(state);
				float newValue = this.miniMaxRecursor(newState, 1, !this.maximize);
				// Better candidates?
				if (flag * newValue > flag * value) {
					value = newValue;
					bestActions.clear();
				}
				// Add it to the list of candidates?
				if (flag * newValue >= flag * value) bestActions.add(action);
			} catch (InvalidActionException e) {
				throw new RuntimeException("Invalid action!");
			}
		}
		// If there are more than one best actions, pick one of the best randomly
		Collections.shuffle(bestActions);
		return bestActions.get(0);
	}
```

$decide$函数决定当前走哪一步，方法很暴力，对于每种可能的走法都计算一遍minimax，取最大的即可。思想堪称朴实无华。

---

接下来将详细解释关键的$minimaxRecursor()$函数:

```java
	public float miniMaxRecursor(State state, int depth, boolean maximize) {
		// Has this state already been computed?
		if (computedStates.containsKey(state)) 
                    // Return the stored result
                    return computedStates.get(state);
		// Is this state done?
		if (state.getStatus() != Status.Ongoing)
                    // Store and return
                    return finalize(state, state.heuristic());
		// Have we reached the end of the line?
		if (depth == this.depth)
                    //Return the heuristic value
                    return state.heuristic();

```

1. 首先是如果当前状态我们已经算过了，那么直接返回上一次计算存在HashMap的值即可。
2. 否则，如果状态结束或者深度到了我们指定的超参数Maxdep，也就停止搜索返回heuristic值。

```java
float value = maximize ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
		int flag = maximize ? 1 : -1;
		List<Action> test = state.getActions();
		for (Action action : test) {
			// Check it. Is it better? If so, keep it.
			try {
				State childState = action.applyTo(state);
				float newValue = this.miniMaxRecursor(childState, depth + 1, !maximize);
				//Record the best value
                if (flag * newValue > flag * value) 
                    value = newValue;
        } catch (InvalidActionException e) {
                            //Should not go here
            throw new RuntimeException("Invalid action!");
        }
    }
    // Store so we don't have to compute it again.
    return finalize(state, value);
}
```

上述代码非常巧妙地利用maximize作为开来判断当前应该取最小值还是最大值，并且利用不等式两边同时乘以负一会变号的手段，保证了对于min和max情况下的代码一致性。


而整体结构依然是喜闻乐见的递归搜索，对于每一个可能的走法，都进行一次递归搜索，直到到达最大深度或者游戏结束。结构上倒没有什么特别之处。

------

### 2.2 Task2 加入alpha-beta剪枝

第一步修改传参
```java
float newValue = this.miniMaxRecursor(newState, 1, !this.maximize,Float.NEGATIVE_INFINITY,Float.POSITIVE_INFINITY);
```

这是修改剪枝后的主体部分
```java
public float miniMaxRecursor(State state, int depth, boolean maximize,float alpha,float beta) {
		// Has this state already been computed?
		if (computedStates.containsKey(state))
                    // Return the stored result
                    return computedStates.get(state);
		// Is this state done?
		if (state.getStatus() != Status.Ongoing)
                    // Store and return
                    return finalize(state, state.heuristic());
		// Have we reached the end of the line?
		if (depth == this.depth)
                    //Return the heuristic value
                    return state.heuristic();
                
		// If not, recurse further. Identify the best actions to take.
		float value = maximize ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
		int flag = maximize ? 1 : -1;
		List<Action> test = state.getActions();
		for (Action action : test) {
			// Check it. Is it better? If so, keep it.
			try {
				State childState = action.applyTo(state);
				float newValue = this.miniMaxRecursor(childState, depth + 1, !maximize,alpha,beta);
				//Record the best value
				if (flag * newValue > flag * value){
					value = newValue;
                    //here!!!!!!!!!!!!!!!!!!!
					if(maximize){
						if(value>alpha){
							if(value>beta) return finalize(state, value);
							alpha = value;
						}
					}
					else{
						if(value<beta){
							if(value<alpha) return finalize(state, value);
							beta = value;
						}
					}
				}

			} catch (InvalidActionException e) {
                                //Should not go here
				throw new RuntimeException("Invalid action!");
			}
		}
		// Store so we don't have to compute it again.
		return finalize(state, value);
	}
```
**经过调整搜索时的$MaxDep$，当$MaxDep$设为较大的值比如$8,9,10$，且在棋局稍微复杂的情况下，alpha-beta剪枝的效果非常明显，搜索时间从30s左右大大减少到3s。**

由此可见，alpha-beta剪枝的效果非常明显，能够大大减少搜索时间。

--------

### 2.3 Task3 理解并改进heuristic函数

这是预先给出的heuristic函数：

```java
	@Override
	public float heuristic() {
		//System.out.printf("%f %f %f %f\n",this.pieceDifferential(), this.moveDifferential(), this.cornerDifferential(), this.stabilityDifferential());
		Status s = this.getStatus();
		int winconstant = 0;
		switch (s) {
		case PlayerOneWon:
			winconstant = 5000;
			break;
		case PlayerTwoWon:
			winconstant = -5000;
			break;
		default:
			winconstant = 0;
			break;
		}
		return this.pieceDifferential() +
		   8 * this.moveDifferential() +
		  300 * this.cornerDifferential() +
		   1 * this.stabilityDifferential() + 
		   winconstant;
	}
```

很直观朴素的计分方法：

1. 是否胜利 * 5000
2. 棋子差距 * 1
3. 可以下的位置数论差 * 8
4. 四个角的棋子差 * 300
5. 可以翻转的棋子的差 * 1

总结一下，是基于棋子灵活度，棋子数量，以及棋子与棋盘的交互来计算的。

至于优化，我们可以更进一步也更智慧地思考以上问题。

![Alt text](010.png)


6. 既然四个角的棋子是最有价值的，那么我们不光关心自己拿到四个角，还要防止对手拿到四个角。于是对于白色叉的格子，我们拿到它只能帮助对手拿到四个角，于是我们对白叉格子赋值为-100。
7. 对于边缘格子，其被取代的方式非常有限，于是我们也希望多拿一些边缘格子，赋值为10

```java
// 6,7条对应的函数
private float secondcorneDifferential(){
	float diff=0;
	short[] corners = new short[4];
	corners[0] = getSpotOnLine(hBoard[1], (byte)1);
	corners[1] = getSpotOnLine(hBoard[1], (byte)(dimension - 2));
	corners[2] = getSpotOnLine(hBoard[dimension - 2], (byte)1);
	corners[3] = getSpotOnLine(hBoard[dimension - 2], (byte)(dimension - 2));
	for (short corner : corners) if (corner != 0) diff += corner == 2 ? 1 : -1;
	return diff;
}
private float edgeDifferent(){
	float diff=0;
	for(int i=1;i<=dimension-2;i++){
		short Left=getSpotOnLine(hBoard[i], (byte)0);
		short Right=getSpotOnLine(hBoard[i], (byte)(dimension-1));
		diff += Left==2?1:-1;diff+= Right==2?1:-1;
	}
	for(int i=1;i<=dimension-2;i++){
		short Up=getSpotOnLine(hBoard[0], (byte)(i));
		short Down=getSpotOnLine(hBoard[dimension-1], (byte)(i));
		diff+= Up==2?1:-1;diff+= Down==2?1:-1;
	}
	return diff;
}
```

![Alt text](011.png)

8. 游戏开始后，在前面12步中，也就是抛开开局系统给出的四颗棋子外（红色方框1内），最好不要把棋子放在红色方框2之外。这个部分的宗旨是先占满方框2，把对方逼出方框。让我们有希望拿到边缘点。

```java
private float FirstDifferent(){
		float diff=0;
		for(int i=2;i<=dimension-3;i++){
			short Left=getSpotOnLine(hBoard[i], (byte)2);
			short Right=getSpotOnLine(hBoard[i], (byte)(dimension-3));
			diff += Left==2?1:-1;diff+= Right==2?1:-1;
		}
		for(int i=2;i<=dimension-3;i++){
			short Up=getSpotOnLine(hBoard[2], (byte)(i));
			short Down=getSpotOnLine(hBoard[dimension-3], (byte)(i));
			diff+= Up==2?1:-1;diff+= Down==2?1:-1;
		}
		return diff;
	}
```

最后我们的heuristic函数长这样：

```java
return this.pieceDifferential() +
		   8 * this.moveDifferential() +
		  300 * this.cornerDifferential() +
		   1 * this.stabilityDifferential() +
		        -100 * this.secondcorneDifferential() +
				10 * this.edgeDifferent() +
				5 * this.FirstDifferent() +
```

经过测试，确实能在游戏过程中很大程度上完成上述设置的要求，证明了我们的优化是有效的。


-------

### 2.4 Task4 MTD搜索法的分析

MTD-f 算法的全称是Memory-enhanced Test Driver，是一种搜索算法，它是一种迭代加深搜索算法，它的特点是在每次迭代中，都会使用一个零窗口搜索来确定当前的最佳值，然后使用这个值来作为下一次迭代的窗口的中心。

我们先从代码层理解MTD到底在做什么？

```java
	public Action decide(State state) {
		startTimeMillis = System.currentTimeMillis();
		transpositionTable = new HashMap<State, SearchNode>(10000);
		return iterative_deepening(state);
	}
```

记录一个时钟，并记下转移过程中的索引表，这样遇到同样的位置就不用重新做一遍了。

接下来，进入迭代加深过程，在这个过程中，我们可以找到一个优秀的解法。

删掉一些无关紧要的部分：

**对代码的一些分析在注释里**
```java

// 对代码的一些分析在注释里

int d;
//对于限定的深度下每一个深度都要跑一遍
for (d = 1; d < maxdepth; d++) {
	int alpha = LOSE; int beta = WIN; int actionsExplored = 0;
	for (ActionValuePair a : actions) {
		//枚举所有的actions，注意这里的actions有一个pair对，除了action动作外还有这个action下的最优value
		State n;
		try {
			n = a.action.applyTo(root);
			
			int value;
			if (USE_MTDF)
			//这里进入关键的MTDF评估函数，传入的参数是当前的状态，当前的最优值，当前的深度
				value = MTDF(n, (int) a.value, d);
			else {
				int flag = maximizer ? 1 : -1;
				value = -AlphaBetaWithMemory(n, -beta , -alpha, d - 1, -flag);
			}
			actionsExplored++;
			// Store the computed value for move ordering
			a.value = value;
			
		} catch (InvalidActionException e) {
			e.printStackTrace();
		} catch (OutOfTimeException e) {
			System.out.println("Out of time");
			// revert to the previously computed values. 
			//HOWEVER, if our best value is found to be catastrophic, keep its value.
			// TODO: this should keep all found catastrophic values, not just the first!

			//在至少有两个action探索过后，我们就可以进行比较了！

			boolean resetBest = true;
			if (actionsExplored > 1) {
				ActionValuePair bestAction = actions.get(0);
				// check to see if the best action is worse than another action
				for (int i=0; i < actionsExplored; i++) {
					if (bestAction.value < actions.get(i).value) {
						// don't reset the first choice
						resetBest = false;
						break;
					}
				}	
			}
			//如果我们发现了更好的action，那么就把之前的action的value恢复到之前的状态，否则除了第一个action，其他的都恢复到之前的状态
			if (resetBest) {
				for (ActionValuePair ac: actions) {
					ac.value = ac.previousValue;
				}
			} else {
				for (int i=1; i < actionsExplored; i++) {
					actions.get(i).value = actions.get(i).previousValue;
				}
			}
			break;
		}
	}
	// 提前排好序，为下一轮的迭代做好准备
	Collections.sort(actions, Collections.reverseOrder());
	
	// 更新previousvalue
	for (ActionValuePair a: actions) {
		a.previousValue = a.value;
	}
}
```

**上述内容会在给定的时间内进行迭代，每一次迭代都会更新当前的最优值，直到时间到了，我们就可以返回当前的最优值了。（所以会按照深度迭代，在时间范围内尽可能多地在更深的地方搜索）**

接下来是关键的MTDF函数：

```java
private int MTDF(State root, int firstGuess, int depth)
		throws OutOfTimeException {
	int g = firstGuess;
	int beta;
	int upperbound = WIN;
	int lowerbound = LOSE;

	int flag = maximizer ? 1 : -1;

	while (lowerbound < upperbound) {
		if (g == lowerbound) {
			beta = g + 1;
		} else {
			beta = g;
		}
		// Traditional NegaMax call, just with different bounds
		g = -AlphaBetaWithMemory(root, beta - 1, beta, depth, -flag);
		if (g < beta) {
			upperbound = g;
		} else {
			lowerbound = g;
		}
	}

	return g;
}
```

$MTD-f$算法采用零窗口搜索方法，旨在确定局面的最优值，初始时将最佳值范围设置为负无穷到正无穷。通过一系列零窗口搜索，不断缩小这个范围，最终确定当前局面的最优值。

其采用一种与期望搜索相似的方法，但在$Alpha-Beta$搜索中对初始值进行智能调整。它的核心思想是通过使搜索窗口尽可能窄来提高搜索效率，一直使用$beta = alpha + 1$来调用$Alpha-Beta$搜索。这种"零宽度"搜索的目的是与Alpha值进行比较，如果搜索的返回值不超过Alpha，那么确切值也不会超过Alpha；反之亦然，如果确切值大于Alpha，搜索结果也会大于Alpha。这种方法有助于更快速地确定最佳局面值。

由于在搜索过程中上下界差只有1，所以可以迅速剪枝，并以极快的速度返回解。

有关$MTD-f$的更多细节：

1. 试探值并不一定设成零，可以用迭代加深的形式由浅一层的MTD(f)搜索给出
2. 因为要进行多次多轮的探索，所以置换表技术相当重要。
3. 虽然要跑很多次，但是$MTD-f$用极端的限制进行$Alpha-Beta$搜索大大减少了搜索时间，所以效率还是很高的。

最后，**MTD与MiniMax的比较**

- 相同点：

1. 都是基于minimax，或者说MTD是minimax的拓展做法

- 不同点：
  
1. MTD使用了迭代加深，每次迭代都在更新最优的策略行为
2. MTD使用了置换表技术和Alpha-Beta优化，同时也使用了零窗口搜索，大大减少了搜索时间。


## 3 结语

本次实验，我充分分析并实践了minimax和其的拓展做法MTD，同时也对alpha-beta剪枝有了更深刻的理解。其强大的性能让我对搜索博弈有了新的认识，收货斐然。