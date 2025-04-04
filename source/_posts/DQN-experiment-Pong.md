---
title: 'DQN experiment: Pong'
tags: RL
Category:
  - RL
abbrlink: cb58cbf7
date: 2024-02-03 15:39:32
---

# <center> DQN theory and practice </center>
#### <center> detect0503@gmail.com </center>

**original paper: [Playing Atari with Deep Reinforcement Learning
](https://arxiv.org/pdf/1312.5602.pdf)**

# Abstract

This is a simple experiment to train a DQN agent to play atari game Pong. This paper proposes a novel approach, directly using the entire game screen as input for the model. The model autonomously determines what features to extract, and even the specific features it learns are not of primary concern. What matters is that the model can make correct actions in various states, and we don't necessarily need to know the exact features it has learned. 

# Introduction

Since traditional reinforcement learning algorithms are not suitable for high-dimensional state space, the DQN algorithm is proposed to solve this problem. The DQN algorithm uses a deep neural network to approximate the Q function, and uses experience replay and target network to stabilize the training process. The DQN algorithm has achieved good results in many atari games, and has become a milestone in the field of reinforcement learning.

Our goal is to train a DQN agent who can play atari game Pong. The agent will receive a reward of +1 if it wins the game, and a reward of -1 if it loses the game. The whole game lasts for at most 21*2-1 =41 turns. The higher the score, the better the agent's performance.(in fact, if we finally win the contest, we will get a reward of +1, otherwise, we will get a reward of -1. And we won't gain anything unless the contest is done. In this way, our agent can play different games better with the same algorithm framework.)

In this article, I'll reproduce the DQN algorithm and analyze why it works compared to traditional reinforcement learning algorithms.



# Related work

- some traditional table based reinforcement learning algorithms

drawbacks: can't handle high-dimensional state space

- with some prior knowledge

drawbacks: won't be a method that can be used in a wide range of games

- TD-Gammon approxiate the state value function $v^{\star}(s)$ rather than the action value function $q^{\star}(s,a)$

drawbacks: slow and not SOTA

- on-policy methods

Let's not discuss it for now

# Method on paper

## Apporximate Q-function

Most of the time, the state space of the environment is too large to store the Q function in a table. Therefore, we need to use a function approximator to approximate the Q function. In the DQN algorithm, the Q function is approximated by a deep neural network. The input of the neural network is the state of the environment, and the output is the Q value of each action.

Sometimes, it's hard to map the game into a fixed-size state space. Since using histories of arbitrary length as inputs to a neural network can be difficult, our Q-function instead works on fixed length representation of histories produced by a function $\phi$. The function $\phi$ takes the last 4 frames and stacks them to produce a 4x84x84 image.

**In summary, our network takes a stack of 4 frames as input and outputs a Q value for each action.**

In previous work, there are some methods taking the advantage of DL as well. All of these methods involve manually extracting features while simultaneously separating the background (somewhat like to dividing 128 colors into 128 layers and then annotating what each color represents). Although this approach is effective for Atari games where different colors typically correspond to different categories of objects, DeepMind chose a different approach. **They opted for the neural network to autonomously learn how to separate the background from game objects rather than relying on manual feature extraction.**

## Experience replay

The experience replay is a technique that stores the agent's experiences in a replay buffer, and then samples a batch of experiences from the replay buffer to train the neural network. The experience replay can break the correlation between the experiences, and stabilize the training process. Pay attention that experience replay is a **off-policy** method.

The correctness of experience replay is based on the assumption that the experiences are independent and identically distributed, similar to the stochastic gradient descent. 

With experience replay, 
- the agent can learn from the same experience multiple times, which means data can be reused efficiently and easing the problem of loss of experience data.
- The experience replay can break the correlation between the experiences, and stabilize the training process. 

(what's more, we can introduce the concept of priority experience replay, which can be used to sample the experiences with higher TD error, and thus improve the training efficiency.)


## Design of the loss function

The loss function of the DQN algorithm is the mean squared error between the Q value predicted by the neural network and the Q value calculated by the Bellman equation. The loss function is defined as follows:

$$L_i(\theta_i) = E_{s,a\sim p(\cdot)}[(y_i-Q(s,a;\theta_i))^2]$$


where $y_i$ is denoted as the target Q value, and is calculated by the Bellman equation:

$$y_i=E_{s,a\sim p(\cdot)}[r+\gamma max_{a'}Q(s',a';\theta_{i-1})|s,a]$$

Since it's off-policy methods, we set policy $\pi$ to be the $\epsilon-greedy$, asuring that the agent can explore the environment.

(One more thing to mention is that the target Q value is calculated by the target network, which is a copy of the Q network. The target network is updated by the Q network every $C$ steps. In this way, the overstate problem can be eased.)

------

# Experiment

![Alt text](DQN0/image0.png)

### data preprocessing

Working directly with raw Atari frames, which are 210 × 160 pixel images with a 128 color palette,can be computationally demanding.

The raw frames are preprocessed by first converting their RGB representation to gray-scale and down-sampling it to a 110×84 image. 

The final input representation is obtained by cropping an 84 × 84 region of the image that roughly captures the playing area. 

the function $\phi$ from algorithm 1 applies this preprocessing to the last 4 frames of a history and stacks them to produce the input to the Q-function.

### network architecture

When describing a neural network, the hierarchical structure is typically built from the input layer to the output layer. Here is the hierarchical structure of the aforementioned DQN model:

1. Input Layer:
- Number of input channels for the input images: `in_channels`, defaulted to 4, representing the stacking of the latest 4 frames of images.

2. First Convolutional Layer (`self.conv1`):
- Input channels: `in_channels` (same as the input layer)
- Output channels: 32
- Convolutional kernel size: 8x8
- Stride: 4
- Output size: Calculated through the convolution operation

3. Second Convolutional Layer (`self.conv2`):
- Input channels: 32 (from the output of the first convolutional layer)
- Output channels: 64
- Convolutional kernel size: 4x4
- Stride: 2
- Output size: Calculated through the convolution operation

4. Third Convolutional Layer (`self.conv3`):
- Input channels: 64 (from the output of the second convolutional layer)
- Output channels: 64
- Convolutional kernel size: 3x3
- Stride: 1
- Output size: Calculated through the convolution operation

5. Fully Connected Layer (`self.fc4`):
- Input size: Flattening the output of the convolutional layers into a one-dimensional vector (`7 * 7 * 64`)
- Output size: 512

6. Output Layer (`self.fc5`):
- Input size: 512 (from the output of the fully connected layer)
- Output size: `num_actions` (representing the values for each possible action)



# DQN experiment details

This is a simple experiment to train a DQN agent to play atari game Pong. I attempt to use the same hyperparameters as the original [DQN paper](https://arxiv.org/abs/1312.5602). Unfortunately, I'm not familar with the usage of the atari environment api in atari game and pytorch. However, I implemented the DQN algorithm with the help of Chatbot. 

#### requirements
python 3.10
gym 0.23.0
pytorch 2.2.0         
pytorch-cuda 11.8           
pytorch-mutex 1.0                  
torchaudio 2.2.0
torchtriton 2.2.0      
torchvision 0.17.0

#### game 

PongNoFrameskip-v4


### Performance

```py
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.05
eps_decay = 30000
frames = 2000000
USE_CUDA = True
learning_rate = 2e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 10000
win_reward = 18    
win_break = True
```

use the hyperparameter above to train the agent, and the result is as follows:

![Alt text](DQN0/image.png)

it only gains 16.7 out of 21, seeming that the agent is not good enough.

So I tried to change the hyperparameter, and added some techniques.

- $\epsilon$-greedy, $\epsilon$ descending so quickly, which means the agent can't explore the environment enough. So I made descending more slowly, while making the terminal $\epsilon$ smaller.

```py
epsilon_max = 1
epsilon_min = 0.02
eps_decay = 200000 # 增加了随机性
```

- larger replay buffer

```py
batch_size = 64 # 增加了经验回收容量
```

- adjust the optimizer and add warm-up scheduler
```py
self.optimizer = optim.AdamW(self.DQN.parameters(), lr=lr, weight_decay=0.01)
self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_total)
```


- cancel the gradient clip
```py
#for param in self.DQN.parameters():
#   param.grad.data.clamp_(-1, 1)
```

Ultimately, the agent can gain 20.1 out of 21, which is a good result.


![Alt text](DQN0/image-1.png)

![alt text](DQN0/image-2.png)

![alt text](DQN0/image-3.png)