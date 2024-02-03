---
title: 'DQN experiment: Pong'
date: 2024-02-03 15:39:32
tags: RL
Category: 
- RL
---

# <center> DQN theory and practice </center>
#### <center> detect0503@gmail.com </center>

**original paper: [Playing Atari with Deep Reinforcement Learning
](https://arxiv.org/pdf/1312.5602.pdf)**

# Abstract

This is a simple experiment to train a DQN agent to play atari game Pong. 

# Introduction

Since traditional reinforcement learning algorithms are not suitable for high-dimensional state space, the DQN algorithm is proposed to solve this problem. The DQN algorithm uses a deep neural network to approximate the Q function, and uses experience replay and target network to stabilize the training process. The DQN algorithm has achieved good results in many atari games, and has become a milestone in the field of reinforcement learning.

Our goal is to train a DQN agent who can play atari game Pong. The agent will receive a reward of +1 if it wins the game, and a reward of -1 if it loses the game. The whole game lasts for 20 turns. The higher the score, the better the agent's performance.(in fact, if we finally win the contest, we will get a reward of +1, otherwise, we will get a reward of -1. And we won't gain anythin unless the contest is done. In this way, our agent can play different games better with the same algorithm framework.)

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

![Alt text](image.png)

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

#### game 

PongNoFrameskip-v4












