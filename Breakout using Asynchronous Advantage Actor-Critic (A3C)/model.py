"""
AI for Breakout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

#Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

#Making the A3C brain
#Consists of fully connected layers, eyes, and memories
#Two nerual networks for the critic and the actor
#Small standard deviation of the weights to the actor and a large standard deviation of the weightsto the critic allows 
#for good management of exploration vs exploitation
class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) #Eyes
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #Eyes
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #Eyes
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) #Eyes
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) #Memory
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) #The 256 comes from the output of the lstm layer, output = V(s)
        self.actor_linear = nn.Linear(256, num_outputs) #The 256 comes from the output of the lstm layer, output = Q(s, a)
        self.apply(weights_init) #Initialize random weights
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) #Standard deviation of the weights, smaller for the actor
        self.actor_linear.bias.data.fill_(0) #Double check that the biases are initialized to 0, may not need
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) #Standard deviation of the weights, larger for the critic
        self.critic_linear.bias.data.fill_(0) #Double check that the biases are initialized to 0, may not need
        self.lstm.bias_ih.data.fill_(0) #Biases or the lstm layer are initialized to 0
        self.lstm.bias_hh.data.fill_(0) #Biases of the lstm layer are initialized to 0
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs #hx are the hidden states/nodes of the lstm and cx are the cell states/nodes of the lstm
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)