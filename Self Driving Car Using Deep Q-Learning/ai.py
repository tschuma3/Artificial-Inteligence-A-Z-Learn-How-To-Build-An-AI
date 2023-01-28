import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the Neural Network
class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Implementing Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

#Implementing Deep Q Learning
class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer =  optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    #Takes an action based on the state given
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100) #T=100
        action = probs.multinomial()
        return action.data[0, 0]

    #Implements the Deep Q Learning equation 
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_output = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_output + batch_reward #gamma * Q + R
        td_loss = F.smooth_l1_loss(outputs, target) #Uses the Hubber Loss
        self.optimizer.zero_grad() #Stochastic Gradient Descent and Initializes the Optimizer
        td_loss.backward(retain_variables=True) #Backward Propagation 
        self.optimizer.step()

    #Updates the data
    def update(self, reward, new_signal):
        #Gets a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #Updates the memory, keeping everything as a Tensor
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #Takes an action
        action = self.select_action(new_state)
        #Lets the model learn from 100 states, next states, rewards, and action
        if len(self.memory.memory) > 100: #self.memory.mermory == self . memory(of the Dnq class) . memory(of the ReplayMemory class)
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) 
        #Updates the last action, last state, last reward, and appends the reward to the reward_window list
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        #Deletes the first reward if it is over 1000 elements
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        #Returns the action
        return action

    #Gives a score based on the reward window
    def score(self):
        #Calculates the mean
        return sum(self.reward_window) / (len(self.reward_window) + 1.) #The '+ 1.' allows for the system to never be 0, therefor keeping it from crashing

    #Saves the current model
    def save(self):
        #Saves the last versions of the weights and optimizer
        #Saved in a dictionary
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                     'last_brain.pth')

    #Loads the last saved model
    def load(self):
        #Makes sure the 'last_brain.pth' exists
        if os.path.isfile(r'last_brain.pth'):
            print("=> loading chechpoint... ")
            checkpoint = torch.load(r'last_brain_pth')
            #Updates the model parameters and the optimizer based on the 'last_brain.pth'
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("no checkpoint found...")