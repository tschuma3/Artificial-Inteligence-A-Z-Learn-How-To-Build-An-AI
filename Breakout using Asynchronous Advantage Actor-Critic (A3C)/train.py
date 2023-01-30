"""
Trains the AI
"""

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

#Implementing a function to make sure the models share the same gradient
#Function may not be necessary
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

#Training function
#Desynchronize using rank to shift each seed so each agent is desynchronize 
def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) #Shifting the seed by the rank to desynchronize
    env = create_atari_env(params.evn_name, video=True)
    env.seed(params.seed + rank) #Allign the agent to a seed to have it's own environment, uses the desynchronized seed from two lines above
    model = ActorCritic(env.observation_space.shape[0], env.action_space) #This is the brain
    state = env.reset() #State is a numpy array of size 1*42*42, in black and white
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict()) #Gets the shared model and the parameters of the model
        #If done, then initialize cell nodes and hidden nodes to 0 
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = [] #Will be the output of the critic
        log_probs = []
        rewards = []
        entropies = []
        #Iterates over the number of steps of the exploration
        for step in range(params.num_steps):
            #value = output of the critic (critic.linear(x)), action_values = output of the actor (actor.linear(x)), tuple of (hx, cx) = tuple of (hx, cx) 
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = F.softmax(action_values) #Gets the probibilities
            log_prob = F.log_softmax(action_values) #Gets the log of the probibilities
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data #Takes and selects a random draw from the probibilities to get an action
            log_prob = log_prob.gather(1, Variable(action)) #Updates the log probibilities
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.numpy()) #Plays the selected action
            done = (done or episode_length >= params.max_episode_length) #Makes sure the agent isn't stuck in a state
            reward = max(min(reward, 1), -1) #Clamps the reward between -1 and 1
            #If the game is done then restart the environment
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            #If done, stop the exploration
            if done:
                break
        R = torch.zeros(1, 1) #Cumulative reward
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx))) #Updates the shared neural network
            R = value.data
        values.append(Variable(R)) #Updates the values with Variable(R)
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1) #Generalized advantage estimation, A(a,s) = Q(a,s) - V(s)
        #Iterates through the rewards
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i] #End of the for loop --> R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
            advantage = R - values[i]
            #Calculate value loss
            value_loss = value_loss + 0.5 * advantage.pow(2) 
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD #End of the fro loop --> gae = sum_i (gamma*tau)^i * TD(i)
            #Calculate policy loss
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] #End of the for loop --> policy_loss = - sum_i log(pi_i) * gae + 0.01 * H_i
        optimizer.zero_grad() 
        #Backward propagation
        (policy_loss + 0.5 * value_loss).backward() #Gives twice as much importance to the policy loss then the value loss
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) #Clamps the values of the gradient to be between 0 and 40
        ensure_shared_grads(model, shared_model) #The model and shared model shares the gradient, may not be needed
        optimizer.step()