# Importing the usual suspects
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import itertools
import numpy as np 

# Creating the environment
env = gym.make("CartPole-v1")
n_state_features = env.observation_space.sample().shape[0]
n_actions = env.action_space.n
n_episodes = 20000
n_episodes_information_interval = 10
n_steps_between_training = 50

# Hyperparameters
learning_rate_actor = 0.0005
learning_rate_critic = 0.001
gamma = 0.99

################################################################################################################################
# Handling cuda availability
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    gpus = [0]
    torch.cuda.set_device(gpus[0])
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

################################################################################################################################
# Defining the Neural Networks = Function Approximators

class Actor(nn.Module):
    def __init__(self, n_state_features, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state_features, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 32)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(32, n_actions)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, input_state, softmax_dim):
        x = self.fc1(input_state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        probabilities = F.softmax(x, dim = softmax_dim) # Why is this set equal to 0 ?
        return probabilities

actor = Actor(n_state_features, n_actions)
optimizer_actor = optim.Adam(actor.parameters(), lr = learning_rate_actor)

################################################################################################################################
# Critic neural network 
# The critic approximates the state value function 
class Critic(nn.Module):
    def __init__(self, n_state_features, n_output = 1): # Why do we set this equal to 1 ?
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_state_features, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(64, n_output)
        nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, input_state): # What is the value to be in a state ? 
        x = self.fc1(input_state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        state_value = self.fc3(x)
        return state_value

critic = Critic(n_state_features)
optimizer_critic = optim.Adam(critic.parameters(), lr = learning_rate_critic)

################################################################################################################################
# Define how training will take place

def train_actor_and_critic(training_data):

    current_states = FloatTensor([d[0] for d in training_data])
    actions = LongTensor([d[1] for d in training_data])
    rewards = FloatTensor([d[2] for d in training_data])
    next_states = FloatTensor([d[3] for d in training_data])
    is_done = FloatTensor([0.0 if d[4] else 1.0 for d in training_data])

    state_values_for_current_states = critic.forward(current_states)
    # When we want to strip the gradient logic from tensors we call detach() on them
    state_values_for_next_states = critic.forward(next_states).detach()
    td_target = rewards.reshape(-1, 1) + gamma * state_values_for_next_states * is_done.reshape(-1, 1)

    loss_critic = F.mse_loss(state_values_for_current_states, td_target)  
    delta = (td_target - state_values_for_current_states).detach()
    loss_actor = -torch.log(actor.forward(current_states, softmax_dim = 1).gather(1, actions.reshape(-1, 1))) * delta

    # Optimize the critic
    optimizer_critic.zero_grad()
    loss_critic.backward()
    optimizer_critic.step()
    # Optimize the actor
    optimizer_actor.zero_grad()
    loss_actor.mean().backward()
    optimizer_actor.step()    


# Run sample episodes gathering data and training the loop
display_epoch_reward = 0
for episode_id in range(n_episodes):    
    # Sample initial state
    initial_state = current_state = env.reset()
    is_done = False

    training_data = []
    while not is_done: 
        for _ in range(n_steps_between_training):
            # Which action should be taken ? We ask the Actor which provides the policy 
            action_probs = actor.forward(FloatTensor(current_state), softmax_dim = 0)
            sampler = Categorical(probs = action_probs)
            action = sampler.sample().item()     

            # Take the action chosen 
            next_state, reward, is_done, _ = env.step(action)
            training_data.append((current_state, action, reward, next_state, is_done))

            display_epoch_reward += reward
            if is_done:
                break
            else:
                current_state = next_state
        
        # Perform the training with these n_steps_between_training number of samples
        train_actor_and_critic(training_data)
        training_data = []
    # Print statistics of what we have gained in this generation after it having been trained
    if (((episode_id + 1) % n_episodes_information_interval) == 0) and episode_id != 0:
        print("Episode: {}/{}, Average Episode Score in the last {} episodes: {} ".format(episode_id +1, n_episodes, n_episodes_information_interval, display_epoch_reward / n_episodes_information_interval))
        display_epoch_reward = 0

# # Saving the model parameters
# PATH = "./actor_critic_actor_weights.pytorch"
# torch.save(actor.state_dict(), PATH)
# PATH = "./actor_critic_critic_weights.pytorch"
# torch.save(critic.state_dict(), PATH)






