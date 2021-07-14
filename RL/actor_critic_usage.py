import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

import gym
import itertools
import matplotlib.pyplot as plt
from IPython import display
import time

import numpy as np

# Creation of the environment
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_state_variables = len(env.observation_space.sample())

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
# Define the architecture of the Neural Network
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



################################################################################################################################
# Instantiate the network and load the pretrained weights
nn = Actor(n_state_variables, n_actions)
PATH = "./actor_critic_actor_weights.pytorch"
nn.load_state_dict(torch.load(PATH))

import matplotlib.pyplot as plt
from IPython import display
import time
import itertools

state = env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
for timestep in itertools.count():
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action_probabilities = nn.forward(FloatTensor(state), softmax_dim = 0)
    action = action_probabilities.argmax(dim = 0).item()
    # action = np.random.choice(np.delete(np.arange(env.action_space.n), action))  # Do not include best action in actions to choose from
    # action = env.action_space.sample() # Randomly choose an action
    next_state, reward, done, info =env.step(action)     
    time.sleep(0.01)
    if done:
        print("The duration of the episode/game is : {} timestemps.".format(timestep+1))
        break
    else:
        state = next_state
plt.clf()
plt.close("all")
env.close()      