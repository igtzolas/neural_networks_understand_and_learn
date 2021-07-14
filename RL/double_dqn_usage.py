
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import itertools
import matplotlib.pyplot as plt
from IPython import display
import time


# Creation of the environment
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_state_variables = len(env.observation_space.sample())

################################################################################################################################
# Define the architecture of the Neural Network
class QNet(nn.Module):   
    
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_state_variables, 256)
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        return x

################################################################################################################################
# Instantiate the network and load the pretrained weights
nn = QNet()
PATH = "./double_dqn_weights.pytorch"
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
    action = torch.argmax(nn.forward(torch.FloatTensor(state))).item()
    # action = env.action_space.sample() # Illustrating how it is to play at randome
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