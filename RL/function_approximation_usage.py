"""
This program makes use of already trained models which have been trained on the function approximation algorithm 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import itertools
import matplotlib.pyplot as plt
from IPython import display
import time

import matplotlib.pyplot as plt
from IPython import display
import time
import itertools

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

import pickle

# Creation of the environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
n_state_variables = len(env.observation_space.sample())

scaler = pickle.load(open("./function_approximation_scaler.sav", "rb"))
featurizer = pickle.load(open("./function_approximation_featurizer.sav", "rb"))
n_features = featurizer.transform(env.reset().reshape(1, -1)).shape[1]
################################################################################################################################
# Define the architecture of the Neural Network
class Estimator(nn.Module):
    def __init__(self, input_features, n_outputs = 1):
        super(Estimator, self).__init__()
        self.fc1 = nn.Linear(input_features, n_outputs)
        torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, input):
        q_values = self.fc1(input)
        return q_values



estimators = [Estimator(400, 1) for _ in range(env.action_space.n)]

################################################################################################################################
# Instantiate the network and load the pretrained weights

PATHS = ["./weights_function_approximation_{}.pytorch".format(i+1) for i,_ in enumerate(estimators)]
[estimators[i].load_state_dict(torch.load(PATH)) for i, PATH in enumerate(PATHS)]


state = env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
for timestep in itertools.count():
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    state = torch.FloatTensor(featurizer.transform(scaler.transform(state.reshape(1, -1))))
    q_values = [estimator.forward(state).item() for estimator in estimators]
    action = np.argmax(q_values)
    # action = np.random.choice(np.delete(np.arange(env.action_space.n), action))    
    # action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)     
    time.sleep(0.01)
    if done:
        print("The duration of the episode/game is : {} timestemps.".format(timestep+1))
        break
    else:
        state = next_state
plt.clf()
plt.close("all")
env.close()      