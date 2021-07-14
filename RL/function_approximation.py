
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np
import itertools

import sklearn
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.kernel_approximation

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

import gym
env = gym.make("MountainCar-v0")
import pickle

learning_rate = 0.005
epsilon=0.2
epsilon_decay=0.99
gamma = 0.99

# State is the position and the velocity
n_initial_observations = 10000
initial_observations = [env.observation_space.sample() for _ in range(n_initial_observations)]
np_initial_observations = np.array(initial_observations)

# Preprocessing steps 

# Scaling 
scaler = sklearn.preprocessing.StandardScaler()
# Train the scaler
scaler.fit(initial_observations)
print("The scaler has found the following mean values : {}".format(scaler.mean_))
print("The scaler has found the folloiwng variance values : {}".format(scaler.var_))
print("The scaler has found the folloiwng standard deviation values : {}".format(scaler.scale_))
# Analytical calculation of the mean and standard deviation
print("The scaler has found the following mean values : {}".format(np_initial_observations.mean(axis = 0)))
print("The scaler has found the folloiwng variance values : {}".format(np.power(np_initial_observations - np_initial_observations.mean(axis = 0), 2).mean(axis = 0)))
print("The scaler has found the folloiwng standard deviation values : {}".format(np.sqrt(np.power(np_initial_observations - np_initial_observations.mean(axis = 0), 2).mean(axis = 0))))

# Have the 'trained' scaler transform the initial_observations
scaled_initial_observations = scaler.transform(initial_observations)
# Understanding how the scaler works
print("Scaled initial observation is : {}".format(scaled_initial_observations[0, :]))
print("Scaled initial observation is calculated analytically is : {}".format(((np_initial_observations - scaler.mean_) / scaler.scale_)[0, :]))

# Kernelizing
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", sklearn.kernel_approximation.RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", sklearn.kernel_approximation.RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", sklearn.kernel_approximation.RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", sklearn.kernel_approximation.RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(initial_observations))
n_features = featurizer.transform(scaled_initial_observations[0,:].reshape(1, -1)).shape[1]


class Estimator(nn.Module):
    def __init__(self, input_features, n_outputs = 1):
        super(Estimator, self).__init__()
        self.fc1 = nn.Linear(input_features, n_outputs)
        torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, input):
        q_values = self.fc1(input)
        return q_values

estimators = [Estimator(n_features, 1) for _ in range(env.action_space.n)]
optimizers = [optim.Adam(estimator.parameters(), lr = learning_rate) for estimator in estimators]

def make_epsilon_greedy_policy(estimators, epsilon, n_actions):
    def choose_next_action(current_state):
        # Closure of estimator network and epsilon values
        A = np.ones(n_actions, dtype=float) * epsilon / n_actions
        q_values = [estimator.forward(current_state).item() for estimator in estimators]
        best_action = np.argmax(q_values)
        # print(q_values)
        # # print(best_action)
        # print(best_action)
        A[best_action] += (1.0 - epsilon)
        return (q_values, A)
    return choose_next_action

# for episode_id in range(100):
#     current_state = env.reset()
#     current_state = torch.FloatTensor(featurizer.transform(scaler.transform(current_state.reshape(1, -1))))
#     q_values_current = q_values_estimator.forward(current_state)
#     q_values_td_target = torch.zeros_like(q_values_current)
#     # Define the loss
#     loss = F.mse_loss(q_values_current, q_values_td_target)
#     # Update the weights of the estimator network
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

episode_reward = 0
n_episodes = 400
for episode_id in range(n_episodes):
    # Make the policy which will be used to take decisions
    policy = make_epsilon_greedy_policy(estimators, epsilon * (epsilon_decay**episode_id), env.action_space.n)

    initial_state = current_state = env.reset()
    current_state = torch.FloatTensor(featurizer.transform(scaler.transform(current_state.reshape(1, -1))))

    for _ in itertools.count():
        
        q_values_current, action_probabilities = policy(current_state)
        action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(featurizer.transform(scaler.transform(next_state.reshape(1, -1))))

        episode_reward += reward

        if done: 
            td_target = reward
        else:
            q_values_next = [estimator.forward(next_state).item() for estimator in estimators] 
            td_target = reward + gamma * max(q_values_next)

        estimator_to_update = estimators[action]
        optimizer_to_use = optimizers[action]

        # Define the loss
        loss = F.mse_loss(estimator_to_update.forward(current_state), torch.FloatTensor([td_target]))

        # Update the weights of the estimator network
        optimizer_to_use.zero_grad()
        loss.backward()
        optimizer_to_use.step()

        if done:
            break
        current_state = next_state


    if (episode_id + 1) % 10 == 0:
        print("{}/{}, the average episode reward is : {}".format(episode_id +1, n_episodes, episode_reward / 10))
        episode_reward = 0

# Saving the model parameters
# PATHS = ["./weights_function_approximation_{}.pytorch".format(i+1) for i,_ in enumerate(estimators)]
# [torch.save(estimators[i].state_dict(), PATH) for i, PATH in enumerate(PATHS)]
# pickle.dump(scaler, open("function_approximation_scaler.sav", "wb"))
# pickle.dump(featurizer, open("function_approximation_featurizer.sav", "wb"))