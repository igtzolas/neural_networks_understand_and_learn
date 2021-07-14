'''
Implementation of the DQN Algorithm
Program code based on the idea that there is a father-son relationship between the Q networks
'''
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np 
import visdom


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
# Hyper-parameters of the program

learning_rate = lr = 0.0005
# Number of experiences to pick up in each training step
batch_size = 32
gamma = 0.99
# Think generations - parent -son metaphor
n_episodes = 10000
n_episodes_information_interval = 20
max_episode_steps = max_episode_duration = 501

# Creation of the environment
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_state_variables = len(env.observation_space.sample())

# Visdom options 
line_options = {
    "width" : 700, 
    "height" : 400,
}
vis = visdom.Visdom()

################################################################################################################################
# Define the architecture of the Neural Network
class PolicyNet(nn.Module):   
    
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_state_variables, 128)
        self.fc2 = nn.Linear(128, n_actions)
    
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x).unsqueeze(1)
        x = F.softmax(x, dim = 0) # This is because we want the policy network to produce probabilities
        return x

################################################################################################################################
# Function to calculate the returns by which each policy is to be evaluated 
# The way this is implemented causality is enforced 
# (i.e. each policy decision is multiplied / weighted by the following rewards (=reward to go) and not previous rewards)
def calculate_returns(rewards, discount_factor, perform_normalization = True):    
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)        
    returns = torch.tensor(returns)
    if perform_normalization:
        returns = (returns - returns.mean()) / returns.std()    
    return returns

################################################################################################################################
# Instantiating the Policy Neural Network
policy_network = PolicyNet()
# We will be optimizing the policy network parameters so we need an optimizer for that
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

##################################################################################################
# Start the training
display_epoch_reward = 0
for episode_id in range(n_episodes):
    # Sample an initial state
    start_state = env.reset()
    current_state = start_state

    # Data for current episode
    # Will be used to estimate the gradient by the end of the episode
    rewards = []
    chosen_action_log_probabilities = []

    episode_reward = 0.0
    episode_length = 0

    for _ in range(max_episode_steps):
        # Find out which action to take based on the stochastic policy (= policy network function)
        action_probabilities = policy_network.forward(FloatTensor(current_state))   
        sampler = Categorical(action_probabilities.squeeze(1))
        action_in_current_state = sampler.sample() # This is a tensor
        # Take the action sampled based on policy network output probabilities
        next_state, reward, is_done, info = env.step(action_in_current_state.item())

        rewards.append(reward)
        chosen_action_log_probabilities.append(sampler.log_prob(action_in_current_state))
        
        display_epoch_reward += reward

        episode_reward += reward
        episode_length += 0 
        # In case the episode/game is complete break out of the loop
        if is_done:
            break
        current_state = next_state
    
    # Visualizations 
    line_options["title"] = "Policy Gradient - Episode Score"
    line_options["xlabel"] = "# Episodes"
    line_options["ylabel"] = "Episode Score"
    vis.line(X=np.array(episode_id).reshape(-1, 1), Y=np.array(episode_reward).reshape(-1, 1) , env = "main", win = "Policy Gradient - Episode Score", opts= line_options, update="append")
    line_options["title"] = "Policy Gradient - Episode Score" 
    
    # Time for TRAINING !!! 
    # Now that the episode has finished / run to completion we do the following : 
    discounted_rewards = calculate_returns(rewards, gamma, perform_normalization = True)
    discounted_rewards_tensorized = FloatTensor(discounted_rewards)
    chosen_action_log_probabilities_tensorized = torch.stack(chosen_action_log_probabilities)
    # Elementwise multiplication 
    gradient_average = (discounted_rewards_tensorized * chosen_action_log_probabilities_tensorized).mean() # Maximize this
    gradient_average_negated = -gradient_average # Minimize this instead of maximizing the positive value (the optimizer is a minimizer)

    optimizer.zero_grad()
    gradient_average_negated.backward()
    optimizer.step()

    # Print statistics of what we have gained in this generation after it having been trained
    if (((episode_id + 1) % n_episodes_information_interval) == 0) and episode_id != 0:
        print("Episode: {}/{}, Average Episode Score in the last {} episodes: {} ".format(episode_id +1, n_episodes, n_episodes_information_interval, display_epoch_reward / n_episodes_information_interval))
        display_epoch_reward = 0



# # Saving the model parameters
# PATH = "./policy_gradient_reinforce_weights.pytorch"
# torch.save(policy_network.state_dict(), PATH)
