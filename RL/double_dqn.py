'''
Implementation of the Double DQN Algorithm
Program code based on the idea that there is a father-son relationship between the Q networks
'''
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import namedtuple
from collections import deque
import numpy as np
import visdom # Facebook visualization library

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

# Capacity of experience buffer
buffer_size = 50000
# Only when the experience buffer has this many of experiences we will proceed in training
minumum_number_of_experiences_in_buffer = 2000 
# Number of experiences to pick up in each training step
batch_size = 32

gamma = 0.99 # When you don't know what value of gamma to have, set it to 0.99

# Think generations - parent -son metaphor
n_generations = 500
n_episodes_per_generation = 20 #
max_episode_steps = max_episode_duration = 500

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
# Buffer in which experiences accumulated over generations will be held
# Only most recent experiences are held - implicit forget mechanism
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "is_done_mask"])
class ExperienceBuffer():    
    def __init__(self, buffer_size):
         self.experiences_over_generations = deque(maxlen=buffer_size)

    def add_experience_to_buffer(self, experience):
        self.experiences_over_generations.append(experience)
    
    def sample_from_buffer(self, n_experiences):
        experiences = random.sample(self.experiences_over_generations, n_experiences)
        return experiences
    
    def __len__(self):
        return len(self.experiences_over_generations)

experience_buffer = ExperienceBuffer(buffer_size)

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
# Define the policy that will be followed - Here a greedy policy is defined     
def greedy_policy(q_network, n_actions):
    def greedy_policy_action(state, epsilon): 
        # Pick a random point between 0 and 1
        if random.random() < epsilon: # If the point is less than epsilon act by choosing an action at random (Exploration of state action space)
            action = random.choice(range(n_actions))
        else: # Act by choosing the action with the highest action value
            action = q_network.forward(FloatTensor(state)).argmax().item() # Get the item from a torch single value tensor 
        return action
    return greedy_policy_action

################################################################################################################################
# Define how training takes place
# It is the son that is being trained 
# The parent only provides advice on the value of the actions taken by the son

def train(nn_son, nn_parent, training_loops_per_episode = 10):
    for _ in range(training_loops_per_episode):    
        # Sample experiences from the experience buffer and turn them into Tensors
        training_experiences = experience_buffer.sample_from_buffer(batch_size)
        states = FloatTensor([training_experience.state for training_experience in training_experiences])
        actions = LongTensor([training_experience.action for training_experience in training_experiences])
        rewards = FloatTensor([training_experience.reward for training_experience in training_experiences])
        next_states = FloatTensor([training_experience.next_state for training_experience in training_experiences])
        is_done_mask = FloatTensor([training_experience.is_done_mask for training_experience in training_experiences])

        # Forward pass of the son network to get q_values
        q_values_son = nn_son.forward(states)
        # Based on the actions having been chosen, keep respective q_values
        q_values_son_chosen = q_values_son.gather(1, actions.reshape(-1,1)).reshape(-1, 1)

        # What is the father advice ? 
        actions_with_max_q_values_son = q_values_son.argmax(dim=1, keepdim=True)
        q_values_parent_advice = rewards.reshape(-1, 1) + gamma * nn_parent.forward(next_states).gather(1, actions_with_max_q_values_son).reshape(-1,1) * is_done_mask.reshape(-1, 1)  
        loss = F.smooth_l1_loss(q_values_son_chosen, q_values_parent_advice)

        # Performing the optimization 
        optimizer.zero_grad() # Zero-ing out partial derivatives from previous steps (pytorch idiosyncracy)
        loss.backward() # Perform backward pass calculating partial derivatives
        optimizer.step() # Update the parameters controlled by the optimizer based on partial derivatives calculated from previous step


################################################################################################################################
# Instantiating the Neural Networks
nn_parent = QNet()
nn_son = QNet()
# We will be optimizing the son's parameters
optimizer = optim.Adam(nn_son.parameters(), lr=learning_rate)
# greedy_policy : Closure with the son neural network
policy = greedy_policy(nn_son, n_actions)

##################################################################################################
# Start the training
for generation_id in range(n_generations):
    epsilon = 0.05 # We can decay the epsilon value to allow for more exploration in initial episodes and less as time progresses
    
    # Previous son becomes a parent
    # Parent has son which inherits current knowledge
    # This is the point in which synchronization between father and son takes place
    nn_parent.load_state_dict(nn_son.state_dict())

    # Keep track of the total reward in this generation
    generation_score = 0.0

    # The son will have n_episodes_per_generation experiences in his lifetime
    for episode_id in range(n_episodes_per_generation):

        episode_score = 0.0
        # Sample an initial state
        start_state = env.reset()
        current_state = start_state
        # Play episode / game
        # We will play the game for at most max_episode_steps 
        # (a game might take too much time to end and we would like to be learning something along the way)
        for _ in range(max_episode_steps):
            # Son performs a forward pass to get 
            action_in_current_state = policy(current_state, epsilon)
            next_state, reward, is_done, info = env.step(action_in_current_state)
            is_done_mask = 0.0 if is_done else 1.0
            experience = Experience(current_state, action_in_current_state, reward, next_state, is_done_mask)
            episode_score += reward
            experience_buffer.add_experience_to_buffer(experience)
            # In case the episode/game is complete break out of the loop
            if is_done:
                break
            current_state = next_state        
        
        generation_score += episode_score

        line_options["title"] = "Double DQN - Episode Score"
        line_options["xlabel"] = "# Episodes"
        line_options["ylabel"] = "Episode Score"
        running_episode = generation_id * n_episodes_per_generation + episode_id
        vis.line(X=np.array(running_episode).reshape(-1, 1), Y=np.array(episode_score).reshape(-1, 1) , env = "main", win = "Double DQN - Episode Score", opts= line_options, update="append")

        # Now that the episode is finished proceed to training
        # Training should take place only if the experience buffer contains a predefined number of experiences
        if len(experience_buffer) >= minumum_number_of_experiences_in_buffer:
            train(nn_son, nn_parent)


    # Print statistics of what we have gained in this generation after it having been trained
    print("Generation : {}, Average Generation Score : {}, Experience Buffer Size = {}".format(generation_id + 1, generation_score / n_episodes_per_generation, len(experience_buffer)))
    # Visualization of training results
    average_generation_score = generation_score / n_episodes_per_generation
    line_options["title"] = "Double DQN - Average Generation Score"
    line_options["xlabel"] = "# Generations"
    line_options["ylabel"] = "Average Generation Score"
    vis.line(X=np.array(generation_id).reshape(-1, 1), Y=np.array(average_generation_score).reshape(-1, 1) , env = "main", win = "Double DQN - Average Generation Score", opts= line_options, update="append")

# # Saving the model parameters
# PATH = "./double_dqn_weights.pytorch"
# torch.save(nn_son.state_dict(), PATH)

