import gym
import itertools
import matplotlib.pyplot as plt
from IPython import display
import time

###################################################################################################################################################
# Creation of the environment 
# Environment on which to train
# https://gym.openai.com/envs/CartPole-v1/
env = gym.make("CartPole-v1")

# Actions : 
env.action_space.n
env.observation_space.sample()
# State : x, x_dot, theta, theta_dot = state
# position, velocity, angular_position, angular_velocity

###################################################################################################################################################
# Play a game in which actions are chosen randomly

env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
total_reward = 0
for timestep in itertools.count():
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    next_state, reward, done, info =env.step(action)     
    total_reward += reward
    time.sleep(0.1)
    if done:
        print("The duration of the episode/game is : {} timestemps.".format(timestep+1))
        print("The total reward for the game is : {}.".format(total_reward))
        break
plt.clf()
plt.close("all")
env.close()

###################################################################################################################################################

