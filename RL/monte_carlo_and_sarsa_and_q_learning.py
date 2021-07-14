'''
Illustration of tabular q learning algorithm 
This is an off policy algorithm which means that while the agent follows one policy to act and choose which action to take next,
the q values are updated based on the policy which chooses the best action in the next state as perceived up to the current point
'''

import numpy as np 

import gym
import itertools
from collections import defaultdict

import visdom
vis = visdom.Visdom()

from lib.envs.windy_gridworld import WindyGridworldEnv

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

env = WindyGridworldEnv()

Q_values_sarsa = defaultdict(lambda: np.zeros(env.action_space.n))
for state in range(env.observation_space.n):
    Q_values_sarsa[state]
Q_values_heatmap_sarsa = np.zeros((3*7, 3*10), dtype = np.int32)

Q_values_monte_carlo = defaultdict(lambda: np.zeros(env.action_space.n))
for state in range(env.observation_space.n):
    Q_values_monte_carlo[state]
Q_values_heatmap_monte_carlo = np.zeros((3*7, 3*10), dtype = np.int32)

Q_values_q_learning = defaultdict(lambda: np.zeros(env.action_space.n))
for state in range(env.observation_space.n):
    Q_values_q_learning[state]
Q_values_heatmap_q_learning = np.zeros((3*7, 3*10), dtype = np.int32)

visdom_options = {
    "colormap" : "Jet", # "Greys",
    "width" : 500, 
    "height" : 500,
    "xtickmin" : 3,
    "xtickvals" : [3*el + 1 for el in range(10)],
    "ytickvals" : [3*el + 1 for el in range(7)],
    # "columnnames" : range(10),
    'layoutopts': { "plotly" : {
        "annotations" : [el for el in range(70)],
        "axis": {"showgrid" : True, "ticklen" : 50},
                                'xaxis': dict(ticklen= 5),
                                'yaxis': dict(showgrid=True) }},
}

line_options = {
    "width" : 500, 
    "height" : 500,
}

monte_carlo_window_name = "Monte Carlo"
vis.heatmap(X=np.flipud(Q_values_heatmap_monte_carlo), win = monte_carlo_window_name, env = "main",  opts = visdom_options)
sarsa_window_name = "Sarsa"
vis.heatmap(X=np.flipud(Q_values_heatmap_sarsa), win = sarsa_window_name, env = "main",  opts = visdom_options)
q_learning_window_name = "Q Learning"
vis.heatmap(X=np.flipud(Q_values_heatmap_q_learning), win = q_learning_window_name, env = "main",  opts = visdom_options)

epsilon=0.1
policy_monte_carlo = make_epsilon_greedy_policy(Q_values_monte_carlo, epsilon, env.action_space.n)
policy_sarsa = make_epsilon_greedy_policy(Q_values_sarsa, epsilon, env.action_space.n)
policy_q_learning = make_epsilon_greedy_policy(Q_values_q_learning, epsilon, env.action_space.n)

def sarsa(env, episodes=1, discount_factor=1.0, alpha=1):
    global Q_values_sarsa
    # The policy we're following
    policy = policy_sarsa
    
    for episode_id in range(episodes):
        episode_reward = 0
        episode_length = 0

        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # import ipdb; ipdb.set_trace()
        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)

            episode_length += 1
            episode_reward += reward
            
            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            
            # TD Update
            td_target = reward + discount_factor * Q_values_sarsa[next_state][next_action]
            td_delta = td_target - Q_values_sarsa[state][action]
            Q_values_sarsa[state][action] += alpha * td_delta
            # visualize(Q_values_heatmap_sarsa, Q_values_sarsa, sarsa_window_name, episode_id)
            # import ipdb; ipdb.set_trace()
            # print("Action : {}, State : {}".format(action, state))


            if done:
                break
                
            action = next_action
            state = next_state 

    return (episode_length, episode_reward)


def q_learning(env, discount_factor=1.0, alpha=0.5):
    global Q_values_q_learning
    # The policy we're following
    policy = policy_q_learning

    episode_reward = 0
    episode_length = 0
    
    # Reset the environment and pick the first action
    state = env.reset()
    
    # One step in the environment
    # total_reward = 0.0
    for t in itertools.count():
        
        # Take a step
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(action)

        episode_length += 1
        episode_reward += reward        
        
        # TD Update
        best_next_action = np.argmax(Q_values_q_learning[next_state])    
        td_target = reward + discount_factor * Q_values_q_learning[next_state][best_next_action]
        td_delta = td_target - Q_values_q_learning[state][action]
        Q_values_q_learning[state][action] += alpha * td_delta
            
        if done:
            break
            
        state = next_state
    
    return (episode_length, episode_reward)


def mc_control_epsilon_greedy(env, discount_factor=1.0):
    
    global Q_values_monte_carlo
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The policy we're following
    policy = policy_monte_carlo
    
    episode_reward = 0
    episode_length = 0

    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples
    episode = []
    state = env.reset()
    for t in range(1500): #itertools.count():
        probs = policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        episode.append((state, action, reward))
        
        if done:
            break
        state = next_state

    # Find all (state, action) pairs we've visited in this episode
    # We convert each state to a tuple so that we can use it as a dict key
    sa_in_episode = set([(x[0], x[1]) for x in episode])
    for state, action in sa_in_episode:
        sa_pair = (state, action)
        # Find the first occurance of the (state, action) pair in the episode
        first_occurence_idx = next(i for i,x in enumerate(episode)
                                    if x[0] == state and x[1] == action)
        # Sum up all rewards since the first occurance
        G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
        # Calculate average return for this state over all sampled episodes
        returns_sum[sa_pair] += G
        returns_count[sa_pair] += 1.0
        Q_values_monte_carlo[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

    
    return (episode_length, episode_reward)


def visualize(Q_values_heatmap, Q_values, window, episode_id):
    for state_vis in range(env.observation_space.n):
        matrix_column_id = int((state_vis* 3) % 30)
        matrix_row_id = 3 * int(state_vis*3 / 30)
        for action_vis in range(env.action_space.n):
            if action_vis == 0: # going up   
                Q_values_heatmap[matrix_row_id, matrix_column_id + 1] = np.round(Q_values[state_vis][action_vis]) 
            elif action_vis == 1: # going right 
                Q_values_heatmap[matrix_row_id + 1, matrix_column_id + 2] = np.round(Q_values[state_vis][action_vis])
            elif action_vis == 2: # going down
                Q_values_heatmap[matrix_row_id + 2, matrix_column_id + 1] = np.round(Q_values[state_vis][action_vis])
            else: # action == 3, going left
                Q_values_heatmap[matrix_row_id + 1, matrix_column_id] = np.round(Q_values[state_vis][action_vis])
    visdom_options["title"] = "{} Algorithm, Episodes : {}".format(window, episode_id + 1)
    vis.heatmap(X = np.flipud(Q_values_heatmap), win = window, env = "main",  opts = visdom_options)


def execute_1_episode(episode_id):
    global line_options
    (episode_length_monte_carlo, episode_reward_monte_carlo) = mc_control_epsilon_greedy(env)
    (episode_length_sarsa, episode_reward_sarsa) = sarsa(env)
    (episode_length_q_learning, episode_reward_q_learning) = q_learning(env)
    visualize(Q_values_heatmap_monte_carlo, Q_values_monte_carlo, monte_carlo_window_name, episode_id)
    visualize(Q_values_heatmap_sarsa, Q_values_sarsa, sarsa_window_name, episode_id)
    visualize(Q_values_heatmap_q_learning, Q_values_q_learning, q_learning_window_name, episode_id)
    line_options["title"] = "Monte Carlo Episode Length"
    vis.line(X=np.array(episode_id).reshape(-1, 1), Y=np.array(episode_length_monte_carlo).reshape(-1, 1) , env = "main", win = "Monte Carlo Episode Length", opts= line_options, update="append")
    line_options["title"] = "Sarsa Episode Length"
    vis.line(X=np.array(episode_id).reshape(-1, 1), Y=np.array(episode_length_sarsa).reshape(-1, 1) , env = "main", win = "Sarsa Episode Length", opts= line_options, update="append")
    line_options["title"] = "Q Learning Episode Length"
    vis.line(X=np.array(episode_id).reshape(-1, 1), Y=np.array(episode_length_q_learning).reshape(-1, 1) , env = "main", win = "Q Learning Episode Length", opts= line_options, update="append")

execution_hook = (execute_1_episode(episode_id)  for episode_id in itertools.count())

# temp = [next(execution_hook) for _ in range(500)]