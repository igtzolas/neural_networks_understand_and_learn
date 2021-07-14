
import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# The final action-value function.
# A nested dictionary that maps state -> (action -> action-value).
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# The policy we're following
policy = make_epsilon_greedy_policy(Q, env.action_space.n)

# Keeps track of sum and count of returns for each state
# to calculate an average. We could use an array to save all
# returns (like in the book) but that's memory inefficient.
returns_sum = defaultdict(float)
returns_count = defaultdict(float)

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):    
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("Episode {}/{}.".format(i_episode, num_episodes), end="\r")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
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
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # The policy is improved implicitly by changing the Q dictionary
    
    return 

mc_control_epsilon_greedy(env, num_episodes=500000, discount_factor = 0.95, epsilon=0.1)

for behavior in ["trained_policy", "random"]:
    games_won = 0
    games_lost = 0
    games_even = 0
    for game_id in range(10000):
        state = initial_state = env.reset()
        is_game_finished = False
        while True:
            score, dealer_score, usable_ace = state
            action = np.argmax(policy(state, 0))
            if behavior == "trained_policy":
                action = np.argmax(policy(state, 0))
            else:
                action = int(np.random.choice(env.action_space.n, 1))
            next_state, reward, is_game_finished, _ = env.step(action)
            if is_game_finished:
                if reward == -1:
                    games_lost += 1
                elif reward == 1:
                    games_won += 1
                else:
                    games_even += 1
                break
            state = next_state
    print("Games Won : {}, Games Lost: {}, Games Even : {}".format(games_won, games_lost, games_even))
