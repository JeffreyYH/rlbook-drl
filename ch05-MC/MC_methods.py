from ast import Return
import sys
import gym
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from blackjack_example import plot_value_function, handcrafted_episode


def generate_episode(policy, max_steps):
    """each element in the episode is a [state, action, reward] tuple
    each state consists of : score, dealer_score, usable_ace """
    init_state = env.reset()
    curr_state = init_state
    episode = []
    for step in range(max_steps):
        action = policy[curr_state]
        next_state, reward, done, _ = env.step(action)
        episode.append((curr_state, action, reward))
        if done:
            break
        else:
            curr_state = next_state
    return episode


def first_visit_MC_prediction(num_episodes, max_steps, discount_factor=1.0):
    # returns for state s (could be multiple returns in one state), value function
    returns = defaultdict(list)
    V = defaultdict(float)

    # generate episodes with the policy
    for i_episode in range(num_episodes):
        # generate one spisode
        # episode = generate_episode(policy, num_steps)
        episode = handcrafted_episode()

        # for each state in this episode
        states_in_episode = [ep[0] for ep in episode]
        rewards_in_episode = [ep[2] for ep in episode]
        states_visited = []
        for i_state in range(len(states_in_episode)):
            s = states_in_episode[i_state]
            if s in states_visited:
                continue
            else:
                states_visited.append(s)
            # compute return following the first occurrence of s
            num_reward = 0
            G = 0
            for i_reward in range(i_state, len(rewards_in_episode)):
                G += (discount_factor**num_reward) * rewards_in_episode[i_reward]
                num_reward += 1
            returns[s].append(G)
            # average all the returns in state s
            V[s] = np.mean(returns[s])
    return V


def MC_ES(env, num_episodes=10000, max_steps=10000):
    """ Monte Carlo ES (Exploring Starts) """
    gamma = 1.0
    Returns = {}
    for s in range(env.nS):
        Returns[s] = {}
        for a in range(env.nA):
            Returns[s][a] = []
    Q = np.zeros((env.nS, env.nA))
    # here we use deterministic policy 
    policy_MCES = np.zeros(env.nS, dtype=int)
    for i in range(env.nS):
        policy_MCES[i] = np.random.choice(env.nA)

    for i_episode in tqdm(range(num_episodes)):
        episode = generate_episode(policy_MCES, max_steps)
        G = 0
        T = len(episode)
        for t in reversed(range(T)):
            s_t = episode[t][0]
            a_t = episode[t][1]
            r_t = episode[t][2]
            G = gamma * G + r_t
            for i in range(t):
                if s_t == episode[i][0] and a_t == episode[i][1]:
                    Returns[s_t][a_t].append(G)
                    Q[s_t][a_t] = np.mean(np.array(Returns[s_t][a_t]))
                    policy_MCES[s_t] = np.argmax(Q[s_t, :])
    
    print(Q)
    return policy_MCES


def OnPolicy_first_visit_MC_control(num_episodes=100, max_steps=100, epislon=0.01):
    """ On-policy first-visit MC control """
    policy_FVMCC = []
    for i_episode in range(num_episodes):
        episode = generate_episode(policy_FVMCC, max_steps)
        G = 0
        T = len(episode)
        for t in range(T):
            pass


if __name__ == "__main__":
    # env_name = "Blackjack-v1"
    env_name = "FrozenLake-v1"
    env = gym.make(env_name)
    print(env.observation_space)

    MC_ES(env, num_episodes=10000, max_steps=10000)

    if env_name == "Blackjack-v1":
        V_10k = first_visit_MC_prediction(num_episodes=10000, max_steps=100)
        plot_value_function(V_10k, title="10,000 episodes")

        V_500k = first_visit_MC_prediction(num_episodes=500000, max_steps=100)
        plot_value_function(V_500k, title="500,000 episodes")



