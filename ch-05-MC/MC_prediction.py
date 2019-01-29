import sys
import gym
import numpy as np
from collections import defaultdict
if "../" not in sys.path: sys.path.append("../")
from lib import plotting

# handcrafted episode for debugging purpose
def handcrafted_episode():
    return [((7, 5, False), 1, 0), ((9, 5, False), 1, 0), ((13, 5, False), 1, 0), ((7, 5, False), 1, 0), ((18, 5, False), 1, 0), ((21, 5, False), 0, 1.0)]

# sample policy: Stick (action 0) if the score is 20 or 21, hit (action 1) otherwise
def policy(observation):
    score, dealer_score, usable_ace = observation
    if score == 20 or score == 21:
        return 0
    else:
        return 1

def first_visit_MC_prediction(num_episodes, num_steps, discount_factor=1.0):
    # returns for state s (could be multiple returns in one state), value function
    returns = defaultdict(list)
    V = defaultdict(float)

    # generate episodes with the policy
    for i_episode in range(num_episodes):
        # each element in the episode is a [state, action, reward] tuple
        # each state consists of : score, dealer_score, usable_ace
        init_state = env.reset()
        curr_state = init_state
        episode = []
        for step in range(num_steps):
            action = policy(curr_state)
            next_state, reward, done, _ = env.step(action)
            episode.append((curr_state, action, reward))
            if done:
                break
            else:
                curr_state = next_state

        #===============================#
        # episode = handcrafted_episode()
        #===============================#

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

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    # print(env.observation_space)
    V_10k = first_visit_MC_prediction(num_episodes=10000, num_steps=100)
    plotting.plot_value_function(V_10k, title="10,000 episodes")

    V_500k = first_visit_MC_prediction(num_episodes=500000, num_steps=100)
    plotting.plot_value_function(V_500k, title="500,000 episodes")



