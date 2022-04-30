from ast import Return
import sys, time
import gym
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse
from collections import namedtuple
from blackjack_example import plot_value_function, handcrafted_episode
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from ch04_DP.DP import Tabular_DP
# from lib.envs.gridworld import GridworldEnv

# register a new deterministic environment
from gym.envs.registration import register
register(
    id='FrozenLake-Deterministic-v1',
    # entry_point='gym.envs.toy_text:FrozenLakeEnv',
    entry_point='lib.envs.myFrozenLake:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

class Tabular_MC:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes = 10000
        self.max_steps = 1000
        self.epislon = 0.1
        self.gamma = 0.99
        self.env_nA = self.env.action_space.n
        self.env_nS = self.env.observation_space.n
        self.tabularUtils = TabularUtils(self.env)


    def generate_episode(self, policy, use_ES):
        """ each element in the episode is a [state, action, reward] tuple
        inputs: 
            policy: pi(a|s)
            use_ES: use exploring starts or not
        notes:
            for blackjack env: each state consists of : score, dealer_score, usable_ace 
        """
        init_state = self.env.reset()
        curr_state = init_state
        episode = []
        for step in range(self.max_steps):
            if step == 0 and use_ES:
                action = np.random.choice(self.env_nA)
            else:
                action = self.tabularUtils.epsilon_greedy_policy(policy[curr_state, :])
            next_state, reward, done, _ = self.env.step(action)
            episode.append((curr_state, action, reward))
            if done:
                break
            else:
                curr_state = next_state
        return episode
    

    def first_visit_MC_prediction(self, policy):
        # returns for state s (could be multiple returns in one state), value function
        returns = defaultdict(list)
        V = defaultdict(float)

        # generate episodes with the policy
        for i_episode in range(self.num_episodes):
            # generate one spisode
            episode = self.generate_episode(policy, False)
            # episode = handcrafted_episode()

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
                    G += (self.gamma**num_reward) * rewards_in_episode[i_reward]
                    num_reward += 1
                returns[s].append(G)
                # average all the returns in state s
                V[s] = np.mean(returns[s])

        # V = sorted(V.items())
        return V


    def init_policy(self):
        """ initialize policy with random action for each state"""
        policy = np.zeros([self.env_nS, self.env_nA])
        # for s in range(self.env_nS):
        #     a = np.random.choice(self.env_nA)
        #     policy[s, :] = self.action_to_onehot(a) 
        return policy


    def MC_ES(self):
        """ Monte Carlo ES (Exploring Starts) """
        # TODO: does not work so well
        Returns = {}
        for s in range(self.env_nS):
            Returns[s] = {}
            for a in range(self.env_nA):
                Returns[s][a] = []
        Q = np.zeros((self.env_nS, self.env_nA))
        # here we use deterministic policy 
        policy = self.init_policy()

        for i_episode in tqdm(range(self.num_episodes)):
            # generate s_0, a_0 to fulfil exploring start assumption
            episode = self.generate_episode(policy, True)
            G = 0
            T = len(episode)
            visited_SA = []
            for t in reversed(range(T)):
                s_t = episode[t][0]
                a_t = episode[t][1]
                r_t = episode[t][2]
                G = self.gamma * G + r_t
                if [s_t, a_t] in visited_SA:
                    continue
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.mean(np.array(Returns[s_t][a_t]))
                a_optimal = np.argmax(Q[s_t, :])
                policy[s_t, :] = self.tabularUtils.action_to_onehot(a_optimal)
                visited_SA.append([s_t, a_t])
        # print(Q)
        return Q, policy


    def OnPolicy_first_visit_MC_control(self):
        """ On-policy first-visit MC control """
        # TODO: does not work so well
        Returns = {}
        for s in range(self.env_nS):
            Returns[s] = {}
            for a in range(self.env_nA):
                Returns[s][a] = []
        Q = np.zeros((self.env_nS, self.env_nA))
        policy = self.init_policy()

        for i_episode in tqdm(range(self.num_episodes)):
            episode = self.generate_episode(policy, False)
            G = 0
            T = len(episode)
            visited_SA = []
            for t in reversed(range(T)):
                s_t = episode[t][0]
                a_t = episode[t][1]
                r_t = episode[t][2]
                G = self.gamma * G + r_t
                if [s_t, a_t] in visited_SA:
                    continue
                Returns[s_t][a_t].append(G)
                Q[s_t][a_t] = np.mean(np.array(Returns[s_t][a_t]))
                a_optimal = np.argmax(Q[s_t, :])
                for a in range(self.env_nA):
                    if a == a_optimal:
                        policy[s_t, a] = 1 - self.epislon + self.epislon/self.env_nA
                    else:
                        policy[s_t, a] = self.epislon / self.env_nA
                visited_SA.append([s_t, a_t])

        # print(Q)
        # print(policy)
        return Q, policy


    def offPolicy_MC_control(self):
        """ off-policy MC control via importance sampling"""
        Q = np.zeros((self.env_nS, self.env_nA))
        C = np.zeros((self.env_nS, self.env_nA))
        pi = self.init_policy()
        for i_episode in tqdm(range(self.num_episodes)):
            b = pi
            episode = self.generate_episode(b, False)
            G = 0.0
            W = 1.0
            T = len(episode)
            for t in reversed(range(T)): 
                s_t = episode[t][0]
                a_t = episode[t][1]
                r_t = episode[t][2]
                G = self.gamma * G + r_t
                C[s_t, a_t] += W
                Q[s_t, a_t] += (W/C[s_t, a_t]) * (G - Q[s_t, a_t])
                pi[s_t, :] = self.tabularUtils.action_to_onehot(np.argmax(Q[s_t, :]))
                if a_t != np.argmax(pi[s_t,:]):
                    continue
                W = W/b[s_t, a_t]
        
        return Q, pi


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="FrozenLake-Deterministic-v1", 
                        # default="gridworld", 
                        choices=["Blackjack-v1", "gridworld", "FrozenLake-Deterministic-v1", "FrozenLake-v1"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args = parse_arguments()
    if args.env_name == "gridworld":
        args.env = GridworldEnv() 
    else:
        args.env = gym.make(args.env_name)

    tabular_utils = TabularUtils(args.env)
    dp = Tabular_DP(args)
    V_optimal_VI, policy_optimal = dp.value_iteration()
    print("ground truth value function and policy by VI:")
    print(V_optimal_VI)
    print(tabular_utils.onehot_policy_to_deterministic_policy(policy_optimal))
    print("\n")

    # testing first-visit MC prediction
    MC_agent = Tabular_MC(args)
    V_MC = MC_agent.first_visit_MC_prediction(policy_optimal)
    V_MC_np = np.zeros(args.env.observation_space.n)
    for key, value in V_MC.items():
        V_MC_np[key] = value
    print(V_MC_np)
    print("mean abs error of first visit MC prediction: %5f \n" %np.mean(np.abs(V_MC_np - V_optimal_VI)))

    # testing MC exploring start
    Q_MCES, policy_MCES = MC_agent.MC_ES()
    V_MCES = tabular_utils.Q_value_to_state_value(Q_MCES, policy_MCES)
    print(V_MCES)
    print(tabular_utils.onehot_policy_to_deterministic_policy(policy_MCES))
    print("mean abs error of value function by MC exploring start: %5f \n" %np.mean(np.abs(V_MCES - V_optimal_VI)))

    # testing first-visit MC control
    Q_FVMCC, policy_FVMCC = MC_agent.OnPolicy_first_visit_MC_control()
    V_FVMCC = tabular_utils.Q_value_to_state_value(Q_FVMCC, policy_FVMCC)
    print(V_FVMCC)
    print(tabular_utils.onehot_policy_to_deterministic_policy(policy_FVMCC))
    print("mean abs error of value function by first-visit MC control: %5f \n" %np.mean(np.abs(V_FVMCC - V_optimal_VI)))
    
    # testing off-policy MC control
    Q_OffMCC, policy_OffMCC = MC_agent.offPolicy_MC_control()
    V_OffMCC = tabular_utils.Q_value_to_state_value(Q_OffMCC, policy_OffMCC)
    print(V_OffMCC)
    print(tabular_utils.onehot_policy_to_deterministic_policy(policy_OffMCC))
    print("mean abs error of value function by off-policy MC control: %5f \n" %np.mean(np.abs(V_OffMCC - V_optimal_VI)))

    learned_policy = policy_FVMCC
    tabular_utils.render(learned_policy)


    if args.env_name == "Blackjack-v1":
        V_10k = MC_agent.first_visit_MC_prediction(num_episodes=10000, max_steps=100)
        plot_value_function(V_10k, title="10,000 episodes")

        V_500k = MC_agent.first_visit_MC_prediction(num_episodes=500000, max_steps=100)
        plot_value_function(V_500k, title="500,000 episodes")



