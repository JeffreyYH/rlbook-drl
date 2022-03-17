import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from ch04_DP.DP import Tabular_DP


class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 1.0
        self.alpha = 0.05
        self.tabularUtils = TabularUtils(self.env)
    

    def TD0_prediction(self, policy):
        """ policy evalution with TD(0)"""
        V_est = np.zeros(self.env.nS)
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            while not done:
                a = np.argmax(policy[s])
                s_next, r, done, _ = self.env.step(a)
                V_est[s] = V_est[s] + self.alpha * (r + self.gamma * V_est[s_next] - V_est[s])
                s = s_next
        return V_est


    def sarsa(self):
        """sarsa: on-policy TD control"""
        Q = np.zeros((self.env.nS, self.env.nA))
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            a = self.tabularUtils.epsilon_greedy_policy(Q[s, :]) 
            while not done:
                s_next, r, done, _ = self.env.step(a)
                a_next = self.tabularUtils.epsilon_greedy_policy(Q[s_next, :]) 
                Q[s][a] = Q[s][a] + self.alpha * (r + self.gamma * Q[s_next][a_next] - Q[s][a])
                s = s_next
                a = a_next
        
        greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)

        return Q, greedy_policy
        
    
    def Q_learning(self):
        """ 
        Q learning: off-policy TD control
        we use the same Q to represent two different policy
        1. the behavior policy: epsilon-greedy
        2. the target policy: greedy, which is used to update the Q function
        """
        Q = np.zeros((self.env.nS, self.env.nA))
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            while not done:
                a = self.tabularUtils.epsilon_greedy_policy(Q[s, :]) 
                s_next, r, done, _ = self.env.step(a)
                Q[s][a] = Q[s][a] + self.alpha * (r + self.gamma * np.max(Q[s_next, :]) - Q[s][a])
                s = s_next
        
        greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)

        return Q, greedy_policy
    

    def double_Q_learning(self):
        Q1 = np.zeros((self.env.nS, self.env.nA))
        Q2 = np.zeros((self.env.nS, self.env.nA))
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            while not done:
                a = self.tabularUtils.epsilon_greedy_policy(Q1[s, :]+Q2[s, :]) 
                s_next, r, done, _ = self.env.step(a)
                Q1[s][a] = Q1[s][a] + self.alpha * (r + self.gamma * Q2[s_next, np.argmax(Q1[s_next, :])] - Q1[s][a])
                Q2[s][a] = Q2[s][a] + self.alpha * (r + self.gamma * Q1[s_next, np.argmax(Q2[s_next, :])] - Q2[s][a])
                s = s_next
        
        Q_final = (Q1+Q2)/2
        greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q_final)

        return Q_final, greedy_policy



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="FrozenLake-v1", 
                        choices=["gridworld", "FrozenLake-v1"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)

    dp = Tabular_DP(args)
    V_optimal_VI, policy_optimal = dp.value_iteration()
    print(V_optimal_VI)
    print(policy_optimal)

    td = Tabular_TD(args)
    V_est_TD0 = td.TD0_prediction(policy_optimal)
    print(V_est_TD0)

    Q_sarsa, policy_sarsa = td.sarsa()
    print(policy_sarsa)

    Q_qlearing, policy_qlearing = td.Q_learning()
    print(policy_qlearing)

    Q_dbQlearing, policy_dbQlearing = td.double_Q_learning()
    print(policy_dbQlearing)