import sys
import gym
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from ch04_DP.DP import Tabular_DP


class Tabular_Planning_Learning:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=100
        self.gamma = 1.0
        self.alpha = 0.05
        self.tabularUtils = TabularUtils(self.env)
        self.num_iter_model = 1000

    def Dyna_Q(self):
        Q = np.zeros((self.env.nS, self.env.nA))
        model = {}
        states_list = []
        actions_list = []
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            while not done:
                a = self.tabularUtils.epsilon_greedy_policy(Q[s, :]) 
                s_next, r, done, _ = self.env.step(a)
                Q[s][a] = Q[s][a] + self.alpha * (r + self.gamma * np.max(Q[s_next, :]) - Q[s][a])
                model[(s, a)] = (r, s_next)
                states_list.append(s)
                actions_list.append(a)
                for i in range(self.num_iter_model):
                    s_tr = random.sample(states_list, 1)[0]
                    a_tr = random.sample(actions_list, 1)[0]
                    r_tr, s_next_tr = model[(s_tr, a_tr)]
                    Q[s_tr][a_tr] = Q[s_tr][a_tr] + self.alpha * (r_tr + self.gamma * np.max(Q[s_next_tr, :]) - Q[s_tr][a_tr])

        greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)

        print(greedy_policy)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="FrozenLake-v1", 
                        choices=["gridworld", "FrozenLake-v1"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)
    planning_learning = Tabular_Planning_Learning(args)
    planning_learning.Dyna_Q()


