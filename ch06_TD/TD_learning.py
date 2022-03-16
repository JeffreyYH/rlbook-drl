import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from ch04_DP.DP import Tabular_DP

class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 1.0
        self.alpha = 0.05
    
    def TD0_prediction(self, policy):
        """ policy evalution with TD(0)"""
        V_est = np.zeros(self.env.nS)
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            while not done:
                a = policy[s]
                s_next, r, done, _ = self.env.step(a)
                V_est[s] = V_est[s] + self.alpha * (r + self.gamma * V_est[s_next] - V_est[s])
                s = s_next
        return V_est


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
