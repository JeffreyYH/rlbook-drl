import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from ch04_DP.DP import Tabular_DP


class Tabular_Planning_Learning:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 1.0
        self.tabularUtils = TabularUtils(self.env)
    
    def Dyna_Q(self):
        Q = np.zeros((self.env.nS, self.env.nA))
        # model = 
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            a = np.argmax(Q[s, :])
            while not done:
                s_next, r, done, _ = self.env.step(a)
                Q[s][a] = Q[s][a] + self.alpha * (r + self.gamma * np.max(Q[s_next, :]) - Q[s][a])