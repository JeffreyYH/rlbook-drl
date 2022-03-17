import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from ch04_DP.DP import Tabular_DP
from TD_learning import Tabular_TD


class Tabular_nStepTD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 1.0
        self.alpha = 0.05
        self.n = 5
        self.tabularUtils = TabularUtils(self.env)
    
    def nStepTD_prediction(self, policy, n):
        V_est = np.zeros(self.env.nS)
        for epi in range(self.num_episode):
            done = False
            t = 0
            s = self.env.reset()
            T = float("inf")
            S = []
            S.append(s) # append intial state
            R = []
            R.append(0)
            while True:
                if t < T:
                    a = policy[s]
                    s_next, r, done, _ = self.env.step(a)
                    S.append(s_next)
                    R.append(r)
                    if done:
                        T = t+1
                # the time step whose state’s estimate is being updated
                τ = t - n + 1
                if τ >= 0:
                    # compute return G
                    G = 0
                    for i in range(τ+1, min(τ+n, T)+1):
                        G += (self.gamma**(i-τ-1)) * R[i]
                    if τ + n < T:
                        G = G + (self.gamma**n) * V_est[S[τ+n]]
                    V_est[S[τ]] = V_est[S[τ]] + self.alpha * (G - V_est[S[τ]])

                # move to next time step
                s = s_next
                t += 1

                if τ == T-1:
                    break

        return V_est

    