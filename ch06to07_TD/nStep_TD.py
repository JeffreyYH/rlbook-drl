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
        self.tabularUtils = TabularUtils(self.env)
    
    def nStepTD_prediction(self, policy, n):
        V_est = np.zeros(self.env.nS)
        for epi in range(self.num_episodes):
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
                    a = np.argmax(policy[s])
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


    # TODO: evaluate n-step sarsa
    def nStep_sarsa(self, n):
        Q = np.zeros((self.env.nS, self.env.nA))
        for epi in range(self.num_episodes):
            S = []; R = []; A = []
            R.append(0)
            s = self.env.reset()
            S.append(s) # append intial state
            a = self.tabularUtils.epsilon_greedy_policy(Q[s, :]) 
            A.append(a)
            T = float("inf")
            t = 0
            while True:
                # print("time step is %d" %t)
                if t < T:
                    s_next, r, done, _ = self.env.step(a)
                    S.append(s_next); R.append(r)
                    if done:
                        T = t+1
                    else:
                        a = self.tabularUtils.epsilon_greedy_policy(Q[s, :]) 
                        A.append(a)
                tau = t - n + 1
                if tau >= 0:
                    # compute return G
                    G = 0
                    for i in range(tau+1, min(tau+n, T)+1):
                        G += (self.gamma**(i-tau-1)) * R[i]
                    if tau + n < T:
                        G = G + (self.gamma**n) * Q[S[tau+n], A[tau+n]]
                    Q[S[tau], A[tau]] = Q[S[tau], A[tau]] + self.alpha * (G - Q[S[tau], A[tau]])

                # move to next time step
                s = s_next
                t += 1

                if tau == T-1:
                    break
        
        greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)

        return Q, greedy_policy


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

    nstep_td = Tabular_nStepTD(args)
    n = 5
    V_est_nstepTD = nstep_td.nStepTD_prediction(policy_optimal, n)
    print(V_est_nstepTD)

    print("mean abs error of n-step TD prediction: %5f" %np.mean(np.abs(V_est_nstepTD - V_optimal_VI)))

    Q_nStepSarsa, policy_nStepSarsa = nstep_td.nStep_sarsa(n)
    print(policy_nStepSarsa)
