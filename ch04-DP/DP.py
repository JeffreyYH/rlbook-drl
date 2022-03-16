import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
import lib.envs.lake_envs as lake_env

class Tabular_DP:
    def __init__(self, args):
        self.env = args.env
        self.discount_factor = 1.0
        self.theta = 1e-5
    

    def policy_eval (self, policy):
        # initialize value function V, for each state s, V(s) = 0
        V = np.zeros(self.env.nS)
        # print(V)
        while True:
            # for each state
            Delta = 0
            for s in range(self.env.nS):
                pre_v = V[s]
                V_s = 0
                # for each action in current state
                for a in range(self.env.nA):
                    # get the probability of taking action a at current state s
                    P_a = policy[s, a]
                    # for each possible NEXT state taking action a at current state s
                    for P_trans, s_next, reward, is_done in self.env.P[s][a]:
                        V_s += P_a * P_trans * (reward + self.discount_factor * V[s_next])
                V[s] = V_s

                # see if the value function converge
                Delta = max(Delta, abs(pre_v - V[s]))

            if Delta < self.theta:
                break
        return V
    

    def compute_q_value(self, s, V):
        q_s = np.zeros(self.env.nA)
        # all each possible action a, get the action-value function
        for a in range(self.env.nA):
            curr_q = 0
            for P_trans, s_next, reward, is_done in self.env.P[s][a]:
                curr_q += P_trans * (reward + self.discount_factor * V[s_next])
            q_s[a] = curr_q
        return q_s


    def policy_iter(self):
        # initialize the policy
        # policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        policy = np.zeros([self.env.nS, self.env.nA])

        while True:
            policy_stable = True
            # get the value function for current policy
            V = self.policy_eval(policy)

            for s in range(self.env.nS):
                # action from the policy before policy improvement
                old_a = np.argmax(policy[s, :])

                # compute action-value function q(s,a) by one step of lookahead
                q_s = self.compute_q_value(s, V)
                # choose the best action and greedily improve the policy
                best_a = np.argmax(q_s)
                policy[s, :] = np.eye(self.env.nA)[best_a, :]

                if old_a != best_a:
                    policy_stable = False

            print (V)

            if policy_stable:
                return V, policy
    

    def value_iter(self):
        # initialize the value function
        V = np.zeros(self.env.nS)
        while True:
            Delta = 0
            for s in range(self.env.nS):
                v = V[s]
                # we have to compute q[s] in each iteration from scratch
                # and compare it with the q value in previous iteration
                q_s = self.compute_q_value(s, V)

                # choose the optimal action and optimal value function in current state
                V[s] = max(q_s)
                Delta = max(Delta, np.abs(V[s] - v))

            if Delta < self.theta:
                break

        V_optimal = V

        # output the deterministic policy with optimal value function
        policy_optimal = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            q_s = self.compute_q_value(s, V_optimal)
            # choose optimal action
            a_optimal = np.argmax(q_s)
            policy_optimal[s, a_optimal] = 1

        return V_optimal, policy_optimal


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        # default="FrozenLake-v1", 
                        # default="Deterministic-4x4-FrozenLake-v0", 
                        default="gridworld", 
                        choices=["gridworld", "FrozenLake-v1", 'Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.env_name == "gridworld":
        args.env = GridworldEnv() 
    else:
        args.env = gym.make(args.env_name)
    dp = Tabular_DP(args)

    # test policy iteration
    V_optimal, policy_optimal = dp.policy_iter()
    print("Optimal value function: ")
    print(V_optimal.reshape([4, 4]))
    print("Optimal policy: ")
    print(policy_optimal)

    # test value iteration
    V_optimal, policy_optimal = dp.value_iter()
    print("Optimal value function: ")
    print(V_optimal.reshape([4, 4]))
    print("Optimal policy: ")
    print(policy_optimal)
