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
        self.gamma = 1.0
        self.theta = 1e-5
        self.max_iterations = 1000


    def compute_q_value_cur_state(self, s, value_func):
        q_s = np.zeros(self.env.nA)
        # all each possible action a, get the action-value function
        for a in range(self.env.nA):
            curr_q = 0
            for P_trans, s_next, reward, is_done in self.env.P[s][a]:
                curr_q += P_trans * (reward + self.gamma * value_func[s_next])
            q_s[a] = curr_q
        return q_s

    
    def policy_evaluation(self, value_func, policy):
        n_iter = 0
        for n_iter in range(1, self.max_iterations+1):
            # for each state
            Delta = 0
            for s in range(self.env.nS):
                pre_v_s = value_func[s]
                V_s = 0
                # for each action in current state
                for a in range(self.env.nA):
                    # get the probability of taking action a at current state s
                    P_a = policy[s, a]
                    # for each possible NEXT state taking action a at current state s
                    for P_trans, s_next, reward, is_done in self.env.P[s][a]:
                        V_s += P_a * P_trans * (reward + self.gamma * value_func[s_next])

                value_func[s] = V_s
                # see if the value function converge
                Delta = max(Delta, abs(pre_v_s - value_func[s]))

            if Delta < self.theta:
                break

        return value_func, n_iter


    def policy_improvement(self, value_func, policy):
        policy_stable = True
        policy_new = policy
        for s in range(self.env.nS):
            # action from the policy before policy improvement
            old_a = np.argmax(policy[s])
            # compute action-value function q(s,a) by one step of lookahead
            q_s = self.compute_q_value_cur_state(s, value_func)
            # choose the best action and greedily improve the policy
            best_a = np.argmax(q_s)
            policy_new[s, :] = np.eye(self.env.nA)[best_a, :]

            if old_a != best_a:
                policy_stable = False

        return policy_stable, policy_new


    def policy_iteration(self):
        policy = np.zeros([self.env.nS, self.env.nA])
        value_func = np.zeros(self.env.nS)

        # iteratively evaluate the policy and improve the policy
        total_num_policy_eval = 0
        for num_policy_iter in range(1, self.max_iterations+1):
            # evaluate policy and improve policy
            value_func, num_policy_eval = self.policy_evaluation(value_func, policy)
            policy_stable, policy = self.policy_improvement(value_func, policy)

            total_num_policy_eval += num_policy_eval
            if policy_stable:
                print("num of policy iteration:%d and policy evaluation: %d" %(num_policy_iter, total_num_policy_eval))
                return value_func, policy
        

    def value_iteration(self):
        # initialize the value function
        value_func = np.zeros(self.env.nS)
        for n_iter in range(1, self.max_iterations+1):
            Delta = 0
            for s in range(self.env.nS):
                pre_v_s = value_func[s]
                # we have to compute q[s] in each iteration from scratch
                # and compare it with the q value in previous iteration
                q_s = self.compute_q_value_cur_state(s, value_func)

                # choose the optimal action and optimal value function in current state
                value_func[s] = max(q_s)
                Delta = max(Delta, np.abs(value_func[s] - pre_v_s))

            if Delta < self.theta:
                break
        
        print("num of iteration is: %d" %n_iter)
        V_optimal = value_func

        # output the deterministic policy with optimal value function
        policy_optimal = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            q_s = self.compute_q_value_cur_state(s, V_optimal)
            # choose optimal action
            a_optimal = np.argmax(q_s)
            policy_optimal[s, a_optimal] = 1.0

        return V_optimal, policy_optimal


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        # default="FrozenLake-v1", 
                        default="Deterministic-4x4-FrozenLake-v0", 
                        # default="gridworld", 
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
    print("================Running policy iteration=====================")
    V_optimal, policy_optimal = dp.policy_iteration()
    print("Optimal value function: ")
    print(V_optimal)
    print("Optimal policy: ")
    print(policy_optimal)
    print("\n")

    # test value iteration
    print("================Running value iteration=====================")
    V_optimal, policy_optimal = dp.value_iteration()
    print("Optimal value function: ")
    print(V_optimal)
    print("Optimal policy: ")
    print(policy_optimal)
