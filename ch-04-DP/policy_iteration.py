import sys
if "../" not in sys.path: sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
import numpy as np

env = GridworldEnv()

# hyper-parameters
discount_factor = 1
theta = 1e-5

def policy_eval (policy):
    # initialize value function V, for each state s, V(s) = 0
    V = np.zeros(env.nS)
    # print(V)
    while True:
        # for each state
        Delta =  0
        for s in range(env.nS):
            v = V[s]
            V_s = 0
            # for each action in current state
            for a in range(env.nA):
                # get the probability of taking action a at current state s
                P_a = policy[s, a]
                # for each possible NEXT state taking action a at current state s
                for P_trans, s_next, reward, is_done in env.P[s][a]:
                    V_s += P_a * P_trans * (reward + discount_factor * V[s_next])

            V[s] = V_s

            # see if the value function converge
            Delta = max(Delta, abs(v - V[s]))

        if Delta < theta:
            break
    return V

def compute_q_value(s, V):
    q_s = np.zeros(env.nA)
    # all each possible action a, get the action-value function
    for a in range(env.nA):
        curr_q = 0
        for P_trans, s_next, reward, is_done in env.P[s][a]:
            curr_q += P_trans * (reward + discount_factor * V[s_next])
        q_s[a] = curr_q
    return q_s

def policy_iter():
    # initialize the policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        policy_stable = True
        # get the value function for current policy
        V = policy_eval(policy)

        for s in range(env.nS):
            # action from the policy before policy improvement
            old_a = np.argmax(policy[s, :])

            # compute action-value function q(s,a) by one step of lookahead
            q_s = compute_q_value(s, V)
            # choose the best action and greedily improve the policy
            best_a = np.argmax(q_s)
            policy[s, :] = np.eye(env.nA)[best_a, :]

            if old_a != best_a:
                policy_stable = False

        if policy_stable:
            return V, policy

if __name__ == "__main__":
    V_optimal, policy_optimal = policy_iter()
    print("Optimal value function: ")
    print(V_optimal.reshape(env.shape))
    print("Optimal policy: ")
    print(policy_optimal)



