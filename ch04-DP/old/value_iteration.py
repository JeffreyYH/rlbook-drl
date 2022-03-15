import sys
if "../" not in sys.path: sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
import numpy as np

env = GridworldEnv()

# hyper-parameters
discount_factor = 1
theta = 1e-5

def compute_q_value(s, V):
    q_s = np.zeros(env.nA)
    # all each possible action a, get the action-value function
    for a in range(env.nA):
        curr_q = 0
        for P_trans, s_next, reward, is_done in env.P[s][a]:
            curr_q += P_trans * (reward + discount_factor * V[s_next])
        q_s[a] = curr_q
    return q_s

def value_iter():
    # initialize the value function
    V = np.zeros(env.nS)
    while True:
        Delta = 0
        for s in range(env.nS):
            v = V[s]
            # we have to compute q[s] in each iteration from scratch
            # and compare it with the q value in previous iteration
            q_s = compute_q_value(s, V)

            # choose the optimal action and optimal value function in current state
            V[s] = max(q_s)
            Delta = max(Delta, np.abs(V[s] - v))

        if Delta < theta:
            break

    V_optimal = V

    # output the deterministic policy with optimal value function
    policy_optimal = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        q_s = compute_q_value(s, V_optimal)
        # choose optimal action
        a_optimal = np.argmax(q_s)
        policy_optimal[s, a_optimal] = 1

    return V_optimal, policy_optimal


if __name__ == "__main__":
    V_optimal, policy_optimal = value_iter()
    print("Optimal value function: ")
    print(V_optimal.reshape(env.shape))
    print("Optimal policy: ")
    print(policy_optimal)


