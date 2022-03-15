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
        Delta = 0
        for s in range(env.nS):
            pre_v = V[s]
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
            Delta = max(Delta, abs(pre_v - V[s]))

        if Delta < theta:
            break
    return V


if __name__ == "__main__":
    # policy should be a nS*nA matrix,
    # each row is the probability of taking which action under current state
    random_policy = np.ones([env.nS, env.nA])/env.nA

    print("Final Value function: ")
    V = policy_eval(random_policy)
    print(V.reshape(env.shape))


