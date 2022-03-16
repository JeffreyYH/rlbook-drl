# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import sys
if "../" not in sys.path: sys.path.append("../")
import lib.envs.lake_envs as lake_env


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)

# added by yf
def compute_q_value_cur_state(env, gamma, s, V):
    q_s = np.zeros(env.nA)
    # all each possible action a, get the action-value function
    for a in range(env.nA):
        curr_q = 0
        for P_trans, s_next, reward, is_done in env.P[s][a]:
            if is_done:
                curr_q += reward
            else:
                curr_q += P_trans * (reward + gamma * V[s_next])
        q_s[a] = curr_q
    return q_s

def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.

    # initialize policy
    policy = np.ones(env.nS)

    # compute Q value
    Q_value = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # all each possible action a, get the action-value function
        Q_value[s, :] = compute_q_value_cur_state(env, gamma, s, value_function)

        # get the optimal action based on q value
        optimal_a = np.argmax(Q_value[s, :])
        policy[s] = optimal_a

    return policy





def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for n_iter in range(1, max_iterations+1):
        Delta = 0
        for s in range(env.nS):
            pre_v_s = value_func[s]
            # we have to compute q[s] in each iteration from scratch
            # and compare it with the q value in previous iteration
            q_s = compute_q_value_cur_state(env, gamma, s, value_func)

            # choose the optimal action and optimal value function in current state
            value_func[s] = max(q_s)
            Delta = max(Delta, np.abs(value_func[s] - pre_v_s))

        if Delta < tol:
            break
    return value_func, n_iter



######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 1.2 & 1.3

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)


    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 1.2 & 1.3

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1],
                xticklabels = np.arange(1, env.nrow+1))
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None

import timeit
if __name__ == "__main__":
    # envs = ['Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0']
    envs = ['Deterministic-4x4-FrozenLake-v0']
    # Define num_trials, gamma and whatever variables you need below.
    gamma = 1.0
    # methods = ["sync", "async_ordered", "async_randperm"]
    methods = ["async_ordered"]
    print('\n ================ running policy_iteration ====================== \n ')
    for method in methods:
      for env_name in envs:
        print("runinng %s and %s \n" %(method, env_name))
        env = gym.make(env_name)
        start_pi = timeit.default_timer()

        # if method == "sync":
        #   policy_new, value_func, num_policy_iter, total_num_policy_eval = policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
        if method == "async_ordered":
          policy_new, value_func, num_policy_iter, total_num_policy_eval = policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3)
        # if method == "async_randperm":
        #   policy_new, value_func, num_policy_iter, total_num_policy_eval = policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3)

        stop_pi = timeit.default_timer()
        print('Time of policy iteration: ', stop_pi - start_pi)
        print("number of policy iter with \n", num_policy_iter)
        print("total number of policy evaluation \n", total_num_policy_eval)
        # print('The optimal policy in letters are ')
        # display_policy_letters(env, policy_new)
        # value_func_heatmap(env, value_func)
        # print("\n")

    # print('\n ================ running value_iteration ====================== \n ')
    # # methods = ["sync", "async_ordered", "async_randperm", "custom"]
    # methods = ["async_ordered"]
    # for method in methods:
    #   for env_name in envs:
    #     print("runinng %s and %s \n" %(method, env_name))
    #     env = gym.make(env_name)
    #     start_vi = timeit.default_timer()

    #     # if method == "sync":
    #     #   value_func, n_value_iter = value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
    #     if method == "async_ordered":
    #       value_func, n_value_iter = value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3)
    #     # if method == "async_randperm":
    #     #   value_func, n_value_iter = value_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3)
    #     # if method == "custom":
    #     #   value_func, n_value_iter = value_iteration_async_custom(env, env_name, gamma, max_iterations=int(1e3), tol=1e-3)
    #     stop_vi = timeit.default_timer()
    #     print('Time of value iteration: ', stop_vi - start_vi)
    #     print("number of value iter\n", n_value_iter)
    #     optimal_policy = value_function_to_policy(env, gamma, value_func)
    #     display_policy_letters(env, optimal_policy)
    #     value_func_heatmap(env, value_func)
    #     print("\n")