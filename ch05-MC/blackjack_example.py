import sys
from unicodedata import name
import gym
if "../" not in sys.path: sys.path.append("../")
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))


def handcrafted_episode():
    """ 
    handcrafted episode for debugging purpose
    each element in the episode is a [state, action, reward] tuple
    each state consists of : score, dealer_score, usable_ace 
    """
    return [((7, 5, False), 1, 0), ((9, 5, False), 1, 1.0), ((13, 5, False), 1, 0), \
            ((7, 5, False), 1, 1.0), ((18, 5, False), 1, 1.0), ((21, 5, False), 0, 1.0)]


# sample policy: Stick (action 0) if the score is 20 or 21, hit (action 1) otherwise
def policy(observation):
    score, dealer_score, usable_ace = observation
    if score == 20 or score == 21:
        return 0
    else:
        return 1


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    num_episode = 1000
    num_win = 0
    num_draw = 0
    num_lose = 0
    for i_episode in range(num_episode):
        observation = env.reset()
        for t in range(1000):
            print_observation(observation)
            action = policy(observation)
            print("Taking action: {}".format( ["Stick", "Hit"][action]))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print("Game end. Reward: {}\n".format(float(reward)))
                if reward == 1:
                    num_win += 1
                elif reward == 0:
                    num_draw += 1
                elif reward == -1:
                    num_lose += 1
                break

    # showing statistics
    winning_rate = num_win/num_episode
    drawing_rate = num_draw/num_episode
    losing_rate = num_lose/num_episode
    print("Winning rate: %.2f" % winning_rate)
    print("drawing rate: %.2f" % drawing_rate)
    print("losing rate: %.2f" % losing_rate)