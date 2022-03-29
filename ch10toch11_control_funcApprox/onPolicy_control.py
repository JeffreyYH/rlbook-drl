import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from lib.models import LinearModel


class OnPolicy_Control:
    def __init__(self, args):
        self.env = args.env
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.num_episodes=10000
        self.gamma = 1.0
        self.alpha = 0.05
        self.tabularUtils = TabularUtils(self.env)
        self.funcApprox_type = args.funcApprox_type
        if self.funcApprox_type == "linear":
            self.Q_func = LinearModel(self.nA, self.nS)

    
    def episodic_semiGradient_sarsa(self):
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            a = self.tabularUtils.epsilon_greedy_policy(self.Q_func.forward(s)) 
            while not done:
                s_next, r, done, _ = self.env.step(a)
                a_next = self.tabularUtils.epsilon_greedy_policy(self.Q_func.forward(s)) 
                # TODO: deal with the gradient of Q
                self.Q_func.W = self.Q_func.W + self.alpha * ( r + self.gamma * self.Q_func.forward(s_next)[a_next] - self.Q_func.forward(s)[a]) 
                s = s_next
                a = a_next
        
        # greedy_policy = self.tabularUtils.Q_value_to_greedy_policy(Q)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="CartPole-v0", 
                        choices=["CartPole-v0", "MountainCar-v0"])
    parser.add_argument("--funcApprox_type", dest="funcApprox_type",type=str,
                        default="linear", 
                        choices=["linear", "MLP", "CNN"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)
    onpolicycontrol = OnPolicy_Control(args)
    onpolicycontrol.episodic_semiGradient_sarsa()