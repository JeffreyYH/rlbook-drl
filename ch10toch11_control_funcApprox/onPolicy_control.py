import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
import torch
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils
from lib.models import LinearModel, MLP


class OnPolicy_Control:
    def __init__(self, args):
        self.env = args.env
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.num_episodes=1000
        self.gamma = 1.0
        self.alpha = 0.001
        self.tabularUtils = TabularUtils(self.env)
        self.funcApprox_type = args.funcApprox_type
        if self.funcApprox_type == "linear":
            self.Q_func = LinearModel(self.nS, self.nA)
        elif self.funcApprox_type == "MLP":
            self.Q_func = MLP(self.nS, self.nA)

    
    def episodic_semiGradient_sarsa(self):
        " episodic semi-gradient Sarsa for estimating optimal Q value"
        loss_format = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.Q_func.parameters(), lr=self.alpha)
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            s_ts = torch.from_numpy(s).float()
            a = self.tabularUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
            R_epi = 0
            while not done:
                s_next, r, done, _ = self.env.step(a)
                if done:
                    TD_target = torch.tensor(r).float()
                else:
                    a_next = self.tabularUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
                    s_next_ts = torch.from_numpy(s_next).float()
                    TD_target = torch.tensor(r + self.gamma * (self.Q_func(s_next_ts)[a_next]).cpu().detach().numpy()).float()
                optimizer.zero_grad()
                L = loss_format(TD_target, self.Q_func(s_ts)[a])
                L.backward()
                optimizer.step()

                # update for this iteration
                s = s_next
                s_ts = torch.from_numpy(s).float()
                a = a_next

                # calculate total episode reward
                R_epi += r

            # episode wrap-up
            if epi % 10 == 0:
                print("Episode %d reward: %d" %(epi, R_epi))



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="CartPole-v0", 
                        choices=["CartPole-v0", "MountainCar-v0"])
    parser.add_argument("--funcApprox_type", dest="funcApprox_type",type=str,
                        default="MLP", 
                        choices=["linear", "MLP"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)
    onpolicycontrol = OnPolicy_Control(args)
    onpolicycontrol.episodic_semiGradient_sarsa()