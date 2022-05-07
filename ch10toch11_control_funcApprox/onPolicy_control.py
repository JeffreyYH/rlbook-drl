import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
import torch
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import BoxUtils
from lib.models import LinearModel, MLP


class OnPolicy_Control:
    def __init__(self, args):
        self.env = args.env
        self.env_nS = self.env.observation_space.shape[0]
        self.env_nA = self.env.action_space.n
        self.num_episodes=1000
        self.gamma = 0.99
        self.alpha = 0.001
        self.boxUtils = BoxUtils(self.env)
        self.funcApprox_type = args.funcApprox_type
        if self.funcApprox_type == "linear":
            self.Q_func = LinearModel(self.env_nS, self.env_nA)
        elif self.funcApprox_type == "MLP":
            self.Q_func = MLP(self.env_nS, self.env_nA)
    

    def test_policy(self, render):
        """ test the learned policy"""
        done = False
        s = self.env.reset()
        s_ts = torch.from_numpy(s).float()
        a = self.boxUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
        R_epi = 0
        while not done:
            s_next, r, done, _ = self.env.step(a)

            # calculate total episode reward
            R_epi += r
            
            a_next = self.boxUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
            s_next_ts = torch.from_numpy(s_next).float()

            # update for this iteration
            s = s_next
            s_ts = torch.from_numpy(s).float()
            a = a_next

            # render
            if render:
                self.env.render()

        return R_epi


    def episodic_semiGradient_sarsa(self):
        " episodic semi-gradient Sarsa for estimating optimal Q value"
        loss_format = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.Q_func.parameters(), lr=self.alpha)
        for epi in range(self.num_episodes):
            done = False
            s = self.env.reset()
            s_ts = torch.from_numpy(s).float()
            a = self.boxUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
            R_epi = 0
            while not done:
                s_next, r, done, _ = self.env.step(a)
                if done:
                    TD_target = torch.tensor(r).float()
                else:
                    a_next = self.boxUtils.epsilon_greedy_policy(self.Q_func(s_ts).cpu().detach().numpy())
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
                R_epi_test = self.test_policy(render=False)
                print("Episode %d, testing reward: %d" %(epi, R_epi_test))
        
        # render at the end
        self.test_policy(render=True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="CartPole-v1", 
                        choices=["CartPole-v1", "CartPole-v0", "MountainCar-v0"])
    parser.add_argument("--funcApprox_type", dest="funcApprox_type",type=str,
                        default="MLP", 
                        choices=["linear", "MLP"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)
    onpolicycontrol = OnPolicy_Control(args)
    onpolicycontrol.episodic_semiGradient_sarsa()