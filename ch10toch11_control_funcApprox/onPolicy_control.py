import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse
if "../" not in sys.path: sys.path.append("../")
from lib.common_utils import TabularUtils


class OnPolicy_Control:
    def __init__(self) -> None:
        pass

    
    def episodic_semiGradient_sarsa(self):
        pass



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        default="CartPole-v0", 
                        choices=["CartPole-v0", "MountainCar-v0"])
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)