import sys
import gym
import numpy as np
from tqdm import tqdm
import argparse


class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.gamma = 1.0