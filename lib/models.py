import os, sys, copy, argparse
import numpy as np
import matplotlib.pyplot as plt
import gym  
import torch
import collections, tqdm

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
    
    def forward(self, input):
        x = self.linear1(input)
        return x


class MLP(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x