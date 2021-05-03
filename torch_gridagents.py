import math 
import numpy as np 
import random
import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn.init as init

from typing import Tuple, Callable

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
    
    def forward(self, *x):
        return x

    # def train(self, loss, optimizer):
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()


class DQNAgent(RLNetwork):
    def __init__(self, batch_size=8, a_dim=4, s_dim=16):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, a_dim)
        )
    
    # Called with either one element to determine next action, or a batch during optimization.
    def forward(self, state: torch.Tensor):
        q_s = self.linear_relu_stack(state)
        return q_s
    
    def get_action_epsgreedy(self, state: torch.Tensor, steps_done: int, a_dim=4):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.forward(state).argmax().view(1,1)
        else:
            return torch.tensor([[random.randrange(a_dim)]], device=device, dtype=torch.long)


        


