import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

from typing import Tuple, Callable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
    
    def forward(self, *x):
        return x

    def train(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DQNAgent(RLNetwork):
    def __init__(self, a_dim=4, s_dim=16):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(s_dim+a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        x = torch.cat([states, actions])
        q_sa = self.linear_relu_stack(x)
        return q_sa
    
    def get_action(state: torch.Tensor):
        pass


