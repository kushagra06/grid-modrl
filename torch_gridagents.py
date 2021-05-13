import math 
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn.init as init

from typing import Tuple, Callable

from utils import ReplayMemory, Transition

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vector_to_number(state):
    numbers = torch.nonzero(state==1, as_tuple=False)
    ans = torch.index_select(numbers, dim=1, index=torch.LongTensor([1]))#select col=1
    return ans.flatten()
    # np_state = state.detach().cpu().numpy()

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
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.1)

    
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

    def optimize_model(self, memory, target_agent):
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)#.unsqueeze(1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        state_action_values = self.forward(state_batch).gather(dim=1, index=action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0].detach() #max Q

        # print("next_state_values: ", next_state_values)
        # print("next_state: ", batch.next_state)
        # print("non_final_next_states: ", non_final_next_states)
        # expected_state_action_values = reward_batch if done else reward_batch + GAMMA * next_state_values
        expected_state_action_values = reward_batch + GAMMA * next_state_values * (1. - done_batch)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Arbitrator(RLNetwork):
    def __init__(self, batch_size=8, a_dim=2, s_dim=16):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, a_dim),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.005)
    
    def forward(self, state: torch.Tensor):
        coeff_s = self.linear_relu_stack(state)
        return coeff_s
    
    def optimize(self, memory, pi_tensors, ret):
        transitions = []
        for i in range(len(memory)):
            transitions.append(memory.memory.popleft())
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        self.optimizer.zero_grad()
        loss = self.loss(state_batch, action_batch, pi_tensors, ret)
        loss.backward(retain_graph=True)
        self.optimizer.step()

    ## change
    def loss(self, state, action, pi_modules, q_k):
        s_dim, a_dim = 16, 4
        n_modules = pi_modules.size()[0]
        coeff = self.forward(state)
        state_number = vector_to_number(state)
        action = action.type(torch.LongTensor)
        # sa = torch.stack((state_number, action), dim=1)
        pi_modules_sa = pi_modules[:,state_number,action]
        loss = 0.
        for s in range(state_number.shape[0]):
            pi_arb = 0.
            for m in range(n_modules):
                pi_arb += coeff[s,m] * pi_modules_sa[m,s]
            loss += q_k[s] * torch.log(pi_arb + 1e-08)         

        return -loss





        


