import math 
import numpy as np 
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.nn.init as init

from typing import Tuple, Callable

from utils import ReplayMemory, Transition, Transition_done, ReplayMemory2

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 3
BATCH_SIZE = 8
GAMMA = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vector_to_number(state):
    numbers = torch.nonzero(state==1, as_tuple=False)
    ans = torch.index_select(numbers, dim=1, index=torch.LongTensor([1]))#select col=1
    return ans.flatten()

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
    def __init__(self, batch_size=32, a_dim=4, s_dim=16):
        super().__init__()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(s_dim, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, a_dim)
        # )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, a_dim),
            nn.Softplus()
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.01)
        self.batch_size = batch_size

    
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
        if len(memory) < self.batch_size:
            return
        
        transitions = memory.sample(self.batch_size)
        batch = Transition_done(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        state_action_values = self.forward(state_batch).gather(dim=1, index=action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values = target_agent(next_state_batch).max(1)[0].detach()
        # next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0].detach() #max Q

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
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001)
    
    def forward(self, state: torch.Tensor):
        coeff_s = self.linear_relu_stack(state)
        return coeff_s
    
    
    def optimize(self, memory, mods_agents, ret, nn=True):
        transitions = []
        for i in range(len(memory)):
            transitions.append(memory.memory.popleft())
        batch = Transition_done(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        self.optimizer.zero_grad()
        if nn:
            loss = self.loss2(state_batch, action_batch, mods_agents, ret)
        else:
            loss = self.loss1(state_batch, action_batch, mods_agents, ret)
        loss.backward(retain_graph=True)
        self.optimizer.step()


    def loss1(self, state, action, pi_modules, q_k):
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

    def loss2(self, state, action, mods_agents, q_k=10):
        s_dim, a_dim = 16, 4
        n_states = state.shape[0]
        n_modules = len(mods_agents)
        coeff = self.forward(state)
        q_vals = [mods_agents[m](state) for m in range(n_modules)]
        pi_mods = [F.softmax(q_s, dim=1) for q_s in q_vals]
        pi_mods_tensors = torch.stack(pi_mods)
        action = action.expand(n_modules, -1) # Repeating actions for all modules
        action = action.unsqueeze(-1) # matching dims of action and pi_mods_tensors
        pi_mods_sa = torch.gather(pi_mods_tensors, dim=2, index=action) # picking action vals according to the sampled actions
        pi_mods_sa = pi_mods_sa.squeeze(-1) # matching dims with coeff 
        
        loss = 0.
        for s in range(n_states):
            for m in range(n_modules):
                pi_arb = torch.sum(coeff[s, m] * pi_mods_sa[m, s])
            loss += q_k[s] * torch.log(pi_arb + 1e-8)

        return loss



        


