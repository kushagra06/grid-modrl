import numpy as np 
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch_gridagents import DQNAgent
from grid import GridEnv
from utils import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_UPDATE = 3
BATCH_SIZE = 8
GAMMA = 0.9

memory = ReplayMemory(10000)

def get_state_vector(s, s_dim=16):
    x = np.zeros(s_dim)
    x[s] = 1.
    x = np.expand_dims(x, 0) 
    return torch.FloatTensor(x)

def optimize_model(agent, target_agent, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)#.unsqueeze(1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent(state_batch).gather(dim=1, index=action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + GAMMA * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run(env, n_epi=50):
    agent = DQNAgent(BATCH_SIZE).to(device)
    target_agent = DQNAgent(BATCH_SIZE).to(device)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    optimizer = optim.RMSprop(agent.parameters())
    epi_durations = []
    returns = []
    steps = 0
    for epi in range(n_epi):
        env.reset()
        cumulative_r = 0.
        for t in count():
            state = get_state_vector(env.cur_state)
            action = agent.get_action_epsgreedy(state, steps)
            np_action = action.detach().cpu().numpy()[0][0] #convert tensor to np
            s, a, s_, r, done = env.step(np_action)
            cumulative_r += r
            steps += 1
            reward = torch.tensor([r], device=device)
            next_state = get_state_vector(env.cur_state)

            memory.push(state, action, next_state, reward)

            optimize_model(agent, target_agent, optimizer)

            if done:
                epi_durations.append(t+1)
                break
        
        returns.append(cumulative_r)
        if epi % TARGET_UPDATE == 0:
            print("epi {} over".format(epi))        
            target_agent.load_state_dict(agent.state_dict())

    return returns

def main():
    env = GridEnv()
    returns = run(env)
    plt.plot(returns)
    plt.show()
    print('Done')

    


if __name__ == "__main__":
    main()