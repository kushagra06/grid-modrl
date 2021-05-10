import numpy as np 
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_gridagents import DQNAgent, Arbitrator
from grid import GridEnv
from utils import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_UPDATE = 3
BATCH_SIZE = 8
GAMMA = 0.9

memory = ReplayMemory(10000)



def get_pi(modules_list, s_dim=16, a_dim_mods=4):
    n_modules = len(modules_list)
    # pi_tensors = torch.FloatTensor()
    pi_list = []
    for m in range(n_modules):
        pi = torch.FloatTensor()
        for s in range(s_dim):
            s_tensor = get_state_vector(s)
            q_s = modules_list[m](s_tensor)
            pi_s = F.softmax(q_s, dim=1)
            pi = torch.cat((pi, pi_s), dim=0)
        # pi_tensors = torch.cat((pi_tensors, pi), dim=0)
        pi_list.append(pi)

    return torch.stack(pi_list)

def get_state_vector(s, s_dim=16):
    # if s==None:
    #     return torch.full((s_dim,), fill_value=None, dtype=None)
    
    x = np.zeros(s_dim)
    x[s] = 1.
    x = np.expand_dims(x, 0) 
    return torch.FloatTensor(x)

def optimize_model(agent, target_agent, optimizer, done):
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
    next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0].detach() #max Q

    # print("next_state_values: ", next_state_values)
    # print("next_state: ", batch.next_state)
    # print("non_final_next_states: ", non_final_next_states)
    expected_state_action_values = reward_batch if done else reward_batch + GAMMA * next_state_values
    # expected_state_action_values = reward_batch + GAMMA * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_dqn(env, n_epi=50):
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
            r = 100 if done else 1
            cumulative_r += r
            steps += 1
            reward = torch.FloatTensor([r], device=device)
            next_state = get_state_vector(s_)

            memory.push(state, action, next_state, reward)

            optimize_model(agent, target_agent, optimizer, done)

            if done:
                epi_durations.append(t+1)
                break
        
        returns.append(cumulative_r)
        print("epi {} over".format(epi))        

        # if epi % TARGET_UPDATE == 0:
        #     print("epi {} over".format(epi))        
        #     target_agent.load_state_dict(agent.state_dict())
    
    # torch.save(agent.state_dict(), "./pytorch_models/dqn_4x4_g7.pt")
    return returns

def test_arb(arb_env, module_files, n_epi=1):
    n_modules = len(module_files)
    modules_list = []
    for i in range(n_modules):
        m = DQNAgent()
        m.load_state_dict(torch.load(module_files[i]))
        m.eval()
        modules_list.append(m)
    
    pi_tensors = get_pi(modules_list)
    s_dim, a_dim = 16, 4
    arb = Arbitrator()

    for epi in range(n_epi):
        arb_env.reset()
        cumulative_r = 0.
        for t in count():
            state = get_state_vector(arb_env.cur_state)
            coeff = torch.flatten(arb(state))
            pi_k = torch.zeros(s_dim, a_dim)
            for m in range(n_modules):
                pi_k += coeff[m] * pi_tensors[m]
            a = np.random.choice(4, p=pi_k[arb_env.cur_state].detach().cpu().numpy())
            s, a, s_, r, done = arb_env.step(a)
            r = 100. if done else 1.
            cumulative_r += r
            reward = torch.FloatTensor([r], device=device)
            next_state = get_state_vector(s_)

            memory.push(state, coeff, next_state, reward)

            if done:
                break
        
        # if epi%4 == 0:
        arb.optimize()
        returns.append(cumulative_r)
        print("epi {} over".format(epi))


    return returns


def main():
    env = GridEnv()
    # returns = run_dqn(env)
    # plt.plot(returns)
    # plt.show()
    # print('Done')

    module_files = ["./pytorch_models/dqn_4x4.pt", "./pytorch_models/dqn_4x4_g7.pt"]
    returns = test_arb(env, module_files)





if __name__ == "__main__":
    main()

## Notes:
# if done then expected_q_vals = r else r + gamma * next_q_values ???