import numpy as np 
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_gridagents import DQNAgent, Arbitrator
from grid import GridEnv
from utils import ReplayMemory, ReplayMemory2, Transition, Transition_done

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_UPDATE = 3
BATCH_SIZE = 8
GAMMA = 0.9


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
    # expected_state_action_values = reward_batch if done else reward_batch + GAMMA * next_state_values
    expected_state_action_values = reward_batch + GAMMA * next_state_values

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

def test_arb(arb_env, modules_list, n_epi=250, max_steps=500):
    n_modules = len(modules_list)
    pi_tensors = get_pi(modules_list)
    # print(pi_tensors )
    # pi_list = [] 
    # pi_list.append(torch.from_numpy(modules_list[0]))
    # pi_list.append(torch.from_numpy(modules_list[1]))
    # pi_tensors = torch.stack(pi_list)

    s_dim, a_dim = 16, 4
    arb = Arbitrator().to(device)
    returns = []
    all_rets = []
    memory = ReplayMemory(10000)
    for epi in range(n_epi):
        arb_env.reset()
        cumulative_r = []
        steps = 0
        while steps < max_steps:
            state = get_state_vector(arb_env.cur_state)
            coeff = arb(state)
            pi_k = torch.zeros(s_dim, a_dim)
            for m in range(n_modules):
                pi_k += coeff[0][m] * pi_tensors[m]
            a = np.random.choice(4, p=pi_k[arb_env.cur_state].detach().cpu().numpy())
            s, a, s_, r, done = arb_env.step(a)
            cumulative_r.append(r)
            reward = torch.FloatTensor([r], device=device)
            next_state = get_state_vector(s_)
            steps += 1
            memory.push(state, torch.FloatTensor([a], device=device), next_state, reward)

            if done:
                state = get_state_vector(arb_env.cur_state)
                coeff = arb(state)
                pi_k = torch.zeros(s_dim, a_dim)
                for m in range(n_modules):
                    pi_k += coeff[0][m] * pi_tensors[m]

                a = np.random.choice(4, p=pi_k[arb_env.cur_state].detach().cpu().numpy())
                state = get_state_vector(arb_env.cur_state)
                next_state = state
                r = 100.
                steps += 1
                reward = torch.FloatTensor([r], device=device)
                cumulative_r.append(r)
                memory.push(state, torch.FloatTensor([a], device=device), next_state, reward)
                break
        
        rets = []
        return_so_far = 0
        for t in range(len(cumulative_r) - 1, -1, -1):
            return_so_far = cumulative_r[t] + 0.9 * return_so_far
            rets.append(return_so_far)                
        # The returns are stored backwards in time, so we need to revert it
        rets = list(reversed(rets))
        all_rets.extend(rets)
        print("epi {} over".format(epi))
        if epi%7==0:
            arb.optimize(memory, pi_tensors, torch.FloatTensor(all_rets))
            all_rets = []
            memory = ReplayMemory(10000)
        returns.append(sum(cumulative_r))


    return returns


def learn_mod_arb(env_list, n_epi=100, max_steps=500):
    n_modules = len(env_list[1:])
    s_dim, a_dim = 16, 4
    returns_arb = []
    arb_env = env_list[0]
    arb = Arbitrator().to(device)
    arb_memory = ReplayMemory2(1000)
    mods_agents = []
    mods_target_agents = []
    mods_memory = []
    for i in range(n_modules):
        mods_agents.append(DQNAgent().to(device))
        mods_memory.append(ReplayMemory2(1000))
        mods_target_agents.append(DQNAgent().to(device))
        mods_target_agents[i].load_state_dict(mods_agents[i].state_dict())
        mods_target_agents[i].eval()
    
    for epi in range(n_epi):
        step = 0
        for env in env_list:
            env.reset()
        cumulative_r = []
        while step < max_steps:
            #state, action, next_state, reward, done: Tensors
            #s,a,s_,r,done: numbers
            state = get_state_vector(arb_env.cur_state)
            coeff = arb(state)
            q_vals = [mods_agents[m](state) for m in range(n_modules)]
            pi_s_mods = [F.softmax(q_s, dim=1) for q_s in q_vals]
            pi_s_tensor = torch.Tensor(a_dim)
            for m in range(n_modules):
                pi_s_tensor += coeff[m] * pi_s_mods[m]
            
            pi_s_np = pi_s_tensor.detach().cpu().numpy()
            a_arb = np.random.choice(4, p=pi_s_np[arb_env.cur_state])
            s_arb, a_arb, new_s_arb, r_arb, done_arb = arb_env.step(a_arb)
            cumulative_r_arb.append(r_arb)
            step += 1
            action_arb = torch.FloatTensor([a_arb], device=device)
            next_state_arb = get_state_vector(new_s_arb)
            reward_arb = torch.FloatTensor([r_arb], device=device)
            done_arb = torch.Tensor([done], device=device)
            arb_memory.push(state, action_arb, next_state_arb, reward_arb, done_arb)

            for m in range(total_modules):
                s, a, s_, r, done = env_list[m+1].env.step(a_arb)
                reward = torch.FloatTensor([r], device=device)
                next_state = get_state_vector(s_)
                action = torch.FloatTensor([a], device=device)
                done = torch.Tensor([done], device=device)
                mods_memory[m].push(state, action, next_state, reward, done)
                #mods_agents[m] -> optimize
                #change dqn model to include optimizer and optimize
            

        
        


def main():
    env = GridEnv()
    # returns = run_dqn(env)
    # print('Done')

    # pi_1 = np.load('./npy_files/m1_pi.npy')
    # pi_2 = np.load('./npy_files/m2_pi.npy')
    # modules_list = [pi_1,pi_2]


    # module_files = ["./pytorch_models/dqn_4x4.pt", "./pytorch_models/dqn_4x4_g7.pt"]
    # modules_list = []
    # for i in range(len(module_files)):
    #     m = DQNAgent()
    #     m.load_state_dict(torch.load(module_files[i]))
    #     m.eval()
    #     modules_list.append(m)
    # returns = test_arb(env, modules_list)


    env1 = GridEnv()
    env2 = GridEnv(goal=7)
    env_list = [evn1, env1, env2]

    returns = learn_mod_arb(env_list)
    plt.plot(returns)
    plt.show()







if __name__ == "__main__":
    main()

## Notes:
# if done then expected_q_vals = r else r + gamma * next_q_values ???