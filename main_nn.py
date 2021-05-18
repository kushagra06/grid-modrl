import math
import random
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

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 3
BATCH_SIZE = 8
GAMMA = 0.9


def get_pi_epsgreedy(q_vals, steps, a_dim=4):
    n_modules = len(q_vals)
    sample = random.random()
    pi = []
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            for m in range(n_modules):
                pi.append(q_vals[m].argmax().view(1,1))
    else:
        for m in range(n_modules):
            pi.append(torch.tensor([[random.randrange(a_dim)]], device=device, dtype=torch.long))
    
    return torch.cat(pi, dim=1)


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
    x = np.zeros(s_dim)
    x[s] = 1.
    x = np.expand_dims(x, 0) 
    return torch.FloatTensor(x)


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
    s_dim, a_dim = 16, 4 
    n_modules = len(modules_list)
    
    pi_tensors = get_pi(modules_list)
    arb = Arbitrator().to(device)
    returns = []
    all_rets = []
    memory = ReplayMemory(10000)
    for epi in range(n_epi):
        arb_env.reset()
        r_list = []
        steps = 0
        while steps < max_steps:
            state = get_state_vector(arb_env.cur_state)
            coeff = arb(state)
            pi_k = torch.zeros(s_dim, a_dim)
            for m in range(n_modules):
                pi_k += coeff[0][m] * pi_tensors[m]
            a = np.random.choice(4, p=pi_k[arb_env.cur_state].detach().cpu().numpy())
            s, a, s_, r, done = arb_env.step(a)
            r_list.append(r)
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
                # state = get_state_vector(arb_env.cur_state)
                next_state = state
                r = 100.
                steps += 1
                reward = torch.FloatTensor([r], device=device)
                r_list.append(r)
                memory.push(state, torch.FloatTensor([a], device=device), next_state, reward)
                break
        
        rets = []
        return_so_far = 0
        for t in range(len(r_list) - 1, -1, -1):
            return_so_far = r_list[t] + 0.9 * return_so_far
            rets.append(return_so_far)                
        # The returns are stored backwards in time, so we need to revert it
        rets = list(reversed(rets))
        all_rets.extend(rets)
        print("epi {} over".format(epi))
        if epi%7==0:
            arb.optimize(memory, pi_tensors, torch.FloatTensor(all_rets))
            all_rets = []
            memory = ReplayMemory(10000)
        returns.append(sum(r_list))


    return returns


def learn_mod_arb(env_list, n_epi=500, max_steps=500):
    n_modules = len(env_list[1:])
    s_dim, a_dim = 16, 4
    arb_env = env_list[0]
    arb = Arbitrator().to(device)
    arb_memory = ReplayMemory2(100000)
    mods_agents = []
    mods_target_agents = []
    mods_memory = []
    mods_returns = []
    for i in range(n_modules):
        mods_agents.append(DQNAgent().to(device))
        mods_memory.append(ReplayMemory2(100000))
        mods_target_agents.append(DQNAgent().to(device))
        mods_target_agents[i].load_state_dict(mods_agents[i].state_dict())
        mods_target_agents[i].eval()
        mods_returns.append([])
    
    all_rets = []
    returns = []
    total_steps = 0
    for epi in range(n_epi):
        step = 0
        for env in env_list:
            env.reset()
        r_list = []
        mod_r_list = [[] for _ in range(n_modules)] 
        while step < max_steps:
            # state, action, next_state, reward, done: Tensors
            # s, a, s_ ,r , done: numbers
            state = get_state_vector(arb_env.cur_state)
            coeff = arb(state)
            q_vals = [mods_agents[m](state) for m in range(n_modules)]
            pi_s_mods = get_pi_epsgreedy(q_vals, total_steps)
            a_arb = torch.sum(coeff * pi_s_mods)
            a_arb = torch.round(a_arb).item()
            # pi_s_mods = [F.softmax(q_s, dim=1) for q_s in q_vals]
            # pi_s_tensor = torch.zeros((1, a_dim))
            # for m in range(n_modules):
            #     pi_s_tensor += coeff[0][m] * pi_s_mods[m]
            
            # pi_s_np = pi_s_tensor.flatten().detach().cpu().numpy()
            # a_arb = np.random.choice(4, p=pi_s_np)

            s_arb, a_arb, new_s_arb, r_arb, done_ar = arb_env.step(a_arb)
            r_list.append(r_arb)
            step += 1
            total_steps += 1
            action_arb = torch.LongTensor([a_arb], device=device)
            next_state_arb = get_state_vector(new_s_arb)
            reward_arb = torch.FloatTensor([r_arb], device=device)
            done_arb = torch.Tensor([done_ar], device=device)

            for m in range(n_modules):
                s, a, s_, r, d = env_list[m+1].step(a_arb)
                mod_r_list[m].append(r)
                reward = torch.FloatTensor([r], device=device)
                next_state = get_state_vector(s_)
                action = torch.LongTensor([[a]], device=device)
                done = torch.Tensor([d], device=device)
                if d:
                    d = 0.
                    done = torch.Tensor([d], device=device)
                    mods_memory[m].push(state, action, next_state, reward, done)
                    mods_agents[m].optimize_model(mods_memory[m], mods_target_agents[m])
                    d = 1.
                    done = torch.Tensor([d], device=device)

                    state = get_state_vector(env_list[m+1].cur_state)
                    coeff = arb(state)
                    q_vals = [mods_agents[m](state) for m in range(n_modules)]
                    pi_s_mods = get_pi_epsgreedy(q_vals, total_steps)
                    a_arb = torch.sum(coeff * pi_s_mods)
                    a_arb = torch.round(a_arb).item()
                    
                    # pi_s_mods = [F.softmax(q_s, dim=1) for q_s in q_vals]
                    # pi_s_tensor = torch.zeros((1, a_dim))
                    # for m in range(n_modules):
                    #     pi_s_tensor += coeff[0][m] * pi_s_mods[m]
                    
                    # pi_s_np = pi_s_tensor.flatten().detach().cpu().numpy()
                    # a_arb = np.random.choice(4, p=pi_s_np)
                    action = torch.LongTensor([[a_arb]], device=device)
                    next_state = state
                    r = 100.
                    mod_r_list[m].append(r)

                    reward = torch.FloatTensor([r], device=device)
                    mods_memory[m].push(state, action, next_state, reward, done)
                    mods_agents[m].optimize_model(mods_memory[m], mods_target_agents[m])
                else:
                    mods_memory[m].push(state, action, next_state, reward, done)
                    mods_agents[m].optimize_model(mods_memory[m], mods_target_agents[m])
                
                if epi % TARGET_UPDATE == 0:
                    mods_target_agents[m].load_state_dict(mods_agents[m].state_dict())

            if done_ar:
                done_ar = 0.
                done_arb = torch.Tensor([done_ar], device=device)
                arb_memory.push(state, action_arb, next_state_arb, reward_arb, done_arb)
                done_ar = 1.
                
                state = get_state_vector(arb_env.cur_state)
                coeff = arb(state)
                q_vals = [mods_agents[m](state) for m in range(n_modules)]
                pi_s_mods = get_pi_epsgreedy(q_vals, total_steps)
                a_arb = torch.sum(coeff * pi_s_mods)
                a_arb = torch.round(a_arb).item()
                
                # pi_s_mods = [F.softmax(q_s, dim=1) for q_s in q_vals]
                # pi_s_tensor = torch.zeros((1, a_dim))
                # for m in range(n_modules):
                #     pi_s_tensor += coeff[0][m] * pi_s_mods[m]
                
                # pi_s_np = pi_s_tensor.flatten().detach().cpu().numpy()
                # a_arb = np.random.choice(4, p=pi_s_np)
                action_arb = torch.LongTensor([a_arb], device=device)
                next_state_arb = state
                r_arb = 100.
                r_list.append(r_arb)
                reward_arb = torch.FloatTensor([r_arb], device=device)
                done_arb = torch.Tensor([done_ar], device=device)
                step +=1
                total_steps += 1

                arb_memory.push(state, action_arb, next_state_arb, reward_arb, done_arb)
                break
            else:
                arb_memory.push(state, action_arb, next_state_arb, reward_arb, done_arb)

        rets = []
        return_so_far = 0
        for t in range(len(r_list) - 1, -1, -1):
            return_so_far = r_list[t] + 0.9 * return_so_far
            rets.append(return_so_far)                
        # The returns are stored backwards in time, so we need to revert it
        rets = list(reversed(rets))
        all_rets.extend(rets)
        if epi%7==0:
            arb.optimize(arb_memory, mods_agents, torch.FloatTensor(all_rets))
            all_rets = []
            arb_memory = ReplayMemory2(100000)

        returns.append(sum(r_list))
        for m in range(n_modules):
            mods_returns[m].append(sum(mod_r_list[m]))

        if epi%100==0:
            print("epi {} done".format(epi))
    
    return returns, mods_returns
        

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
    env_list = [env1, env1, env2]

    returns, mods_returns = learn_mod_arb(env_list)
    plt.plot(returns)
    plt.show()

    plt.plot(mods_returns[0])
    plt.show()
    plt.plot(mods_returns[1])
    plt.show()

    print("mods_returns[0]: ", mods_returns[0])
    print("mods_returns[1]: ", mods_returns[1])


if __name__ == "__main__":
    main()

## Notes:
# if done then expected_q_vals = r else r + gamma * next_q_values ???