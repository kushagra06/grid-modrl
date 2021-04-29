import numpy as np 
import torch
from itertools import count
from torch_gridagents import DQNAgent
from grid import GridEnv
from utils import ReplayMemory, Transition

TARGET_UPDATE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state_vector(s, s_dim=16):
    x = np.zeros(s_dim)
    x[s] = 1. 
    return x

def run(env, agent, n_epi=50):
    memory = ReplayMemory(10000)
    epi_durations = []
    for epi in range(n_epi):
        env.reset()
        for t in count():
            state = get_state_vector(env.cur_state)
            action = agent.get_action(state)
            s, a, s_, r, done = env.step(action)
            reward = torch.tensor([r], device=device)
            next_state = get_state_vector(env.cur_state)

            memory.push(state, action, next_state, reward)

            optimize_model()

            if done:
                epi_durations.append(t+1)
                break

        if epi % TARGET_UPDATE == 0:
            target_net.load_state_dict(agent.state_dict())



def main():
    env = GridEnv()
    agent = DQNAgent()
    run(env, agent)
    print('Done')

    


if __name__ == "__main__":
    main()