import numpy as np 
import gym
from grid import GridEnv
from gridAgents import PolMixAgent, QModule
import matplotlib.pyplot as plt


def run(agent, n_epi=500, max_steps=50, learn=True):
    returns = []

    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        agent.env.reset()
        while (not epi_over) and (step < max_steps):
            a = agent.get_action(agent.env.cur_state)
            s, a, new_s, r, done = agent.env.step(a)

            if learn == True:
                agent.update(s, a, new_s, r)
            
            cumulative_r += r
            step += 1

            if done:
                epi_over = False
    
        returns.append(cumulative_r)

    return returns



def main():
    env1 = GridEnv(goal=15)
    env2 = GridEnv(goal=12)
    agent1 = QModule(env1)
    # agent2 = QModule(env2)
    returns1 = run(agent1)
    # returns2 = run(agent2)
    np.save('m1', agent1.q)
    print(agent1.q)
    # np.save('m2', agent2.q)
    plt.plot(returns1)
    plt.show()
    # plt.plot(returns2)
    # plt.show()


if __name__ == "__main__":
    main()