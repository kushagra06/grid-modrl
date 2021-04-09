import numpy as np 
import gym
from grid import GridEnv
from gridAgents import PolMixAgent, QModule, PolMixModule
import matplotlib.pyplot as plt
import tensorflow as tf


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


def test_arb(arb_agent, n_epi=500, max_steps=50, learn=True):
    returns = []

    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        arb_agent.env.reset()
        while (not epi_over) and (step < max_steps):
            prev_state = arb_agent.cur_state
            a_arb = arb_agent.get_action(prev_state)
            # print("a arb: ", a_arb)
            a_env = np.random.randint(0, 4)
            s, a, new_s, r, done = arb_agent.env.step(a_env)
            arb_agent.update_state(new_s)

            if epi%8 == 0:
                arb_agent.update()
            else:
                pi1 = tf.convert_to_tensor(np.random.choice(4, size=1))
                pi2 = tf.convert_to_tensor(np.random.choice(4, size=1))
                arb_agent.add_to_onpolicy_buffer(prev_state, a, r, arb_agent.cur_state, done, pi1, pi2)

            cumulative_r += r
            step += 1

            epi_over = False if done else True
        
        returns.append(cumulative_r)
    
    return returns


def main():
    env1 = GridEnv(goal=15)
    # env2 = GridEnv(goal=12)
    agent1 = QModule(env1)
    # # agent2 = QModule(env2)
    returns1 = run(agent1)
    # # returns2 = run(agent2)
    # np.save('m1', agent1.q)
    print(agent1.q)
    # # np.save('m2', agent2.q)
    plt.plot(returns1)
    plt.show()
    # plt.plot(returns2)
    # plt.show()

    # pol_mix_agent = PolMixAgent(env1, env2)
    # arb_agent = PolMixModule(env1)
    # arb_returns = test_arb(arb_agent)
    # print(arb_returns)



if __name__ == "__main__":
    main()