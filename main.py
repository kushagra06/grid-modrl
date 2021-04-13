import numpy as np 
import gym
from grid import GridEnv
from gridAgents import PolMixAgent, QModule, PolMixModule
import matplotlib.pyplot as plt
import tensorflow as tf


def run(agent, n_epi=500, max_steps=500, learn=True):
    returns = []

    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        agent.env.reset()
        # while (not epi_over) and (step < max_steps):
        while step < max_steps:
            a = agent.get_action(agent.env.cur_state)
            s, a, new_s, r, done = agent.env.step(a)

            if learn == True:
                agent.update(s, a, new_s, r)
            
            cumulative_r += r
            step += 1

            # if done:
            #     epi_over = False
    
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
                x = np.random.rand(4)
                pi = x/np.sum(x)
                # pi1 = tf.convert_to_tensor(pi)
                # pi2 = tf.convert_to_tensor(pi)
                arb_agent.add_to_onpolicy_buffer(prev_state, a, r, arb_agent.cur_state, done, pi, pi)

            cumulative_r += r
            step += 1

            epi_over = False if done else True
        
        returns.append(cumulative_r)
    
    return returns

def get_occupancy_measure(pi_k, gamma=0.9):
	total_states = env.observation_space.n
	total_actions = env.action_space.n
	b0 = np.ones(total_states) / total_states
	c = np.zeros(total_states)
	t_mat = np.zeros([total_states, total_states])
	for j in range(total_states):
		for s in range(total_states):
			val = 0
			for a in range(total_actions):
				val += env.P[a,s,j] * pi_k[s,a]
			c[s] = val
		t_mat[j] = c
	I = np.eye(total_states)
	A = I - gamma * t_mat
	x = np.linalg.solve(A, b0)

	x_sa = np.zeros([total_states, total_actions])
	for s in range(total_states):
		for a in range(total_actions):
			x_sa[s,a] = x[s] * pi_k[s,a]

	return x, x_sa


def solve_coeff(q_k, dsa_k, pi1_k, pi2_k):
    constr = {'type':'eq', 'fun':lambda_constr}
	bnds = [(0+1e-8, 1+1e-8)] * total_states
    
    res = optimize.minimize(pol_loss, x0=coeff0, args=(d_sa, q_k, pi1_k, pi2_k), method='SLSQP', constraints=constr, bounds=bnds, 
                            options={'disp':False})
	
	obj_val = pol_loss(res.x, pi_k, q_k, ds_k, dsa_k)

	new_pi = np.reshape(res.x, (total_states, total_actions))

	return new_pi, obj_val

def optimize_pi(pi_k, pi1_k, pi2_k):
    q_k = get_q(pi_k)
    d_s, d_sa = get_occupancy_measure(pi_k)
    coeff = solve_coeff(q_k, d_sa, pi1_k, pi2_k)

def test_tab_arb(env1, env2, q1, q2, n_epi=500, max_steps=200, learn=True):
    returns = []
    
    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        env1.reset()
        while (step < max_steps):
            pi1_k, pi2_k = get_pi(q1[env1.cur_state]), get_pi(q2[env2.cur_state])
            pi_k = coeff1[env1.cur_state] * pi1_k + coeff2[env2.cur_state] * pi2_k
            a_env = sample_pi(pi_k)
            s1, a1, s_1, r1, done1 = env1.step(a_env) #module1 and arb
            s2, a2, s_2, r2, done2 = env2.step(a_env)
            coeff1, coeff2 = optimize_pi(pi_k, pi1_k, pi2_k)

            cumulative_r += r1
            step += 1

        returns.append(cumulative_r)
    
    return returns




def main():

    # agent1 = QModule(env1)
    # agent2 = QModule(env2)
    # returns1 = run(agent1)
    # returns2 = run(agent2)
    # np.save('m1_q1', agent1.q)
    # print(agent1.q)
    # np.save('m2_q2', agent2.q)
    # print(agent2.q)
    # plt.plot(returns1)
    # plt.show()
    # plt.plot(returns2)
    # plt.show()

    # pol_mix_agent = PolMixAgent(env1, env2)
    # arb_agent = PolMixModule(env1)
    # arb_returns = test_arb(arb_agent)
    # print(arb_returns)

    q1 = np.load('m1_q1.npy')
    q2 = np.load('m2_q2.npy')
    tab_arb_returns = test_tab_arb(q1, q2)


if __name__ == "__main__":
    env1 = GridEnv(goal=15)
    env2 = GridEnv(goal=12)
    main()