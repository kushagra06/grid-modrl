import numpy as np
from scipy import optimize 
from grid import GridEnv
# from gridAgents import PolMixAgent, QModule, PolMixModule
# import matplotlib.pyplot as plt
# import tensorflow as tf


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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def get_pi(q):
    pi = []
    for s in range(env1.observation_space.n):
        pi_s = softmax(q[s])
        pi.append(pi_s)

    return np.asarray(pi)

def sample_pi(pi, cur_state):
    new_pi = np.empty((env1.observation_space.n, env1.action_space.n))
    for s in range(env1.observation_space.n):
        new_pi[s] = pi[s]/np.sum(pi[s])

    return new_pi, np.random.choice(4, p=new_pi[cur_state])

def get_occupancy_measure(pi_k, gamma=0.9):
	total_states = env1.observation_space.n
	total_actions = env1.action_space.n
	b0 = np.ones(total_states) / total_states
	c = np.zeros(total_states)
	t_mat = np.zeros([total_states, total_states])
	for j in range(total_states):
		for s in range(total_states):
			val = 0
			for a in range(total_actions):
				val += env1.P[a,s,j] * pi_k[s,a]
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


def get_constraints_matrix(total_vars, total_constraints):
    A = np.zeros((total_constraints, total_vars))
    j = 0
    for i in range(total_constraints):
        A[i][j] = 1
        A[i][j+total_constraints] = 1
        j += 1
    
    return A

def coeff_constr(coeff):
    total_constraints = env1.observation_space.n 
    total_vars = env1.observation_space.n * 2
    b = np.ones(total_constraints)
    A = get_constraints_matrix(total_vars, total_constraints)
    return (A.dot(coeff)-b)

def arb_loss(coeff, d_sa, q_k, pi1_k, pi2_k):
    total_states = env1.observation_space.n
    total_actions = env1.action_space.n
    loss = 0.
    for s in range(total_states):
        coeff1_s, coeff2_s = coeff[env1.cur_state], coeff[env1.cur_state+total_states]
        for a in range(total_actions):
            pi_arb = coeff1_s*pi1_k[s,a] + coeff2_s*pi2_k[s,a]
            logpi_arb = 0 if (pi_arb<=0 or np.isinf(pi_arb) or np.isnan(pi_arb)) else np.log(pi_arb)
            loss += d_sa[s,a] * q_k[s,a] * logpi_arb
    
    return -loss


def solve_coeff(coeff1, coeff2, q_k, dsa_k, pi1_k, pi2_k):
    total_variables = env1.observation_space.n * 2
    constr = {'type':'eq', 'fun':coeff_constr}
    bnds = [(0, 1+1e-8)] * total_variables

    f0 = np.concatenate((coeff1,coeff2))

    res = optimize.minimize(arb_loss, x0=f0, args=(dsa_k, q_k, pi1_k, pi2_k), method='SLSQP', constraints=constr, bounds=bnds, options={'disp':False})
    obj_val = arb_loss(res.x, dsa_k, q_k, pi1_k, pi2_k)
    # print(obj_val)
    return res.x

def get_q(pi_k, gamma=0.9):
    total_states = env1.observation_space.n
    total_actions = env1.action_space.n

    coeffients = []
    for s in range(total_states):
        for a in range(total_actions):
            for s_ in range(total_states):
                for a_ in range(total_actions):
                    coeffients.append(env1.P[(a,s,s_)] * pi_k[s_,a_])

    I = np.eye(total_states*total_actions, total_states*total_actions)

    A = I - gamma * np.asarray(coeffients).reshape((total_states*total_actions, total_states*total_actions))

    R = []
    for s in range(total_states):
        for a in range(total_actions):
            val = 0
            for s_ in range(total_states):
                if s_ == env1.goal:
                    r = 10
                else:
                    r = 1
                val += env1.P[(a,s,s_)] * r
            R.append(val)  

    R = np.asarray(R).reshape((total_states*total_actions, 1))
    x = np.linalg.solve(A, R)
    x = np.asarray(x).reshape((total_states, total_actions))

    return x


def optimize_pi(pi_k, pi1_k, pi2_k, coeff1, coeff2):
    total_states = env1.observation_space.n
    q_k = get_q(pi_k)
    d_s, d_sa = get_occupancy_measure(pi_k)
    coeff = np.concatenate((coeff1, coeff2))
    obj_val = arb_loss(coeff, d_sa, q_k, pi1_k, pi2_k)
    coeff = solve_coeff(coeff1, coeff2, q_k, d_sa, pi1_k, pi2_k)
    return coeff[:total_states], coeff[total_states:]


def test_tab_arb(q1, q2, n_epi=60, max_steps=100, learn=True):
    returns = []
    coeff1, coeff2 = np.full(env1.observation_space.n, 0.5), np.full(env1.observation_space.n, 0.5)
    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        env1.reset()
        while (step < max_steps):
            pi1_k, pi2_k = get_pi(q1), get_pi(q2)
            pi_k = coeff1[env1.cur_state] * pi1_k + coeff2[env1.cur_state] * pi2_k
            pi_k, a_env = sample_pi(pi_k, env1.cur_state)
            s1, a1, s_1, r1, done1 = env1.step(a_env) #module1 and arb
            s2, a2, s_2, r2, done2 = env2.step(a_env)
            coeff1, coeff2 = optimize_pi(pi_k, pi1_k, pi2_k, coeff1, coeff2)
            cumulative_r += r1
            step += 1

        print("epi {}".format(epi))
        print("coeff1: ", coeff1)
        print("\n")

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