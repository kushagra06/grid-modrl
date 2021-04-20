import numpy as np
from scipy import optimize 
from grid import GridEnv
from gridAgents import PolMixAgent, QModule, PolMixModule
import matplotlib.pyplot as plt
import tensorflow as tf
from columnar import columnar


def run(agent, n_epi=500, max_steps=500, learn=True):
    returns = []

    for epi in range(n_epi):
        path = [0]
        cumulative_r = 0
        step = 0
        epi_over = False
        agent.env.reset()
        # while (not epi_over) and (step < max_steps):
        while (not epi_over) and (step < max_steps):
            a = agent.get_action(agent.env.cur_state)
            s, a, new_s, r, done = agent.env.step(a)
    
            path.append(new_s)

            if learn == True:
                agent.update(s, a, new_s, r)
            
            cumulative_r += r
            step += 1
            
            if done == 1:
                epi_over = True

          
        agent.eps *= 0.99
        returns.append(cumulative_r)
        # print(path)
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

def get_greedy_pi(q):
    pi = np.zeros([env1.observation_space.n, env1.action_space.n])
    for s in range(env1.observation_space.n):
        a = np.argmax(q[s])
        pi[s,a] = 1
    
    return pi

def get_pi(q_list, total_modules):
    pi_list = []
    for m in range(total_modules):
        pi = []
        for s in range(env1.observation_space.n):
            pi_s = softmax(q_list[m][s])
            pi.append(pi_s)
        pi = np.asarray(pi)
        pi_list.append(pi)
    
    return np.asarray(pi_list)

def sample_pi(pi, cur_state):
    return pi, np.random.choice(4, p=pi[cur_state])

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
    for i in range(total_constraints):
        for m in range(total_modules):
            A[i][i+m*total_constraints] = 1
    return A

def coeff_constr(coeff):
    total_constraints = env1.observation_space.n 
    total_vars = env1.observation_space.n * total_modules
    b = np.ones(total_constraints)
    A = get_constraints_matrix(total_vars, total_constraints)
    return (A.dot(coeff)-b)

def arb_loss(coeff, dsa_k, q_k, pi_array):
    total_variables = env1.observation_space.n * total_modules
    total_states = env1.observation_space.n
    total_actions = env1.action_space.n
    loss = 0.
    coeff_s = np.zeros(total_modules)
    
    for s in range(total_states):       
        i = 0
        indices = list(range(s,total_variables-total_states+s+1, total_states))
        for a in range(total_actions):
            pi_arb = np.sum(coeff[indices] * pi_array[:,s,a])
            logpi_arb = 0 if (pi_arb<=0 or np.isinf(pi_arb) or np.isnan(pi_arb)) else np.log(pi_arb)
            loss += dsa_k[s,a] * q_k[s,a] * logpi_arb
    return -loss


def solve_coeff(coeff, q_k, dsa_k, pi_array):
    total_variables = env1.observation_space.n * total_modules
    constr = {'type':'eq', 'fun':coeff_constr}
    bnds = [(0., 1)] * total_variables

    init_guess = np.concatenate(coeff)
    res = optimize.minimize(arb_loss, x0=init_guess, args=(dsa_k, q_k, pi_array), method='SLSQP', constraints=constr, bounds=bnds, options={'disp':False})
    # obj_val = arb_loss(res.x, dsa_k, q_k, pi_array)
    
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
                if s == s_ and s == env1.goal:
                    r = 10
                else:
                    r = 1
                val += env1.P[(a,s,s_)] * r ## reward signal for the arb = global r
            R.append(val)  

    R = np.asarray(R).reshape((total_states*total_actions, 1))
    x = np.linalg.solve(A, R)
    x = np.asarray(x).reshape((total_states, total_actions))

    return x


def optimize_pi(pi_k, pi_array, coeff):
    q_k = get_q(pi_k)
    d_s, d_sa = get_occupancy_measure(pi_k)
    coeff = solve_coeff(coeff, q_k, d_sa, pi_array)
    
    return np.reshape(coeff, newshape=(total_modules, env1.observation_space.n))


def get_init_coeff(total_modules):
    coeff = np.full((total_modules, env1.observation_space.n,), fill_value=1./total_modules)
    return coeff

def test_tab_arb(q_list, n_epi=1, max_steps=500, learn=True):
    returns = []
<<<<<<< HEAD
    coeff1, coeff2 = np.full(env1.observation_space.n, 0.5), np.full(env1.observation_space.n, 0.5)
    pi1_k, pi2_k = get_pi(q1), get_pi(q2)
=======
    coeff = get_init_coeff(total_modules)
    pi_array = get_pi(q_list, total_modules)
>>>>>>> 35dba4e140b8c832b60288a517bc50b491d42ad3
    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        for m in range(total_modules):
            env_list[m].reset()

        pi_k = np.zeros((env1.observation_space.n, env1.action_space.n))
        for i in range(total_modules):
            pi_k += coeff[i][env1.cur_state] * pi_array[i]

        # run an episode with pi_k
        while (not epi_over) and (step < max_steps):
            a_env = np.random.choice(4, p=pi_k[env1.cur_state]) # sample from pi_k
            s1, a1, s1_, r1, done1 = env1.step(a_env) # module1 and arb same
            if s1==s1_ and s1==env1.goal:
                r1 = 10
            else:
                r1 = 1

            cumulative_r += r1
            step += 1

            if done1:
                epi_over = True
        
        coeff = optimize_pi(pi_k, pi_array, coeff)
        
        # if epi%10 == 0:
        print("Done: {}, cumulative_r: {}, coeff1: {}\n".format(epi, cumulative_r, coeff[0]))
        returns.append(cumulative_r)

<<<<<<< HEAD
    np.save('coeff1', coeff1)
    np.save('coeff2', coeff2)
=======
    # np.save('coeff1_greedy', coeff1)
    # np.save('coeff2_greedy', coeff2)
>>>>>>> 35dba4e140b8c832b60288a517bc50b491d42ad3
    return returns


def main():
    np.set_printoptions(precision=4, suppress=True)

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

    # agent3 = QModule(env3)
    # agent4 = QModule(env4)
    # returns3, returns4 = run(agent3), run(agent4)
    # np.save('m3_q3', agent3.q)
    # np.save('m4_q4', agent4.q)
    # plt.plot(returns3)
    # plt.show()
    # plt.plot(returns4)
    # plt.show()
    # q3, q4 = np.load('m3_q3.npy'), np.load('m4_q4.npy')
    # pi3, pi4 = get_pi(q3), get_pi(q4)
    # print("pi3")
    # print(pi3)
    # print("\n")
    # print("pi4")
    # print(pi4)

    # pol_mix_agent = PolMixAgent(env1, env2)
    # arb_agent = PolMixModule(env1)
    # arb_returns = test_arb(arb_agent)
    # print(arb_returns)

    q1 = np.load('m1_q1.npy')
    q2 = np.load('m2_q2.npy')
<<<<<<< HEAD
    # tab_arb_returns = test_tab_arb(q1, q2)
    # plt.plot(tab_arb_returns)
    # plt.show()
    pi1 = get_pi(q1)
    pi2 = get_pi(q2)
    coeff1 = np.load('coeff1.npy')
    coeff2 = np.load('coeff2.npy')

    np.set_printoptions(precision=2, suppress=True)

    print("pi1")
    print(pi1)
    print("\n")

    print("pi2")
    print(pi2)
    print("\n")

    pi_arb = np.empty((16, 4))
    for s in range(16):
        pi_arb[s] = coeff1[s]*pi1[s] + coeff2[s]*pi2[s]
    print("pi_arb")
    print(pi_arb)

    np.save('m1_pi_greedy', pi1)
    np.save('m2_pi_greedy', pi2)
    np.save('pi_arb_greedy', pi_arb)
=======
    q3 = np.load('m3_q3.npy')
    q4 = np.load('m4_q4.npy')
    tab_arb_returns = test_tab_arb([q1, q2])
    plt.plot(tab_arb_returns)
    plt.show()
    # pi1 = get_pi(q1)
    # pi2 = get_pi(q2)
    # pi3 = get_pi(q3)
    # pi4 = get_pi(q4)
    
    # coeff1 = np.load('coeff1.npy')
    # coeff2 = np.load('coeff2.npy')


    # # headers_pi = ["LEFT", "UP", "RIGHT", "DOWN"]
    # # table_pi1 = columnar(pi1.tolist(), headers_pi, no_borders=True)
    # # table_pi2 = columnar(pi2.tolist(), headers_pi, no_borders=True)
    # # pi12 = np.hstack((pi1, pi2))
    # # headers_pi12 = ["LEFT", "UP", "RIGHT", "DOWN", "LEFT", "UP", "RIGHT", "DOWN"]
    # # table_pi12 = columnar(pi12.tolist(), headers_pi12)
    # # print(table_pi12)
    
    # print("pi1")
    # print(pi1)
    # print("\n")

    # print("pi2")
    # print(pi2)
    # print("\n")

    # pi_arb = np.empty((16, 4))
    # for s in range(16):
    #     pi_arb[s] = coeff1[s]*pi1[s] + coeff2[s]*pi2[s]
    # print("pi_arb")
    # print(pi_arb)

    # np.save('m1_pi_greedy', pi1)
    # np.save('m2_pi_greedy', pi2)
    # np.save('pi_arb_greedy', pi_arb)
>>>>>>> 35dba4e140b8c832b60288a517bc50b491d42ad3

if __name__ == "__main__":
    env1 = GridEnv(goal=15)
    env2 = GridEnv(goal=12)
    env3 = GridEnv(goal=3)
    env4 = GridEnv(goal=9)
    env_list = [env1, env2]#, env3]#, env4]
    total_modules = len(env_list)
    main()


## Notes ##
# Beware of numerical errors while initializing np arrays
# Use np arrays everywhere (instead of lists)
# Values inside log
# Bounds should be properly set