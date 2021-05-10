import numpy as np
from scipy import optimize 
from grid import GridEnv
from gridAgents import QModule
import matplotlib.pyplot as plt
import tensorflow as tf
import time
# from columnar import columnar


ALPHA_ARB = 0.1
GAMMA = 0.9

def run(agent, n_epi=200, max_steps=500, learn=True):
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
    
    for s in range(total_states):       
        i = 0
        ## e.g. for 4 modules, for s=0, indices=[0,16,32,48] -> 0 to 48 in steps of 16
        indices = list(range(s, total_variables-total_states+s+1, total_states)) 
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

    coefficients = []
    for s in range(total_states):
        for a in range(total_actions):
            for s_ in range(total_states):
                for a_ in range(total_actions):
                    coefficients.append(env1.P[(a,s,s_)] * pi_k[s_,a_])

    I = np.eye(total_states*total_actions, total_states*total_actions)

    A = I - gamma * np.asarray(coefficients).reshape((total_states*total_actions, total_states*total_actions))

    R = []
    for s in range(total_states):
        for a in range(total_actions):
            val = 0
            for s_ in range(total_states):
                r = 10 if (s==s_ and s==env1.goal) else 1
                # if s == s_ and s == env1.goal:
                #     r = 10
                # else:
                #     r = 1
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


def get_init_coeff(total_modules, arb_env):
    coeff = np.full((total_modules, arb_env.observation_space.n,), fill_value=1./total_modules)
    # coeff[0][:] = 0.01
    # coeff[1][:] = 0.99 
    return coeff


def value_iteration(env, eps=1e-05, gamma=0.9):
    total_states, total_actions = env.observation_space.n, env.action_space.n
    q = np.zeros((total_states, total_actions))
    v = np.zeros(total_states)

    while True:
        delta = 0.
        for s in range(total_states):
            v_old = v[s]
            for a in range(total_actions):
                val = 0
                q_old = q[s,a]
                for s_ in range(total_states):
                    r = 10 if (s==s_ and s==env.goal) else 1
                    val += env.P[a,s,s_] * (r + gamma * np.max(q[s_]))
                delta = max(delta, abs(q[s,a]-val))
                q[s,a] = val
            # v[s] = np.max(q[s])
            # delta = max(delta, np.abs(v[s]-v_old))
        if delta < eps:
            break
    
    return q

def learn_mod_arb(env_list, n_epi=100, max_steps=500):
    mods_agents = []
    returns_mods = []
    for _ in range(total_modules):
        returns_mods.append([])
    returns_arb = []
    arb_env = env_list[0]
    coeff = get_init_coeff(total_modules, arb_env)
    for i in range(1, len(env_list)):
        mods_agents.append(QModule(env_list[i]))
    print("total_modules: ", total_modules)

    for epi in range(n_epi):
        step = 0
        cumulative_r_arb = 0.
        epi_over = False
        for env in env_list:
            env.reset()
        
        q_list = []
        for i in range(len(mods_agents)):
            mods_agents[i].reset()
            q_list.append(mods_agents[i].q)
        
        pi_array = get_pi(q_list, total_modules)
        pi_arb = np.zeros((arb_env.observation_space.n, arb_env.action_space.n))
        for s in range(arb_env.observation_space.n):
            for m in range(total_modules):
                pi_arb[s] += coeff[m][s] * pi_array[m][s]
            pi_arb[s] = pi_arb[s]/np.sum(pi_arb[s])
        
        while (not epi_over) and (step < max_steps):
            a_arb = np.random.choice(4, p=pi_arb[arb_env.cur_state])
            s_arb, a_arb, new_s_arb, r_arb, done_arb = arb_env.step(a_arb)
            r_arb = 10. if (s_arb == new_s_arb and s_arb == arb_env.goal) else 1.
            cumulative_r_arb += r_arb
            step += 1

            for m in range(total_modules):
                # a = mods_agents[m].get_action(mods_agents[m].env.cur_state)
                s, a, s_, r, done = mods_agents[m].env.step(a_arb)
                mods_agents[m].update(s, a, s_, r)
                mods_agents[m].cumulative_r += r

            if done_arb:
                epi_over = True
        
        # if epi>=1:
        new_coeff = optimize_pi(pi_arb, pi_array, coeff)
        coeff = (1. - ALPHA_ARB) * coeff + ALPHA_ARB * new_coeff
        # coeff = coeff + ALPHA_ARB * (GAMMA * new_coeff - coeff)
        if epi%9 == 0:
            print("Done: {}, cumulative_r: {}, coeff1: {}\n".format(epi, cumulative_r_arb, coeff[0]))
        
        returns_arb.append(cumulative_r_arb)
        for m in range(total_modules):
            mods_agents[m].eps *= 0.99
            returns_mods[m].append(mods_agents[m].cumulative_r)

    return returns_arb, returns_mods


def test_tab_arb(q_list, arb_env, n_epi=20, max_steps=500):
    returns = []
    coeff = get_init_coeff(total_modules, arb_env)
    pi_array = get_pi(q_list, total_modules)
    print("total_modules", total_modules)
    for epi in range(n_epi):
        cumulative_r = 0
        step = 0
        epi_over = False
        arb_env.reset()      
        # run an episode with pi_k
        while (not epi_over) and (step < max_steps):
            pi_k = np.zeros((arb_env.observation_space.n, arb_env.action_space.n))
            for i in range(total_modules):
                pi_k += coeff[i][arb_env.cur_state] * pi_array[i]
            a_env = np.random.choice(4, p=pi_k[arb_env.cur_state]) # sample from pi_k
            s1, a1, s1_, r1, done1 = arb_env.step(a_env) # module1 and arb same
            r1 = 10. if (s1==s1_ and s1==arb_env.goal) else 1.

            cumulative_r += r1
            step += 1

            if done1:
                epi_over = True
        
        coeff = optimize_pi(pi_k, pi_array, coeff)
        
        # if epi%10 == 0:
        print("Done: {}, cumulative_r: {}, coeff1: {}\n".format(epi, cumulative_r, coeff[0]))
        returns.append(cumulative_r)

    # np.save('coeff1', coeff1)
    # np.save('coeff2', coeff2)
    return returns


def main():
    np.set_printoptions(precision=4, suppress=True)

    # q4_1_vi = value_iteration(env1)
    # env4_2 = GridEnv(goal=12)
    # q4_2_vi = value_iteration(env4_2)
    # np.save('m1_q_4_vi', q4_1_vi)
    # np.save('m2_q_4_vi', q4_2_vi)
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
    # agent5 = QModule(env5)
    # returns5 = run(agent5)
    # np.save('m5_q5', agent5.q)

    # env6_1 = GridEnv(grid_size=6, goal=35)
    # agent6_1 = QModule(env6_1)
    # returns6_1 = run(agent6_1)
    # np.save('m1_q_6', agent6_1.q)

    # env6_2 = GridEnv(grid_size=6, goal=10)
    # agent6_2 = QModule(env6_2)
    # returns6_2 = run(agent6_2)
    # np.save('m2_q_6', agent6_2.q)

    # env5_1 = GridEnv(grid_size=5, goal=24)
    # q5_1_vi = value_iteration(env5_1)
    # np.save('m1_q_5_vi', q5_1_vi)

    # pi5_1_vi = get_pi([q5_1_vi], 1)
    # print(pi5_1_vi)
    # print("\n")
    # q5_1 = np.load('m1_q_5.npy')
    # pi5_1 = get_pi([q5_1], 1)
    # print(pi5_1) 
    # agent5_1 = QModule(env5_1)
    # returns5_1 = run(agent5_1)
    # # np.save('m1_q_5', agent5_1.q)

    # env5_2 = GridEnv(grid_size=5, goal=12)
    # q5_2_vi = value_iteration(env5_2)
    # print(q5_2_vi)
    # print("\n")
    # pi5_2_vi = get_pi([q5_2_vi], 1)
    # print(pi5_2_vi)
    # np.save('m2_q_5_vi', q5_2_vi)
    # agent5_2 = QModule(env5_2)
    # returns5_2 = run(agent5_2)
    # np.save('m2_q_5', agent5_2.q)
    
    # env5_3 = GridEnv(grid_size=5, goal=17)
    # agent5_3 = QModule(env5_3)
    # returns5_3 = run(agent5_3)
    # np.save('m3_q_5', agent5_3.q)

    # env5_4 = GridEnv(grid_size=5, goal=20)
    # agent5_4 = QModule(env5_4)
    # returns5_4 = run(agent5_4)
    # np.save('m4_q_5', agent5_4.q)

    # env7_1 = GridEnv(grid_size=7, goal=48)
    # agent7_1 = QModule(env7_1)
    # returns7_1 = run(agent7_1)
    # np.save('m1_q_7', agent7_1.q)

    # env7_2 = GridEnv(grid_size=7, goal=24)
    # agent7_2 = QModule(env7_2)
    # returns7_2 = run(agent7_2)
    # np.save('m2_q_7', agent7_2.q)


    # q4_1_vi, q4_2_vi = np.load('m1_q_4_vi.npy'), np.load('m2_q_4_vi.npy')
    # q5_1_vi, q5_2_vi = np.load('m1_q_5_vi.npy'), np.load('m2_q_5_vi.npy')
    # q7_1 = np.load('m1_q_7.npy')
    # q7_2 = np.load('m2_q_7.npy')
    # q5_1 = np.load('m1_q_5.npy')
    # q5_2 = np.load('m2_q_5.npy')
    # q5_3 = np.load('m3_q_5.npy')
    # q5_4 = np.load('m4_q_5.npy')
    # q6_1 = np.load('m1_q_6.npy')
    # q6_2 = np.load('m2_q_6.npy')
    # q1 = np.load('m1_q1.npy')
    # q2 = np.load('m2_q2.npy')
    # q3 = np.load('m3_q3.npy')
    # q4 = np.load('m4_q4.npy')
    # q5 = np.load('m5_q5.npy')
    # q_rand = np.random.rand(env1.observation_space.n, env1.action_space.n)
    env_list = [env1, env1, env2] # index0 -> arb
    start = time.time()
    # tab_arb_returns = test_tab_arb([q4_1_vi, q4_2_vi], arb_env=env1)#q5_2, q5_3, q5_4], arb_env=env1)#, q_rand, q_rand, q_rand])
    returns_arb, returns_mods = learn_mod_arb(env_list)
    end = time.time()
    print("Total time taken: ", end-start)
    plt.plot(returns_arb)
    plt.show()
    
    # pi1 = get_pi(q1)
    # pi2 = get_pi(q2)
    # coeff1 = np.load('coeff1.npy')
    # coeff2 = np.load('coeff2.npy')  

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

if __name__ == "__main__":
    env1 = GridEnv(goal=15)
    env2 = GridEnv(goal=12)
    # env3 = GridEnv(goal=3)
    # env4 = GridEnv(goal=9)
    # env5 = GridEnv(goal=10)
    # env_list = [env1, env2, env3, env4, env5]
    # env1 = GridEnv(grid_size=7, goal=48)
    # env1 = GridEnv(grid_size=6, goal=35)
    # env1 = GridEnv(grid_size=5, goal=24)
    # env2 = GridEnv(grid_size=5, goal=12)
    # env3 = GridEnv(grid_size=5, goal=8)
    total_modules = 2
    main()


## Notes ##
# Beware of numerical errors while initializing np arrays
# Use np arrays everywhere (instead of lists)
# Values inside log
# Bounds should be properly set