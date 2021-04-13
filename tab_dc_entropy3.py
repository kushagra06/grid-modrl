# from __future__ import print_function
import numpy as np
import gym
from scipy.special import entr
from scipy.stats import entropy
import gym_gridworlds

from scipy import optimize
import matplotlib.pyplot as plt 


def get_constraints_matrix(total_constraints, total_vars, vars_per_constraint):
	constraints_mat = []
	for i in range(0, total_vars, vars_per_constraint):
		row = list(np.zeros(total_vars))
		for j in range(i, i+4):
			row[j] = 1
		constraints_mat.append(row)

	return constraints_mat

def pi_constraints(pi):
	total_constraints = total_states
	vars_per_constraint = total_actions
	total_vars = total_states * total_actions
	b = np.ones(total_constraints)
	constraints_mat = get_constraints_matrix(total_constraints, total_vars, vars_per_constraint)
	A = np.array(constraints_mat)
	ret = A.dot(pi) - b
	return ret

def get_occupancy_measure(pi_k, gamma):
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

def pol_loss(pi, pi_k, q_k, ds_k, dsa_k, alpha):
	term1, term2 = 0.0, 0.0
	i, j = 0, 0
	for s in range(total_states):
		H_k = entr(pi_k[s]).sum()
		H = entr(pi[j:j+4]).sum()
		logH = 0 if (H<=0 or np.isinf(H) or np.isnan(H)) else np.log(H)
		logH_k = 0 if (H_k<=0 or np.isinf(H_k) or np.isnan(H_k)) else np.log(H_k)
		term2 += alpha * (logH - logH_k) * ds_k[s] * H_k 
		j += 4
		for a in range(total_actions):
			logpi = 0 if pi[i]==0 else np.log(pi[i])
			logpi_k = 0 if pi_k[s][a]==0 else np.log(pi_k[s][a])
			term1 += (logpi - logpi_k) * dsa_k[s][a] * (q_k[s][a] + alpha * H_k)
			i += 1
	
	return -(term1 + term2)

def pol_solve(pi_k, q_k, gamma, alpha, disp_val): 	
	total_vars = total_states * total_actions
	total_constraints = total_states
	vars_per_constraint = total_actions 
	ds_k, dsa_k = get_occupancy_measure(pi_k, gamma)
	# pi0 = pi_k	
	pi0 = pi_k.flatten()
	cons = {'type':'eq', 'fun':pi_constraints}
	bnds = [(0+1e-8, 1+1e-8)] * total_vars
	
	res = optimize.minimize(pol_loss, x0=pi0, args=(pi_k, q_k, ds_k, dsa_k, alpha), method='SLSQP', constraints=cons, bounds=bnds, options={'disp':disp_val})
	
	obj_val = pol_loss(res.x, pi_k, q_k, ds_k, dsa_k, alpha)

	new_pi = np.reshape(res.x, (total_states, total_actions))

	return new_pi, obj_val

def kl_div(pi, q_k, alpha):
	entropy_term = entr(pi).sum()
	# p2 = np.exp(q_k) #/ np.sum(np.exp(q_k))
	# kl = entropy(pi, p2)
	cross_entropy_term = - np.sum(pi * (1./alpha) * q_k)
	return (-entropy_term + cross_entropy_term)

def sum_pi_cons(pi):
	return (np.sum(pi) - 1.0)

def soft_pol_impr(pi_k, q_k, gamma, alpha, disp_val):
	new_pi = np.zeros([total_states, total_actions])
	bnds = [(0+1e-8, 1+1e-8)] * total_actions
	cons = {'type':'eq', 'fun':sum_pi_cons}

	total_kl_obj = 0
	for s in range(total_states):
		res = optimize.minimize(kl_div, x0=pi_k[s], args=(q_k[s], alpha), method='SLSQP', constraints=cons, bounds=bnds, options={'disp':disp_val})
		new_pi[s] = res.x
		total_kl_obj += kl_div(new_pi[s], q_k, alpha) 
	
	return new_pi, total_kl_obj

######## can be formulated as a LP ###############
def soft_pol_eval(pi, gamma, eps, alpha):
	v = np.zeros(total_states)
	q = np.zeros([total_states, total_actions])

	# while True:
	# 	delta = 0
	# 	for s in range(total_states):
	# 		v_val = 0
	# 		logpi = np.zeros(total_actions)
	# 		for a in range(total_actions):
	# 			logpi[a] = 0 if pi[s,a]<=0 else np.log(pi[s,a])
	# 			q_val = 0
	# 			for (s_, p) in enumerate(env.P[a][s]):
	# 				r = 1 if s==0 else 0
	# 				q_val += p * (r + gamma * v[s_])
	# 			q[s][a] = q_val
			
	# 		for a in range(total_actions):
	# 			v_val += pi[s,a] * (q[s][a] - alpha * logpi[a])

	# 		delta = max(delta, np.abs(v_val - v[s]))

	# 		v[s] = v_val
		
	# 	if delta < eps:
	# 		break

	# return q, v

	while True:
		delta = 0
		for s in range(total_states):
			for a in range(total_actions):
				q_val = 0
				for (s_, p) in enumerate(env.P[a][s]):
					r = 1 if s==0 else 0
					q_val += p * (r + gamma * v[s_])
				q[s][a] = q_val
				
		for s in range(total_states):
			v_val = 0
			for a in range(total_actions):
				logpi = 0 if pi[s,a]<=0 else np.log(pi[s,a])
				v_val += pi[s,a] * (q[s][a] - alpha * logpi)
			delta = max(delta, np.abs(v_val - v[s]))
			v[s] = v_val

		if delta < eps: 
			break

	return q, v


def get_mdp_obj(pi, gamma, alpha):
	mdp_obj = 0
	ds, dsa = get_occupancy_measure(pi, gamma)
	for s in range(total_states):
		H = entr(pi[s]).sum()
		r = 1 if s==0 else 0
		for a in range(total_actions):
			mdp_obj += dsa[s][a] * (r + alpha * H)

	return mdp_obj

def solve(gamma, n_itr, tol_norm, tol_obj, alpha, eps, disp_val, soft_pol_impr_flag, print_always):
	pi = np.ones([total_states, total_actions])/total_actions 
	b0 = np.ones(total_states) / total_states
	pol_obj_vals = []
	mdp_obj_vals = []
	oldpol_obj_val = 1e5
	for itr in range(n_itr):
		
		q, v = soft_pol_eval(pi, gamma, eps, alpha)
		
		if soft_pol_impr_flag: #false by default
			new_pi, pol_obj_val = soft_pol_impr(pi, q, gamma, alpha, disp_val)
		else:
			new_pi, pol_obj_val = pol_solve(pi, q, gamma, alpha, disp_val)
		
		mdp_obj = get_mdp_obj(pi, gamma, alpha)

		if print_always:
			print("Iteration {}".format(itr))
			# print("pi: ",pi)
			# print("new_pi: ", new_pi)
			print("DC objective value: ", pol_obj_val)
			print("MDP objective value (from occupancy measure): ", mdp_obj)
			# print("MDP objective value (from value function): ", np.sum(v * b0))
			print("\n")

		pol_obj_vals.append(pol_obj_val)
		mdp_obj_vals.append(mdp_obj)
		
		if itr%(n_itr/10)==0 and not print_always:
			print("Iteration {}/{}".format(itr, n_itr))

		if abs(pol_obj_val - oldpol_obj_val) <= tol_obj:
			print('Converged at iteration {}'.format(itr))
			break
		oldpol_obj_val = pol_obj_val

		# if np.linalg.norm(pi-new_pi)  <= tol_norm:
		# 	print('Converged at itrsode %d' %(itr+1))
		# 	break
		
		pi = np.copy(new_pi)

	return pi, q, v, pol_obj_vals, mdp_obj_vals


def best_pol_directions(pi):
	d = {}
	d[0] = 'UP'
	d[1] = 'RIGHT'
	d[2] = 'DOWN'
	d[3] = 'LEFT'
	directions = []
	for s in range(total_states):
		a = np.argmax(pi[s])
		directions.append(d[a])

	return directions

def plotting_util(pol_obj_vals, figname, figdir, xlabel, ylabel, dosave):
	fig, ax = plt.subplots()
	ax.plot(pol_obj_vals)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if dosave:
		plt.savefig(figdir+figname)

def main(args):
	opt_pi, opt_q, opt_v, pol_obj_vals, mdp_obj_vals = solve(args.gamma, args.n_itr, args.tol_norm, args.tol_obj, 
		float(args.alpha), args.eps, args.disp_val, args.soft_pol_impr, args.print_always)
	
	print("OPT POLICY: ", opt_pi)
	print("OPT Q: ", opt_q)
	print("OPT V: ", opt_v)
	print("\n")

	best_directions = best_pol_directions(opt_pi)
	for (s, d) in enumerate(best_directions):
		print("State {} : {}".format(s,d))

	if float(args.alpha) == 0:
		figname_dc = "dc_obj_noH.png"
		figname_mdp = "mdp_obj_noH.png"
	else:
		if args.soft_pol_impr:
			figname_dc = "kl_obj_H.png"
			figname_mdp = "mdp_obj_H_kl.png"
		else:
			figname_dc = "dc_obj_H.png"
			figname_mdp = "mdp_obj_H.png"
	
	plotting_util(pol_obj_vals, figname_dc, args.figdir, xlabel="Number of iterations", ylabel="DC (policy) objective", dosave=True)
	plotting_util(mdp_obj_vals, figname_mdp, args.figdir, xlabel="Number of iterations", ylabel="MDP objective", dosave=True)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="DC-Entropy in tabular setting (15 state gridworld)")
	parser.add_argument('--env', help="Environment", type=str, default='Gridworld-v0')
	parser.add_argument('--n_itr', help="Total number of iterations for the program", default=100)
	parser.add_argument('--gamma', help="Discount factor", default=0.90)
	parser.add_argument('--tol_norm', help="Stopping criterion for the program using policy norm", default=1e-3)
	parser.add_argument('--tol_obj', help="Stopping criterion for the program using DC objective", default=1e-3)
	parser.add_argument('--alpha', help="Temperature for entropy", default=1)
	parser.add_argument('--eps', help="Stopping criterion for policy evaluation", default=1e-5)
	parser.add_argument('--disp_val', help="Display DC objective value/status", default=False)
	parser.add_argument('--soft_pol_impr', help="Do soft policy improvement using KL divergence", default=False)
	parser.add_argument('--print_always', help="Print objective values at each iteration", default=True)
	parser.add_argument('--figdir', help="Directory to save plots (as a string)", required=True)

	args = parser.parse_args()
	env = gym.make(args.env)
	total_states, total_actions = env.observation_space.n, env.action_space.n
	np.random.seed(0)

	main(args)
