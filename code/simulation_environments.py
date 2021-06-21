from gurobipy import *
import numpy as np 
import sys
import time
from numba import jit


def make_T_from_q(Q):
	
	T = np.zeros(Q.shape)

	T[:,:,-1] = Q[:,:,-1]
	for i in list(range(Q.shape[2]-1))[::-1]:
		T[:,:,i] = Q[:,:,i] - Q[:,:,i+1]
	
	return T


# Check that a T respects:
# - Q is increasing in s wrt to a and k
# - Q is increasing in a wrt to s and k
def check_T_strict(T):
	S = T.shape[0]
	A = T.shape[1]
	Q = np.zeros((S,A,S))

	for s in range(S):
		for a in range(A):
			for k in range(S):
				Q[s,a,k] = T[s,a,k:].sum()


	# Covers the p11 > p01
	for k in range(S):
		for a in range(A):

			non_decreasing_in_S = True
			previous_value = 0
			for s in range(S):
				non_decreasing_in_S = Q[s,a,k] >= previous_value
				if not non_decreasing_in_S:
					return False
				previous_value = Q[s,a,k]

	# Ensure that action effects 
	# does this check preclude the first? No.
	# I think this covers p11a > p11p but need to verify
	for s in range(S):
		for k in range(S):

			non_decreasing_in_a = True
			previous_value = 0
			for a in range(A):
				non_decreasing_in_a = Q[s,a,k] >= previous_value
				if not non_decreasing_in_a:
					return False
				previous_value = Q[s,a,k]


	return True


# Check that a T respects:
# - Q is increasing in s wrt to a and k
def check_T_puterman(T):
	S = T.shape[0]
	A = T.shape[1]
	Q = np.zeros((S,A,S))

	for s in range(S):
		for a in range(A):
			for k in range(S):
				Q[s,a,k] = T[s,a,k:].sum()


	# Covers the p11 > p01
	for k in range(S):
		for a in range(A):

			non_decreasing_in_S = True
			previous_value = 0
			for s in range(S):
				non_decreasing_in_S = Q[s,a,k] >= previous_value
				if not non_decreasing_in_S:
					return False
				previous_value = Q[s,a,k]

	return True

def no_check(T):
	return True

def random_T(S,A,check_function=check_T_strict):

	T = None
	T_passed_check = False
	count_check_failures = 0
	while (not T_passed_check):
		count_check_failures += 1
		if count_check_failures %1000 == 0:
			print('count_check_failures:',count_check_failures)
		T = np.random.dirichlet(np.ones(S), size=(S,A))
		T_passed_check = check_function(T)

	return T






def get_full_random_experiment(N, S, A, REWARD_BOUND):

	T = np.zeros((N,S,A,S))
	for i in range(N):
		T[i] = random_T(S,A,check_function=no_check)

	R = np.sort(np.random.rand(N, S), axis=1)*REWARD_BOUND
	# R = np.array([np.arange(S) for _ in range(N)])


	C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

	B = N/2*A


	return T, R, C, B

def jitter(T):
	for i in range(T.shape[0]):
		for j in range(T.shape[1]):
			noise = np.random.rand(T.shape[2])*0.05
			T[i,j] = T[i,j]+noise
			T[i,j] = T[i,j] / T[i,j].sum()
	return T




# Process needs to be acted on all the time to stay adhering
def get_needy_eng14(S,A,recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

	T = np.zeros((S,A,S))

	
	# process needs to be acted on all the time to stay adhering
	S = 2
	
	s = 0
	for a in range(0,A):
		pr = recovery_prob*(a+1)/(a+2)
		T[s,a] = [1.0-pr, pr]


	s = 1
	for a in range(0,A):
		pf  = 0
		if a == 1:
			pf= 1-(0.75/2+0.25)
		elif a == 2:
			pf = 1-0.99
		if a == 0:
			pf = fail_prob_passive

		T[s,a] = [pf, 1-pf]

	return T

# These processes improve when acted on, drop when not
def get_reliable_eng14(S,A, p00passive = 0.75, p01active=0.75, p10passive=0.25, p11active=0.85, base_action_effect=0.25 ):

	T = np.zeros((S,A,S))

	
	# process does better when acted on, worse when not
	S = 2
	
	s = 0
	a = 0
	T[s,a] = [p00passive, 1-p00passive]

	for a in range(1,A):
		pr  = 0
		if a == 1:
			pr = 0.75 + 0.125
		elif a == 2:
			pr = 0.99
		T[s,a] = [1.0-pr, pr]


	s = 1
	a = 0
	T[s,a] = [p10passive, 1-p10passive]

	for a in range(1,A):
		pr  = 0
		if a == 1:
			pr = 0.75 + 0.125
		elif a == 2:
			pr = 0.99
		T[s,a] = [1.0-pr, pr]

	return T


def get_eng14_experiment(N, A, percent_greedy, REWARD_BOUND, 
	recovery_prob=0.4, fail_prob_passive=0.75, fail_prob_act=0.4,
	p00passive = 0.95, p01active=0.60, p10passive=0.25, p11active=0.6, base_action_effect=0.4):

	S = 2
	A = 3
	T = np.zeros((N,S,A,S))

	num_greedy = int(N*percent_greedy)
	for i in range(num_greedy):
		#cliff, no ladder
		T[i] = get_needy_eng14(S, A, recovery_prob=recovery_prob, fail_prob_passive=fail_prob_passive, fail_prob_act=fail_prob_act)
		# print("getting nonrecov")
	for i in range(num_greedy, N):

		#cliff with ladder
		T[i] = get_reliable_eng14(S,A, p00passive=p00passive, p01active=p01active,
	 p10passive=p10passive, p11active=p11active, base_action_effect=base_action_effect)
		# print("getting good on their own")


	R = np.array([np.zeros(S) for _ in range(N)])

	# set rewards
	R[:,0] = 0
	R[:,1] = 1


	print(T)
	# 1/0
	C = np.arange(A)
	# np.save('eng_14_N%s_rewards.npy'%N,R)
	# 1/0

	return T, R, C



# Patient needs to be acted on all the time to stay adhering
def get_needy_eng15_history(history_length, A, base_states, recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

	HL = history_length
	num_states = 2**HL

	state_effect = 0.05
	weights = np.linspace(1,2, history_length)
	state_weights = weights*state_effect/weights.sum()

	T = np.zeros((num_states,A,num_states))

	# now to enumerate all possible states transitions...
	historied_states = [seq for seq in itertools.product((0,1), repeat=HL)]
	state_dict = dict([(state, i) for i,state in enumerate(historied_states)])

	for current_state in historied_states:
		
		current_state_ind = state_dict[current_state]

		active_base_state = current_state[-1]

		current_state_shifted_by_1 = list(current_state[1:])
		for next_base_state in base_states:
			next_historied_state = tuple(current_state_shifted_by_1+[next_base_state])
			next_state_ind = state_dict[next_historied_state]
			state_bonus = state_weights.dot(current_state)
			for a in range(0,A):
				if active_base_state == 0 and next_base_state == 0:
					pr = recovery_prob*(a+1)/(a+2)
					pr += state_bonus
					T[current_state_ind, a, next_state_ind] = 1 - pr 
				elif active_base_state == 0 and next_base_state == 1:
					pr = recovery_prob*(a+1)/(a+2)
					pr += state_bonus
					T[current_state_ind, a, next_state_ind] = pr

				elif active_base_state == 1 and next_base_state == 0:

					pf  = 0
					if a == 1:
						pf= 1-(0.75/2+0.25)
					elif a == 2:
						pf = 1-0.94
					if a == 0:
						pf = fail_prob_passive
					pf -= state_bonus

					T[current_state_ind, a, next_state_ind] = pf

				elif active_base_state == 1 and next_base_state == 1:

					pf  = 0
					if a == 1:
						pf= 1-(0.75/2+0.25)
					elif a == 2:
						pf = 1-0.94
					if a == 0:
						pf = fail_prob_passive
					pf -= state_bonus

					T[current_state_ind, a, next_state_ind] = 1-pf

	return T










@jit(nopython=True)
def collapse_history_matrix(T, history_length, shorter_length, longer_states, shorter_states):
	num_s_shorter = 2**shorter_length
	T_shorter = np.zeros((num_s_shorter, num_s_shorter))

	shorter_ind_list = []
	num_shorter_states = len(shorter_states)
	num_longer_states = len(longer_states)

	for j in range(num_shorter_states):
		shorter_state = shorter_states[j]
		inds_to_combine = []

		for i in range(num_longer_states):
			longer_state = longer_states[i]
			suffix = longer_state[-shorter_length:]
			# print(suffix)
			if np.array_equal(suffix,shorter_state):
				inds_to_combine.append(i)
				# print(long_state)
		# print()
		shorter_ind_list.append(inds_to_combine)

	shorter_ind_list=np.array(shorter_ind_list)

	for i in range(num_shorter_states):
		inds_i = shorter_ind_list[i]
		for j in range(num_shorter_states):
			inds_j = shorter_ind_list[j]
			collapsed_row = T[inds_i].sum(axis=0)
			collapsed_entry = collapsed_row[inds_j].sum()
			T_shorter[i,j] = collapsed_entry
	# print(T_shorter)
	for i in range(num_shorter_states):
		row = T_shorter[i]
		T_shorter[i] = row/row.sum()
	# print(T_shorter)


	return T_shorter


# @jit(nopython=True)
def advance_simulation(current_state, T):
	next_state=np.argmax(np.random.multinomial(1, T[current_state]))
	return next_state


def simulate_start_states(T, history_length):
	
	longer_states = np.array([seq for seq in itertools.product((0,1), repeat=history_length)])

	current_states = np.ones(T.shape[0],dtype=object)

	for shorter_length in range(1,history_length):
		shorter_states = [seq for seq in itertools.product((0,1), repeat=shorter_length)]
		shorter_state_dict = dict([(state, i) for i,state in enumerate(shorter_states)])

		for arm in range(T.shape[0]):
			T_arm = collapse_history_matrix(T[arm,:,0], history_length, shorter_length, longer_states, shorter_states)
			current_state_string = current_states[arm]
			if type(current_state_string) == int:
				current_state_string = (current_state_string,)
			current_state_index = shorter_state_dict[current_state_string]
			next_state_index = advance_simulation(current_state_index, T_arm)
			next_state_string = shorter_states[next_state_index]
			new_state_tuple = tuple(list(current_state_string) + list(next_state_string[-1:]))
			current_states[arm] = new_state_tuple

	longer_state_dict = dict([(tuple(state), i) for i,state in enumerate(longer_states)])
	current_states = np.array([longer_state_dict[current_state] for current_state in current_states])


	return current_states




def get_tb_patients_with_history(N, num_actions, history_length, action_diff_mult, REWARD_BOUND):


	fname = '../data/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy'%(N, num_actions, history_length, action_diff_mult)
	T = np.load(fname)

	# Get one reward if most recent state is adhering, 0 otherwise
	R_single = np.ones(2**history_length)
	inds = np.arange(2**(history_length-1))
	inds*=2
	R_single[inds] -= 1

	R = np.array([ R_single for _ in range(N)])

	# R = np.cumsum(R,axis=1)
	np.set_printoptions(threshold=np.inf)
	np.set_printoptions(linewidth=np.inf)


	start_states = simulate_start_states(T,history_length)



	C = np.arange(num_actions)
	B = N/5

	return T, R, C, B, start_states







def get_tb_patients_plus_needy_with_history(N, num_actions, history_length, action_diff_mult, REWARD_BOUND,
	percent_greedy, file_root=None,
	recovery_prob=0.4, fail_prob_passive=0.75, fail_prob_act=0.4):


	base_states = [0, 1]
	HL = history_length
	S = 2**HL
	

	# fname = '../data/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy'%(N, num_actions, history_length, action_diff_mult)
	fname = file_root+'/data/frequentist/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy'%(N, num_actions, history_length, action_diff_mult)
	T = np.load(fname)

	num_greedy = int(N*percent_greedy)
	for i in range(num_greedy):

		T[i] = get_needy_eng15_history(history_length, num_actions, base_states, recovery_prob=recovery_prob, fail_prob_passive=fail_prob_passive, fail_prob_act=fail_prob_act)

	# Get one reward if most recent state is adhering, 0 otherwise
	R_single = np.ones(2**history_length)
	inds = np.arange(2**(history_length-1))
	inds*=2
	R_single[inds] -= 1

	R = np.array([ R_single for _ in range(N)])

	# R = np.cumsum(R,axis=1)
	np.set_printoptions(threshold=np.inf)
	np.set_printoptions(linewidth=np.inf)


	start_states = simulate_start_states(T,history_length)



	C = np.arange(num_actions)
	B = N/4
	print(T)
	# np.save('eng_15_S%s_N%s_rewards.npy'%(S,N),R)

	# 1/0
	return T, R, C, B, start_states


