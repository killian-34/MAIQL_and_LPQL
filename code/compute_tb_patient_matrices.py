import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.cluster import KMeans
from collections import Counter
import sys

print('HISTORY_LENGTH = int(sys.argv[1])')
print('n_samples = int(sys.argv[2])')
print('action_diff_mult = int(sys.argv[3])')


np.set_printoptions(threshold=sys.maxsize, linewidth=1000000)


df = pd.read_csv('the_adherence_file.csv') # this is a placeholder, real data not included

sequences = df['AdherenceSequence'].values

# num_transitions = max([len(x) for x in sequences]) # this is 366 which is way more than we need so
num_transitions = 168
window_size = 168


# for i in range(0, num_transitions - window_size + 1, window_size):
# d[(0,0)] = 0
# d[(0,1)] = 0
# d[(1,0)] = 0
# d[(1,1)] = 0

HISTORY_LENGTH = int(sys.argv[1])
HL = HISTORY_LENGTH

import itertools 
states = [seq for seq in itertools.product((0,1), repeat=HL)]
print(states)




AT_LEAST_N_DAYS = 30

patient_probs = []
patient_counts = []
for sequence in sequences:


	patient_transition_counts = np.zeros((len(states), len(states)))
	patient_transition_probs = np.zeros((len(states), len(states)))
	if  len(sequence) > AT_LEAST_N_DAYS:
		for j in range(HL-1, window_size):
			if j + 1 < len(sequence):
				curr_state = tuple(sequence[j-HL+1:j+1])
				next_state = tuple(sequence[j-HL+2:j+2])
				patient_transition_counts[state_dict[curr_state]][state_dict[next_state]] += 1

		patient_transition_counts = np.array(patient_transition_counts).astype(float)
		totals = patient_transition_counts.sum(axis=1)

		for i, total in enumerate(totals):
			if total > 0:
				patient_transition_probs[i] = patient_transition_counts[i] / total

		patient_probs.append(patient_transition_probs)
		patient_counts.append(patient_transition_counts)


num_states = 2**HL

data_probs = np.array(patient_probs).reshape(-1, num_states*num_states)
patient_counts = np.array(patient_counts)



def create_prior_mask(history_length, base_states):
	# Create priors 

	# we are going to sample from dirichlets using the counts estimated above as priors
	# but there are two things we need to account for:
	#  1) all but 2 entries will need to be 0 in our transition probs (and will be 0 in our counts),
	#     since you can only transition to a state that has "your history"+"adhere" 
	#     or "your history"+"not adhere". But sampling from dirichlet with a prior of 0 is not possible
	#     so we will add 1e-6 to force the samples to effectively give 0 for states without counts
	#  2) despite the clustering, some clustered_transition_counts will still have all-0 rows simply 
	#     because so many patients always adhere in this dataset. To account for this, we will add 1 
	#     to the prior of all *possible* transitions (i.e., the two valid transitions per row)
	#     essentially assuming a uniform prior.
	HL = history_length
	num_states = 2**HL

	# the "prior mask" that we will add to all counts
	effective_zero_prior = 1e-6
	prior_mask = np.zeros((num_states, num_states)) + effective_zero_prior

	# now to enumerate all possible states transitions...
	historied_states = [seq for seq in itertools.product((0,1), repeat=HL)]

	for current_state in historied_states:
		
		current_state_ind = state_dict[current_state]

		current_state_shifted_by_1 = list(current_state[1:])
		for next_base_state in base_states:
			next_historied_state = tuple(current_state_shifted_by_1+[next_base_state])
			next_state_ind = state_dict[next_historied_state]
			prior_mask[current_state_ind, next_state_ind] += 1

	return prior_mask



# Patient needs to be acted on all the time to stay adhering
def get_needy_eng14(history_length, A, recovery_prob=0.05, fail_prob_passive=0.95, fail_prob_act=0.05):

	HL = history_length
	num_states = 2**HL

	state_effect = 0.05
	weights = np.linspace(1,2, history_length)
	state_weights = weights*state_effect/weights.sum()

	T = np.zeros((num_states,A,num_states))

	# now to enumerate all possible states transitions...
	historied_states = [seq for seq in itertools.product((0,1), repeat=HL)]

	for current_state in historied_states:
		
		current_state_ind = state_dict[current_state]

		active_base_state = current_state[-1]

		current_state_shifted_by_1 = list(current_state[1:])
		for next_base_state in base_states:
			next_historied_state = tuple(current_state_shifted_by_1+[next_base_state])
			next_state_ind = state_dict[next_historied_state]
			for a in range(0,A):
				if active_base_state == 0 and next_base_state == 0:
					pr = recovery_prob*(a+1)/(a+2)
					state_bonus = state_weights.dot(current_state)
					pr += state_bonus
					T[current_state_ind, a, next_state_ind] = 1 - pr 
				elif active_base_state == 0 and next_base_state == 1:
					pr = recovery_prob*(a+1)/(a+2)
					state_bonus = state_weights.dot(current_state)
					pr += state_bonus
					T[current_state_ind, a, next_state_ind] = pr

				elif active_base_state == 1 and next_base_state == 0:

					pf = fail_prob_act*((a+2)/(a+1) - 1)
					if a == 0:
						pf = fail_prob_passive
					state_bonus = state_weights.dot(current_state)
					pf -= state_bonus

					T[current_state_ind, a, next_state_ind] = pf

				elif active_base_state == 1 and next_base_state == 1:

					pf = fail_prob_act*((a+2)/(a+1) - 1)
					if a == 0:
						pf = fail_prob_passive
					state_bonus = state_weights.dot(current_state)
					pf -= state_bonus

					T[current_state_ind, a, next_state_ind] = 1-pf

				



def sample_from_tb_data(n_samples, clustered_transition_counts, cluster_sizes, history_length, base_states, num_actions, action_diff_mult=2):


	# need to set up for diminishing return for higher actions
	multiplier_list_a =  np.concatenate([[1], np.sort(np.random.rand(num_actions-1)*(action_diff_mult-1)+1)[::-1]]).cumsum()
	multiplier_list_a =  multiplier_list_a/multiplier_list_a[1]
	multiplier_list_na = 1/multiplier_list_a
	

	multiplier_list = list(zip(multiplier_list_na, multiplier_list_a))


	num_states = clustered_transition_counts.shape[1]
	sampled_transition_probs = np.zeros((n_samples, num_states, num_states))
	sampled_transition_probs = np.zeros((n_samples, num_actions, num_states, num_states))

	# want to sample from each cluster based on its relative size
	cluster_sampling_probs = cluster_sizes / cluster_sizes.sum()

	prior_mask = create_prior_mask(history_length, base_states)

	cluster_priors = clustered_transition_counts + prior_mask
	n_clusters = len(cluster_sizes)
	clusters_to_sample = np.random.choice(np.arange(n_clusters), size=n_samples, p=cluster_sampling_probs)
	

	# impose action effects within this method by multiplying priors so that we can sample a valid
	# simplex, instead of fixing things manually after sampling

	for i, cluster_ind in enumerate(clusters_to_sample):

		print('i', i)
		for row in range(num_states):
			for a, mults in enumerate(multiplier_list):
				print(mults)
				action_adjusted_prior = np.copy(cluster_priors[cluster_ind, row])
				current_state = states[row]
				next_state = tuple(list(current_state[1:]) + [0])
				next_state_ind = state_dict[next_state]
				print(current_state)
				print(action_adjusted_prior[next_state_ind:next_state_ind+len(base_states)])
				action_adjusted_prior[next_state_ind:next_state_ind+len(base_states)] *= mults
				print(action_adjusted_prior[next_state_ind:next_state_ind+len(base_states)])
				sampled_transition_probs[i, a, row] = np.random.dirichlet(action_adjusted_prior)
				print()
			print()



	return sampled_transition_probs


###### run clustering
# In this case, th larger N_CLUSTERS, the more "peaks" of the distribution we will try to approximate
# but the fewer data points we will have to estimate the distribution in each cluster
# -- however, we may want more clusters to allow for some samples to come from uncommon
#    but "diverse" transition matrices, e.g., they don't always adhere or not 
# SENSITIVITY ANALYSIS ON THIS PARAM
N_CLUSTERS = 10 
print("running kmeans")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(data_probs)
centers = kmeans.cluster_centers_
# print('centers')
# print(centers)
counts_per_cluster = Counter(kmeans.labels_)
print('finished kmeans')
print(kmeans.labels_[:4])

##### get aggregate counts among clusters
clustered_transition_counts = np.zeros((N_CLUSTERS, num_states, num_states))
for i, label in enumerate(kmeans.labels_):
	clustered_transition_counts[label] += patient_counts[i]


# extract the cluster sizes from the counter dict
cluster_sizes = np.array([counts_per_cluster[i] for i in range(N_CLUSTERS)])

n_samples = int(sys.argv[2])
base_states = [0,1]
num_actions = 3
action_diff_mult = int(sys.argv[3])
T = sample_from_tb_data(n_samples, clustered_transition_counts, cluster_sizes, HL, base_states, num_actions, action_diff_mult=action_diff_mult)
# change to S,A,S
T = np.swapaxes(T,1,2)
print(T.shape)

fname = '../data/patient_T_matrices_n%s_a%s_HL%s_adm%s.npy'%(n_samples, num_actions, HL, action_diff_mult)
np.save(fname, T)

# check what the distribution of entries for the all adhere and all not-adhere states
for a in range(num_actions):
	# print(T[:,a,-1,:])
	all_adhere = T[:,-1,a,-1]
	all_not_adhere = T[:,0,a,0]

	import pandas as pd
	pd.DataFrame(all_adhere).hist(bins=500)
	plt.show()

	pd.DataFrame(all_not_adhere).hist(bins=500)
	plt.show()




