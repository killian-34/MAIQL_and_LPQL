import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import jit
import itertools 



########## WIBQL (a.k.a. Borkar et al.)

@jit(nopython=True)
def wibql_qlearn_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, average_mode, Q, C, Cp, 
  isa_counter, lam, gamma, currentIndices, lams_over_time, update_lam,
  lambda_lb, lambda_ub, update_lam_at_end, learn_divisor):   
    
    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]

      if a > 0:
        a = 1

      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / learn_divisor)
      
      
      for s_i in range(num_s):
        if average_mode: 
          f = Q[arm].mean()

          Q[arm, s_i, state, a] += update_mult*(R[arm, state] - lam[arm, state] + Q[arm, s_i, nextState].max() - f - Q[arm, s_i, state, a])
        else:
          Q[arm, s_i, state, a] += update_mult*(R[arm, state] - lam[arm, state] + gamma*Q[arm, s_i, nextState].max() - Q[arm, s_i, state, a])


      if iteration % (n_arms) == 0 and update_lam:
        update_mult = Cp / (1 + np.ceil(iteration*np.log(iteration)/ learn_divisor) )
        
        lam[arm, state] += update_mult*(Q[arm, state, state, 1] - Q[arm, state, state, 0])
        lam[arm, state] = min(lambda_ub, lam[arm, state])
        lam[arm, state] = max(lambda_lb, lam[arm, state])
  

      #Create a list of current Whittle Indices based on each arm's current state
      if update_lam:
        currentIndices[arm] = lam[arm, nextState]
      

    # if we are learning from replays update each lambda once after all the replays
    if update_lam_at_end:
      for state in range(num_s):
        update_mult = Cp / (1 + np.ceil(iteration*np.log(iteration)/ learn_divisor) )
        lam[arm, state] += update_mult*(Q[arm, state, state, 1] - Q[arm, state, state, 0])# - lam[arm, state])
        lam[arm, state] = min(lambda_ub, lam[arm, state])
        lam[arm, state] = max(lambda_lb, lam[arm, state])
    


class RMABQL_WIBQL(object):
  def __init__(self, n_arms, m, eps, C, Cp, gamma, 
    iterations, n_states, n_actions, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    lambda_lb=None, lambda_ub=None,
    learn_divisor=500, epsilon_divisor=500):

    self.n_arms = n_arms # Number of arms 
    self.m = m # number of arms to be selected (budget)
    self.eps = eps # Used for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.Cp = Cp # starting Learning parameter for lambda values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the bandits environment
    self.average_mode = average_mode
    self.replay_buffer_size = replay_buffer_size
    self.lambda_lb = lambda_lb
    self.lambda_ub = lambda_ub
    self.learn_divisor=learn_divisor
    self.epsilon_divisor=epsilon_divisor

    # last dimension is 4 because we will only store (arm, a, s, s) pairs, then lookup reward from table
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    self.s = n_states
    self.a = n_actions

    self.Q = np.zeros((n_arms, self.s, self.s, self.a)) # Stores Q values
    self.lam = np.zeros((n_arms, self.s)) # Stores lambda values


    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t
    self.isa_replay_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t
    self.currentIndices = np.zeros((n_arms)) # Stores the current values of Whittle Indices of the currentState for each arm. 
    self.count_state = np.zeros(((self.iterations+1), self.s)) # Stores the number of times a state is pulled observed till time t (cumulative)

    self.LamsOverTime = None#np.zeros((n_arms, self.iterations, self.s)) # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t

    self.initial_exploration = initial_exploration
    self.eps_decay = eps_decay
  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/self.epsilon_divisor),0.01)

    if self.initial_exploration:
      if self.iterations<100:
        eps = 0.9
    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices


  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      if a > 0:
        a=1
      self.isa_counter[arm, state_log[arm][0], a] += 1

    actions[actions>1] = 1
    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState


    update_lam = True
    update_lam_at_end = False
    wibql_qlearn_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.LamsOverTime, update_lam, self.lambda_lb, self.lambda_ub, update_lam_at_end,
      self.learn_divisor)


    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def plot_indexes(self):
    import matplotlib.pyplot as plt
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', weight='bold')
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    

    fig,ax = plt.subplots(3, 4, figsize=(14,8))
    ax = ax.reshape(-1)
    colors = ['r','g','b','c']
    # wi_vals = [-0.5, 0.5, 1, -1]
    for arm in range(self.n_arms):
      for state in range(self.s):
        if arm == 0:
          ax[arm].plot(self.LamsOverTime[arm,:,state], color=colors[state], alpha=0.5, label='S:%s'%state)
        else:
          ax[arm].plot(self.LamsOverTime[arm,:,state], color=colors[state], alpha=0.5)
        

    plt.suptitle('ArXiv QWIC: Lambdas')
    fig.legend(ncol=4,loc='lower center')
    plt.savefig('indices_over_time_arm%s_lams_arxiv_qwic.png'%arm,dpi=200)
    plt.show()



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    sample_weights = (sample_weights.max() - sample_weights)**10

    sample_weights = sample_weights / sample_weights.sum()

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    for replay in replays:
      arm, state, a, _ = replay
      self.isa_replay_counter[arm, state, a] += 1

    update_lam = False
    update_lam_at_end = True
    wibql_qlearn_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.LamsOverTime, update_lam, self.lambda_lb, self.lambda_ub, 
      update_lam_at_end, self.learn_divisor)










########## MAIQL tabular

@jit(nopython=True)
def qlearn_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, average_mode, Q, C, Cp, 
  isa_counter, lam, gamma, currentIndices, lams_over_time, update_lam,
  lambda_lb, lambda_ub, update_lam_at_end, learn_divisor):
    # Update values using most recent (s, a, r, s) pairs

    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]


      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / learn_divisor)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      for a_i in range(1, num_a):
        for s_i in range(num_s):
          if average_mode: 
            f = Q[arm, s_i, a_i].mean()
            Q[arm, s_i, a_i, state, a] += update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + Q[arm, s_i, a_i, nextState].max() - f - Q[arm, s_i, a_i, state, a])
          else:
            Q[arm, s_i, a_i, state, a] += update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + gamma*Q[arm, s_i, a_i, nextState].max() - Q[arm, s_i, a_i, state, a])
      
      # update lambda - how to update lambda in multi-action setting
      if iteration % (n_arms) == 0 and update_lam:
        update_mult = Cp / (1 + np.ceil(iteration*np.log(iteration)/ learn_divisor) )

        # since a lambda depends on a and a-1, update the lambda if either action is taken
        for a in range(a,min(a+2, num_a)):
          value = update_mult*(Q[arm, state, a, state, a] - Q[arm, state, a, state, a - 1])/(costs[a] - costs[a - 1])


          lam[arm, a, state] += value
          lam[arm, a, state] = min(lambda_ub, lam[arm, a, state])
          lam[arm, a, state] = max(lambda_lb, lam[arm, a, state])

      if update_lam:
        currentIndices[arm] = lam[arm, :, nextState]




    # if we are learning from replays update each lambda once after all the replays
    if update_lam_at_end:
      update_mult = Cp / (1 + np.ceil(iteration*np.log(iteration)/ learn_divisor) )
      for a in range(1,num_a):
        for state in range(num_s):
          value = update_mult*(Q[arm, state, a, state, a] - Q[arm, state, a, state, a - 1])/(costs[a] - costs[a - 1])

          # print(arm, a_i, s_i, Q[arm, s_i, a_i, s_i, a_i], Q[arm, s_i, a_i, s_i, a_i-1], costs[a_i], costs[a_i-1], isa_counter[arm, s_i, a_i], isa_counter[arm, s_i, a_i-1])
          lam[arm, a, state] += value
          lam[arm, a, state] = min(lambda_ub, lam[arm, a, state])
          lam[arm, a, state] = max(lambda_lb, lam[arm, a, state])




class MultiActionQLTabular(object):
  def __init__(self, n_arms, m, eps, C, Cp, gamma, 
    iterations, n_states, n_actions, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    lambda_lb=None, lambda_ub=None,
    learn_divisor=500, epsilon_divisor=500):

    self.n_arms = n_arms # Number of arms 
    self.m = m # number of arms to be selected (budget)
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.Cp = Cp # starting Learning parameter for lambda values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode
    self.replay_buffer_size = replay_buffer_size
    self.lambda_lb = lambda_lb
    self.lambda_ub = lambda_ub
    self.learn_divisor=learn_divisor
    self.epsilon_divisor=epsilon_divisor

    # last dimension is 4 because we will only store (arm, a, s, s) pairs, then lookup reward from table
    # print(self.replay_buffer_size)
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period


    self.s = n_states
    self.a = n_actions

    # Stores Q values - need a new set of Q(s,a) for each action since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    self.Q = np.zeros((n_arms, self.s, self.a, self.s, self.a)) 
    self.lam = np.zeros((n_arms, self.a, self.s)) # Stores lambda values
    

    # if not initial_Q_values_as_0:
    #   # Initialize to Q values
    #   for i in np.arange(n_arms):
    #     for j in np.arange(len(arms[i].mdp.s)):
    #       self.Q[i][j][0] =  arms[i].mdp.r[j]
    #       self.Q[i][j][1] =  arms[i].mdp.r[j]

    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t
    self.isa_replay_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t
    self.currentIndices = np.zeros((n_arms, self.a)) # Stores the current values of Whittle Indices of the currentState for each arm. 
    self.count_state = np.zeros(((self.iterations+1), self.s)) # Stores the number of times a state is pulled observed till time t (cumulative)

    self.LamsOverTime = None#np.zeros((n_arms, self.iterations, self.s, self.a)) # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t

    self.initial_exploration = initial_exploration
    self.eps_decay = eps_decay
  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/self.epsilon_divisor),0.01)

    if self.initial_exploration:
      if self.iterations<100:
        eps = 0.9
    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    update_lam = True

    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState


    # # Update values using most recent (s, a, r, s) pairs
    update_lam_at_end = False
    qlearn_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.LamsOverTime, update_lam, self.lambda_lb, self.lambda_ub, update_lam_at_end,
      self.learn_divisor)

    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    sample_weights = (sample_weights.max() - sample_weights)**5

    sample_weights = sample_weights / sample_weights.sum()

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    for replay in replays:
      arm, state, a, _ = replay
      self.isa_replay_counter[arm, state, a] += 1

    update_lam = False
    update_lam_at_end = True
    qlearn_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.LamsOverTime, update_lam, self.lambda_lb, self.lambda_ub,
      update_lam_at_end, self.learn_divisor)



  def plot_indexes(self, wi_vals):
    print('isa counters')
    print('do')
    for arm in range(self.n_arms):
      print(self.isa_counter[arm])
    # print(self.isa_counter[arm, state_log[arm][0], a])
    import matplotlib.pyplot as plt
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', weight='bold')
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    for a_i in range(1,self.a):
      fig,ax = plt.subplots(3,4, figsize=(14,8))
      ax = ax.reshape(-1)
      colors = ['r','g','b','c']
      # wi_vals = [-0.5, 0.5, 1, -1]
      for arm in range(12):#self.n_arms):
        for state in range(self.s):
          if arm == 0:
            ax[arm].plot(self.LamsOverTime[arm,:,state,a_i], color=colors[state], alpha=0.5, label='S:%s,A:%s'%(state,a_i))
          else:
            ax[arm].plot(self.LamsOverTime[arm,:,state,a_i], color=colors[state], alpha=0.5)
          
          ax[arm].plot([0, self.iterations], [wi_vals[arm, state, a_i], wi_vals[arm, state, a_i]], color=colors[state], linestyle='--')

      plt.suptitle('MAQL: Lambdas, Action: %s'%a_i)
      fig.legend(ncol=4,loc='lower center')
      plt.savefig('indices_over_time_arm%s_lams_maql.png'%arm,dpi=200)
      plt.show()










@jit(nopython=True)
def lpql_tabular_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, n_lams, 
  average_mode, Q, C, isa_counter, gamma, lams, learn_divisor):
    # Update values using most recent (s, a, r, s) pairs

    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]


      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / learn_divisor)
      # print('Q update:%s'%update_mult)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      for lam_i in range(n_lams):
        if average_mode: 
          f = Q[arm, lam_i].mean()
          Q[arm, lam_i, state, a] += update_mult*(R[arm, state] - costs[a]*lams[lam_i] + Q[arm, lam_i, nextState].max() - f - Q[arm, lam_i, state, a])
        else:
          Q[arm, lam_i, state, a] += update_mult*(R[arm, state] - costs[a]*lams[lam_i] + gamma*Q[arm, lam_i, nextState].max() - Q[arm, lam_i, state, a])




class LPQLTabular(object):
  def __init__(self, n_arms, m, eps, C, gamma, 
    iterations, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    learn_divisor=500, epsilon_divisor=500):

    self.n_arms = n_arms # Number of arms 
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode # Not implemented right now
    self.replay_buffer_size = replay_buffer_size
    self.n_lams = n_lams
    self.learn_divisor=learn_divisor
    self.epsilon_divisor=epsilon_divisor

    # last dimension is 4 because we will only store (arm, s, a, s) pairs, then lookup reward from table
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    self.s = n_states
    self.a = n_actions

    # Stores Q values - need a new set of Q(s,a) for each action since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    self.Q = np.zeros((n_arms, self.n_lams, self.s, self.a)) 
    self.lambda_lim = lambda_lim
    self.lams = np.linspace(0, self.lambda_lim, self.n_lams)
    

    # if not initial_Q_values_as_0:
    #   # Initialize to Q values
    #   for i in np.arange(n_arms):
    #     for j in np.arange(len(arms[i].mdp.s)):
    #       self.Q[i][j][0] =  arms[i].mdp.r[j]
    #       self.Q[i][j][1] =  arms[i].mdp.r[j]

    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t

    self.eps_decay = eps_decay

  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/self.epsilon_divisor),0.005)

    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState


    # # Update values using most recent (s, a, r, s) pairs
    lpql_tabular_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.learn_divisor)

    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    # print('sample max',sample_weights.max())
    sample_weights = (sample_weights.max() - sample_weights)**10

    sample_weights = sample_weights / sample_weights.sum()

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    lpql_tabular_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.learn_divisor)






@jit(nopython=True)
def vfnc_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, average_mode, Q, C, isa_counter, gamma, learn_divisor):
    # Update values using most recent (s, a, r, s) pairs

    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]


      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / learn_divisor)
      # print('Q update:%s'%update_mult)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      if average_mode: 
        f = Q[arm].mean()
        Q[arm, state, a] += update_mult*(R[arm, state] + Q[arm, nextState].max() - f - Q[arm, state, a])
      else:
        Q[arm, state, a] += update_mult*(R[arm, state] + gamma*Q[arm, nextState].max() - Q[arm, state, a])




class VfNCQL(object):
  def __init__(self, n_arms, m, eps, C, gamma, 
    iterations, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    learn_divisor=500, epsilon_divisor=500):

    self.n_arms = n_arms # Number of arms 
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode # Not implemented right now
    self.replay_buffer_size = replay_buffer_size
    self.learn_divisor=learn_divisor
    self.epsilon_divisor=epsilon_divisor

    # last dimension is 4 because we will only store (arm, s, a, s) pairs, then lookup reward from table
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    self.s = n_states
    self.a = n_actions

    # Stores Q values - need a new set of Q(s,a) for each action since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    self.Q = np.zeros((n_arms, self.s, self.a)) 

    # if not initial_Q_values_as_0:
    #   # Initialize to Q values
    #   for i in np.arange(n_arms):
    #     for j in np.arange(len(arms[i].mdp.s)):
    #       self.Q[i][j][0] =  arms[i].mdp.r[j]
    #       self.Q[i][j][1] =  arms[i].mdp.r[j]

    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t


    self.eps_decay = eps_decay

  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/self.epsilon_divisor),0.005)

    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState


    # # Update values using most recent (s, a, r, s) pairs
    vfnc_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.learn_divisor)

    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    # print('sample max',sample_weights.max())
    sample_weights = (sample_weights.max() - sample_weights)**10

    sample_weights = sample_weights / sample_weights.sum()


    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    vfnc_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.learn_divisor)






@jit(nopython=True)
def maiql_approx_tabular_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, n_lams, average_mode, Q, C, isa_counter, gamma, lams, learn_divisor):
    # Update values using most recent (s, a, r, s) pairs

    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]


      # Extension of WI Q-Learning - https://arxiv.org/pdf/2004.14427.pdf
      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / learn_divisor)
      # print('Q update:%s'%update_mult)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      for lam_i in range(n_lams):
        if average_mode: 
          f = Q[arm, lam_i].mean()
          Q[arm, lam_i, state, a] += update_mult*(R[arm, state] - costs[a]*lams[lam_i] + Q[arm, lam_i, nextState].max() - f - Q[arm, lam_i, state, a])
        else:
          Q[arm, lam_i, state, a] += update_mult*(R[arm, state] - costs[a]*lams[lam_i] + gamma*Q[arm, lam_i, nextState].max() - Q[arm, lam_i, state, a])

@jit(nopython=True)
def maiql_approx_tabular_get_indexes(Q, current_state, n_arms, num_a, lams):
  indexes_out = np.zeros((n_arms, num_a))
  for arm in range(n_arms):
    for a in range(1, num_a):
      Qa2 = Q[arm,:,current_state[arm],a]
      Qa1 = Q[arm,:,current_state[arm],a-1]
      Qdiff = np.abs(Qa2 - Qa1)
      indexes_out[arm,a] = lams[np.argmin(Qdiff)]

  return indexes_out



class MAIQLApproxTabular(object):
  def __init__(self, n_arms, m, eps, C, gamma, 
    iterations, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    learn_divisor=500, epsilon_divisor=500):

    self.n_arms = n_arms # Number of arms 
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode # Not implemented right now
    self.replay_buffer_size = replay_buffer_size
    self.n_lams = n_lams
    self.learn_divisor=learn_divisor
    self.epsilon_divisor=epsilon_divisor

    # last dimension is 4 because we will only store (arm, s, a, s) pairs, then lookup reward from table
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    # will use this as a fast way to look up 
    # self.replay_buffer_identities = np.zeros((self.replay_buffer_size, n_arms), dtype=object)

    self.s = n_states
    self.a = n_actions

    # Stores Q values - need a new set of Q(s,a) for each action since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    self.Q = np.zeros((n_arms, self.n_lams, self.s, self.a)) 
    self.lambda_lim = lambda_lim
    self.lams = np.linspace(0, self.lambda_lim, self.n_lams)

    # if not initial_Q_values_as_0:
    #   # Initialize to Q values
    #   for i in np.arange(n_arms):
    #     for j in np.arange(len(arms[i].mdp.s)):
    #       self.Q[i][j][0] =  arms[i].mdp.r[j]
    #       self.Q[i][j][1] =  arms[i].mdp.r[j]

    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t


    self.eps_decay = eps_decay

  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/self.epsilon_divisor),0.005)

    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self, current_state):

    # self.Q = np.zeros((n_arms, self.n_lams, self.s, self.a))
    inds = maiql_approx_tabular_get_indexes(self.Q, current_state,  self.n_arms, self.a, self.lams)
    # print(inds)
    return inds
    



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState


    # # Update values using most recent (s, a, r, s) pairs
    maiql_approx_tabular_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.learn_divisor)

    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]



    sample_weights = (sample_weights.max() - sample_weights)**10

    sample_weights = sample_weights / sample_weights.sum()
    # print('weights',sample_weights[:5])
    # print('num samples',self.num_experiences_in_buffer)

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    maiql_approx_tabular_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.learn_divisor)















###########
###### Beta implementations (not guaranteed to work, not in paper)
###########

LPQL_LQFA_DIVISOR = 1e6
LPQL_LQFA_EPSILON_DIVISOR = 1e6
@jit(nopython=True)
def lpql_lqfa_helper(tuples, R, iteration, costs, num_a, num_s, n_arms, n_lams, average_mode, Q, C, isa_counter, gamma, lams, state_list):
    # Update values using most recent (s, a, r, s) pairs

    n_tuples = len(tuples)
    for i in range(n_tuples):
      arm, state, a, nextState = tuples[i]


      # Extension of WI Q-Learning - https://arxiv.org/pdf/2004.14427.pdf
      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / LPQL_LQFA_DIVISOR)
      # print('Q update:%s'%update_mult)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      for lam_i in range(n_lams):
        if average_mode: 
            pass # TODO: implement 
            # f = Q[arm, s_i, a_i].mean()
            # Q[arm, s_i, a_i, state, a] += update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + Q[arm, s_i, a_i, nextState].max() - f - Q[arm, s_i, a_i, state, a])
        else:
          x_s = state_list[state]
          x_a = np.zeros(num_a)
          x_a[a] = 1
          # x_curr = np.concatenate([[1],x_s,x_a])
          x_curr = np.ones(len(Q[arm, lam_i]))
          x_curr[1:1+len(x_s)] = x_s
          x_curr[1+len(x_s):] = x_a

          x_s_next = state_list[nextState]
          max_next_Q = -np.inf
          for i in range(num_a):
            x_a = np.zeros(num_a)
            x_a[a] = 1
            # x = np.concatenate([[1],x_s,x_a])
            x = np.ones(len(Q[arm, lam_i]))
            x[1:1+len(x_s)] = x_s
            x[1+len(x_s):] = x_a
            max_next_Q = max(max_next_Q, Q[arm, lam_i].dot(x))

          # for linear function, grad q is just the feature representation, so 
          grad_Q = x_curr
          # TODO: update this to SARSA, but for now, pick the action that gives the largest Q?
          # print('update thing')
          # print(grad_Q)
          # print(update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + gamma*max_next_Q - Q[arm, s_i, a_i].dot(x_curr)))

          Q[arm, lam_i] += update_mult*(R[arm, state] - costs[a]*lams[lam_i] + gamma*max_next_Q - Q[arm, lam_i].dot(x_curr))*grad_Q



# (Beta) implementation of LPQL that uses linear q-function approximation
class LPQL_LQFA(object):
  def __init__(self, n_arms, m, eps, C, gamma, 
    iterations, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500, history_length=None ):

    self.n_arms = n_arms # Number of arms 
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode # Not implemented right now
    self.replay_buffer_size = replay_buffer_size
    self.n_lams = n_lams

    # last dimension is 4 because we will only store (arm, s, a, s) pairs, then lookup reward from table
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    
    self.num_weights = 1+history_length+n_actions # one for const term, one for each day of history, and enough to 1-hot the actions
    self.weights = np.random.normal(0,1,self.num_weights)

    # will use this as a fast way to look up 
    # self.replay_buffer_identities = np.zeros((self.replay_buffer_size, n_arms), dtype=object)

    self.s = n_states
    self.a = n_actions

    
    self.state_list = [seq for seq in itertools.product((0,1), repeat=history_length)]


    # Stores Q functions - need a new Q for each s,a since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    # that is applied to all s,a
    self.Q = np.zeros((n_arms, self.n_lams, len(self.weights))) 

    self.lambda_lim = lambda_lim
    self.lams = np.linspace(0, self.lambda_lim, self.n_lams)


    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t

    # self.QOverTime = np.zeros((n_arms, self.iterations, self.s, self.a, self.s, self.a)) # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t

    self.eps_decay = eps_decay

  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/LPQL_LQFA_EPSILON_DIVISOR),0.01)

    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 

  def get_all_Q(self, arm, s):

    Q_out = np.zeros((self.n_lams, self.a))

    x_s = self.state_list[s]
    
    # x_curr = np.concatenate([[1],x_s,x_a])
    x_curr = np.ones(self.num_weights)
    x_curr[1:1+len(x_s)] = x_s
    for a in range(self.a):
      x_a = np.zeros(self.a)
      x_a[a] = 1
      x_curr[1+len(x_s):] = x_a
      Q_out[:,a] = self.Q[arm].dot(x_curr)


    return Q_out



  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState

    # print("Thing")
    # print(self.Q[0,1,1])

    # # Update values using most recent (s, a, r, s) pairs
    lpql_lqfa_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.state_list)

    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)



  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    # print('sample max',sample_weights.max())
    sample_weights = (sample_weights.max() - sample_weights)**10
    # print(sample_weights[i].sum())
    # print('replays',self.replay_buffer[:5])
    # print('weights',sample_weights[:5])
    # print('counters',self.isa_counter)

    sample_weights = sample_weights / sample_weights.sum()
    # print('weights',sample_weights[:5])
    # print('num samples',self.num_experiences_in_buffer)

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    lpql_lqfa_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.n_lams, self.average_mode, self.Q, self.C, 
      self.isa_counter, self.gamma, self.lams, self.state_list)



############# MAIQL with linear QFA

MAIQL_LQFA_DIVISOR = 2000
MAIQL_LQFA_EPSILON_DIVISOR = 250

@jit(nopython=True)
def linear_qfa_qlearn_helper(tuples, R, iteration, costs, 
  num_a, num_s, n_arms, average_mode, Q, C, Cp, isa_counter, lam, gamma, 
  currentIndices, state_list):
    # Update values using most recent (s, a, r, s) pairs
    num_tuples = len(tuples)

    for i in range(num_tuples):
      arm, state, a, nextState = tuples[i]

      # Extension of WI Q-Learning - https://arxiv.org/pdf/2004.14427.pdf
      # update Q function
      update_mult = C / np.ceil(isa_counter[arm, state, a] / MAIQL_LQFA_DIVISOR)

      # print('Q update:%s'%update_mult)
      
      # need to update all $a$ Q functions, except the 0th
      # and every state needs its own copies of Q
      for a_i in range(1, num_a):
        for s_i in range(num_s):
          if average_mode: 
            pass # TODO: implement 
            # f = Q[arm, s_i, a_i].mean()
            # Q[arm, s_i, a_i, state, a] += update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + Q[arm, s_i, a_i, nextState].max() - f - Q[arm, s_i, a_i, state, a])
          else:
            x_s = state_list[state]
            x_a = np.zeros(num_a)
            x_a[a] = 1
            # x_curr = np.concatenate([[1],x_s,x_a])
            x_curr = np.ones(len(Q[arm, s_i, a_i]))
            x_curr[1:1+len(x_s)] = x_s
            x_curr[1+len(x_s):] = x_a

            x_s_next = state_list[nextState]
            max_next_Q = -np.inf
            for i in range(num_a):
              x_a = np.zeros(num_a)
              x_a[a] = 1
              # x = np.concatenate([[1],x_s,x_a])
              x = np.ones(len(Q[arm, s_i, a_i]))
              x[1:1+len(x_s)] = x_s
              x[1+len(x_s):] = x_a
              max_next_Q = max(max_next_Q, Q[arm, s_i, a_i].dot(x))

            # for linear function, grad q is just the feature representation, so 
            grad_Q = x_curr
            # TODO: update this to SARSA, but for now, pick the action that gives the largest Q?
            # print('update thing')
            # print(grad_Q)
            # print(update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + gamma*max_next_Q - Q[arm, s_i, a_i].dot(x_curr)))

            Q[arm, s_i, a_i] += update_mult*(R[arm, state] - costs[a]*lam[arm, a_i, s_i] + gamma*max_next_Q - Q[arm, s_i, a_i].dot(x_curr))*grad_Q
      
      # update lambda - how to update lambda in multi-action setting
      if iteration % n_arms == 0:
        update_mult = Cp / (1 + np.ceil(iteration*np.log(iteration)/ MAIQL_LQFA_DIVISOR) )
        # print('lam update:%s'%update_mult)
        # print('a',a)
        for a_i in range(1, num_a):
          for s_i in range(num_s):
            x_s = state_list[s_i]

            x_a = np.zeros(num_a)
            x_a[a_i] = 1
            x_high = np.ones(len(Q[arm, s_i, a_i]))
            x_high[1:1+len(x_s)] = x_s
            x_high[1+len(x_s):] = x_a

            x_a = np.zeros(num_a)
            x_a[a_i-1] = 1
            x_low = np.ones(len(Q[arm, s_i, a_i]))
            x_low[1:1+len(x_s)] = x_s
            x_low[1+len(x_s):] = x_a
            lam[arm, a_i, s_i] += update_mult*(Q[arm, s_i, a_i].dot(x_high) - Q[arm, s_i, a_i].dot(x_low))/(costs[a_i] - costs[a_i - 1])


      currentIndices[arm] = lam[arm, :, nextState]



# (Beta) Implementation of MAIQL that uses linear q-function approximation
class MultiActionQL_LQFA(object):
  def __init__(self, n_arms, m, eps, C, Cp, gamma, 
    iterations, n_states, n_actions, initial_exploration = False, 
    initial_Q_values_as_0 = False, eps_decay = False, average_mode = False,
    replay_buffer_size = 10e5, num_replays_per_dream=100, replay_period=500,
    history_length=4 ):

    self.n_arms = n_arms # Number of arms 
    self.m = m # number of arms to be selected (budget)
    self.eps = eps # USed for the \epsilon-greedy selection of arms
    self.C = C # starting Learning parameter for Q values
    self.Cp = Cp # starting Learning parameter for lambda values
    self.gamma = gamma # Discount factor
    self.iterations = iterations # Number of iterations for simulating the Bandits environment
    self.average_mode = average_mode
    self.replay_buffer_size = replay_buffer_size

    # last dimension is 4 because we will only store (arm, a, s, s) pairs, then lookup reward from table
    print(self.replay_buffer_size)
    self.replay_buffer = np.zeros((self.replay_buffer_size, 4),dtype=int)
    self.num_experiences_in_buffer = 0

    self.num_replays_per_dream = num_replays_per_dream
    self.replay_period = replay_period

    self.num_weights = 1+history_length+n_actions # one for const term, one for each day of history, and enough to 1-hot the actions
    self.weights = np.random.normal(0,1,self.num_weights)

    # will use this as a fast way to look up 
    # self.replay_buffer_identities = np.zeros((self.replay_buffer_size, n_arms), dtype=object)

    self.s = n_states
    self.a = n_actions

    
    self.state_list = [seq for seq in itertools.product((0,1), repeat=history_length)]


    # Stores Q functions - need a new Q for each s,a since 
    # we need a unique index (i.e., value of lagrange multiplier) for each action
    # that is applied to all s,a
    self.Q = np.zeros((n_arms, self.s, self.a, len(self.weights))) 
    self.lam = np.zeros((n_arms, self.a, self.s)) # Stores lambda values
    

    self.isa_counter = np.zeros((n_arms, self.s, self.a)) # Stores the number of times each (arm, state, action) pair is observed till time t
    self.currentIndices = np.zeros((n_arms, self.a)) # Stores the current values of Whittle Indices of the currentState for each arm. 
    self.count_state = np.zeros(((self.iterations+1), self.s)) # Stores the number of times a state is pulled observed till time t (cumulative)

    self.LamsOverTime = None#np.zeros((n_arms, self.iterations, self.a, self.s)) # Stores the values of Whittle Indices (of an arm at each state), which is learnt till time t

    self.initial_exploration = initial_exploration
    self.eps_decay = eps_decay
  
  def check_random(self, iteration, random_stream=None):
        
    eps = self.eps
    if self.eps_decay:
      # eps = eps/np.ceil(iteration/self.n_arms)
      eps = max(self.eps/np.ceil(iteration/MAIQL_LQFA_EPSILON_DIVISOR),0)

    if self.initial_exploration:
      if self.iterations<100:
        eps = 0.9
    
    p = None
    if random_stream is not None:
      p = random_stream.random()
    else:
      p = np.random.random() 
    # print(eps)

    return p < eps 


  # need to return nxa array of indices
  def get_indexes(self):

    return self.currentIndices



  # action is dimension n
  # state_log is nx2 array with prev state and current state
  # R is nxs
  def qlearn(self, actions, state_log, R, iteration, costs, add_samples_to_replay_buffer=True, random_stream=None):
       
    # Take actions based on the selection
    for arm, a in enumerate(actions):
      self.isa_counter[arm, state_log[arm, 0], a] += 1
      # self.QOverTime[arm, iteration] = self.Q[arm]
    # print("Thing")
    # print(self.Q[0,1,1])

    state = state_log[:, 0]
    nextState = state_log[:, 1]
    nsas_tuples = np.zeros((self.n_arms, 4),dtype=int)
    nsas_tuples[:,0] = np.arange(self.n_arms)
    nsas_tuples[:,1] = state
    nsas_tuples[:,2] = actions
    nsas_tuples[:,3] = nextState

    # # Update values using most recent (s, a, r, s) pairs
    linear_qfa_qlearn_helper(nsas_tuples, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.state_list)


    # add to replay buffer
    if add_samples_to_replay_buffer:
      self.add_to_replay_buffer(state_log, actions, random_stream=random_stream)


  def add_to_replay_buffer(self, state_log, actions, random_stream=None):
    
    i = 0
    while i < self.n_arms and self.num_experiences_in_buffer < self.replay_buffer_size:
      self.replay_buffer[self.num_experiences_in_buffer] = [i, state_log[i,0], actions[i], state_log[i,1]]
      self.num_experiences_in_buffer += 1
      i+=1

    # if we hit replay buffer size limit before storing all the memories from this round
    if i < self.n_arms:

      # randomly sample the experiences to replace --
      # this will naturally favor replacing more common experiences
      inds = random_stream.randint(0, self.replay_buffer_size, self.n_arms - i)
      for ind in inds:
        self.replay_buffer[ind] = [i, state_log[i,0], actions[i], state_log[i,1]]
        i+=1


  def learn_from_replays(self, iteration, R, costs, random_stream=None):
    
    # want to put larger weights on replays that have been seen less often
    sample_weights = np.zeros(self.num_experiences_in_buffer)


    for i in range(self.num_experiences_in_buffer):
      arm, s, a, sp = self.replay_buffer[i]
      # want to put larger weights on replays that have been seen less often
      sample_weights[i] = self.isa_counter[arm, s, a]


    # print('sample max',sample_weights.max())
    sample_weights = (sample_weights.max() - sample_weights)**10
    

    sample_weights = sample_weights / sample_weights.sum()
    # print('weights',sample_weights[:5])
    # print('num samples',self.num_experiences_in_buffer)

    choices = np.arange(self.num_experiences_in_buffer)
    size= min(self.num_replays_per_dream, self.num_experiences_in_buffer)
    inds = random_stream.choice(choices, size=size, p=sample_weights)
    replays = self.replay_buffer[inds]

    linear_qfa_qlearn_helper(replays, R, iteration, costs, self.a, self.s, self.n_arms, self.average_mode, self.Q, self.C, self.Cp, 
      self.isa_counter, self.lam, self.gamma, self.currentIndices, self.state_list)

  def plot_weights(self):
    print('isa counters')
    print('do')
    for arm in range(self.n_arms):
      print(self.isa_counter[arm])
    # print(self.isa_counter[arm, state_log[arm][0], a])
    import matplotlib.pyplot as plt
    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    plt.rc('font', weight='bold')
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    for s_i in range(2):
      for a_i in range(1, self.a):
        fig,ax = plt.subplots(2,3, figsize=(14,8))
        ax = ax.reshape(-1)
        colors = ['r','g','b','c']
        wi_vals = [-0.5, 0.5, 1, -1]
        linestyles = ['-',':','-.','--']
        for arm in range(2):
          for weight in range(self.num_weights):
            if arm == 0:
              ax[arm].plot(self.QOverTime[arm,:,s_i, a_i, weight], alpha=0.5, label='weight:%s'%(weight))
            else:
              ax[arm].plot(self.QOverTime[arm,:,s_i, a_i, weight], alpha=0.5)
            
            # ax[arm].plot([0, self.iterations], [0, 0], color=colors[s_i], linestyle='--')
        fig.suptitle('MAIQL: Q weights S:%s, A:%a'%(s_i, a_i))
        fig.legend(ncol=4,loc='lower center')
        plt.savefig('q_weights_over_time_arm%s_state%s_action%a_wi_maql.png'%(arm,s_i,a_i),dpi=200)
        plt.show()




