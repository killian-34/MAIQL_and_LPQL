import numpy as np
import pandas as pd
import time

import utils
from itertools import product

import lp_methods

import os
import argparse
import tqdm 
import itertools

import mdptoolbox
from numba import jit


import rmab_ql
import simulation_environments

import matplotlib.pyplot as plt


index_policies = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
TS_policies = [21, 22, 25]


def takeAction(current_states, T, actions, random_stream):

    N=len(current_states)

    ###### Get next state
    next_states=np.zeros(current_states.shape)
    for i in range(N):

        current_state=int(current_states[i])
        next_state=np.argmax(random_stream.multinomial(1, T[i, current_state, int(actions[i]), :]))
        next_states[i]=next_state

    return next_states


def getActions(N, T_hat, R, C, B, k, valid_action_combinations=None, current_state=None, 
                policy_option=0, gamma=0.95, indexes=None,
                output_data=None, True_T=None, qlearning_objects=None, learning_random_stream=None, t=None,
                action_to_play=None, exact_knapsack=True):

   
    if policy_option==0:
        ################## Nobody
        return np.zeros(N)

    elif policy_option==1:
        ################## Everybody
        return np.ones(N)

    elif policy_option==2:
        ################## Random
        # Randomly pick from all valid options
        choices = np.arange(valid_action_combinations.shape[0])
        choice = np.random.choice(choices)
        return valid_action_combinations[choice]



    # Fast random, inverse weighted
    elif policy_option==3:

        actions = np.zeros(N,dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:
            
            # select an action at random
            num_valid_actions_left = len(C[C<=B-current_action_cost])
            p = 1/(C[C<=B-current_action_cost]+1)
            p = p/p.sum()
            p = None
            a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
            current_action_cost += C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                break

            actions[arm] = a


        return actions

    # 2-action random
    elif policy_option==4:

        actions = np.zeros(N,dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:
            
            # select an action at random
            num_valid_actions_left = len(C[C<=B-current_action_cost])
            a = action_to_play
            current_action_cost += C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                break

            actions[arm] = a


        return actions


    # discrete-chunk random
    elif policy_option==5:

        actions = np.zeros(N,dtype=int)

        # can choose each action up to num_actions - 1 times
        choice_list = np.zeros((C.shape[0]-1,N),dtype=int)
        for i in range(0, C.shape[0]-1):
            choice_list[i] = np.arange(N)
        choice_list = choice_list.reshape(-1)

        current_action_cost = 0
        process_order = np.random.choice(choice_list, int(B), replace=False)
        for arm in process_order:

            actions[arm] += 1


        return actions

    # Hawkins
    elif policy_option==21 or policy_option == 42:

        T = None
        if policy_option == 21:
            T = T_hat
        elif policy_option == 42:
            T = True_T

        # print(policy_option)
        # print(T)

        actions = np.zeros(N,dtype=int)

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros((N, C.shape[0], T.shape[1]))

        L_vals, lambda_val = lp_methods.hawkins(T, R, C, B, current_state, lambda_lim=lambda_lim, gamma=gamma)
        


        


        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] - lambda_val*C[a] + gamma*L_vals[i].dot(T[i,s,a])
        output_data['hawkins_lambda'].append(lambda_val)

        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B, exact_knapsack=exact_knapsack)

        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)

        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")



        # return  np.random.randint(low=0, high=C.shape[0], size=N)
        return actions



    # LP to compute the index policies (online vs. oracle version)
    elif policy_option==22 or policy_option == 23:

        T = None
        if policy_option == 22:
            T = T_hat
        elif policy_option == 23:
            T = True_T

        # print(policy_option)
        # print(T)

        actions = np.zeros(N,dtype=int)

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros((N, C.shape[0]))

        for a in range(1,C.shape[0]):
            L_vals, indexes[:,a] = lp_methods.lp_to_compute_index(T, R, C, B, current_state, a, lambda_lim=lambda_lim, gamma=gamma)


        # compute all indexes
        if t==1 and policy_option==23:
            all_indexes = np.zeros((N,T.shape[1],T.shape[2]))
            for s in range(T.shape[1]):
                state_vec = np.ones(N,dtype=int)*s
                for a in range(1,C.shape[0]):
                    # print(state_vec)
                    _, all_indexes[:,s,a] = lp_methods.lp_to_compute_index(T, R, C, B, state_vec, a, lambda_lim=lambda_lim, gamma=gamma)


            output_data['lp-oracle-index'] = all_indexes

        output_data['lp_index_method_values'].append(indexes)



        
        # placeholder is not for the 0th action, just to make sure the last line of loop doesn't break
        placeholder = np.ones((N,1))*(-np.inf)
        indexes = np.concatenate([indexes[:,1:], placeholder], axis=1)
        # print(indexes)
        current_action_cost = 0
        plus_one_action_indexes = indexes[:, 0]

        # Break if all arms assigned most expensive action (i.e., budget too large)
        while not (actions.astype(int) == C.shape[0]-1).all():

            # greedily select next best action
            ind_of_max = np.argmax(plus_one_action_indexes)

            # increase budget
            current_action_cost -= C[actions[ind_of_max]]
            actions[ind_of_max] += 1
            current_action_cost += C[actions[ind_of_max]]

            # if the next selection takes us over budget, break and undo the action
            if current_action_cost > B:
                actions[ind_of_max] -= 1
                break

            if current_action_cost == B:
                break

            # else, shift by one the index for that arm
            plus_one_action_indexes[ind_of_max] = indexes[ind_of_max, actions[ind_of_max]]
            # print(ind_of_max, actions[ind_of_max])

        
        # print(actions)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")
       

        return actions





    # VfNc - Value function, No costs
    elif policy_option==24 or policy_option == 25:
        actions = np.zeros(N)
        
        T = None
        indexes=None
        if policy_option == 24:
            T = True_T
            indexes = output_data['Oracle_vfnc_indexes']
        elif policy_option == 25:
            T = T_hat
        



            V = np.zeros((N,T.shape[1]))

            for i in range(N):

                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] = R[i]

                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='full', epsilon=output_data['mdp_epsilon'])

                mdp.run()

                V[i] = np.array(mdp.V)

            # print('value iteration run time:',time.time()-start)
            indexes = np.zeros((N,C.shape[0],T.shape[1]))
            for i in range(N):
                for a in range(C.shape[0]):
                    for s in range(T.shape[1]):
                        indexes[i,a,s] = R[i,s] + gamma*V[i].dot(T[i,s,a])


        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        start = time.time()
        indexes = np.zeros((N, C.shape[0], T.shape[1]))

        # start = time.time()
        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B, exact_knapsack=exact_knapsack)

        actions = np.argmax(decision_matrix, axis=1)
        # print(actions)
        # 1/0

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

        return actions


    # VFNC QL
    elif policy_option in [26]:
        vfnc_object = qlearning_objects['vfnc_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if vfnc_object.check_random(t, random_stream=learning_random_stream):
            
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=3, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions


        Q = vfnc_object.Q
        Q_current = np.zeros((N, C.shape[0]))
        for arm in range(N):
            Q_current[arm] = Q[arm,current_state[arm]]


        decision_matrix = lp_methods.action_knapsack(Q_current, C, B, exact_knapsack=exact_knapsack)


        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)

        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")


        return actions






    # WIBQL
    elif policy_option in [46, 51]:
        wiqbl_object = qlearning_objects['wiqbl_object']
        st = time.time()
        actions = np.zeros(N,dtype=int)

        action_to_play = 1
        if policy_option == 46:
            action_to_play = 1
        elif policy_option == 51:
            action_to_play = 2

        # with prob epsilon, explore randomly
        # This call will also decay epsilon        
        if wiqbl_object.check_random(t, random_stream=learning_random_stream):
            # print('Doing a random')
            return getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=4, indexes=indexes, output_data=output_data, True_T=True_T,
                             t=t, qlearning_objects=qlearning_objects, action_to_play=action_to_play)



        placeholder = np.ones((N,1))*(-np.inf)
        indexes_per_arm = wiqbl_object.get_indexes()
        sorted_arm_inds = np.argsort(indexes_per_arm)[::-1]

        # print(wiqbl_object.get_indexes())
        # print(wiqbl_object.lam)



        playable_arms = int(B//(C[action_to_play]))
        arms_to_play = sorted_arm_inds[:playable_arms]

        actions[arms_to_play] = action_to_play
        
        # print(actions)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

       

        return actions



    # Multi-action q learning
    elif policy_option in [48]:

        maiql_tabular_object = qlearning_objects['maiql_tabular_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if maiql_tabular_object.check_random(t, random_stream=learning_random_stream):

            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                         policy_option=3, indexes=indexes, output_data=output_data, True_T=True_T, 
                         t=t, qlearning_objects=qlearning_objects)
        
            return actions


       

        # placeholder is not for the 0th action, just to make sure the last line of loop doesn't break
        placeholder = np.ones((N,1))*(-np.inf)
        indexes = np.concatenate([maiql_tabular_object.get_indexes()[:,1:], placeholder], axis=1)
        # print(indexes)
        current_action_cost = 0
        plus_one_action_indexes = indexes[:, 0]
        # print(maiql_tabular_object.lam)

        # Break if all arms assigned most expensive action (i.e., budget too large)
        while not (actions.astype(int) == C.shape[0]-1).all():
            # print('p1a:',plus_one_action_indexes)
            # greedily select next best action
            ind_of_max = np.argmax(plus_one_action_indexes)

            # increase budget
            current_action_cost -= C[actions[ind_of_max]]
            actions[ind_of_max] += 1
            current_action_cost += C[actions[ind_of_max]]

            # if the next selection takes us over budget, break and undo the action
            if current_action_cost > B:
                actions[ind_of_max] -= 1
                break

            if current_action_cost == B:
                break

            # else, shift by one the index for that arm
            plus_one_action_indexes[ind_of_max] = indexes[ind_of_max, actions[ind_of_max]]
            # print(ind_of_max, actions[ind_of_max])

        # print(indexes)
        # print(actions)
        # print(current_state)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")
       

        return actions



    # Multi-action q learning, linear q function approximation
    elif policy_option in [49]:
        maiql_lqfa_object = qlearning_objects['maiql_lqfa_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if maiql_lqfa_object.check_random(t, random_stream=learning_random_stream):
            # print('Doing a random')
            if N <= 10:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=5, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            else:
                actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=5, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions


       

        # placeholder is not for the 0th action, just to make sure the last line of loop doesn't break
        placeholder = np.ones((N,1))*(-np.inf)
        indexes = np.concatenate([maiql_lqfa_object.get_indexes()[:,1:], placeholder], axis=1)
        # print(indexes)
        current_action_cost = 0
        plus_one_action_indexes = indexes[:, 0]

        # Break if all arms assigned most expensive action (i.e., budget too large)
        while not (actions.astype(int) == C.shape[0]-1).all():
            # print('p1a:',plus_one_action_indexes)
            # greedily select next best action
            ind_of_max = np.argmax(plus_one_action_indexes)

            # increase budget
            current_action_cost -= C[actions[ind_of_max]]
            actions[ind_of_max] += 1
            current_action_cost += C[actions[ind_of_max]]

            # if the next selection takes us over budget, break and undo the action
            if current_action_cost > B:
                actions[ind_of_max] -= 1
                break

            if current_action_cost == B:
                break

            # else, shift by one the index for that arm
            plus_one_action_indexes[ind_of_max] = indexes[ind_of_max, actions[ind_of_max]]
            # print(ind_of_max, actions[ind_of_max])

        
        # print(actions)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")
       

        return actions



    # Multi-action q learning, tabular, constrained
    elif policy_option in [50]:
        maiql_tabular_constrained_object = qlearning_objects['maiql_tabular_constrained_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if maiql_tabular_constrained_object.check_random(t, random_stream=learning_random_stream):
            
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=5, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions


       

        # placeholder is not for the 0th action, just to make sure the last line of loop doesn't break
        placeholder = np.ones((N,1))*(-np.inf)
        indexes = np.concatenate([maiql_tabular_constrained_object.get_indexes()[:,1:], placeholder], axis=1)
        # print(indexes)
        current_action_cost = 0
        plus_one_action_indexes = indexes[:, 0]

        # Break if all arms assigned most expensive action (i.e., budget too large)
        while not (actions.astype(int) == C.shape[0]-1).all():
            # print('p1a:',plus_one_action_indexes)
            # greedily select next best action
            ind_of_max = np.argmax(plus_one_action_indexes)

            # increase budget
            current_action_cost -= C[actions[ind_of_max]]
            actions[ind_of_max] += 1
            current_action_cost += C[actions[ind_of_max]]

            # if the next selection takes us over budget, break and undo the action
            if current_action_cost > B:
                actions[ind_of_max] -= 1
                break

            if current_action_cost == B:
                break

            # else, shift by one the index for that arm
            plus_one_action_indexes[ind_of_max] = indexes[ind_of_max, actions[ind_of_max]]
            # print(ind_of_max, actions[ind_of_max])

        
        # print(actions)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")
       

        return actions




    # LQ Qlearning tabular
    elif policy_option in [60]:
        lpql_tabular_object = qlearning_objects['lpql_tabular_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if lpql_tabular_object.check_random(t, random_stream=learning_random_stream):
            
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=3, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions


        Q = lpql_tabular_object.Q
        V_current = np.zeros((N,lpql_tabular_object.n_lams))
        Q_current = np.zeros((N,lpql_tabular_object.n_lams, C.shape[0]))
        for arm in range(N):
            # get the value function for each arm of the current state, as a function of lambda
            # index 1 is ':' to ask for the estimates of Q for various values of lambda
            Q_current[arm] = Q[arm,:,current_state[arm]]
            V_current[arm] = Q_current[arm].max(axis=1)
       
        sum_of_V_per_lam = V_current.sum(axis=0)
        sum_of_V_per_lam_shift = np.zeros(sum_of_V_per_lam.shape)
        sum_of_V_per_lam_shift[0:-1] = sum_of_V_per_lam[1:]
        sum_of_V_per_lam_shift[-1] = sum_of_V_per_lam_shift[-2]
        sum_of_V_per_lam_diffs = sum_of_V_per_lam_shift - sum_of_V_per_lam

        lams = lpql_tabular_object.lams
        lams_shift = np.zeros(lams.shape)
        lams_shift[0:-1] = lams[1:]
        lams_shift[-1] = lams_shift[-2]
        lams_diff = lams_shift - lams

        # find the value of lambda that makes the sum of slopes of V wrt lambda
        # greater than the negative slope on the linear lambda term of hawkins 2.5
        sum_of_V_slopes = sum_of_V_per_lam_diffs / lams_diff
        slope_to_beat =  -B / (1- lpql_tabular_object.gamma)


        slope_is_gtoet_slope_to_beat = sum_of_V_slopes >= slope_to_beat
        lam_index = 0
        for i in range(lpql_tabular_object.n_lams):
            if slope_is_gtoet_slope_to_beat[i]:
                lam_index = i
                break

        # print(Q_current)
        # print(Q_current.shape)
        Q_current = Q_current[:,lam_index]
        # print(Q_current)


        decision_matrix = lp_methods.action_knapsack(Q_current, C, B, exact_knapsack=exact_knapsack)

        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)

        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")


        # return  np.random.randint(low=0, high=C.shape[0], size=N)
        return actions



    # LQ Qlearning lqfa
    elif policy_option in [61]:
        lpql_lqfa_object = qlearning_objects['lpql_lqfa_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if lpql_lqfa_object.check_random(t, random_stream=learning_random_stream):
            
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=3, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions


        Q = lpql_lqfa_object.Q
        V_current = np.zeros((N,lpql_lqfa_object.n_lams))
        Q_current = np.zeros((N,lpql_lqfa_object.n_lams, C.shape[0]))
        for arm in range(N):
            # get the value function for each arm of the current state, as a function of lambda
            # index 1 is ':' to ask for the estimates of Q for various values of lambda
            Q_current[arm] = lpql_lqfa_object.get_all_Q(arm,current_state[arm])
            V_current[arm] = Q_current[arm].max(axis=1)
       
        sum_of_V_per_lam = V_current.sum(axis=0)
        sum_of_V_per_lam_shift = np.zeros(sum_of_V_per_lam.shape)
        sum_of_V_per_lam_shift[0:-1] = sum_of_V_per_lam[1:]
        sum_of_V_per_lam_shift[-1] = sum_of_V_per_lam_shift[-2]
        sum_of_V_per_lam_diffs = sum_of_V_per_lam_shift - sum_of_V_per_lam

        lams = lpql_lqfa_object.lams
        lams_shift = np.zeros(lams.shape)
        lams_shift[0:-1] = lams[1:]
        lams_shift[-1] = lams_shift[-2]
        lams_diff = lams_shift - lams

        # find the value of lambda that makes the sum of slopes of V wrt lambda
        # greater than the negative slope on the linear lambda term of hawkins 2.5
        sum_of_V_slopes = sum_of_V_per_lam_diffs / lams_diff
        slope_to_beat =  -B / (1- lpql_lqfa_object.gamma)


        slope_is_gtoet_slope_to_beat = sum_of_V_slopes >= slope_to_beat
        lam_index = 0
        for i in range(lpql_lqfa_object.n_lams):
            if slope_is_gtoet_slope_to_beat[i]:
                lam_index = i
                break

        # print(Q_current)
        # print(Q_current.shape)
        Q_current = Q_current[:,lam_index]
        # print(Q_current)


        decision_matrix = lp_methods.action_knapsack(Q_current, C, B, exact_knapsack=exact_knapsack)


        # print(decision_matrix)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)

        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")

        return actions


    # MAIQL Approx tabular
    elif policy_option in [70]:
        maiql_approx_tab_object = qlearning_objects['maiql_approx_tab_object']

        actions = np.zeros(N,dtype=int)

        # with prob epsilon, explore randomly
        # This call will also decay epsilon 
        if maiql_approx_tab_object.check_random(t, random_stream=learning_random_stream):
            
            actions = getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=current_state,
                             policy_option=3, indexes=indexes, output_data=output_data, True_T=True_T, 
                             t=t, qlearning_objects=qlearning_objects)
            return actions



        # placeholder is not for the 0th action, just to make sure the last line of loop doesn't break
        placeholder = np.ones((N,1))*(-np.inf)
        indexes = np.concatenate([maiql_approx_tab_object.get_indexes(current_state)[:,1:], placeholder], axis=1)
        # print(indexes)
        current_action_cost = 0
        plus_one_action_indexes = indexes[:, 0]

        # Break if all arms assigned most expensive action (i.e., budget too large)
        while not (actions.astype(int) == C.shape[0]-1).all():
            # print('p1a:',plus_one_action_indexes)
            # greedily select next best action
            ind_of_max = np.argmax(plus_one_action_indexes)

            # increase budget
            current_action_cost -= C[actions[ind_of_max]]
            actions[ind_of_max] += 1
            current_action_cost += C[actions[ind_of_max]]

            # if the next selection takes us over budget, break and undo the action
            if current_action_cost > B:
                actions[ind_of_max] -= 1
                break

            if current_action_cost == B:
                break

            # else, shift by one the index for that arm
            plus_one_action_indexes[ind_of_max] = indexes[ind_of_max, actions[ind_of_max]]
            # print(ind_of_max, actions[ind_of_max])


        # print(actions)
        # print(C)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")
       

        return actions





# Need to update this for multi-action
def update_counts(actions, state_log, counts):
    for arm, a in enumerate(actions):
        a=int(a)
        s = state_log[arm, 0]
        sprime = state_log[arm, 1]
        counts[arm, s, a, sprime] += 1
    return counts



def thompson_sampling(N, T_shape, priors, counts, random_stream):

    T_hat = np.zeros(T_shape)
    for i in range(N):
        for j in range(T_hat.shape[1]):
            for k in range(T_hat.shape[2]):
                params = priors[i, j, k, :] + counts[i, j, k, :]
                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat

def thompson_sampling_constrained(N, priors, counts, random_stream):

    T_hat = np.zeros((N,2,2,2))
    for i in range(N):
    	# While sampled T_hat is not valid or has not been sampled yet...
    	while (not verify_T_matrix(T_hat[i]) or T_hat[i].sum() == 0):
	        for j in range(T_hat.shape[1]):
	            for k in range(T_hat.shape[2]):
	                params = priors[i, j, k, :] + counts[i, j, k, :]
	                T_hat[i, j, k, :] = random_stream.dirichlet(params)
    return T_hat


def simulateExperiment(N, L, T, R, C, B, k, policy_option,
                        action_logs={}, cumulative_state_log=None, 
                        seedbase=None, savestring='trial', learning_mode=False, 
                        world_random_seed=None, learning_random_seed=None, verbose=False,
                        file_root=None, gamma=0.95,
                        output_data=None, start_state=None, do_plot=None, n_lams=None, pname=None,
                        config=None, exact_knapsack=True):



    learning_random_stream = np.random.RandomState()
    if learning_mode > 0:
        learning_random_stream.seed(learning_random_seed)

    world_random_stream = np.random.RandomState()
    world_random_stream.seed(world_random_seed)


    # set up thompson sampling
    T_hat = None
    priors = np.ones(T.shape)
    counts = np.zeros(T.shape)

    last_called = np.zeros(N).astype(int)


    qlearning_objects = {}



    eps = config['epsilon_wibql']
    C_one = config['C_one_wibql']
    Cp = config['Cp_wibql']
    n_states = T.shape[1]
    n_actions = 2
    average_mode = True

    replay_buffer_size = int(config['replay_buffer_size_wibql'])
    num_replays_per_dream=int(config['num_replays_per_dream_wibql'])
    replay_period=int(config['replay_period_wibql'])
    
    lambda_lb = -R.max()/(C[C>0].min()*(1-gamma))
    if 'lambda_bound_wibql' in config.keys():
        lambda_lb = config['lambda_bound_wibql']

    learn_divisor = config['learn_divisor_wibql']
    epsilon_divisor = config['epsilon_divisor_wibql']

    
    lambda_ub = -lambda_lb

    wiqbl_object = rmab_ql.RMABQL_WIBQL(N, k, eps, C_one, Cp, gamma, 
        L, n_states, n_actions, initial_exploration = False, 
        initial_Q_values_as_0 = True, eps_decay = True, average_mode = average_mode,
        replay_buffer_size=replay_buffer_size, num_replays_per_dream=num_replays_per_dream, 
        replay_period=replay_period, lambda_lb=lambda_lb, lambda_ub=lambda_ub,
        learn_divisor=learn_divisor, epsilon_divisor=epsilon_divisor)
    qlearning_objects['wiqbl_object'] = wiqbl_object
    # borkarQ





    eps = config['epsilon_maiql']

    C_one = config['C_one_maiql']
    Cp = config['Cp_maiql']
    n_states = T.shape[1]
    n_actions = T.shape[2]
    average_mode = False

    # No replays
    replay_buffer_size = int(config['replay_buffer_size_maiql'])
    num_replays_per_dream=int(config['num_replays_per_dream_maiql'])
    replay_period=int(config['replay_period_maiql'])
    
    lambda_lb = -R.max()/(C[C>0].min()*(1-gamma))
    if 'lambda_bound_maiql' in config.keys():
        lambda_lb = config['lambda_bound_maiql']

    learn_divisor = config['learn_divisor_maiql']
    epsilon_divisor = config['epsilon_divisor_maiql']

    lambda_ub = -lambda_lb

    maiql_tabular_object = rmab_ql.MultiActionQLTabular(N, k, eps, C_one, Cp, gamma, 
        L, n_states, n_actions, initial_exploration = False, 
        initial_Q_values_as_0 = True, eps_decay = True, average_mode = average_mode,
        replay_buffer_size=replay_buffer_size, num_replays_per_dream=num_replays_per_dream, 
        replay_period=replay_period, lambda_lb=lambda_lb, lambda_ub=lambda_ub,
        learn_divisor=learn_divisor, epsilon_divisor=epsilon_divisor)
    qlearning_objects['maiql_tabular_object'] = maiql_tabular_object
    #MaiqlQ



    eps = config['epsilon_lpql']
    C_one = config['C_one_lpql']


    n_states = T.shape[1]
    n_actions = T.shape[2]
    average_mode = False

    # No replays

    replay_buffer_size = int(config['replay_buffer_size_lpql'])
    num_replays_per_dream=int(config['num_replays_per_dream_lpql'])
    replay_period=int(config['replay_period_lpql'])

    lambda_lim = R.max()/(C[C>0].min()*(1-gamma))


    if 'lambda_bound_lpql' in config.keys():
        lambda_lim = config['lambda_bound_lpql']

    learn_divisor = config['learn_divisor_lpql']
    epsilon_divisor = config['epsilon_divisor_lpql']


    lpql_tabular_object = rmab_ql.LPQLTabular(N, k, eps, C_one, gamma, 
        L, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
        initial_Q_values_as_0 = True, eps_decay = True, average_mode = average_mode,
        replay_buffer_size=replay_buffer_size, num_replays_per_dream=num_replays_per_dream, 
        replay_period=replay_period, learn_divisor=learn_divisor, epsilon_divisor=epsilon_divisor)
    # print(lpql_tabular_object.lams)
    # 1/0
    qlearning_objects['lpql_tabular_object'] = lpql_tabular_object
    # lpqlq



    eps = config['epsilon_lam0']
    C_one = config['C_one_lam0']

    n_states = T.shape[1]
    n_actions = T.shape[2]
    average_mode = False

    replay_buffer_size = int(config['replay_buffer_size_lam0'])
    num_replays_per_dream=int(config['num_replays_per_dream_lam0'])
    replay_period=int(config['replay_period_lam0'])

    learn_divisor = config['learn_divisor_lam0']
    epsilon_divisor = config['epsilon_divisor_lam0']
    

    vfnc_object = rmab_ql.VfNCQL(N, k, eps, C_one, gamma, 
        L, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
        initial_Q_values_as_0 = True, eps_decay = True, average_mode = average_mode,
        replay_buffer_size=replay_buffer_size, num_replays_per_dream=num_replays_per_dream, 
        replay_period=replay_period,
        learn_divisor=learn_divisor, epsilon_divisor=epsilon_divisor)
    qlearning_objects['vfnc_object'] = vfnc_object


    eps = config['epsilon_maiql_aprx']
    C_one = config['C_one_maiql_aprx']
    
    n_states = T.shape[1]
    n_actions = T.shape[2]
    average_mode = False

    # No replays
    replay_buffer_size = int(config['replay_buffer_size_maiql_aprx'])
    num_replays_per_dream=int(config['num_replays_per_dream_maiql_aprx'])
    replay_period=int(config['replay_period_maiql_aprx'])

    lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

    if 'lambda_bound_maiql_aprx' in config.keys():
        lambda_lim = config['lambda_bound_maiql_aprx']

    learn_divisor = config['learn_divisor_maiql_aprx']
    epsilon_divisor = config['epsilon_divisor_maiql_aprx']

    maiql_approx_tab_object = rmab_ql.MAIQLApproxTabular(N, k, eps, C_one, gamma, 
        L, n_states, n_actions, n_lams, lambda_lim, initial_exploration = False, 
        initial_Q_values_as_0 = True, eps_decay = True, average_mode = average_mode,
        replay_buffer_size=replay_buffer_size, num_replays_per_dream=num_replays_per_dream, 
        replay_period=replay_period,
        learn_divisor=learn_divisor, epsilon_divisor=epsilon_divisor)
    qlearning_objects['maiql_approx_tab_object'] = maiql_approx_tab_object



    if policy_option == 24:
        V = np.zeros((N,T.shape[1]))

        for i in range(N):
            T_i = np.swapaxes(T[i],0,1)
            R_i = np.zeros(T_i.shape)
            for x in range(R_i.shape[0]):
                for y in range(R_i.shape[1]):
                    R_i[x,:,y] = R[i]

            mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='full', epsilon=output_data['mdp_epsilon'])

            mdp.run()

            V[i] = np.array(mdp.V)
        indexes = np.zeros((N,C.shape[0],T.shape[1]))
        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] + gamma*V[i].dot(T[i,s,a])
        output_data['Oracle_vfnc_indexes'] = indexes


    state_log=np.zeros((N,L), dtype=int)
    actions_record=np.zeros((N, L-1))

    if action_logs is not None:
        action_logs[policy_option] = []


    indexes = np.zeros((N,C.shape[0]))


    print('Running simulation w/ policy: %s'%policy_option)
    

    if start_state is not None:
        state_log[:,0] = start_state
    else:
        state_log[:,0] = 1


    #######  Run simulation #######
    print('Running simulation w/ policy: %s'%policy_option)
    # make array of nan to initialize observations
    observations = np.full(N, np.nan)
    learning_modes=['no_learning', 'Online']
    print("Learning mode:", learning_modes[learning_mode])
    print("Policy:", pname[policy_option])
        


    # if problem size is small enough, enumerate all valid actions
    # to use for random exploration
    # else, we will use "fast" random
    valid_action_combinations = None
    if policy_option in [2, 4, 48, 49, 41, 43, 44, 45] and N <= 5:
        options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
        valid_action_combinations = utils.list_valid_action_combinations(N,C,B,options)

    # Fill with (s,a,r,s) tuples
    replay_buffer = []
    count_of_action_2_rounds = 0
    # for t in tqdm.tqdm(range(1,L)):
    for t in range(1,L):
        # print(t)
        
        '''
        Learning T_hat from simulation so far
        '''
        #print("Round: %s"%t)

        st = time.time()        
        T_hat = None
        if learning_mode==0:
            T_hat=T
        elif learning_mode == 1 and policy_option in TS_policies:
            # Thompson sampling
            T_hat = thompson_sampling(N, T.shape, priors, counts, random_stream=learning_random_stream)

        
        # EPSILON_CLIP=0.0005
        # T_hat= utils.epsilon_clip(T_hat, EPSILON_CLIP)
        
            #print ("ERROR: ", error(T_hat, T))
            #print ("TO be implemented: Learning mode True")

        #### Epsilon greedy part
        
        actions=getActions(N, T_hat, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:,t-1],
                             policy_option=policy_option,  gamma=gamma,
                             indexes=indexes, output_data=output_data, True_T=T, learning_random_stream=learning_random_stream,
                             t=t, qlearning_objects=qlearning_objects, exact_knapsack=exact_knapsack)
        
                
        # print('action cost')
        # payment = 0
        # for i in range(len(actions)):
        #     payment += C[int(actions[i])]
        # print(payment)


        actions_record[:, t-1]=actions

        # debug
        # print("Actions:")
        # print(actions)
        # print('state')
        # print(state_log[:,t-1])
        if (actions==2).any():
            
            count_of_action_2_rounds+=1

        if action_logs is not None:
            action_logs[policy_option].append(actions.astype(int))



        state_log[:,t] = takeAction(state_log[:,t-1].reshape(-1), T, actions, random_stream=world_random_stream)


        if policy_option in [46, 51]:
            wiqbl_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%wiqbl_object.replay_period == 0 and t > 2:
                wiqbl_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [48]:
            maiql_tabular_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%maiql_tabular_object.replay_period == 0 and t > 1000:
                maiql_tabular_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [49]:
            maiql_lqfa_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%maiql_lqfa_object.replay_period == 0 and t > 2:
                maiql_lqfa_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [50]:
            maiql_tabular_constrained_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%maiql_tabular_constrained_object.replay_period == 0 and t > 2:
                maiql_tabular_constrained_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [60]:
            lpql_tabular_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%lpql_tabular_object.replay_period == 0 and t > 2:
                lpql_tabular_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [61]:
            lpql_lqfa_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%lpql_lqfa_object.replay_period == 0 and t > 2:
                lpql_lqfa_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)


        if policy_option in [26]:
            vfnc_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%vfnc_object.replay_period == 0 and t > 2:
                vfnc_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)

        if policy_option in [70]:
            maiql_approx_tab_object.qlearn(actions, state_log[:, t-1:], R, t, C, random_stream=learning_random_stream)
            if t%maiql_approx_tab_object.replay_period == 0 and t > 2:
                maiql_approx_tab_object.learn_from_replays(t, R, C, random_stream=learning_random_stream)


        if learning_mode == 1:
            update_counts(actions, state_log[:, t-1:], counts)

    # print(state_log)
    print('num action 2 rounds',count_of_action_2_rounds)
    # 1/0





    if cumulative_state_log is not None:
        cumulative_state_log[policy_option] = np.cumsum(state_log.sum(axis=0))

    print("Final Indexes")


    # if policy_option == 48 and do_plot:

    #     maiql_tabular_object.plot_indexes(output_data['lp-oracle-index'])

    #     print("isa counter")
    #     print(maiql_tabular_object.isa_counter)
    #     print("isa replay counter")
    #     print(maiql_tabular_object.isa_replay_counter)
    #     print("sum")
    #     print(maiql_tabular_object.isa_replay_counter+maiql_tabular_object.isa_counter)

    # if policy_option in [46,51] and do_plot:
    #     wiqbl_object.plot_indexes()
    #     print("isa counter")
    #     print(wiqbl_object.isa_counter)
    #     print("isa replay counter")
    #     print(wiqbl_object.isa_replay_counter)
    #     print("sum")
    #     print(wiqbl_object.isa_replay_counter+wiqbl_object.isa_counter)




    return state_log



# example command:
# python3 ma_rmab_online_simulation.py -pc -1 -l 10000 -d eng14 -s 0 -ws 0 -sv testing --n_lams 100 -g 0.95 -lr 1 -N 3 -n 16 --budget_frac 0.5 --config_file config_eng14.csv

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run multi-action RMAB online learning simulations with various methods.')
    parser.add_argument('-n', '--num_arms', default=2, type=int, help='Number of Processes')
    parser.add_argument('-b', '--budget_frac', default=0.5, type=float, help='Budget per round as fraction of n')
    parser.add_argument('-l', '--simulation_length', default=1000, type=int, help='Number of rounds to run simulation')
    parser.add_argument('-N', '--num_trials', default=3, type=int, help='Number of trials to run')
    parser.add_argument('-S', '--num_states', default=2, type=int, help='Number of states per process')
    parser.add_argument('-A', '--num_actions', default=2, type=int, help='Number of actions per process')
    parser.add_argument('-g', '--discount_factor', default=0.95, type=float, help='Discount factor for MDP solvers')
    parser.add_argument('-nl', '--n_lams', default=1000, type=int, help='Number of lambda test points for LPQL and MAIQL-Aprx')

    parser.add_argument('-hl', '--history_length', default=1, type=int, help='History length for tb_data_history simulation')
    parser.add_argument('-adm', '--action_diff_mult', default=3, type=int, help='Parameter that multiplies the difference in action effect between 0 and 1 for tb_data_history')

    parser.add_argument('-d', '--data', default='real', choices=['full_random_online','eng14','eng15'], 
        type=str,help='Method for generating transition probabilities of processes')

    parser.add_argument('-me', '--mdp_epsilon', default=1e-1, type=float, help='Tolerance for Value Iteration')

    parser.add_argument('-s', '--seed_base', type=int, help='Base for the numpy random seed')
    parser.add_argument('-ws','--world_seed_base', default=None, type=int, help='Base for the world random seed')
    parser.add_argument('-ls','--learning_seed_base', default=None, type=int, help='Base for learning algorithm random seeds')

    parser.add_argument('-f', '--file_root', default='./..', type=str,help='Root dir for experiment (should be the dir above this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='Policy to run, default is all policies in policy array')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Used to separate seeding if running slurm_array_id-based workflow')
    parser.add_argument('-sv', '--save_string', default='', type=str, help='Special string to include in saved file name')

    parser.add_argument('-sid', '--slurm_array_id', default=None, type=int, help='Specify the index of the parameter combo (if running many parallel jobs on slurm)')

    parser.add_argument('-lr', '--learning_option', default=1, choices=[0,1], type=int, help='0: No Learning (Ground truth known)\n1: Online\nOnly use 0 for debugging.')
    
    parser.add_argument('-cf', '--config_file', default="config.csv", type=str, help='Config file setting all algorithm hyperparameters')

    args = parser.parse_args()

    config = utils.parse_config_file(args.config_file)

    policies = []

    
    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()

    ##### Save special name
    if args.save_string=='':
        args.save_string=str(time.ctime().replace(' ', '_').replace(':','_'))

    ##### Policies to run
    if args.policy<0:

        # Full algorithm suite from paper
        # policies = [0, 23, 24, 26, 46, 48, 51, 60, 70, 42]

        # policies used for the plots in each simulated domain from the paper
        if args.data == 'eng14':
            policies = [0, 24, 26, 48, 46, 51, 60, 70, 42]
        elif args.data == 'full_random_online':
            policies = [0, 23, 24, 48, 60, 70, 42]
        elif ars.data == 'eng15':
            policies = [0, 24, 26, 48, 60, 70, 42]


    else:
        policies=[args.policy]



    # policy names dict
    pname={
        0: 'No Actions',    2: 'Random',
        3: 'FastRandom', 
        21:'Hawkins-Thompson',
        23:'Oracle-LP-Index',

        24:r'Oracle $\lambda=0$',
        25:'TS-VfNc',
        26:r'QL-$\lambda=0$',

        42:'Oracle LP',

        46:'WIBQL a=1',
        48:'MAIQL',
        51:'WIBQL a=2',

        60:'LPQL',
        70:'MAIQL-Aprx',

    }

    # if set, will only run this many iterations for the policy - this is used for expensive oracles
    truncated_policy_lengths={
            21:5000,
            42:1000,
            22:5000,
            23:1000,
            24:1000
        }


    N = 0
    k = 0




    ##########################################################
    #
    ###############
    #
    # This block created the loop for eng14 (two-process) experiments, slurm-based batching
    #
    ###############


    # for i in 0 46 51 26 70 60 48 24 42; do sbatch --array=0-239 job.run_simulation_eng14.sh $i; done
    NUM_TRIALS = 20
    trial_number_list = [i for i in range(NUM_TRIALS)]
    n_list = [16, 32, 48]
    budget_list = [0.25, 0.5, 1.0, 1.5]
    master_combo_list = list(itertools.product(trial_number_list, n_list, budget_list))
    if args.slurm_array_id is not None:
        combo = master_combo_list[args.slurm_array_id]
        args.trial_number = combo[0]
        args.num_arms = combo[1] # num processes
        args.budget_frac = combo[2]


    ##########################################################




    ##########################################################
    #
    ###############
    #
    # This block created the loop for full_random experiments, slurm-based batching
    #
    ###############


    # # for i in 0 23 70 60 48 24 42; do sbatch --array=0-59 job.run_simulation_full_random.sh $i; done
    # NUM_TRIALS = 20
    # trial_number_list = [i for i in range(NUM_TRIALS)]
    # n_list = [16]
    # action_list = [2, 5, 10]
    # state_list = [5]
    # master_combo_list = list(itertools.product(trial_number_list, n_list, action_list, state_list))
    # if args.slurm_array_id is not None:
    #     combo = master_combo_list[args.slurm_array_id]
    #     args.trial_number = combo[0]
    #     args.num_arms = combo[1] # num processes
    #     args.num_actions = combo[2]
    #     args.num_states = combo[3]

    
    ###########################################################



    ##########################################################
    #
    ###############
    #
    # This block created the loop for eng15 (TB patient-derived) experiments, slurm-based batching
    #
    ###############


    # # for i in 0 26 70 60 48 24 42 46 51; do sbatch --array=0-179 job.run_simulation_eng15.sh $i; done

    # NUM_TRIALS = 20
    # trial_number_list = [i for i in range(NUM_TRIALS)]
    # n_list = [16, 32, 48]
    # budget_list = [0.25]
    # history_length_list = [2, 3, 4]

    # master_combo_list = list(itertools.product(trial_number_list, n_list, budget_list, history_length_list))
    # if args.slurm_array_id is not None:
    #     combo = master_combo_list[args.slurm_array_id]

    #     args.trial_number = combo[0]
    #     args.num_arms = combo[1] # num processes
    #     args.budget_frac = combo[2]
    #     args.history_length = combo[3]

    ###########################################################



    


    # If we pass a trial number, that means we are running this as a job
    # and we want jobs/trials to run in parallel so this does some rigging to enable that,
    # while still keeping separate all the seeds
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0

    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial

    first_learning_seedbase=np.random.randint(0, high=100000)
    if args.learning_seed_base is not None:
        first_learning_seedbase = args.learning_seed_base+add_to_seed_for_specific_trial

    N=args.num_arms
    L=args.simulation_length
    k=0
    savestring=args.save_string
    N_TRIALS=args.num_trials
    LEARNING_MODE=args.learning_option
    
    
    record_policy_actions=[ 5, 6, 7, 21, 27, 24, 30, 31, 32, 33, 34, 35, 36,
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                            50, 60, 100
                           ]

    size_limits={   
                    0:None, 2:8, 3:None, 4:8, 
                    5:None, 6:None, 7:None,
                    19:None, 20:None, 21:None, 22:None, 23:None,
                    24:None, 25:None, 26:None, 27:None, 28:None,
                    30:None, 31:None, 32:None, 33:None, 34:None,
                    35:None, 36:None, 37:None, 38:None, 39:None,
                    40:None, 41:None, 42:None, 43:None, 44:None,
                    45:None, 46:None, 47:None, 48:None, 49:None,
                    50:None, 51:None, 60:None, 61:None, 70:None,
                    100:None
                }

    

    



    # for rapid prototyping
    # use this to avoid updating all the function calls when you need to pass in new
    # algo-specific things or return new data
    output_data = {}

    # list because one for each trial
    output_data['hawkins_lambda'] = []
    output_data['lp_index_method_values'] = []

    output_data['mdp_epsilon'] = args.mdp_epsilon



    state_log=dict([(key,[]) for key in pname.keys()])
    action_logs = {}
    cumulative_state_log = {}

    mean_reward_log=dict([(key,[]) for key in pname.keys()])

    window_size = 500
    mean_reward_log_moving_avg=dict([(key,[]) for key in pname.keys()])

    start=time.time()
    file_root=args.file_root
    
    runtimes = np.zeros((N_TRIALS, len(policies)))
    
    for i in range(N_TRIALS):

        # do_plot = i==0
        do_plot=False

        # use np global seed for rolling random data, then for random algorithmic choices
        seedbase = first_seedbase + i
        np.random.seed(seed=seedbase)

        # Use world seed only for evolving the world (If two algs 
        # make the same choices, should create the same world for same seed)
        world_seed_base = first_world_seedbase + i

        # Use learning seed only for processes involving learning (i.e., exploration vs. exploitation)
        learning_seed_base = first_learning_seedbase + i

        print ("Seed is", seedbase)

        T = None
        R = None
        C = None
        B = None
        start_state = None
        exact_knapsack = True

        # --------------------------------
        
        if args.data =='full_random_online':
            REWARD_BOUND = 1
            start_state = np.zeros(N)
            exact_knapsack = False

            T, R, C, B = simulation_environments.get_full_random_experiment(N, args.num_states, args.num_actions, REWARD_BOUND)


        if args.data =='eng14':
            REWARD_BOUND = 1

            percent_greedy = 0.25

            B=args.budget_frac*N

            T, R, C = simulation_environments.get_eng14_experiment(N, args.num_actions, percent_greedy, REWARD_BOUND)
            print(T.shape)
            print(C)
            args.num_states = T.shape[1]
            args.num_actions = C.shape[0]
            start_state = np.ones(N)

        if args.data =='eng15':
            REWARD_BOUND = 1

            percent_greedy = 0.25

            B=round(N*(percent_greedy))*2
            # B = int(np.ceil(N*.51))
            T, R, C, B, start_state = simulation_environments.get_tb_patients_plus_needy_with_history(N, args.num_actions, args.history_length, args.action_diff_mult, REWARD_BOUND, percent_greedy)
            print(T.shape)
            print(C)
            print(B)
            args.num_states = T.shape[1]
            start_state = np.ones(N)



             
        np.random.seed(seed=seedbase)


        for p,policy_option in enumerate(policies):


            policy_start_time=time.time()
            if size_limits[policy_option]==None or size_limits[policy_option]>N:
                np.random.seed(seed=seedbase)
                

                L_in = L
                if policy_option in truncated_policy_lengths.keys():
                    L_in = truncated_policy_lengths[policy_option]
                    if L_in > L:
                        L_in = L

                state_matrix=simulateExperiment(N, L_in, T, R, C, B, k, policy_option=policy_option, seedbase=seedbase, action_logs=action_logs, 
                                                   cumulative_state_log=cumulative_state_log, learning_mode=LEARNING_MODE, 
                                                   learning_random_seed=learning_seed_base, world_random_seed=world_seed_base, 
                                                   file_root=file_root, output_data=output_data, start_state=start_state, do_plot=do_plot, n_lams=args.n_lams,
                                                   pname=pname, gamma=args.discount_factor, config=config, exact_knapsack=exact_knapsack)



                np.save(file_root+'/logs/state_log/states_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions), state_matrix)
                

                reward_matrix = np.zeros(state_matrix.shape)
                for ind_i in range(state_matrix.shape[0]):
                    for ind_j in range(state_matrix.shape[1]):
                        reward_matrix[ind_i,ind_j] = (args.discount_factor**ind_j)*R[ind_i, state_matrix[ind_i, ind_j]]




                state_log[policy_option].append(np.sum(reward_matrix))
                # mean_reward_log[policy_option].append(np.mean(reward_matrix.cumsum(axis=1),axis=0))
                reward_matrix = np.zeros(state_matrix.shape)



                # longterm average
                for ind_i in range(state_matrix.shape[0]):
                    for ind_j in range(state_matrix.shape[1]):
                        reward_matrix[ind_i,ind_j] = R[ind_i, state_matrix[ind_i, ind_j]]
                
                np.save(file_root+'/logs/results/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions), reward_matrix)

                reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                mean_reward_log[policy_option].append(np.sum(reward_matrix_cumulative,axis=0))


                # convolved_reward_matrix = []

                # # size of convolution/sliding window
                ws = window_size
                reward_matrix = np.zeros(state_matrix.shape)
                for ind_i in range(state_matrix.shape[0]):
                    for ind_j in range(state_matrix.shape[1]):
                        reward_matrix[ind_i,ind_j] = R[ind_i, state_matrix[ind_i, ind_j]]


                    reward_matrix[ind_i] = np.convolve(reward_matrix[ind_i], np.ones(ws)/ws, mode='same')
                    

                mean_reward_log_moving_avg[policy_option].append(np.sum(reward_matrix,axis=0))
                


                # np.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, 
                #     linewidth=np.inf, suppress=None, nanstr=None, 
                #     infstr=None, formatter=None, sign=None, 
                #     floatmode=None, legacy=None)



            policy_end_time=time.time()
            policy_run_time=policy_end_time-policy_start_time
            np.save(file_root+'/logs/runtime/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions), policy_run_time)

            runtimes[i,p] = policy_run_time


        ##### SAVE ALL RELEVANT LOGS #####

        # write out action logs
        for policy_option in action_logs.keys():
            fname = os.path.join(args.file_root,'logs/action_logs/action_logs_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions))
            columns = list(map(str, np.arange(N)))
            df = pd.DataFrame(action_logs[policy_option], columns=columns)
            df.to_csv(fname, index=False)

       

    end=time.time()
    print ("Time taken: ", end-start)


    for i,p in enumerate(policies):
        # print (pname[p],": ", np.mean(state_log[p]))
        print (pname[p],": ", runtimes[:,i].mean())

    # exit()


    if args.policy<0:
       
        policies_to_plot = policies

        # bottom = 0

        labels = [pname[i] for i in policies_to_plot]
        values=[np.mean(mean_reward_log[i],axis=0) for i in policies_to_plot]
        fill_between_0=[np.percentile(mean_reward_log[i], 25, axis=0) for i in policies_to_plot]
        fill_between_1=[np.percentile(mean_reward_log[i], 75, axis=0) for i in policies_to_plot]



        utils.rewardPlot(labels, values, fill_between_0=fill_between_0, fill_between_1=fill_between_1,
            ylabel='Mean cumulative reward',
            title='Data: %s, Patients %s, Budget: %s, S: %s, A: %s' % (args.data, N, B, args.num_states, args.num_actions),
            filename='img/online_trajectories_mean_cumu_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s.pdf'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions), 
            root=args.file_root)



        labels = [pname[i] for i in policies_to_plot]
        values=[np.mean(mean_reward_log_moving_avg[i],axis=0) for i in policies_to_plot]
        fill_between_0=[np.percentile(mean_reward_log_moving_avg[i], 25, axis=0) for i in policies_to_plot]
        fill_between_1=[np.percentile(mean_reward_log_moving_avg[i], 75, axis=0) for i in policies_to_plot]

        utils.rewardPlot(labels, values, fill_between_0=fill_between_0, fill_between_1=fill_between_1,
            ylabel='Moving average reward (ws=%s)'%window_size,
            title='Data: %s, Patients %s, Budget: %s, S: %s, A: %s' % (args.data, N, B, args.num_states, args.num_actions),
            filename='img/online_trajectories_moving_average_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_a%s.pdf'%(savestring, N,args.budget_frac,L_in,policy_option,args.data,seedbase,args.num_states,args.num_actions), 
            root=args.file_root, x_ind_for_line = -window_size)

