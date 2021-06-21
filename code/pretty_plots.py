import matplotlib
# matplotlib.use('pdf')

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib import rcParams
from numba import jit 

#Ensure type 1 fonts are used
import matplotlib as mpl
# mpl.rcParams['ps.useafm'] = True
# mpl.rcParams['pdf.use14corefonts'] = True
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.unicode']=True


# policy names dict
pname={
        0: 'No Calls',    2: 'Random',
        3: 'FastRandom',    4: 'MDPoptimal', 
        21:'Hawkins-Thompson',
        22:'LP-index',      23:'Oracle-LP-Index',

        24:r'Oracle $\lambda=0$',
        25:'TS-VfNc',
        26:r'QL-$\lambda=0$',

        42:'Oracle LP',

        46:'WIBQL a=1',
        47:'QWIC',
        48:'MAIQL',
        49:'MAIQL_LQFA',
        50:'MAIQL_TAB_CON',
        51:'WIBQL a=2',


        60:'LPQL',
        61:'LPQL-LQFA',
        70:'MAIQL-Aprx',


        100:'arpita_sim1_optimal'
    }


# Reduced to the map list for the paper
# color_map = {
#         23:'Oracle-LP-Index',
#         24:r'Oracle $\lambda=0$',
#         26:r'QL-$\lambda=0$',
#         42:'Oracle LP',
#         46:'WIBQL a=1',
#         48:'MAIQL',
#         51:'WIBQL a=2',
#         60:'LPQL',
#         70:'MAIQL-Aprx',
# }

# https://cran.r-project.org/web/packages/unikn/readme/README.html
# color_map = {
#         23:'#59c7eb',
#         24:'#0a9086',
#         42:'#e0607e',
#         26:'#fea090',
#         46:'#3e5496',
#         48:'#ff0000',
#         51:'#efdc60',
#         60:'#8e2043',
#         70:'#000000',
# }

# https://cran.r-project.org/web/packages/unikn/readme/README.html
color_map = {
        23:'#59c7eb',
        24:'#ff0000',
        42:'#57009e',
        26:'#fea090',
        46:'#3e5496',
        48:'#8e2043',
        51:'#ad9621',
        60:'#0a9086',
        70:'#000000',
}

# oracles are non-solid, others solid
linestyle_map = {
        23:'--',
        24:'-.',
        42:':',
        26:'-',
        46:'-',
        48:'-',
        51:'-',
        60:'-',
        70:'-',
}

# https://matplotlib.org/stable/api/markers_api.html
linemarker_map = {
        23:',',                 # pixel (basically none)
        24:',',                 # pixel (basically none)
        42:',',                 # pixel (basically none)
        26:'P',                 # Plus
        46:'d',                 # thin diamond
        48:'v',                 # triangle down
        51:'D',                 # thick diamond
        60:'o',                 # circle
        70:'s',                 # square
}

# Make marker spacings co-prime for everyone that is on a plot together
markevery_map = {
        23:1,
        24:1,
        26:7,
        42:1,
        46:3.5,   # these only show up with 60, 42, 70, 24
        48:3,
        51:6,   # these only show up with 60, 42, 70, 24
        60:2,
        70:5,
}


# SMALL_SIZE = 30
# MEDIUM_SIZE = 36
# BIGGER_SIZE = 36
SMALL_SIZE = 16
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', weight='bold')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def convergencePlotEng14VaryNSingle(savestring, root, firstseed, num_trials, ignore_missing=False):
    

    n_list = [16, 32, 48]
    budget_frac_list = [0.25, 0.5]
    state_size_list = [2] 

    
    policies_to_plot = [42, 60, 70, 48, 26, 46, 51, 24]
    long_len = 50000
    L_list = [1000, long_len, long_len, long_len, long_len, long_len, long_len, 1000]

    
    

    means = np.zeros((len(n_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(n_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(n_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'

    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    
    fig, ax = plt.subplots(3,1,figsize=(9,8))
    ax = ax.reshape(3,1)
    col = 0

    budget_frac = budget_frac_list[1]

    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, prob_size in enumerate(n_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0


                # if we hit the walltime or some other error
                except Exception as e:
                    print(e)
                    pass

            rewards = np.array(rewards)

            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']



    # want maximum of 10 spaces
    MAX_MARKERS = 10
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1

    for x,prob_size in enumerate(n_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x,col].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)


        
      
        
        ax[x,col].grid()

    ax[0,0].set_ylim(9,12.5)
    # ax[0,1].set_ylim(9,12.5)
    ax[1,0].set_ylim(18,25)
    # ax[1,1].set_ylim(18,25)
    ax[2,0].set_ylim(24,38)
    # ax[2,1].set_ylim(24,38)
    
    # ax[0,1].set_yticks([8,9,10])

    plt.setp( ax[0,0].get_xticklabels(), visible=False)
    plt.setp( ax[1,0].get_xticklabels(), visible=False)
    # plt.setp( ax[1].get_xticklabels(), visible=False)
    
    # change xtick scale
    xpos = np.linspace(0,50000, 6)
    xlabel = list(map(str, np.linspace(0,50, 6).astype(int)))
    ax[2,0].set_xticks(xpos)
    ax[2,0].set_xticklabels(xlabel)
    # ax[2,1].set_xticks(xpos)
    # ax[2,1].set_xticklabels(xlabel)

    plt.subplots_adjust(top=0.85)#bottom=0.2, left=0.2, right=0.8, )

    

    fig.supylabel('Mean Cumulative Reward')
    ax[2,0].set_xlabel('Timesteps (x1e3)')
    
    
    all_handles = []
    all_labels = []

    # get all the handles from the first plot, except the last since that's shared
    handles, labels = ax[0,0].get_legend_handles_labels()

    all_handles+=handles
    all_labels+=labels


    fig.legend(all_handles, all_labels, loc='upper center', 
        bbox_to_anchor=[0,0,1,1], ncol=3)
    

    outname = 'eng14_vary_n_results_single_camera_ready.png'

    plt.savefig(outname,dpi=300)
    plt.show()






def convergencePlotEng14(savestring, root, firstseed, num_trials, sv_tag, ignore_missing=False):
    


    n_list = [16]
    budget_frac_list = [0.25, 0.5]
    state_size_list = [2] 

    


    policies_to_plot = [42, 60, 70, 48, 26, 24]
    long_len = 50000
    L_list = [1000, long_len, long_len, long_len, long_len, 1000]

    if sv_tag == 'binary_action':
        policies_to_plot = [42, 60, 70, 46, 51, 24]
        L_list = [1000, long_len, long_len, long_len, long_len, 1000]
    

    means = np.zeros((len(budget_frac_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(budget_frac_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(budget_frac_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'


    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    


    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, budget_frac in enumerate(budget_frac_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0


                # if we hit the walltime or some other error
                except Exception as e:
                    print(e)
                    pass

            rewards = np.array(rewards)

            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']

    fig, ax = plt.subplots(2,1,figsize=(4.5,8))
    ax = ax.reshape(-1)


    # want maximum of 20 spaces
    MAX_MARKERS = 20
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1

    for x,budget_frac in enumerate(budget_frac_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 0
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = len(policies_to_plot)-1
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 1
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)


        
      
        
        ax[x].grid()


    ax[0].set_ylim(6,11)
    ax[1].set_ylim(8,12.5)
    # ax[2].set_ylim(7.5,11)

    plt.setp( ax[0].get_xticklabels(), visible=False)
    # plt.setp( ax[1].get_xticklabels(), visible=False)
    
    # change xtick scale
    xpos = np.linspace(0,50000, 6)
    xlabel = list(map(str, np.linspace(0,50, 6).astype(int)))
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(xlabel)

    plt.subplots_adjust(bottom=0.2, left=0.2, right=0.8, top=0.8 )

    

    fig.supylabel('Mean Cumulative Reward')
    ax[1].set_xlabel('Timesteps (x1e3)')
    

    ax[0].legend( bbox_to_anchor=(0.5,1), loc='lower center', ncol=len(policies_to_plot)//3)

    

    outname = 'eng14_vary_budget_results_%s_camera_ready.png'%sv_tag

    plt.savefig(outname,dpi=300)
    plt.show()
    # plt.clf()






def convergencePlotEng14Dual(savestring, root, firstseed, num_trials, ignore_missing=False):
    
    

    # savestring = 'eng11_trials_resubmit'
    n_list = [16]
    budget_frac_list = [0.25, 0.5]
    state_size_list = [2] # actually the n_actions_list = [10, 20, 30]

    


    policies_to_plot = [42, 60, 70, 48, 26, 24]
    long_len = 50000
    L_list = [1000, long_len, long_len, long_len, long_len, 1000]
    

    means = np.zeros((len(budget_frac_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(budget_frac_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(budget_frac_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'



    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    
    fig, ax = plt.subplots(2,2,figsize=(9,6))
    col = 0

    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, budget_frac in enumerate(budget_frac_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0


                # if we hit the walltime or some other error
                except Exception as e:
                    print(e)
                    pass

            rewards = np.array(rewards)
            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']



    # want maximum of 10 spaces
    MAX_MARKERS = 10
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1

    for x,budget_frac in enumerate(budget_frac_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 0
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = len(policies_to_plot)-1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x,col].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)


        
      
        
        ax[x,col].grid()






    ##############################
    ########## second two plots
    ##############################

    policies_to_plot = [42, 60, 70, 48, 46, 51, 24]
    L_list = [1000, long_len, long_len, long_len, long_len, long_len, 1000]


    means = np.zeros((len(budget_frac_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(budget_frac_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(budget_frac_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'


    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    


    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, budget_frac in enumerate(budget_frac_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0


                # if we hit the walltime or some other error
                except Exception as e:
                    # if not ignore_missing:
                    #     run_times.append(60*60*4) # seconds, minutes, hours
                    print(e)
                    # 1/0
                    pass

            rewards = np.array(rewards)
            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']


    # HOP
    # want maximum of 10 spaces
    MAX_MARKERS = 10
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1
    col = 1

    for x,budget_frac in enumerate(budget_frac_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 0
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = len(policies_to_plot)-1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x,col].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)
        ax[x,col].grid()



    #################################
    ########## second two plots end.
    #################################



    ax[0,0].set_ylim(4,11)
    ax[1,0].set_ylim(9,12.5)
    ax[0,1].set_ylim(8,11)
    ax[1,1].set_ylim(9,12.5)
    
    ax[0,1].set_yticks([8,9,10])

    plt.setp( ax[0,0].get_xticklabels(), visible=False)
    plt.setp( ax[0,1].get_xticklabels(), visible=False)
    # plt.setp( ax[1].get_xticklabels(), visible=False)
    
    # change xtick scale
    xpos = np.linspace(0,50000, 3)
    xlabel = list(map(str, np.linspace(0,50, 3).astype(int)))
    ax[1,0].set_xticks(xpos)
    ax[1,0].set_xticklabels(xlabel)
    ax[1,1].set_xticks(xpos)
    ax[1,1].set_xticklabels(xlabel)

    plt.subplots_adjust(top=0.8, bottom=0.2, left=0.13)

    

    fig.supylabel('Mean Cumulative Reward')
    xlabel_fontsize=20
    ax[1,0].set_xlabel('Timesteps (x1e3)',fontsize=xlabel_fontsize)
    ax[1,1].set_xlabel('Timesteps (x1e3)',fontsize=xlabel_fontsize)

    all_handles = []
    all_labels = []

    # get all the handles from the first plot, except the last since that's shared
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles, labels = handles[:-1], labels[:-1]
    all_handles+=handles
    all_labels+=labels

    # only get the wibql handles from the second plot, and the shared final one
    handles, labels = ax[0,1].get_legend_handles_labels()
    handles, labels = handles[-3:], labels[-3:]
    all_handles+=handles
    all_labels+=labels

    fig.legend(all_handles, all_labels, loc='upper center', 
        bbox_to_anchor=[0,0,1,1], ncol=len(policies_to_plot)//2)
    # ax[0,0].legend( bbox_to_anchor=[0.5, 1, 1, 1], loc='upper center', ncol=len(policies_to_plot)//2)
    # plt.tight_layout()    
    

    outname = 'eng14_vary_budget_results_combined_camera_ready.png'

    plt.savefig(outname,dpi=300)
    plt.show()
    # plt.clf()




def convergencePlotEng14VaryNDual(savestring, root, firstseed, num_trials, ignore_missing=False):
    
    


    n_list = [16, 32, 48]
    budget_frac_list = [0.25, 0.5]
    state_size_list = [2] 

    

    policies_to_plot = [42, 60, 70, 48, 26, 24]
    long_len = 50000
    L_list = [1000, long_len, long_len, long_len, long_len, 1000]

    

    means = np.zeros((len(n_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(n_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(n_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'

    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    
    fig, ax = plt.subplots(3,2,figsize=(9,8))
    col = 0

    budget_frac = budget_frac_list[1]

    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, prob_size in enumerate(n_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0


                # if we hit the walltime or some other error
                except Exception as e:
                    # if not ignore_missing:
                    #     run_times.append(60*60*4) # seconds, minutes, hours
                    print(e)
                    # 1/0
                    pass

            rewards = np.array(rewards)
            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']


    # HOP
    # want maximum of 10 spaces
    MAX_MARKERS = 10
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1

    for x,prob_size in enumerate(n_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x,col].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)


        
      
        
        ax[x,col].grid()



    ##############################
    ########## second two plots
    ##############################

    policies_to_plot = [42, 60, 70, 48, 46, 51, 24]
    L_list = [1000, long_len, long_len, long_len, long_len, long_len, 1000]


    means = np.zeros((len(n_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(n_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(n_list),len(policies_to_plot)),dtype=int)



    datatype = 'eng14'

    state_size = 2
    a = 3
    prob_size = 16
    ws=0
    


    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, prob_size in enumerate(n_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1
                    
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0



                # if we hit the walltime or some other error
                except Exception as e:

                    print(e)
                    # 1/0
                    pass

            rewards = np.array(rewards)

            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']


    # HOP
    # want maximum of 10 spaces
    MAX_MARKERS = 10
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1
    col = 1

    for x,prob_size in enumerate(n_list):
        for i,p in enumerate(policies_to_plot):
            if p==24:
                print(prob_size,budget_frac,ws,L_list[i])
                print(means[x,i,ws:L_list[i]-ws])
            if p not in [24, 42, 23]:
                ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 0
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = len(policies_to_plot)-1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 1
                    ax[x,col].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
            # else:
            #     ax[x,col].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x,col].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)
        ax[x,col].grid()



    #################################
    ########## second two plots end.
    #################################



    ax[0,0].set_ylim(9,12.5)
    ax[0,1].set_ylim(9,12.5)
    ax[1,0].set_ylim(18,25)
    ax[1,1].set_ylim(18,25)
    ax[2,0].set_ylim(24,38)
    ax[2,1].set_ylim(24,38)
    
    # ax[0,1].set_yticks([8,9,10])

    plt.setp( ax[0,0].get_xticklabels(), visible=False)
    plt.setp( ax[0,1].get_xticklabels(), visible=False)
    plt.setp( ax[1,0].get_xticklabels(), visible=False)
    plt.setp( ax[1,1].get_xticklabels(), visible=False)
    # plt.setp( ax[1].get_xticklabels(), visible=False)
    
    # change xtick scale
    xpos = np.linspace(0,50000, 3)
    xlabel = list(map(str, np.linspace(0,50, 3).astype(int)))
    ax[2,0].set_xticks(xpos)
    ax[2,0].set_xticklabels(xlabel)
    ax[2,1].set_xticks(xpos)
    ax[2,1].set_xticklabels(xlabel)

    plt.subplots_adjust(top=0.85)

    

    fig.supylabel('Mean Cumulative Reward')
    ax[2,0].set_xlabel('Timesteps (x1e3)')
    ax[2,1].set_xlabel('Timesteps (x1e3)')

    all_handles = []
    all_labels = []

    # get all the handles from the first plot, except the last since that's shared
    handles, labels = ax[0,0].get_legend_handles_labels()
    handles, labels = handles[:-1], labels[:-1]
    all_handles+=handles
    all_labels+=labels

    # only get the wibql handles from the second plot, and the shared final one
    handles, labels = ax[0,1].get_legend_handles_labels()
    handles, labels = handles[-3:], labels[-3:]
    all_handles+=handles
    all_labels+=labels

    fig.legend(all_handles, all_labels, loc='upper center', 
        bbox_to_anchor=[0,0,1,1], ncol=len(policies_to_plot)//2)

    # plt.tight_layout()    
    

    outname = 'eng14_vary_n_results_combined_camera_ready.png'

    plt.savefig(outname,dpi=300)
    plt.show()
    # plt.clf()










def convergencePlotFullRandom(savestring, root, firstseed, num_trials, ignore_missing=False):
    
    


    actions_list = [2, 5, 10]

    


    policies_to_plot = [42, 24, 60, 70, 48, 23]
    L_list = [1000, 1000, 50000, 50000, 50000, 1000]

    

    means = np.zeros((len(actions_list),len(policies_to_plot), max(L_list)))
    all_runs = np.zeros((len(actions_list), len(policies_to_plot), num_trials, max(L_list)))
    runs_found = np.zeros((len(actions_list),len(policies_to_plot)),dtype=int)



    datatype = 'full_random_online'
    state_size = 5
    prob_size = 16
    budget_frac = 0.5
    ws = 100


    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, a in enumerate(actions_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:


                    reward_matrix = np.load(fname)
                    runs_found[x,i]+=1

                    for ind_i in range(reward_matrix.shape[0]):
                        reward_matrix[ind_i] = np.convolve(reward_matrix[ind_i], np.ones(ws)/ws, mode='same')


                    rewards.append(np.sum(reward_matrix,axis=0))

                    # print(run_times)
                # if we hit the walltime or some other error
                except Exception as e:
                    print(e)

                    pass

            rewards = np.array(rewards)
            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']

    fig, ax = plt.subplots(3,1,figsize=(9,8))
    ax = ax.reshape(-1)

    # want maximum of 20 spaces
    MAX_MARKERS = 20
    markevery_multipler = 50000 // MAX_MARKERS

    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'
    x_ind_for_line = -ws-1


    for x,a in enumerate(actions_list):
        for i,p in enumerate(policies_to_plot):
            if p not in [24, 42, 23]:
                ax[x].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = i
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)

        
        ax[x].grid()


    ax[0].set_ylim(7.5,9.5)
    ax[1].set_ylim(7.5,10.5)
    ax[2].set_ylim(7.5,11)

    plt.setp( ax[0].get_xticklabels(), visible=False)
    plt.setp( ax[1].get_xticklabels(), visible=False)

    # change xtick scale
    xpos = np.linspace(0,50000, 6)
    xlabel = list(map(str, np.linspace(0,50, 6).astype(int)))
    ax[2].set_xticks(xpos)
    ax[2].set_xticklabels(xlabel)

    

    ax[1].set_ylabel('Moving Average Reward (ws=%s)'%ws)
    ax[2].set_xlabel('Timesteps (x1e3)')
    ax[0].legend( bbox_to_anchor=(0.5,1), loc='lower center', ncol=len(policies_to_plot)//2)
    # plt.tight_layout()    
    

    outname = 'full_random_results_camera_ready.png'

    plt.savefig(outname,dpi=300)
    plt.show()
    # plt.clf()
















def convergencePlotEng15(savestring, root, firstseed, num_trials, ignore_missing=False):
    
    


    states_list = [4, 8, 16]

    policies_to_plot = [42, 60, 70, 48, 26, 24]
    long_len = 750000
    cut_len = 100000
    L_list = [1000, long_len, long_len, long_len, long_len, 1000]

    

    means = np.zeros((len(states_list),len(policies_to_plot), cut_len))
    all_runs = np.zeros((len(states_list), len(policies_to_plot), num_trials, cut_len))
    runs_found = np.zeros((len(states_list),cut_len),dtype=int)



    datatype = 'eng15'
    a = 3
    prob_size = 16
    budget_frac = 0.25
    ws = 100
    y_label = "Moving Average Reward (ws=%s)"%ws
    


    for i,p in enumerate(policies_to_plot):
        L = L_list[i]
        for x, state_size in enumerate(states_list):

            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_A%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, a)

                fname = os.path.join(root,filename)

                try:
                    reward_matrix = np.load(fname)
                    cut = min(cut_len,L)
                    reward_matrix = reward_matrix[:,:cut]

                    runs_found[x,i]+=1
                    
                    # mean cumulative reward
                    reward_matrix_cumulative = reward_matrix.cumsum(axis=1)
                    reward_matrix_cumulative = reward_matrix_cumulative / (np.arange(reward_matrix_cumulative.shape[1]) + 1)
                    rewards.append(np.sum(reward_matrix_cumulative,axis=0))
                    ws=0
                    y_label = "Mean Cumulative Reward"
                    
                # if we hit the walltime or some other error
                except Exception as e:

                    print(e)

                    pass

            rewards = np.array(rewards)
            means[x,i,:L] = rewards.mean(axis=0)
            all_runs[x,i,:runs_found[x,i],:L] = rewards



    colors = ['c','r','g','b','k','m','y','r','g','b']

    long_len = cut_len
    L_list = [1000, long_len, long_len, long_len, long_len, 1000]

    fig, ax = plt.subplots(3,1,figsize=(9,10))
    ax = ax.reshape(-1)

    # want maximum of 20 spaces
    MAX_MARKERS = 20
    markevery_multipler = cut_len // MAX_MARKERS
    print('cutlen',cut_len)
    print('mult',markevery_multipler)
    print('map',markevery_map)
    
    linewidth=3
    markersize=16
    alpha=0.7
    markeredgecolor='k'


    for x,state_size in enumerate(states_list):
        for i,p in enumerate(policies_to_plot):
            if p not in [24, 42, 23]:
                # ax[x].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], color=colors[i], alpha=0.75, linewidth=2)
                ax[x].plot(means[x,i,ws:L_list[i]-ws], label=pname[p], alpha=alpha, linewidth=linewidth,
                    color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                    markevery=int(markevery_map[p]*markevery_multipler),
                    markersize=markersize,
                    markeredgecolor=markeredgecolor)
            else:
                if p==42:
                    x_ind_for_line = -ws-1
                    lind = 2
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 0
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p ==24:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = len(policies_to_plot)-1
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)
                elif p == 23:
                    x0 = 0
                    x1 = L_list[lind]-2*ws
                    p_ind = 1
                    ax[x].plot([x0, x1],[means[x, p_ind, L_list[p_ind]+x_ind_for_line], means[x,p_ind,L_list[p_ind]+x_ind_for_line]], 
                        label=pname[p], alpha=0.75, linewidth=linewidth,
                        color=color_map[p], linestyle=linestyle_map[p], marker=linemarker_map[p],
                        markevery=int(markevery_map[p]*markevery_multipler),
                        markersize=markersize)

            fill_between_low = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 25, axis=0)
            fill_between_high = np.percentile(all_runs[x,i,:runs_found[x,i],ws:L_list[i]-ws], 75, axis=0)

            ax[x].fill_between(np.arange(L_list[i]-2*ws), fill_between_low, fill_between_high, color=color_map[p], alpha=0.2)

        
        ax[x].grid()


    ax[0].set_ylim(6,10)
    ax[1].set_ylim(6,11.5)
    ax[2].set_ylim(4,10)

    # hide xticks for top two plots
    plt.setp( ax[0].get_xticklabels(), visible=False)
    plt.setp( ax[1].get_xticklabels(), visible=False)

    # change xtick scale
    xpos = np.linspace(0,100000, 6)
    xlabel = list(map(str, np.linspace(0,100, 6).astype(int)))
    ax[2].set_xticks(xpos)
    ax[2].set_xticklabels(xlabel)

    
    ax[1].set_ylabel('Mean Cumulative Reward')

    ax[2].set_xlabel('Timesteps (x1e3)')
    ax[0].legend( bbox_to_anchor=(0.5,1), loc='lower center', ncol=len(policies_to_plot)//2)
    # plt.tight_layout()    
    


    outname = 'eng15_results_N%s_cumu_camera_ready.png'%prob_size

    plt.savefig(outname,dpi=300)
    plt.show()
    # plt.clf()





num_trials=20
########### eng14

savestring = 'eng14_2_7_3'
# reward_filename = 'eng_14_N%s_rewards.npy'
firstseed=0
root='/n/holylfs/LABS/tambe_lab/jkillian/maiql/'

convergencePlotEng14Dual(savestring, root, firstseed, num_trials, ignore_missing=True)


# sv_tag = 'vs_lam0'
# convergencePlotEng14(savestring, root, firstseed, num_trials, sv_tag, ignore_missing=True)

# sv_tag = 'binary_action'
# convergencePlotEng14(savestring, root, firstseed, num_trials, sv_tag, ignore_missing=True)





########### eng14

savestring = 'eng14_2_7_3'
# reward_filename = 'eng_14_N%s_rewards.npy'
firstseed=0
root='/n/holylfs/LABS/tambe_lab/jkillian/maiql/'

convergencePlotEng14VaryNSingle(savestring, root, firstseed, num_trials, ignore_missing=True)




########### full random


savestring = 'full_random_online_2_7'
firstseed = 0

root='/n/holylfs/LABS/tambe_lab/jkillian/maiql/'

convergencePlotFullRandom(savestring, root, firstseed, num_trials, ignore_missing=True)





######### eng15


savestring = 'eng15_2_7_3'
firstseed = 0

root='/n/holylfs/LABS/tambe_lab/jkillian/maiql/'


convergencePlotEng15(savestring, root, firstseed, num_trials, ignore_missing=True)









