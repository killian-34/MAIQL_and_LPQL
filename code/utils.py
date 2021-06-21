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


@jit(nopython=True)
def swap_axes(T,R):
    # T_i = np.swapaxes(T,0,1)
    T_i = np.transpose(T,(1,0,2))
    R_i = np.zeros(T_i.shape)
    for x in range(R_i.shape[0]):
        for y in range(R_i.shape[1]):
            R_i[x,:,y] = R

    return T_i, R_i

@jit(nopython=True)
def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0])
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options


def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)

def parse_config_file(fname):
    df = pd.read_csv(fname)
    config = dict([ (row[0], row[1]) for row in df.values])
    return config



def rewardPlot(labels, values, fill_between_0=None, fill_between_1=None, ylabel='Average Adherence out of 180 days',
            title='Adherence simulation for 20 patients/4 calls', filename='image.png', root='.', x_ind_for_line=-1):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    # x = np.arange(len(labels))  # the label locations
    
    colors = ['r','g','b','k','c','m','y','#875600','#380087','#ffad94']

    fig, ax = plt.subplots(figsize=(8,5))
    for i in range(len(labels)):
        ax.plot(values[i], label=labels[i], color=colors[i], alpha=0.75)
        if fill_between_0 is not None:
            ax.fill_between(np.arange(len(values[i])), fill_between_0[i], fill_between_1[i], color=colors[i], alpha=0.2)

    # x0 = 0
    # x1 = len(values[1])
    # ax.plot([x0, x1],[values[-1][x_ind_for_line], values[-1][x_ind_for_line]], color = colors[len(labels)-1])
    # x0 = 0
    # x1 = len(values[1])
    # ax.plot([x0, x1],[values[-2][x_ind_for_line], values[-2][x_ind_for_line]], color = colors[len(labels)-2])
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    # ax.set_ylim([3.5, 4.5])
    # ax.set_yscale("log")
    ax.set_title(title, fontsize=14)   
    ax.legend()
    
    
    plt.tight_layout() 
    plt.savefig(fname)
    plt.show()

