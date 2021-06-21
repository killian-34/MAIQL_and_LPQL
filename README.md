# MAIQL and LPQL
Q-Learning Multi-action Index and Lagrange Policies for Multi-Action Restless bandits

Code that accompanies paper: Killian JA, Biswas A, Shah S, Tambe M. Q-Learning Lagrange Policies for Multi-Action Restless Bandits. KDD'21.


## Setup

Main file is `code/ma_rmab_online_simulation.py`

#### To install follow these directions:

- Clone the repo
- Create the directory structure necessary to save experiments: `bash make_dirs.sh`
- Install the relevant packages -- see the bottom of this page for a comprehensive list of required packages, as well as full setup instructions starting from a new digital ocean linux server.
- Important: you'll need to install gurobi and gurobipy. An academic license for Gurobi can be obtained for free at https://www.gurobi.com/downloads/end-user-license-agreement-academic/



#### Experimental domains from the paper map to simulation environments as follows:

- Two-process == eng14
- Random == full_random_online
- Medication Adherence == eng15


## Running Experiments

Hyperparameters for each algorithm are set with config files (e.g., `config_eng14.csv`). There is one config file included in the repo for each experimental domain in the paper, set with the hyperparameters used to produce the results in the paper.


You can select different algorithms by editing the `policies` array on line 1390 (see line 1399 for all policy options, reproduced here):
```
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
```

For help on options like number of arms/states or setting the experimental domain, run `python3 ma_rmab_online_simulation.py -h`. Summary reproduced here:
```
  -h, --help            show this help message and exit
  -n NUM_ARMS, --num_arms NUM_ARMS
                        Number of Processes
  -b BUDGET_FRAC, --budget_frac BUDGET_FRAC
                        Budget per round as fraction of n
  -l SIMULATION_LENGTH, --simulation_length SIMULATION_LENGTH
                        Number of rounds to run simulation
  -N NUM_TRIALS, --num_trials NUM_TRIALS
                        Number of trials to run
  -S NUM_STATES, --num_states NUM_STATES
                        Number of states per process
  -A NUM_ACTIONS, --num_actions NUM_ACTIONS
                        Number of actions per process
  -g DISCOUNT_FACTOR, --discount_factor DISCOUNT_FACTOR
                        Discount factor for MDP solvers
  -nl N_LAMS, --n_lams N_LAMS
                        Number of lambda test points for LPQL and MAIQL-Aprx
  -d {full_random_online,eng14,eng15}, --data {full_random_online,eng14,eng15}
                        Method for generating transition probabilities of
                        processes
  -s SEED_BASE, --seed_base SEED_BASE
                        Base for the numpy random seed
  -ws WORLD_SEED_BASE, --world_seed_base WORLD_SEED_BASE
                        Base for the world random seed
  -ls LEARNING_SEED_BASE, --learning_seed_base LEARNING_SEED_BASE
                        Base for learning algorithm random seeds
  -f FILE_ROOT, --file_root FILE_ROOT
                        Root dir for experiment (should be the dir above this
                        script)
  -pc POLICY, --policy POLICY
                        Policy to run, default is all policies in policy array
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        Config file setting all algorithm hyperparameters

```


#### Command to run scaled down experiment with same params as in Fig 3, bottom row in the paper (Two-process setting):

`python3 ma_rmab_online_simulation.py -pc -1 -l 7000 -d eng14 -s 0 -ws 0 -sv testing --n_lams 100 -g 0.95 -lr 1 -N 3 -n 16 --budget_frac 0.5 --config_file config_eng14.csv`

Other two-process plots from the paper can be recreated similarly by adjusting `-n` and `--budget_frac` as needed.


#### Command to run scaled down experiment with same params as in Fig 5, bottom row in the paper (Random setting):

`python3 ma_rmab_online_simulation.py -pc -1 -l 7000 -d full_random_online -s 0 -ws 0 -sv testing --n_lams 100 -g 0.95 -lr 1 -N 3 -n 16 --budget_frac 0.5 --config_file config_full_random.csv -S 5 -A 5`


#### Medication adherence experiments require the adherence frequencies derived from Killian et al. 2019.




## Setup steps, Digital Ocean install
Tested as of June 21, 2021

Using 4 GB Memory / 2 Intel vCPUs / 80 GB Disk / NYC1 - Ubuntu 20.04 (LTS) x64, Python 3.8.5

- `git clone https://github.com/killian-34/MAIQL_and_LPQL`
- `bash make_dirs.sh`
- `apt install python3-pip --fix-missing`
- `pip install numpy pandas matplotlib tqdm numba scipy`
- In a web browser Register for a Gurobi account or login at https://www.gurobi.com/downloads/end-user-license-agreement-academic/ 
- Navigate to https://www.gurobi.com/downloads/ and select `Gurobi Optimizer`
- Review the EULA, then click `I accept the End User License Agreement`
- Identify the latest installation... as of writing, it is 9.1.2, and the following commands will reflect that. However, if the latest version has changed, you can replace 9.1.2 in the following commands with the newer version number and/or links on the Gurobi website.
- Navigate to `https://www.gurobi.com/wp-content/uploads/2021/04/README_9.1.2.txt` and read the README
- Back on the digial ocean server terminal: `mkdir tools`
- `cd tools`
- `wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz`
- `tar xvzf gurobi9.1.2_linux64.tar.gz`
- Add the following lines to your ~/.bashrc file, e.g., via `vim ~/.bashrc`
```
export GUROBI_HOME="/root/tools/gurobi912/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
``` 
- Run `source ~/.bashrc`
- On the browser, navigate to https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- Review the conditions, then click `I accept these conditions`
- Scroll down to **Installation** and copy the command that looks like `grbgetkey 00000000-0000-0000-0000-000000000000`, then paste and run in the server terminal window
- Enter `Y` to select the default options when prompted
- `cd ~/MAIQL_and_LPQL/code`
- `pip install gurobipy`
- Run this command to test: `python3 ma_rmab_online_simulation.py -pc -1 -l 1000 -d eng14 -s 0 -ws 0 -sv testing --n_lams 10 -g 0.95 -N 1 -n 4 --budget_frac 0.5 --config_file config_eng14.csv`



