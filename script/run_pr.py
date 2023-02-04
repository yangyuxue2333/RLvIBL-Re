import random
import sys
import os
SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)
from simulate import *
import numpy as np
import random
import itertools

random.seed(0)

# Prepare working directory
main_dir = os.path.abspath(os.path.dirname(os.getcwd()))
subject_dir = 'data/simulation_pr'
# subject_dir = 'data/test'

ans = [0.05, 0.2, 0.3, 0.4, 0.5]
bll = [.2, .5, .8]
lf = [0.1]
egs =  [0.05, 0.2, 0.3, 0.4, 0.5]
alpha = [.2, .5, .8]
r = [0.1]

ls = list(itertools.product(*[ans, bll, lf, egs, alpha, r]))

# Preapre subject list
subject_ids = np.sort([f.split('/')[-1].split('.')[-2] for f in glob.glob(os.path.join(main_dir, 'data', 'gambling_trials', '*.csv'))])

# Prepare param list
param_list = []
for subject_id in subject_ids:
    for i in range(len(ls)):
        param_set = dict(zip(['ans', 'bll', 'lf', 'egs', 'alpha', 'r'], ls[i]))
        param_set['subject_id'] = subject_id
        param_set['param_id'] = 'param' + str(i)
        param_list.append(param_set)

print('>>> START SIMULATION: <%s> \nPARAM [%d] \t SUBJ [%d] \t NUM SIMULATION [%d]' % (subject_dir, len(ls), len(subject_ids), len(param_list)))

# Start simulation
run_maxLL_pr_pipline(main_dir=main_dir, subject_dir=subject_dir, param_list=param_list)

print('>>>END...MAXLL_FACTOR_VAR: [%s] \n MAXLL_EXCLUDE_NEUTRAl: [%s]' % (str(MAXLL_FACTOR_VAR), str(MAXLL_EXCLUDE_NEUTRAl)))