import sys

SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)
from simulate import *
import numpy as np

# Prepare working directory
main_dir = os.path.abspath(os.path.dirname(os.getcwd()))
subject_dir = 'data/simulation_1condition_exclude_neutral/simulation_pr_best'
# subject_dir = 'data/simulation_2condition_include_neutral/simulation_pr_best'

# Load HCP subject ids
subject_ids = np.sort([f.split('/')[-1].split('.')[-2] for f in glob.glob(os.path.join(main_dir, 'data', 'gambling_trials', '*.csv'))])

# Load best fit parameter sets
param_list = load_best_parameter_list(main_dir=main_dir, log_dir=subject_dir)

print('>>> START SIMULATION: <%s> \n \t SUBJ [%d] \t NUM SIMULATION [%d]' % (subject_dir, len(subject_ids), len(param_list)))

# Start simulation
run_maxLL_pr_pipline(main_dir=main_dir, subject_dir=subject_dir, param_list=param_list, overwrite=True)

print('>>>END...MAXLL_FACTOR_VAR: [%s] \n MAXLL_EXCLUDE_NEUTRAl: [%s]' % (str(MAXLL_FACTOR_VAR), str(MAXLL_EXCLUDE_NEUTRAl)))