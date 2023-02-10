import os
import sys

SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)
from simulate import *

# Prepare working directory
main_dir = os.path.abspath(os.path.dirname(os.getcwd()))
subject_dir = 'data/subject_data'

run_maxLL_pipline(main_dir=main_dir, subject_dir=subject_dir, overwrite=False, analysis_type='subject')

print('>>>END...MAXLL_FACTOR_VAR: [%s] \n MAXLL_EXCLUDE_NEUTRAl: [%s]' % (str(MAXLL_FACTOR_VAR), str(MAXLL_EXCLUDE_NEUTRAl)))