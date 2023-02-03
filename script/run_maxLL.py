import sys
import os
SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)
from simulate import *

# Prepare working directory
main_dir = os.path.abspath(os.path.dirname(os.getcwd()))
subject_dir = 'data/subject_data'

run_maxLL_pipline(main_dir=main_dir, subject_dir=subject_dir)