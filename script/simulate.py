import os

import pandas as pd

from device import *
import numpy as np
import glob
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

MAXLL_FACTOR_VAR = ['BlockType', 'TrialType']
ACTR_PARAMETER_NAME = {'model1': ['ans', 'bll', 'lf'], 'model2':['egs', 'alpha', 'r']}

#----------------------------------------#
# RUN SIMULATION
#----------------------------------------#

def simulate(epoch, subject_id, param_id, model, param_set, verbose, log_dir, special_suffix="", overwrite=True):
    """
    Run simulation
    """
    model_dat_list = []
    for i in range(epoch):
        g = GambleTask()
        g.setup(model=model,
                param_set=param_set,
                reload=True,
                subject_id=subject_id,
                verbose=True)
        model_dat = g.experiment()

        # log simulation data
        model_dat["Epoch"] = i
        param_names = g.get_parameters_name()
        for param_name in param_names:
            model_dat[param_name] = g.get_parameter(param_name)
        model_dat_list.append(model_dat)

    df_model = pd.concat(model_dat_list, axis=0)

    # process response switch
    df_model = process_model_data(df=df_model, model=model)

    if log_dir:
        save_simulation_data(log_dir=log_dir,
                       subject_id=subject_id,
                       param_id=param_id,
                       df_model=df_model,
                       special_suffix=special_suffix,
                       verbose=verbose,
                       overwrite=overwrite)

def save_simulation_data(log_dir, subject_id, param_id, df_model, special_suffix, verbose, overwrite):
    """
    Save samulation data
    """
    parent_dir = os.path.dirname(os.getcwd())

    # calculate aggregate data
    df_model['HCPID'] = subject_id
    model = df_model['model'].unique()[0]
    group_var = ['HCPID'] + MAXLL_FACTOR_VAR + ACTR_PARAMETER_NAME[model]
    df_agg = df_model.groupby(group_var).agg({'ResponseSwitch': 'mean'}).reset_index()

    ### MODEL SIMULATION DATA
    # dest_dir = os.path.join(parent_dir, log_dir, subject_id)
    dest_dir = os.path.join(parent_dir, log_dir, subject_id, param_id)
    log_path = os.path.join(dest_dir, actr.current_model() + special_suffix + ".csv")

    # check exist
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        if verbose: print('...SKIP: ALREADY COMPLETED SIM DATA ... [%s] [%s]' % (subject_id, param_id))
        if overwrite:
            df_model.to_csv(log_path, header=True, index=True, mode='w')
    else:
        # save model simulation data
        df_model.to_csv(log_path, header=True, index=True, mode='w')
    if verbose: print("...COMPLETE SIMULATION...\n")

    ### SIMULATION AGG DATA
    dest_agg_dir = os.path.join(os.path.join(dest_dir, 'aggregate'))
    agg_path = os.path.join(dest_agg_dir, actr.current_model() + special_suffix + ".csv")

    # check exist
    if check_exists_simple(dest_dir=dest_agg_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        if verbose: print('...SKIP: ALREADY COMPLETED SIM DATA ... [%s] [%s]' % (subject_id, param_id))
        if overwrite:
            df_agg.to_csv(agg_path, header=True, index=True, mode='w')
    else:
        df_agg.to_csv(agg_path, header=True, index=True, mode='w')
    if verbose: print("...COMPLETE AGG SIMULATION...\n")

    return df_model, df_agg

#----------------------------------------#
# PROCESS SUBJECT DATA
#----------------------------------------#

def process_subject_data(df, subject_id):
    """
    ResponseSwitch = FutureResponse - CurrentResponse
    Factor variable: [TrialType, BlockType]
    """
    assert 'FutureResponse' in df.columns
    # df['HCPID'] = subject_id
    # df['FutureResponse'] = df['Response'].shift(-1)
    # df['ResponseSwitch'] = df.apply(lambda x: 1 if x['Response'] != x['FutureResponse'] else 0, axis=1)
    df = df.dropna(axis=0, subset=['ResponseSwitch'])
    return df

def save_subject_aggregate_data(main_dir, subject_dir, subject_id, verbose=False):
    """
    Save HCP subjects aggregate data
    """
    ori_path = os.path.join(main_dir, subject_dir, subject_id, 'subject.csv')

    df = pd.read_csv(ori_path, usecols=['HCPID', 'BlockTypeCoded', 'TrialType', 'PreviousFeedback', 'PastResponse',
                                        'CurrentResponse', 'ResponseSwitch'])
    df = df.rename(columns={'BlockTypeCoded': 'BlockType'})
    df['ResponseSwitch'] = df.apply(lambda x: 0 if x['CurrentResponse'] == x['PastResponse'] else 1, axis=1)
    df = df.dropna(subset=['PastResponse'])

    #  calculate agg by group variable
    df_agg = df.groupby(['HCPID'] + MAXLL_FACTOR_VAR).agg({'ResponseSwitch': 'mean'}).reset_index()

    dest_dir = os.path.join(main_dir, subject_dir, subject_id, 'aggregate')
    # simple check
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=1):
        if verbose: print('...SKIP: ALREADY COMPLETED maxLL ... <%s> [%s]' % (dest_dir, subject_id))
        return

    dest_path = os.path.join(dest_dir, 'subject-agg.csv')
    df_agg.to_csv(dest_path)
    if verbose: print('... COMPLETE SUBJECT AGG DATA...[%s]' % (subject_id))
    return df_agg

#----------------------------------------#
# PROCESS MODEL SUMULATION DATA
#----------------------------------------#

def process_model_data(df, model):
    df['model'] = model
    df[ACTR_PARAMETER_NAME[model][0]] = df[ACTR_PARAMETER_NAME[model][0]].fillna(0.0)
    df['FutureResponse'] = df['Response'].shift(-1)
    df['ResponseSwitch'] = df.apply(lambda x: 1 if x['Response'] != x['FutureResponse'] else 0, axis=1)

    # drop the last trial for each block
    # BlockTrial == 7
    df = df[df['BlockTrial'] != df['BlockTrial'].max()]
    return df

def save_model_aggregate_data(main_dir, model_dir, subject_dir, subject_id, model, overwrite):
    """
    Save aggregate model file
    """
    # define destination dir
    dest_dir = os.path.join(main_dir, subject_dir, subject_id, 'likelihood')

    # check if already exist
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        print('...SKIP: ALREADY COMPLETED AGG MODEl DATA ... [%s] [%s]' % (subject_id, model))
        return

    model_paths = glob.glob(os.path.join(main_dir, model_dir, '%s*_%s*_gs.csv') % (str.upper(model), subject_id))
    df_model = pd.concat([pd.read_csv(model_path) for model_path in model_paths], axis=0)
    df_model = process_model_data(df=df_model, model=model)
    df_model_agg = df_model.groupby(MAXLL_FACTOR_VAR + ACTR_PARAMETER_NAME[model]).agg(
        {'ResponseSwitch': ('mean', 'std')}).reset_index()
    df_model_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df_model_agg.columns]

    # save file
    df_model_agg.to_csv(os.path.join(dest_dir, str.upper(model) + '.csv'))
    print('...COMPLETE AGG MODEl DATA ... [%s] [%s]' % (subject_id, model))
    return df_model_agg

def load_best_parameter_list(main_dir, log_dir):
    """
    Load best fit parameter list
    main_dir='/home/chery/Documents/Github/RLvIBL-Re',
    Return: param_list = [{'subject_id': '994273_fnca',
                      'ans': 0.45,
                      'bll': 0.6,
                      'lf': 0.1,
                      'egs': 0.4,
                      'alpha': 0.25,
                      'r': 0.1,
                      'param_id': 'param0'}, ...]
    """
    maxLL = pd.read_csv(os.path.join(main_dir, 'data', 'MODELLogLikelihood.csv'),
                        usecols=['HCPID', 'ans.m1', 'bll.m1', 'lf.m1', 'ans.m2', 'bll.m2', 'lf.m2'])
    maxLL.columns = ['subject_id', 'ans', 'bll', 'lf', 'egs', 'alpha', 'r']
    maxLL['ans'] = maxLL['ans'].fillna(0)
    maxLL = maxLL.drop_duplicates()
    maxLL['param_id'] = '' #['param' + str(i) for i in np.arange(len(maxLL))]
    param_list = maxLL.to_dict('records')
    return param_list


#----------------------------------------#
# CALCULATE MAX LOG-LIKELIHOOD
#----------------------------------------#

def calculate_maxLL(df_merge, param_cols):
    """
    Calcualte maxLL
    param_cols: [ans, bll, lf] or [egs, alpha, r]
    df_merge: df_model + df_subject merged dataframe
    """
    # ResponseSwitch
    df_merge['PSwitch_z'] = df_merge.apply(
        lambda x: (x['ResponseSwitch_mean'] - x['ResponseSwitch']) / max(x['ResponseSwitch_std'], 1e-10), axis=1)
    df_merge['PSwitch_probz'] = df_merge.apply(lambda x: norm.pdf(x['PSwitch_z']), axis=1)
    df_merge['PSwitch_logprobz'] = df_merge.apply(lambda x: np.log(max(x['PSwitch_probz'], 1e-10)), axis=1)

    # factor_cols = ['HCPID', 'BlockType', 'TrialType'] PreviousFeedback
    factor_cols = ['HCPID']
    df_maxLL = df_merge.groupby(factor_cols + param_cols).agg(LL=('PSwitch_logprobz', 'sum')).reset_index().sort_values(by='LL', ascending=False)
    return df_maxLL

def merge_for_maxLL(df_model, df_subject, param_cols):
    """
    Merge subject data and model data
    """
    # deal with fake subject data
    df_subject = df_subject[['HCPID','ResponseSwitch']+MAXLL_FACTOR_VAR]
    df_merge = pd.merge(df_model, df_subject, how='left')

    # NOTE: exclude neutral trials
    try:
        df_merge = df_merge[df_merge['PreviousFeedback'] != 'Neutral']
    except:
        pass

    try:
        df_merge = df_merge[df_merge['TrialType'] != 'Neutral']
    except:
        pass

    df_merge = df_merge.sort_values(by = ['HCPID']+param_cols+MAXLL_FACTOR_VAR)
    return df_merge
#----------------------------------------#
# HERLPER FUNCTION
#----------------------------------------#

def get_subject_ids(exclude_list = None):
    parent_dir = os.path.dirname(os.getcwd())
    subject_dir = os.path.join(parent_dir, 'data/gambling_trials')
    subject_files = glob.glob(subject_dir + "/*.csv")
    subject_ids = np.sort([sub.split("/")[-1].split(".")[0] for sub in subject_files])
    if exclude_list:
        subject_ids = [sub for sub in subject_ids if sub not in exclude_list]
    return subject_ids

def check_exists_simple(dest_dir, special_suffix, num_files, overwrite=False):
    if os.path.exists(dest_dir) and len(glob.glob(os.path.join(dest_dir, '*' + special_suffix))) == num_files:
        return True
    else:
        print('...CREATE', dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        return False

def reshape_recovery_data(df, model):
    data_list = []
    for p in ACTR_PARAMETER_NAME[model]:
        temp = df[['HCPID', p+'.s', p+'.m']].rename(columns={p+'.s':'m.original', p+'.m':'m.recovery'})
        temp['param_name'] = p
        data_list.append(temp)
    res = pd.concat(data_list, axis=0)
    return res

def merge_parameter_recovery_data(main_dir, log_dir='data/simulation_test', model='model1'):
    """
    This function process maxLL data
    - load original maxLL data
    - load recovered maxLL data
    - reshape data for better visualization

    Return df (onlly for model1 or model2)
    param_type 	HCPID 	param_name 	m.recovery 	m.original
            0 	178748_fnca 	ans 	0.45 	0.45
            1 	178748_fnca 	bll 	0.80 	0.65
            2 	178748_fnca 	lf 	    0.10 	0.10
            3 	178849_fnca 	ans 	0.10 	0.20
            ...
    """
    # load maxLL
    m_paths = os.path.join(main_dir, log_dir, '*', '*', 'maxll', str.upper(model) + '-maxLL.csv')
    m_files = glob.glob(m_paths)
    df_maxLL_m = pd.concat([pd.read_csv(f, index_col=0) for f in m_files], axis=0)

    # reshape dataframe
    res = reshape_recovery_data(df=df_maxLL_m, model=model)
    return res


#----------------------------------------#
# RUN PIPELINE
#----------------------------------------#
def run_maxLL_pipline(main_dir, subject_dir):
    """
    Input: 'gambling_clean_data.csv'
    Output: aggregate subject data, aggregate model data, and maxLL data
    """
    subject_ids = get_subject_ids()
    df = pd.read_csv(os.path.join(main_dir, 'data', 'gambling_clean_data.csv'), index_col=0)
    for subject_id in subject_ids:
        df_subject = df[df['HCPID'] == subject_id]
        dest_dir = os.path.join(main_dir, subject_dir, subject_id)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        df_subject = process_subject_data(df=df_subject, subject_id=subject_id)
        df_subject.to_csv(os.path.join(dest_dir, 'subject.csv'))

        # save subject agg data
        save_subject_aggregate_data(main_dir=main_dir, subject_dir=subject_dir, subject_id=subject_id, verbose=False)

        # save model agg data
        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model1', overwrite=True)

        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model2', overwrite=True)

        # save maxLL
        save_maxLL_data(main_dir=main_dir,
               log_dir=subject_dir,
               subject_id=subject_id,
               model='model1',
               overwrite=True)

        save_maxLL_data(main_dir=main_dir,
               log_dir=subject_dir,
               subject_id=subject_id,
               model='model2',
               overwrite=True)

def save_maxLL_data(main_dir, log_dir, subject_id, model, overwrite):
    """
    Calculate maxLL for real HCP subject data and save to maxll folder
    """
    subject_path = os.path.join(main_dir, log_dir, subject_id, 'aggregate', 'subject-agg.csv')
    model_path = os.path.join(main_dir, log_dir, subject_id, 'likelihood', str.upper(model) + '.csv')
    # print(subject_path)

    # load single subject and model data
    df_subject = pd.read_csv(subject_path, index_col=0)
    df_model = pd.read_csv(model_path, index_col=0)

    # df_merge = pd.merge(df_model, df_subject, how='left')
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])
    # df_maxLL['HCPID'] = subject_id

    # save data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, 'maxll')
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=4, overwrite=overwrite):
        print('...SKIP: ALREADY COMPLETED maxLL ... [%s] [%s]' % (model, subject_id))
        return

    # merge  maxLL and aggregate subject data (fake)
    df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + subject_id + '-merged.csv'))
    df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '.csv'))

    # find maxLL
    maxLL = df_maxLL['LL'].max()
    df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + ACTR_PARAMETER_NAME[model]]

    # merge maxLL and subject data
    res = pd.merge(df_model, pd.merge(df_maxLL_top, df_subject, how='outer'), how='inner')
    res.to_csv(os.path.join(dest_dir, str.upper(model) + '-maxLL.csv'))
    print('...COMPLETE maxLL DATA ...[%s] [%s]' % (subject_id, model))
    return res


def save_model_classification(main_dir, subject_dir):
    """
    Combine maxLL data and determine best model
    """
    df_model1 = [pd.read_csv(f, index_col=0) for f in
                 glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*1-maxLL.csv'))]
    df_model2 = [pd.read_csv(f, index_col=0) for f in
                 glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*2-maxLL.csv'))]

    df1 = pd.concat(df_model1)
    df2 = pd.concat(df_model2)

    merge_cols = ['HCPID', 'ResponseSwitch'] + MAXLL_FACTOR_VAR
    df = pd.merge(df1, df2, on=merge_cols, suffixes=('.m1', '.m2')).sort_values(by='HCPID')

    df['LL.diff'] = df.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
    df = df[merge_cols + [c for c in df.columns if c not in merge_cols]]
    df['best_model'] = df.apply(lambda x: 'm1' if x['LL.m1'] > x['LL.m2'] else 'm2', axis=1)

    df.to_csv(os.path.join(main_dir, subject_dir, 'actr_maxLL.csv'))
    print('...COMPLETE MODEL CLASSIFICATION...')

    return df

def run_maxLL_pr_pipline(main_dir, subject_dir, param_list):
    """
    Run one simulation for parameter recovery analysis

    - main_dir='/home/chery/Documents/Github/RLvIBL-Re',
    - subject_dir='data/simulation_test',
    - param_list = [{'subject_id': '994273_fnca',
                      'ans': 0.45,
                      'bll': 0.6,
                      'lf': 0.1,
                      'egs': 0.4,
                      'alpha': 0.25,
                      'r': 0.1,
                      'param_id': 'param0'}, ...]
    """
    # save param_list.csv
    df_param = pd.DataFrame(param_list)
    if not os.path.exists(os.path.join(main_dir, subject_dir)):
        os.mkdir(os.path.join(main_dir, subject_dir))
    df_param.to_csv(os.path.join(main_dir, subject_dir, 'param_list.csv'))
    print('... LOG [param_list.csv]...')

    for param in param_list:
        param_id = param['param_id']
        subject_id = param['subject_id']

        param_set = {key: param[key] for key in param.keys() if key not in ('param_id', 'subject_id')}

        # re-simulate fake subject data
        simulate(epoch=1,
                 subject_id=subject_id,
                 param_id=param_id,
                 model='model1',
                 param_set=param_set,
                 verbose=False,
                 log_dir=subject_dir,
                 special_suffix="",
                 overwrite=False)
        simulate(epoch=1,
                 subject_id=subject_id,
                 param_id=param_id,
                 model='model2',
                 param_set=param_set,
                 verbose=False,
                 log_dir=subject_dir,
                 special_suffix="",
                 overwrite=False)

        # re-process model data
        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model1',
                                  overwrite=False)

        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model2',
                                  overwrite=False)

        # calculate maxLL
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model1',
                           overwrite=False)
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model2',
                           overwrite=False)

def save_maxLL_pr_data(main_dir, log_dir, subject_id, param_id, model, overwrite=False):
    """
    Calculate maxLL for fake subject (parameter recovery analysis) and save to maxll folder
    """

    subject_path = os.path.join(main_dir, log_dir, subject_id, param_id, 'aggregate', str.upper(model) + '.csv')
    model_path = os.path.join(main_dir, log_dir, subject_id, 'likelihood', str.upper(model) + '.csv')
    # print(subject_path)

    # load single subject and model data
    df_subject = pd.read_csv(subject_path, index_col=0)
    df_model = pd.read_csv(model_path, index_col=0)

    # merge df_model and df_subject
    # note: need to remove param_cols [ans, bll, lf]
    # df_merge = pd.merge(df_model, df_subject.drop(columns=param_cols), how='left')
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])

    # save maxLL data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, param_id, 'maxll')
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=4, overwrite=overwrite):
        print('...SKIP: ALREADY COMPLETED maxLL ... [%s] [%s] [%s]' % (model, subject_id, param_id))
        return


    # merge  maxLL and aggregate subject data (fake)
    df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' +subject_id + '-merged.csv'))
    df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '.csv'))

    # find maxLL and keep all rows
    maxLL = df_maxLL['LL'].max()
    df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + ACTR_PARAMETER_NAME[model]]

    # merge maxLL data  and subject data
    res = pd.merge(df_maxLL_top, df_subject[ACTR_PARAMETER_NAME[model]].drop_duplicates(), how='cross', suffixes=('.m', '.s'))
    res.to_csv(os.path.join(dest_dir, str.upper(model) + '-maxLL.csv'))
    print('...COMPLETE maxLL DATA ...[%s] [%s]' % (subject_id, model))
    return res
