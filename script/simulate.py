import os

import pandas as pd

from device import *
import numpy as np
import glob
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

MAXLL_FACTOR_VAR = ['BlockType', 'TrialType']
# MAXLL_FACTOR_VAR = ['TrialType']
MAXLL_EXCLUDE_NEUTRAl = False
ACTR_PARAMETER_NAME = {'model1': ['ans', 'bll', 'lf'], 'model2':['egs', 'alpha', 'r']}

#----------------------------------------#
# RUN SIMULATION
#----------------------------------------#

def simulate(epoch, subject_id, param_id, model, param_set, verbose, log_dir, special_suffix="", overwrite=False):
    """
    Run simulation
    """
    parent_dir = os.path.dirname(os.getcwd())
    model_dat_list = []
    for i in range(epoch):
        #check if exists 2 files MODEL1.csv MODEL2.csv
        dest_dir = os.path.join(parent_dir, log_dir, subject_id, param_id)
        if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
            if verbose: print('...SKIP: ALREADY SIM DATA ... [%s] [%s]' % (subject_id, param_id))
            continue
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

    # if all being simulated
    # skip
    try:
        df_model = pd.concat(model_dat_list, axis=0)
    except:
        return

    # process response switch
    df_model = process_model_data(df=df_model, model=model)

    if log_dir:
        save_simulation_data(main_dir=parent_dir,
                       log_dir=log_dir,
                       subject_id=subject_id,
                       param_id=param_id,
                       df_model=df_model,
                       special_suffix=special_suffix,
                       verbose=verbose,
                       overwrite=overwrite)

def save_simulation_data(main_dir, log_dir, subject_id, param_id, df_model, special_suffix, verbose, overwrite):
    """
    Save samulation data
    """
    ### MODEL SIMULATION DATA
    # dest_dir = os.path.join(parent_dir, log_dir, subject_id)
    dest_dir = os.path.join(main_dir, log_dir, subject_id, param_id)
    log_path = os.path.join(dest_dir, actr.current_model() + special_suffix + ".csv")

    df_model['HCPID'] = subject_id

    # check exist
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        if verbose: print('...SKIP: ALREADY COMPLETED SIM DATA ... [%s] [%s]' % (subject_id, param_id))
        if overwrite:
            df_model.to_csv(log_path, header=True, index=True, mode='w')
    else:
        # save model simulation data
        df_model.to_csv(log_path, header=True, index=True, mode='w')
    if verbose: print("...COMPLETE SIMULATION...\n")

def save_simulation_aggregate_data(main_dir, log_dir, subject_id, param_id, model, special_suffix, verbose, overwrite):
    """
    Save simulation aggregate data
    """
    ### SIMULATION AGG DATA
    # calculate aggregate data
    df_model = pd.read_csv(os.path.join(main_dir, log_dir, subject_id, param_id, str.upper(model) + '.csv'), index_col=0)
    group_var = ['HCPID'] + MAXLL_FACTOR_VAR + ACTR_PARAMETER_NAME[model]
    df_agg = df_model.groupby(group_var).agg({'ResponseSwitch': 'mean'}).reset_index()

    # define path
    dest_agg_dir = os.path.join(os.path.join(main_dir, log_dir, subject_id, param_id, 'aggregate'))
    agg_path = os.path.join(dest_agg_dir,  str.upper(model)  + special_suffix + ".csv")

    # check exist
    if check_exists_simple(dest_dir=dest_agg_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        if verbose: print('...SKIP: ALREADY COMPLETED SIM DATA ... [%s] [%s]' % (subject_id, param_id))
    else:
        df_agg.to_csv(agg_path, header=True, index=True, mode='w')
    if verbose: print("...COMPLETE AGG SIM DATA...[%s]\n" % (model))
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

def save_subject_aggregate_data(main_dir, subject_dir, subject_id, verbose=False, overwrite=False):
    """
    Save HCP subjects aggregate data
        ResponseSwitch = FutureResponse - CurrResponse
        The grouping variable should match MAXLL_FACTOR_VAR
        drop any trial that ResponseSwitch is NaN (the last trial of each block): the ultimate total number of
        trials left = 64 - 8 = 56
    """
    ori_path = os.path.join(main_dir, subject_dir, subject_id, 'subject.csv')

    df = pd.read_csv(ori_path, usecols=['HCPID', 'BlockTypeCoded', 'TrialType', 'ResponseSwitch']).rename(columns={'BlockTypeCoded': 'BlockType'})
    df = df.dropna(subset=['ResponseSwitch'])

    #  calculate agg by group variable
    df_agg = df.groupby(['HCPID'] + MAXLL_FACTOR_VAR).agg({'ResponseSwitch': 'mean'}).reset_index()

    dest_dir = os.path.join(main_dir, subject_dir, subject_id, 'aggregate')
    # check if exist 1 file in aggregate/
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=1, overwrite=overwrite):
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

    # check if exist 2 files MODEL1.csv MODEL2.csv in likelihood/
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
    maxLL = pd.read_csv(os.path.join(main_dir, 'data', 'old_actr_maxLL.csv'),
                        usecols=['HCPID', 'ans', 'bll', 'lf', 'egs', 'alpha', 'r'])
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

    # NOTE: whether or not exclude neutral trials
    if MAXLL_EXCLUDE_NEUTRAl:
        assert 'TrialType'  in MAXLL_FACTOR_VAR
        df_merge =  df_merge[df_merge['TrialType'] != 'Neutral']

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
        if overwrite:
            print("...OVERWRITE...")
            return False
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
    # has param_id sub-folder
    m_files = glob.glob(os.path.join(main_dir, log_dir, '*', '*', 'maxll', str.upper(model) + '-maxLL.csv'))
    # no param_id sub-folder
    if len(m_files) == 0:
        m_files = glob.glob(os.path.join(main_dir, log_dir, '*', 'maxll', str.upper(model) + '-maxLL.csv'))
    df_maxLL_m = pd.concat([pd.read_csv(f, index_col=0) for f in m_files], axis=0)

    # reshape dataframe
    df = reshape_recovery_data(df=df_maxLL_m, model=model)
    return df

def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
    """
    Given a dataframe with at least
    two categorical columns, create a
    confusion matrix of the count of the columns
    cross-counts

    use like:
    >>> confusion_matrix(test_df, 'actual_label', 'predicted_label')
    """
    return (
        df
            .groupby([col1, col2])
            .size()
            .unstack(fill_value=0)
    )

def compare_actr_maxLL(main_dir, log_dir):
    """
    Compare new vs. old actr_maxLL.csv
    """
    # df0 = pd.read_csv(os.path.join(main_dir, 'data/MODELLogLikelihood.csv'), index_col=0, usecols=['HCPID', 'best_model', 'PSwitch.LL.m2', 'PSwitch.LL.m1'])
    # df0 = df0.rename(columns = {'PSwitch.LL.m2':'LL.m2', 'PSwitch.LL.m1':'LL.m1'})
    # df0['LL.diff'] = df0['LL.m1'] - df0['LL.m2']
    # df0 = df0.sort_values(by = 'HCPID')
    try:
        df0 = pd.read_csv(os.path.join(main_dir, 'data/old_actr_maxLL.csv'))
        df1 = pd.read_csv(os.path.join(main_dir, log_dir, 'actr_maxLL.csv'), index_col=0)
        df_compare = pd.merge(df0, df1, suffixes=('.old', '.new'), how='right')
    except:
        print("NO NEW actr_maxLL.csv FOUND...")
        return
    return df_compare, df0, df1

#----------------------------------------#
# RUN PIPELINE
#----------------------------------------#
def run_maxLL_pipline(main_dir, subject_dir, overwrite=True):
    """
    Input: 'gambling_clean_data.csv'
    Output: aggregate subject data, aggregate model data, and maxLL data
    """
    subject_ids = get_subject_ids()
    df = pd.read_csv(os.path.join(main_dir, 'data', 'gambling_clean_data.csv'), index_col=0)
    for subject_id in subject_ids:
        df_subject = df[df['HCPID'] == subject_id]

        # prepare dest fir
        dest_dir = os.path.join(main_dir, subject_dir, subject_id)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # reformat subject data
        df_subject = process_subject_data(df=df_subject, subject_id=subject_id)
        if check_exists_simple(dest_dir=dest_dir, num_files=1, special_suffix=".csv", overwrite=False):
            print("...SKIP: ALREADY REFORMAT SUBJ DATA ... [%s]" % (subject_id))
        else:
            df_subject.to_csv(os.path.join(dest_dir, 'subject.csv'))

        # save subject agg data
        save_subject_aggregate_data(main_dir=main_dir,
                                    subject_dir=subject_dir,
                                    subject_id=subject_id,
                                    verbose=False,
                                    overwrite=overwrite)

        # save model agg data
        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model1', overwrite=overwrite)

        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model2', overwrite=overwrite)

        # save maxLL
        save_maxLL_data(main_dir=main_dir,
               log_dir=subject_dir,
               subject_id=subject_id,
               model='model1',
               overwrite=overwrite)

        save_maxLL_data(main_dir=main_dir,
               log_dir=subject_dir,
               subject_id=subject_id,
               model='model2',
               overwrite=overwrite)

    # combine maxLL
    save_model_classification(main_dir=main_dir, subject_dir=subject_dir)

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

    # use merge method
    # may include/exclude neutral
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])
    # df_maxLL['HCPID'] = subject_id

    # save data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, 'maxll')
    # check if exist 6 files in maxll/
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=6, overwrite=overwrite):
        print('...SKIP: ALREADY CALCULATED maxLL ... [%s] [%s]' % (model, subject_id))
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
    ## TODO: this only works if no param_id folder is found
    df_model1 = [pd.read_csv(f, index_col=0) for f in
                 glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*1-maxLL.csv'))]
    df_model2 = [pd.read_csv(f, index_col=0) for f in
                 glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*2-maxLL.csv'))]

    df1 = pd.concat(df_model1)
    df2 = pd.concat(df_model2)

    merge_cols = ['HCPID'] + MAXLL_FACTOR_VAR
    df = pd.merge(df1, df2, on=merge_cols, suffixes=('.m1', '.m2')).sort_values(by='HCPID')

    df['LL.diff'] = df.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
    df = df[merge_cols + [c for c in df.columns if c not in merge_cols]]
    df['best_model'] = df.apply(lambda x: 'm1' if x['LL.m1'] > x['LL.m2'] else 'm2', axis=1)

    if check_exists_simple(dest_dir=subject_dir, special_suffix='actr_maxLL.csv', num_files=1):
        print('...SKIP... MODEL CLASSIFICATION...')
        return df
    df.to_csv(os.path.join(main_dir, subject_dir, 'actr_maxLL.csv'))
    print('...COMPLETE MODEL CLASSIFICATION...[actr_maxLL.csv]')

    # save simulation info
    df_readme = pd.DataFrame({'MAXLL_FACTOR_VAR': MAXLL_FACTOR_VAR,
                              'MAXLL_EXCLUDE_NEUTRAl': MAXLL_EXCLUDE_NEUTRAl}).reset_index()
    df_readme.to_csv(os.path.join(main_dir, subject_dir, 'README.txt'))
    print('...COMPLETE ...[README.txt]')
    return df

def run_maxLL_pr_pipline(main_dir, subject_dir, param_list, overwrite):
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
                 verbose=True,
                 log_dir=subject_dir,
                 special_suffix="",
                 overwrite=False)
        simulate(epoch=1,
                 subject_id=subject_id,
                 param_id=param_id,
                 model='model2',
                 param_set=param_set,
                 verbose=True,
                 log_dir=subject_dir,
                 special_suffix="",
                 overwrite=False)

        # save model simulation(fake subject) aggregate data
        save_simulation_aggregate_data(main_dir=main_dir,
                                       log_dir=subject_dir,
                                       subject_id=subject_id,
                                       param_id=param_id,
                                       model='model1',
                                       special_suffix="",
                                       verbose=True,
                                       overwrite=overwrite)
        save_simulation_aggregate_data(main_dir=main_dir,
                                       log_dir=subject_dir,
                                       subject_id=subject_id,
                                       param_id=param_id,
                                       model='model2',
                                       special_suffix="",
                                       verbose=True,
                                       overwrite=overwrite)

        # re-process model data
        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model1',
                                  overwrite=overwrite)

        save_model_aggregate_data(main_dir=main_dir,
                                  model_dir='data/model_output_local',
                                  subject_dir=subject_dir,
                                  subject_id=subject_id,
                                  model='model2',
                                  overwrite=overwrite)

        # calculate maxLL
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model1',
                           overwrite=overwrite)
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model2',
                           overwrite=overwrite)

    # combine maxLL
    save_model_classification(main_dir=main_dir, subject_dir=subject_dir)




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
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])

    # save maxLL data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, param_id, 'maxll')
    # check if exist 6 files in maxll
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=6, overwrite=overwrite):
        print('...SKIP: ALREADY CALCULATED maxLL ... [%s] [%s] [%s]' % (model, subject_id, param_id))
        return

    # merge  maxLL and aggregate subject data (fake)
    df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' +subject_id + '-merged.csv'))
    df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '.csv'))

    # find maxLL and keep all rows
    maxLL = df_maxLL['LL'].max()
    df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + ACTR_PARAMETER_NAME[model]]

    # merge maxLL data  and subject data
    res = pd.merge(pd.merge(df_model, df_maxLL_top, on=ACTR_PARAMETER_NAME[model]).dropna(axis=0), df_subject, on=['HCPID'] + MAXLL_FACTOR_VAR, suffixes=('.m', '.s'))
    res.to_csv(os.path.join(dest_dir, str.upper(model) + '-maxLL.csv'))
    print('...COMPLETE maxLL DATA ...[%s] [%s]' % (subject_id, model))
    return res
