import glob
import shutil
import warnings

import numpy as np
from scipy.stats import norm

from device import *

warnings.filterwarnings('ignore')

MAXLL_FACTOR_VAR = ['BlockType', 'TrialType']
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

    # group var has to include both BlockType, TrialType, and ans, bll, lf ...
    # TODO: rename group variable when merge with model data to calculate maxLL
    # otherwise merged df will lose .m, .s when subject fake = model1, model=model2
    group_var = ['HCPID'] + MAXLL_FACTOR_VAR + ACTR_PARAMETER_NAME[model]
    df_agg = df_model.groupby(group_var).agg({'ResponseSwitch': 'mean'}).reset_index()

    # rename ACTR_PARAMETER_NAME: add .s
    df_agg = df_agg.rename(columns={**dict(zip(ACTR_PARAMETER_NAME[model], [p+'.s' for p in ACTR_PARAMETER_NAME[model]]))})

    # define path
    dest_agg_dir = os.path.join(os.path.join(main_dir, log_dir, subject_id, param_id, 'aggregate'))
    agg_path = os.path.join(dest_agg_dir,  str.upper(model)  + special_suffix + ".csv")

    # check exist
    if check_exists_simple(dest_dir=dest_agg_dir, special_suffix='.csv', num_files=2, overwrite=overwrite):
        if verbose: print('...SKIP: ALREADY COMPLETED AGG SIM DATA ... [%s] [%s]' % (subject_id, param_id))
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
        if verbose: print('...SKIP: ALREADY COMPLETED SUBJ AGG ... <%s> [%s]' % (dest_dir, subject_id))
        return

    dest_path = os.path.join(dest_dir, subject_id+'-agg.csv')
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

    # rename ACTR_PARAMETER_NAME: add .m
    df_model_agg = df_model_agg.rename(
        columns={**dict(zip(ACTR_PARAMETER_NAME[model], [p + '.m' for p in ACTR_PARAMETER_NAME[model]]))})

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
    param_cols: [ans, bll, lf] or [egs, alpha, r] depending on model
    df_merge: df_model + df_subject merged dataframe
    Since df_merge contains [ans.m, bll.m]
    """
    # ResponseSwitch
    df_merge['PSwitch_z'] = df_merge.apply(
        lambda x: (x['ResponseSwitch_mean'] - x['ResponseSwitch']) / max(x['ResponseSwitch_std'], 1e-10), axis=1)
    df_merge['PSwitch_probz'] = df_merge.apply(lambda x: norm.pdf(x['PSwitch_z']), axis=1)
    df_merge['PSwitch_logprobz'] = df_merge.apply(lambda x: np.log(max(x['PSwitch_probz'], 1e-10)), axis=1)

    # factor_cols = ['HCPID', 'BlockType', 'TrialType'] PreviousFeedback
    factor_cols = ['HCPID']

    # note: need to aggregate by model parameter
    # param_cols should be [xx.m, xx.m, xx.m]
    try:
        df_maxLL = df_merge.groupby(factor_cols + param_cols).agg(LL=('PSwitch_logprobz', 'sum')).reset_index().sort_values(by='LL', ascending=False)
        return df_maxLL
    except:
        df_maxLL = df_merge.groupby(factor_cols + [c+'.m' for c in param_cols]).agg(LL=('PSwitch_logprobz', 'sum')).reset_index().sort_values(by='LL', ascending=False)
        return df_maxLL

def merge_for_maxLL(df_model, df_subject, param_cols):
    """
    Merge subject data and model data
    param_cols: [ans, bll, lf] depends on model
    """
    # assert all(p.split('.')[0] in param_cols for p in df_subject.columns) and \
    #        all(p.split('.')[0] in param_cols for p in df_model.columns)

    # deal with fake subject data
    # should be ok since df_subject has [ans.s, bll.s, lf.s] should not conflict with df_model
    # df_subject = df_subject[['HCPID','ResponseSwitch']+MAXLL_FACTOR_VAR]
    df_merge = pd.merge(df_model, df_subject, how='left')

    # NOTE: whether or not exclude neutral trials
    if MAXLL_EXCLUDE_NEUTRAl:
        assert 'TrialType' in MAXLL_FACTOR_VAR
        df_merge =  df_merge[df_merge['TrialType'] != 'Neutral']
    df_merge = df_merge.sort_values(by = ['HCPID']+MAXLL_FACTOR_VAR)
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
    # if overwrite is enabled,
    # remove all files within dest_dir
    # re-create new empty dir
    if overwrite:
        print("...OVERWRITE...[%s]" % (dest_dir))
        try:
            shutil.rmtree(dest_dir)
        except:
            print("No DIR", dest_dir)
        finally:
            os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest_dir) and len(glob.glob(os.path.join(dest_dir, '*' + special_suffix))) == num_files:
        return True
    else:
        # if not exist
        # create mew dor
        print("...EMPTY...CREATE DIR: [%s]" % (dest_dir))
        os.makedirs(dest_dir, exist_ok=True)
        return False

def reshape_recovery_data(df, param_cols):
    data_list = []
    # param_cols = [c for c in df.columns if (c.endswith('.m') or c.endswith('.s'))]
    print(param_cols)
    for p in param_cols:
        temp = df[['HCPID', p+'.s', p+'.m']].rename(columns={p+'.s':'m.original', p+'.m':'m.recovery'})
        # temp = df[['HCPID', param_cols]].rename(columns={p+'.s':'m.original', p+'.m':'m.recovery'})
        temp['param_name'] = p
        data_list.append(temp)
    res = pd.concat(data_list, axis=0)
    return res

def merge_parameter_recovery_data(main_dir, log_dir, model='model1', model_fake_subject='model1'):
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

    try:
        # if contains xx/param1
        m_files = glob.glob(os.path.join(main_dir, log_dir, '*',  '*', 'maxll',
                                         '%s-FAKE%s-maxLL.csv' % (str.upper(model), str.upper(model_fake_subject))))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in m_files], axis=0)
    except:
        # not contains xx/param1
        m_files = glob.glob(os.path.join(main_dir, log_dir, '*', 'maxll',
                                         '%s-FAKE%s-maxLL.csv' % (str.upper(model), str.upper(model_fake_subject))))
        df = pd.concat([pd.read_csv(f, index_col=0) for f in m_files], axis=0)
    finally:
        # reshape dataframe
        # res = reshape_recovery_data(df=df)
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
def run_maxLL_pipline(main_dir, subject_dir, overwrite=True, analysis_type='subject'):
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
                                  model='model2', overwrite=False)

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
               overwrite=False)

    # combine maxLL
    save_model_classification(main_dir=main_dir, subject_dir=subject_dir, param_list=None, overwrite=overwrite, analysis_type=analysis_type)

# def save_maxLL_data(main_dir, log_dir, subject_id, model, overwrite):
#     """
#     Calculate maxLL for real HCP subject data and save to maxll folder
#     """
#     subject_path = os.path.join(main_dir, log_dir, subject_id, 'aggregate', 'subject-agg.csv')
#     model_path = os.path.join(main_dir, log_dir, subject_id, 'likelihood', str.upper(model) + '.csv')
#     # print(subject_path)
#
#     # load single subject and model data
#     df_subject = pd.read_csv(subject_path, index_col=0)
#     df_model = pd.read_csv(model_path, index_col=0)
#
#     # use merge method
#     # may include/exclude neutral
#     # merge  maxLL and aggregate subject data (fake)
#     # save data
#     dest_dir = os.path.join(main_dir, log_dir, subject_id, 'maxll')
#
#     # check if exist 6 files in maxll/
#     if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=6, overwrite=overwrite):
#         print('...SKIP: ALREADY CALCULATED maxLL ... [%s] [%s]' % (model, subject_id))
#         return
#
#     # merge df_model and df_subject and save
#     df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
#
#     # calculate maxLL
#     df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])
#
#     # save maxLL data
#     df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + subject_id + '-merged.csv'))
#     df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '.csv'))
#     # df_maxLL['HCPID'] = subject_id
#
#     # find maxLL
#     maxLL = df_maxLL['LL'].max()
#     df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + ACTR_PARAMETER_NAME[model]]
#
#     # merge maxLL and subject data
#     res = pd.merge(df_model, pd.merge(df_maxLL_top, df_subject, how='outer'), how='inner')
#     res.to_csv(os.path.join(dest_dir, str.upper(model) + '-maxLL.csv'))
#     print('...COMPLETE maxLL DATA ...[%s] [%s]' % (subject_id, model))
#     return res

def save_maxLL_data(main_dir, log_dir, subject_id, model, overwrite=False):
    """
    Calculate maxLL for real subject and save to maxll folder
    model: model1 or model2, the simulated model
    model_fake_subject: model1 or model2, the fake subject data file generated by whichever model (model1 or model2)
    """
    subject_path = os.path.join(main_dir, log_dir, subject_id, 'aggregate', subject_id+'-agg.csv')
    model_path = os.path.join(main_dir, log_dir, subject_id, 'likelihood', str.upper(model) + '.csv')
    # print(subject_path)

    # load single subject and model data
    df_subject = pd.read_csv(subject_path, index_col=0)
    df_model = pd.read_csv(model_path, index_col=0)

    # save maxLL data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, 'maxll')

    # check if exist 12 files in maxll
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=6, overwrite=overwrite):
        print('...SKIP: ALREADY CALCULATED maxLL ... [%s] [%s]' % (model, subject_id))
        return

    # merge df_model and df_subject
    # note: need to remove param_cols
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=ACTR_PARAMETER_NAME[model])
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=ACTR_PARAMETER_NAME[model])


    # find maxLL and keep all rows
    maxLL = df_maxLL['LL'].max()

    try:
        # pr analysis
        df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + ACTR_PARAMETER_NAME[model]]
        # subject maxLL
    except:
        df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID', 'LL'] + [c+'.m' for c in ACTR_PARAMETER_NAME[model]]]
    finally:
        # merge maxLL and subject data
        res = pd.merge(df_model, pd.merge(df_maxLL_top, df_subject, how='outer'), how='inner')
        # res['FakeModel'] = model_fake_subject

        # merge  maxLL and aggregate subject data (fake)
        df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + subject_id + '-merged.csv'))
        df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + subject_id + '.csv'))
        res.to_csv(os.path.join(dest_dir, str.upper(model)  + '-' + subject_id + '-maxLL.csv'))
        print('...COMPLETE maxLL DATA ...[%s] [%s]' % (model, subject_id))
        return res


def save_model_classification(main_dir, subject_dir, param_list=None, overwrite=False, analysis_type='subject'):
    """
    Combine maxLL data and determine best model

    For parameter
    """
    assert analysis_type in ('subject', 'pr_best', 'pr_random')

    if analysis_type == 'subject':
        df = estimate_model_class_subject(main_dir=main_dir, subject_dir=subject_dir)
    if analysis_type == 'pr_best':
        df_list = []
        for subject_id in get_subject_ids():
            tempt = estimate_model_class_pr(main_dir=main_dir,
                                            subject_dir=subject_dir,
                                            subject_id=subject_id,
                                            param_id='')
            df_list.append(tempt)
        df = pd.concat(df_list, axis=0)
    if analysis_type == 'pr_random':
        df_list = []
        for subject_id in get_subject_ids():
            param_ids = np.sort([p.split('/')[-1] for p in glob.glob(os.path.join(main_dir,  subject_dir, subject_id, 'param*'))])
            for param_id in param_ids:
                tempt = estimate_model_class_pr(main_dir=main_dir,
                                                subject_dir=subject_dir,
                                                subject_id=subject_id,
                                                param_id=param_id)
                df_list.append(tempt)
        df = pd.concat(df_list, axis=0)
    if param_list:
        df_param = pd.DataFrame(param_list)
        df_param['MAXLL_FACTOR_VAR'] = str(MAXLL_FACTOR_VAR)
        df_param['MAXLL_EXCLUDE_NEUTRAl'] = MAXLL_EXCLUDE_NEUTRAl
        df_param.to_csv(os.path.join(main_dir, subject_dir, 'README.txt'), index=False)
        print('...SAVE README.txt ...')

    if check_exists_simple(dest_dir=subject_dir, special_suffix='actr_maxLL.csv', num_files=1, overwrite=overwrite):
        print('...SKIP... MODEL CLASSIFICATION...')
        return df
    df.to_csv(os.path.join(main_dir, subject_dir, 'actr_maxLL.csv'))
    print('...COMPLETE MODEL CLASSIFICATION...[actr_maxLL.csv]')
    return df

# def estimate_model_class(main_dir, subject_dir, model_fake_subject='model1'):
#     """
#     Estimate the model class
#     """
#
#     # TODO: deal with path to make sure param_id is useable
#     try:
#         # single param_id
#         m1_files = glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*1*%s-maxLL.csv' % (str.upper(model_fake_subject))))
#         m2_files = glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '*2*%s-maxLL.csv' % (str.upper(model_fake_subject))))
#
#         # concat maxLL for all subjects, all param ids
#         df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
#         df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)
#     except:
#         # all param_id
#         m1_files = glob.glob(os.path.join(main_dir, subject_dir, '*', '*', 'maxll', '*1*%s-maxLL.csv' % (str.upper(model_fake_subject))))
#         m2_files = glob.glob(os.path.join(main_dir, subject_dir, '*', '*', 'maxll', '*2*%s-maxLL.csv' % (str.upper(model_fake_subject))))
#
#         # concat maxLL for all subjects, all param ids
#         df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
#         df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)
#     finally:
#
#         # only select max LL for each subject, each model, across param_ids
#         # df1 = df_model1.groupby(['HCPID', 'FakeModel']).agg({'LL':'max'}).reset_index() #.agg(LL = ('LL', 'max')).reset_index()
#         # df2 = df_model2.groupby(['HCPID', 'FakeModel']).agg({'LL':'max'}).reset_index() #.agg(LL = ('LL', 'max')).reset_index()
#         maxLL1 = df_model1.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()
#         maxLL2 = df_model2.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()
#
#
#         # join back to obtain full parameter info
#         df1 = pd.merge(df_model1, maxLL1, how='inner')
#         df2 = pd.merge(df_model2, maxLL2, how='inner')
#
#         # join model1 maxLL and model2 maxLL
#         merge_cols = ['HCPID', 'FakeModel'] + MAXLL_FACTOR_VAR
#         df = pd.merge(df1, df2, on=merge_cols, suffixes=('.m1', '.m2')).sort_values(by='HCPID')
#
#         # determine model class
#         # if LL.m1 > LL.m2 -> BestModel = m1
#         # else BestModel = m2
#         df['LL.diff'] = df.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
#         df = df[merge_cols + [c for c in df.columns if c not in merge_cols]]
#         df['BestModel'] = df.apply(lambda x: 'model1' if x['LL.m1'] > x['LL.m2'] else 'model2', axis=1)
#
#         return df


# def estimate_model_class_best_param(main_dir, subject_dir):
#     """
#     Estimate the model class
#     """
#
#     # try:
#     # single param_id
#     # all files starts with MODEL1- fake model1 and fake model2 estimated using model1
#     m1_files = glob.glob(
#         os.path.join(main_dir, subject_dir, '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model1'))))
#     m2_files = glob.glob(
#         os.path.join(main_dir, subject_dir, '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model2'))))
#
#     # concat maxLL for all subjects, all param ids
#     df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
#     df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)
#     # except:
#     #     # all param_id
#     #     m1_files = glob.glob(
#     #         os.path.join(main_dir, subject_dir, '*', '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model1'))))
#     #     m2_files = glob.glob(
#     #         os.path.join(main_dir, subject_dir, '*', '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model2'))))
#     #
#     #     # concat maxLL for all subjects, all param ids
#     #     df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
#     #     df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)
#
#     # finally:
#     # df_model1['BlockType'] = df_model1.apply(lambda x: pd.Categorical(x['BlockType'], categories=['MostlyReward', 'MostlyPunishment']))
#     df_model1['BlockType'] = pd.Categorical(df_model1['BlockType'], categories=df_model1['BlockType'].unique())
#     df_model1['TrialType'] = pd.Categorical(df_model1['TrialType'], categories=df_model1['TrialType'].unique())
#     df_model2['BlockType'] = pd.Categorical(df_model2['BlockType'], categories=df_model2['BlockType'].unique())
#     df_model2['TrialType'] = pd.Categorical(df_model2['TrialType'], categories=df_model2['TrialType'].unique())
#
#     # drop .s columns
#     m1_s_cols = [c for c in df_model1.columns if c.endswith('.s')]
#     m2_s_cols = [c for c in df_model2.columns if c.endswith('.s')]
#     df_model1_drops = df_model1.drop(columns=m1_s_cols)
#     df_model2_drops = df_model2.drop(columns=m2_s_cols)
#
#     # only select max LL for each subject, each model, across param_ids
#     # df1 = df_model1.groupby(['HCPID', 'FakeModel']).agg({'LL':'max'}).reset_index() #.agg(LL = ('LL', 'max')).reset_index()
#     # df2 = df_model2.groupby(['HCPID', 'FakeModel']).agg({'LL':'max'}).reset_index() #.agg(LL = ('LL', 'max')).reset_index()
#     maxLL1 = df_model1_drops.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()
#     maxLL2 = df_model2_drops.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()
#
#     # outer join maxLL1 and maxLL2
#     # each row is a fake subject
#     # LL.m1 is LL estimated by model1
#     # LL.m2 is LL estimated by model2
#     # BestModel is the determined by higher LL between LL.m1 and LL.m2
#     maxLL = pd.merge(maxLL1, maxLL2, how='outer', on=['HCPID', 'FakeModel'], suffixes=('.m1', '.m2'))
#     maxLL['LL.diff'] = maxLL.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
#     maxLL['BestModel'] = maxLL.apply(lambda x: 'model1' if x['LL.m1'] > x['LL.m2'] else 'model2', axis=1)
#
#     # join back to obtain parameter and simulation data
#     m1_cols = [c for c in df_model1.columns if (c.endswith('.s') or c.endswith('.m'))]
#     m2_cols = [c for c in df_model2.columns if (c.endswith('.s') or c.endswith('.m'))]
#
#     df_join1 = pd.merge(maxLL, df_model1[['HCPID', 'FakeModel'] + m1_cols].drop_duplicates(), how='right')
#     df_join2 = pd.merge(maxLL, df_model2[['HCPID', 'FakeModel'] + m2_cols].drop_duplicates(), how='right')
#     df_join = pd.merge(df_join1, df_join2, how='inner').sort_values(by=['HCPID', 'FakeModel'])
#     return df_join
#
#
#
#         # ## TODO: update
#         #
#         # # join back to obtain full parameter info
#         # df1 = pd.merge(df_model1, maxLL1, how='inner')
#         # df2 = pd.merge(df_model2, maxLL2, how='inner')
#         #
#         # # join model1 maxLL and model2 maxLL
#         # merge_cols = ['HCPID', 'FakeModel'] + MAXLL_FACTOR_VAR
#         # df = pd.merge(df1, df2, on=merge_cols, suffixes=('.m1', '.m2')).sort_values(by='HCPID')
#         #
#         # # determine model class
#         # # if LL.m1 > LL.m2 -> BestModel = m1
#         # # else BestModel = m2
#         # df['LL.diff'] = df.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
#         # df = df[merge_cols + [c for c in df.columns if c not in merge_cols]]
#         # df['BestModel'] = df.apply(lambda x: 'model1' if x['LL.m1'] > x['LL.m2'] else 'model2', axis=1)
#         #
#         # return df

def estimate_model_class_subject(main_dir, subject_dir):
    """

    """
    m1_files = glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model1'))))
    m2_files = glob.glob(os.path.join(main_dir, subject_dir, '*', 'maxll', '%s-*-maxLL.csv' % (str.upper('model2'))))

    # concat maxLL for all subjects, all param ids
    df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
    df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)

    # only select max LL for each subject, each model, across param_ids
    maxLL1 = df_model1.groupby(['HCPID'])['LL'].max().reset_index()
    maxLL2 = df_model2.groupby(['HCPID'])['LL'].max().reset_index()

    # outer join maxLL1 and maxLL2
    # each row is a fake subject
    # LL.m1 is LL estimated by model1
    # LL.m2 is LL estimated by model2
    # BestModel is the determined by higher LL between LL.m1 and LL.m2
    maxLL = pd.merge(maxLL1, maxLL2, how='outer', on=['HCPID'], suffixes=('.m1', '.m2'))
    maxLL['LL.diff'] = maxLL.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
    maxLL['BestModel'] = maxLL.apply(lambda x: 'model1' if x['LL.m1'] > x['LL.m2'] else 'model2', axis=1)

    # join back to obtain parameter and simulation data
    m1_cols = [c for c in df_model1.columns if (c.endswith('.s') or c.endswith('.m'))]
    m2_cols = [c for c in df_model2.columns if (c.endswith('.s') or c.endswith('.m'))]

    df_join1 = pd.merge(maxLL, df_model1[['HCPID'] + m1_cols].drop_duplicates(), how='right')
    df_join2 = pd.merge(maxLL, df_model2[['HCPID'] + m2_cols].drop_duplicates(), how='right')
    df_join = pd.merge(df_join1, df_join2, how='inner').sort_values(by=['HCPID'])
    return df_join


def estimate_model_class_pr(main_dir, subject_dir, subject_id, param_id):
    """
    Estimate model class for each fake subject
    """
    # all files starts with MODEL1- fake model1 and fake model2 estimated using model1
    m1_files = glob.glob(
        os.path.join(main_dir, subject_dir, subject_id, param_id, 'maxll', '%s-*-maxLL.csv' % (str.upper('model1'))))
    m2_files = glob.glob(
        os.path.join(main_dir, subject_dir, subject_id, param_id, 'maxll', '%s-*-maxLL.csv' % (str.upper('model2'))))

    # concat maxLL for all subjects, all param ids
    df_model1 = pd.concat([pd.read_csv(f, index_col=0) for f in m1_files], axis=0)
    df_model2 = pd.concat([pd.read_csv(f, index_col=0) for f in m2_files], axis=0)

    df_model1['BlockType'] = pd.Categorical(df_model1['BlockType'], categories=df_model1['BlockType'].unique())
    df_model1['TrialType'] = pd.Categorical(df_model1['TrialType'], categories=df_model1['TrialType'].unique())
    df_model2['BlockType'] = pd.Categorical(df_model2['BlockType'], categories=df_model2['BlockType'].unique())
    df_model2['TrialType'] = pd.Categorical(df_model2['TrialType'], categories=df_model2['TrialType'].unique())

    # drop .s columns
    m1_s_cols = [c for c in df_model1.columns if c.endswith('.s')]
    m2_s_cols = [c for c in df_model2.columns if c.endswith('.s')]
    df_model1_drops = df_model1.drop(columns=m1_s_cols)
    df_model2_drops = df_model2.drop(columns=m2_s_cols)

    # only select max LL for each subject, each model, across param_ids
    maxLL1 = df_model1_drops.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()
    maxLL2 = df_model2_drops.groupby(['HCPID', 'FakeModel'])['LL'].max().reset_index()

    maxLL = pd.merge(maxLL1, maxLL2, how='outer', on=['HCPID', 'FakeModel'], suffixes=('.m1', '.m2'))
    maxLL['LL.diff'] = maxLL.apply(lambda x: x['LL.m1'] - x['LL.m2'], axis=1)
    maxLL['BestModel'] = maxLL.apply(lambda x: 'model1' if x['LL.m1'] > x['LL.m2'] else 'model2', axis=1)

    m1_cols = [c for c in df_model1.columns if (c.endswith('.s') or c.endswith('.m'))]
    m2_cols = [c for c in df_model2.columns if (c.endswith('.s') or c.endswith('.m'))]

    df_join1 = pd.merge(maxLL, df_model1[['HCPID', 'FakeModel'] + m1_cols].drop_duplicates(), how='right')
    df_join2 = pd.merge(maxLL, df_model2[['HCPID', 'FakeModel'] + m2_cols].drop_duplicates(), how='right')
    df_join = pd.merge(df_join1, df_join2, how='inner').sort_values(by=['HCPID', 'FakeModel'])
    df_join['param_id'] = param_id
    return df_join

def run_maxLL_pr_pipline(main_dir, subject_dir, param_list, overwrite, analysis_type='pr_best'):
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
    if not os.path.exists(os.path.join(main_dir, subject_dir)):
        os.mkdir(os.path.join(main_dir, subject_dir))

    for param in param_list:
        param_id = param['param_id']
        subject_id = param['subject_id']

        param_set = {key: param[key] for key in param.keys() if key not in ('param_id', 'subject_id')}

        # re-simulate fake subject data
        # always set overwrite as False
        # skip simulation process
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
        # agg data may change depending on grouping var and exclude neutral or not
        save_simulation_aggregate_data(main_dir=main_dir,
                                       log_dir=subject_dir,
                                       subject_id=subject_id,
                                       param_id=param_id,
                                       model='model1',
                                       special_suffix="",
                                       verbose=False,
                                       overwrite=overwrite)
        save_simulation_aggregate_data(main_dir=main_dir,
                                       log_dir=subject_dir,
                                       subject_id=subject_id,
                                       param_id=param_id,
                                       model='model2',
                                       special_suffix="",
                                       verbose=True,
                                       overwrite=False)

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
                                  overwrite=False)

        # calculate maxLL
        # NOTE: need to calculate maxLL for both fake subject data -> 4 ...maxLL.csv, 12 files intotal
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model1',
                           model_fake_subject='model1',
                           overwrite=overwrite)
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model1',
                           model_fake_subject='model2',
                           overwrite=False)
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model2',
                           model_fake_subject='model1',
                           overwrite=False)
        save_maxLL_pr_data(main_dir=main_dir,
                           log_dir=subject_dir,
                           subject_id=subject_id,
                           param_id=param_id,
                           model='model2',
                           model_fake_subject='model2',
                           overwrite=False)

    # combine maxLL
    save_model_classification(main_dir=main_dir, subject_dir=subject_dir, param_list=param_list, analysis_type=analysis_type)

def save_maxLL_pr_data(main_dir, log_dir, subject_id, param_id, model, model_fake_subject, overwrite=False):
    """
    Calculate maxLL for fake subject (parameter recovery analysis) and save to maxll folder
    model: model1 or model2, the simulated model
    model_fake_subject: model1 or model2, the fake subject data file generated by whichever model (model1 or model2)
    """


    subject_path = os.path.join(main_dir, log_dir, subject_id, param_id, 'aggregate', str.upper(model_fake_subject) + '.csv')
    model_path = os.path.join(main_dir, log_dir, subject_id, 'likelihood', str.upper(model) + '.csv')
    # print(subject_path)

    # load single subject and model data
    df_subject = pd.read_csv(subject_path, index_col=0)
    df_model = pd.read_csv(model_path, index_col=0)

    # save maxLL data
    dest_dir = os.path.join(main_dir, log_dir, subject_id, param_id, 'maxll')

    # check if exist 12 files in maxll
    if check_exists_simple(dest_dir=dest_dir, special_suffix='.csv', num_files=12, overwrite=overwrite):
        print('...SKIP: ALREADY CALCULATED maxLL ... [%s] [%s] [%s]' % (model, subject_id, param_id))
        return

    # merge df_model and df_subject
    # note: need to remove param_cols
    m_param_cols = [p + '.m' for p in ACTR_PARAMETER_NAME[model]]
    df_merge = merge_for_maxLL(df_model=df_model, df_subject=df_subject, param_cols=m_param_cols)
    df_maxLL = calculate_maxLL(df_merge=df_merge, param_cols=m_param_cols)


    # find maxLL and keep all rows
    maxLL = df_maxLL['LL'].max()
    df_maxLL_top = df_maxLL[df_maxLL['LL'] >= maxLL][['HCPID','LL'] + m_param_cols]

    # merge maxLL data  and subject data
    # add FakeModel column to indicate which model generates this fake subject data file
    res = pd.merge(pd.merge(df_model, df_maxLL_top, on=m_param_cols).dropna(axis=0), df_subject, on=['HCPID'] + MAXLL_FACTOR_VAR, suffixes=('.m', '.s'))
    res['FakeModel'] = model_fake_subject

    # merge  maxLL and aggregate subject data (fake)
    df_merge.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + 'FAKE' + str.upper(model_fake_subject) + '-merged.csv'))
    df_maxLL.to_csv(os.path.join(dest_dir, str.upper(model) + '-' + 'FAKE' + str.upper(model_fake_subject) + '.csv'))
    res.to_csv(os.path.join(dest_dir, str.upper(model)  + '-' + 'FAKE'+str.upper(model_fake_subject) + '-maxLL.csv'))
    print('...COMPLETE maxLL DATA ...[%s] [%s] fake[%s]' % (subject_id, model, model_fake_subject))
    return res
