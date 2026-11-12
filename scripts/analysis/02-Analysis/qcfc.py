# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: fc
#     language: python
#     name: python3
# ---

# %%
import os.path as op
import os 
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir, 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')
config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )

# %%
pkl_dir = config['Paths']['pkl_dir']

# %%
import pickle
import pingouin as pg
import pandas as pd
import numpy as np
import pickle

# %%
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
hpc_qa_df = pd.read_csv(op.join(base_dir, 'data', 'hcp_relative_movement.csv'), index_col=0)

# %%
hpc_qa_df = hpc_qa_df.rename(columns={'rfMRI_REST1_RL': 'RL_1',
                                       'rfMRI_REST1_LR': 'LR_1', 
                                       'rfMRI_REST2_RL': 'RL_2',
                                         'rfMRI_REST2_LR': 'LR_2'}) 
hpc_qa_df.index = [f'sub-{ii}' for ii in list(hpc_qa_df.index)]

# %%
#the sessions are cross-validated out of order, so these need to be swapped. 
hpc_qa_df = hpc_qa_df.rename(columns={'LR_1': 'LR_2', 
                                      'RL_1': 'RL_2', 
                                      'LR_2': 'LR_1', 
                                      'RL_2': 'RL_1'})

# %%
#adopt future option 
pd.set_option('future.no_silent_downcasting', True)

unrestricted_df = pd.read_csv(op.join(base_dir,
                                       'data', 
                                       'restricted', 
                                       'unrestricted_1_27_2022_20_50_7.csv')) 
unrestricted_df.index = [f'sub-{i}' for i in unrestricted_df['Subject']]
unrestricted_df[['Gender']] = unrestricted_df[['Gender']].replace({'M':0, 'F':1}).infer_objects(copy=False)

# %%
restricted_df = pd.read_csv(op.join(base_dir, 
                                    'data', 
                                    'restricted', 
                                    'RESTRICTED_1_31_2022_23_26_45.csv'))
restricted_df.index = [f'sub-{i}' for i in restricted_df['Subject']]

# %%
covariates = ['Gender', 'Age_in_Yrs']
covariate_df = unrestricted_df[['Gender']].join(restricted_df[['Age_in_Yrs']]) 


# %%
def make_fc_df(results_dict, ses):
    """Subjects x edges (flattened full matrix; columns are integer edge ids)."""
    return pd.DataFrame.from_dict(
        {key: np.asarray(val[f'ses-{ses}'], dtype=float).ravel()
         for key, val in results_dict.items()},
        orient='index')


# %%
def make_qcfc_df(fc_df, covariate_df, qa_df, ses):
    connectomes = fc_df if covariate_df is None else fc_df.join(covariate_df)
    # median across all runs belonging to this session (columns ending in f'_{ses}')
    rms_df = qa_df.filter(regex=f'_{ses}$').astype(float).median(axis=1).rename('rms')
    return connectomes.join(rms_df)


# %%
def qcfc(qcfc_df, covariate_df, edge_range):
    covar = None if covariate_df is None else list(covariate_df.columns)
    result_df = pd.DataFrame(np.nan,
                             index=[f'edge-{i}' for i in edge_range],
                             columns=['r', 'p-val'])
    for edge_id in edge_range:
        try:
            metric = pg.partial_corr(data=qcfc_df, x=edge_id, y='rms', covar=covar)
            # pingouin >= 0.7 renamed 'p-val' -> 'p_val'
            metric = metric.rename(columns={'p_val': 'p-val'})
        except np.linalg.LinAlgError:
            continue
        result_df.loc[f'edge-{edge_id}'] = metric.loc['pearson', ['r', 'p-val']].astype(float).values
    return result_df


# %%
def run_qcfc(model, covariate_df, proc_type, ses, qa_df=hpc_qa_df, date_str='2026-10-04',
             resample=False, n_boot=9, seed=0, cv='ses'):
    # cv: 'ses' for HCP (cross-session), 'task' for PNC (rest vs. idemo)
    key = f'{model}_{proc_type}_ses-{ses}'
    result_dict = {key: {}}
    with open(op.join(pkl_dir, proc_type,  f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-{model}_{cv}_relmat.pkl')
              , 'rb') as l:
        connectome_dict = pickle.load(l)

    fc_df = make_fc_df(connectome_dict, ses=ses)
    edge_range = fc_df.columns
    qcfc_df = make_qcfc_df(fc_df, covariate_df, qa_df, ses=ses)

    result_dict[key]['estimate'] = qcfc(qcfc_df, covariate_df, edge_range)

    if resample:
        rng = np.random.default_rng(seed)
        for ii in range(1, n_boot + 1):
            resample_df = qcfc_df.sample(frac=1, replace=True, random_state=rng)
            result_dict[key][f'boot_{ii}'] = qcfc(resample_df, covariate_df, edge_range)

    return result_dict


# %%
qcfc_dict = {}

# %%
# single results date for every dataset/model so all inputs are comparable
date_str = '2026-10-04'

# %%
model = 'lassoBIC'
proc_type = 'MSMAll_FIX'
qcfc_dict.update(run_qcfc(model, covariate_df, proc_type, date_str=date_str, ses='1'))
qcfc_dict.update(run_qcfc(model, covariate_df, proc_type, date_str=date_str, ses='2'))

# %%
model = 'correlation'
proc_type = 'MSMAll_FIX'
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='1'))
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='2'))

# %%
model = 'partial_correlation'
proc_type = 'MSMAll_FIX'
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='2')) 

# %%
model = 'ridgeCV'
proc_type = 'MSMAll_FIX'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='2')) 

# %%
model = 'uoiLasso'
proc_type = 'MSMAll_FIX'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model,covariate_df, proc_type, date_str=date_str, ses='2')) 

# %%
model = 'correlation'
proc_type = 'xcpd'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='2')) 

# %%
model = 'lassoBIC'
proc_type = 'xcpd'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='2')) 

# %%
model = 'uoiLasso'
proc_type = 'xcpd'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='2')) 

# %%
model = 'ridgeCV'
proc_type = 'xcpd'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='2')) 

# %%
model = 'partial_correlation'
proc_type = 'xcpd'
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='1')) 
qcfc_dict.update(run_qcfc(model, covariate_df,proc_type, date_str=date_str, ses='2'))

# %% [markdown]
# ## PNC

# %%
pnc_qa_df = pd.read_csv(op.join(base_dir, 'data', 'pnc_relative_movement.csv'), index_col=0)
pnc_qa_df.index = [f'sub-{ii}' for ii in pnc_qa_df.index]
# PNC was cross-validated across tasks: in calc_fc.py the timeseries are concatenated
# rest then idemo, and fold_0 (-> 'ses-1') is *trained* on idemo, fold_1 (-> 'ses-2') on rest.
pnc_qa_df = pnc_qa_df.rename(columns={'mean_relative_rms_idemo': 'idemo_1',
                                      'mean_relative_rms_rest': 'rest_2'})

# %%
pnc_participants_df = pd.read_csv(op.join(base_dir, 'data', 'study-PNC_desc-participants.tsv'), sep='\t')
pnc_participants_df.index = [f'sub-{i}' for i in pnc_participants_df['participant_id']]
# match HCP covariate coding/names: Gender (M=0, F=1), Age_in_Yrs
pnc_covariate_df = pd.DataFrame({'Gender': pnc_participants_df['sex'].map({'Male': 0, 'Female': 1}),
                                 'Age_in_Yrs': pnc_participants_df['age'].astype(float)})

# %%
proc_type = 'pnc_xcpd'
for model in ['correlation', 'lassoBIC', 'partial_correlation', 'ridgeCV', 'uoiLasso']:
    for ses in ['1', '2']:
        qcfc_dict.update(run_qcfc(model, pnc_covariate_df, proc_type, ses=ses,
                                  qa_df=pnc_qa_df, date_str=date_str, cv='task'))

# %%
mask =np.triu(np.ones_like(np.ones((100,100)), dtype=bool), k=1) 

# %%
qcfc_dict.keys() 

# %%
for key in qcfc_dict.keys(): 
    df = qcfc_dict[key]['estimate']
    # if 'correlation' in key: 
    #     sig_df = df[mask.ravel()].loc[df['p-val'] <= .05]

    #     print(key, '', sig_df.shape[0] / 4950) 
    # else: 
    sig_df = df.loc[df['p-val'] <= .05]

    print(key, '', sig_df.shape[0]/ (10000 - 100)) 

# %%
with open(op.join(results_dir, '2026-10-26_qcfc_dict.pkl'), 'wb') as f: 
    pickle.dump(qcfc_dict, f) 

# %%
