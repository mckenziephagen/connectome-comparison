# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
script_dir = op.join(base_dir , 'scripts')
results_dir =  op.join(base_dir , 'results')
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')

config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )

pkl_dir = config['Paths']['pkls']

# %%
from glob import glob
import pickle
import numpy as np
import pandas as pd


# %%
def make_fc_df(results_dict, model , ses): 
    fc_df = pd.DataFrame(columns=results_dict.keys()) 
    for sub in results_dict.keys(): 
        if 'correlation' in model:
            mat = results_dict[sub][f'ses-{ses}']
            np.fill_diagonal(mat, np.nan)

            fc_df[sub]  = mat[np.triu_indices(mat.shape[0], k=1)].flatten()
        else:
            mat = results_dict[sub][f'ses-{ses}']

            np.fill_diagonal(mat, np.nan)

            fc_df[sub] = results_dict[sub][f'ses-{ses}'].flatten() 
    fc_df = fc_df.T
    return fc_df


# %%
date_str = '2026-10-04'


# %%
proc_type = 'MSMAll_FIX'
out_dir = op.join(results_dir, 'MSMAll-FIX')
os.makedirs(out_dir, exist_ok=True)
for model_str in ['uoiLasso', 'correlation', 'partial_correlation', 
               #   'corr-lassoThresh_sub', 'corr-uoiThresh_sub',
                 'ridgeCV', 'lassoBIC']: 
    pkl_file = f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-{model_str}_ses_relmat.pkl'
    with open(op.join(pkl_dir, proc_type, pkl_file), 'rb') as l:
        results_dict = pickle.load(l)

    fc_df = make_fc_df(results_dict, model = model_str, ses='full') 
    print(fc_df.shape)
    fc_df.to_csv(op.join(out_dir, 
                          f'proc-FIX_model-{model_str}_ses-full_connectome-edges.csv'))

# %% [markdown]
# #### XCPD

# %%
proc_type = 'xcpd' 
out_dir = op.join(results_dir, 'XCPD')
os.makedirs(out_dir, exist_ok=True)
for model_str in ['uoiLasso', 'correlation', 'partial_correlation', 
                 'ridgeCV', 'lassoBIC']: 
    pkl_file = f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-{model_str}_ses_relmat.pkl'
    with open(op.join(pkl_dir, proc_type, pkl_file), 'rb') as l:
        results_dict = pickle.load(l)

    fc_df = make_fc_df(results_dict, model = model_str, ses='full') 
    print(fc_df.shape)
    fc_df.to_csv(op.join(out_dir, 
                         f'proc-XCPD_model-{model_str}_ses-full_connectome-edges.csv'))

# %% [markdown]
# #### PNC

# %%
proc_type = 'pnc_xcpd'
out_dir = op.join(results_dir, 'PNC_XCPD')
os.makedirs(out_dir, exist_ok=True)
for model_str in ['uoiLasso', 'correlation', 'partial_correlation', 
                 'ridgeCV', 'lassoBIC']: 
    pkl_file = f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-{model_str}_task_relmat.pkl'
    with open(op.join(pkl_dir, proc_type, pkl_file), 'rb') as l:
        results_dict = pickle.load(l)
# PNC doesn't have a "full" connectome like HCP does
    fc_df = make_fc_df(results_dict, model = model_str, ses='2') 
    fc_df.to_csv(op.join(out_dir, 
                         f'proc-PNC_model-{model_str}_ses-2_connectome-edges.csv'))
