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
import os 
import os.path as op
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir , 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')

config = configparser.ConfigParser()
config.read(op.join(base_dir,  'config.ini') )
pkl_dir = config['Paths']['pkl_dir']

# %%
from hyppo.discrim import DiscrimOneSample
import numpy as np
import pickle
import pandas as pd
from hyppo.discrim import DiscrimOneSample, DiscrimTwoSample

# %%
date_str = '2026-10-04'

# %%

with open(op.join(pkl_dir, 'MSMAll_FIX', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_ses_relmat.pkl'),   'rb') as f:
        fix_lasso_dict = pickle.load(f)


with open(op.join(pkl_dir, 'MSMAll_FIX', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_ses_relmat.pkl'),   'rb') as f:
        fix_pearson_dict = pickle.load(f)

with open(op.join(pkl_dir, 'MSMAll_FIX', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_ses_relmat.pkl'),   'rb') as f:
        fix_uoi_dict = pickle.load(f)

with open(op.join(pkl_dir, 'MSMAll_FIX', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_ses_relmat.pkl'),   'rb') as f:
        fix_pcorr_dict = pickle.load(f)

with open(op.join(pkl_dir, 'MSMAll_FIX', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_ses_relmat.pkl'),   'rb') as f:
        fix_ridge_dict = pickle.load(f)  


    


# %%
with open(op.join(pkl_dir, 'xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_ses_relmat.pkl'),   'rb') as f:
    xcpd_lasso_dict = pickle.load(f)
    
with open(op.join(pkl_dir, 'xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_ses_relmat.pkl'),   'rb') as f:
        xcpd_pearson_dict = pickle.load(f)

with open(op.join(pkl_dir, 'xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_ses_relmat.pkl'),   'rb') as f:
        xcpd_uoi_dict = pickle.load(f)

with open(op.join(pkl_dir, 'xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_ses_relmat.pkl'),   'rb') as f:
        xcpd_ridge_dict = pickle.load(f)  


with open(op.join(pkl_dir, 'xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_ses_relmat.pkl'),   'rb') as f:
        xcpd_pcorr_dict = pickle.load(f)


# %%
def _complete_subs(*fc_dicts):
    """Subjects present with both sessions in every dict (order from the first)."""
    subs = [k for k in fc_dicts[0]
            if all(k in d and 'ses-1' in d[k] and 'ses-2' in d[k] for d in fc_dicts)]
    n_dropped = len(fc_dicts[0]) - len(subs)
    if n_dropped:
        print(f'Dropping {n_dropped} subjects missing from at least one dict/session')
    return subs


def _stack_sessions(fc_dict, subs):
    """Rows: ses-1 for every subject, then ses-2 for every subject."""
    return np.stack([np.asarray(fc_dict[k][ses], dtype=float).ravel()
                     for ses in ('ses-1', 'ses-2') for k in subs])


def one_samp_discrim(fc_dict):
    subs = _complete_subs(fc_dict)
    x = _stack_sessions(fc_dict, subs)
    y = np.tile(np.arange(len(subs)), 2)
    return DiscrimOneSample().test(x, y)


# %%
def two_samp_discrim(x1_fc_dict, x2_fc_dict):
    subs = _complete_subs(x1_fc_dict, x2_fc_dict)
    x_1 = _stack_sessions(x1_fc_dict, subs)
    x_2 = _stack_sessions(x2_fc_dict, subs)
    y = np.tile(np.arange(len(subs)), 2)
    return DiscrimTwoSample().test(x_1, x_2, y)

# %%
fix_pearson_stat = one_samp_discrim(fix_pearson_dict).stat

# %%
fix_lasso_stat = one_samp_discrim(fix_lasso_dict).stat

# %%
fix_ridge_stat = one_samp_discrim(fix_ridge_dict).stat

# %%
fix_pcorr_stat = one_samp_discrim(fix_pcorr_dict).stat

# %%
fix_uoi_stat = one_samp_discrim(fix_uoi_dict).stat

# %%
two_samp_discrim(fix_pearson_dict, fix_lasso_dict) 

# %%
two_samp_discrim(fix_pearson_dict, fix_ridge_dict) 

# %%
two_samp_discrim(fix_pearson_dict, fix_uoi_dict) 

# %%
two_samp_discrim(fix_pearson_dict, fix_pcorr_dict) 

# %% [markdown]
# #### XCPD

# %%
two_samp_discrim(xcpd_pearson_dict, xcpd_lasso_dict) 

# %%
two_samp_discrim(xcpd_pearson_dict, xcpd_ridge_dict) 

# %%
two_samp_discrim(xcpd_pearson_dict, xcpd_pcorr_dict) 

# %%
two_samp_discrim(xcpd_pearson_dict, xcpd_uoi_dict) 

# %%
xcpd_pearson_stat = one_samp_discrim(xcpd_pearson_dict).stat

# %%
xcpd_lasso_stat = one_samp_discrim(xcpd_lasso_dict).stat

# %%
xcpd_ridge_stat = one_samp_discrim(xcpd_ridge_dict).stat

# %%
xcpd_pcorr_stat = one_samp_discrim(xcpd_pcorr_dict).stat

# %%
xcpd_uoi_stat = one_samp_discrim(xcpd_uoi_dict).stat

# %% [markdown]
# ### PNC

# %%
with open(op.join(pkl_dir, 'pnc_xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_task_relmat.pkl'),   'rb') as f:
    pnc_lasso_dict = pickle.load(f)
    
with open(op.join(pkl_dir, 'pnc_xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_task_relmat.pkl'),   'rb') as f:
        pnc_pearson_dict = pickle.load(f)

with open(op.join(pkl_dir, 'pnc_xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_task_relmat.pkl'),   'rb') as f:
        pnc_uoi_dict = pickle.load(f)

with open(op.join(pkl_dir, 'pnc_xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_task_relmat.pkl'),   'rb') as f:
        pnc_ridge_dict = pickle.load(f)  


with open(op.join(pkl_dir, 'pnc_xcpd', f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_task_relmat.pkl'),   'rb') as f:
        pnc_pcorr_dict = pickle.load(f)

# %%
pnc_pearson_stat = one_samp_discrim(pnc_pearson_dict).stat

# %%
pnc_lasso_stat = one_samp_discrim(pnc_lasso_dict).stat

# %%
two_samp_discrim(pnc_pearson_dict, pnc_pcorr_dict) 

# %%
two_samp_discrim(pnc_pearson_dict, pnc_ridge_dict) 

# %%
two_samp_discrim(pnc_pearson_dict, pnc_lasso_dict) 

# %%
two_samp_discrim(pnc_pearson_dict, pnc_uoi_dict) 

# %%
two_samp_discrim(pnc_pearson_dict, pnc_pcorr_dict) 

# %%
pnc_ridge_stat = one_samp_discrim(pnc_ridge_dict).stat

# %%
pnc_pcorr_stat = one_samp_discrim(pnc_pcorr_dict).stat

# %%
pnc_uoi_stat = one_samp_discrim(pnc_uoi_dict).stat

# %%
fix_discrim_dict = {'fix_lasso': fix_lasso_stat, 
               'fix_pearson': fix_pearson_stat, 
               'fix_uoi': fix_uoi_stat, 
               'fix_ridge': fix_ridge_stat, 
               'fix_pcorr': fix_pcorr_stat} 


# %%
xcpd_discrim_dict = {'xcpd_lasso': xcpd_lasso_stat, 
               'xcpd_pearson': xcpd_pearson_stat, 
               'xcpd_uoi': xcpd_uoi_stat, 
               'xcpd_ridge': xcpd_ridge_stat, 
               'xcpd_pcorr': xcpd_pcorr_stat} 


# %%
pnc_discrim_dict = {'pnc_lasso': pnc_lasso_stat, 
               'pnc_pearson': pnc_pearson_stat, 
               'pnc_uoi': pnc_uoi_stat, 
               'pnc_ridge': pnc_ridge_stat, 
               'pnc_pcorr': pnc_pcorr_stat} 


# %%
fix_discrim_dict

# %%
xcpd_discrim_dict

# %%
results_df = pd.DataFrame({'MSMAll_FIX': list(fix_discrim_dict.values()), 
              'PNC': list(pnc_discrim_dict.values()), 
              'XCPD': list(xcpd_discrim_dict.values())}, index=['LASSO', 'Pearson', 'UoI', 'Ridge', 'Partial Correlation'])


# %%
results_df.to_csv(op.join(results_dir, 'plotting', 'stat-discrim.csv') )
