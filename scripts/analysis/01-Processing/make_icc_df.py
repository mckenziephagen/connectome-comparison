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

# %% [markdown]
# Organizes connectomes into long format for reliabilitiy analyses in R. 
#
# **Inputs:**
#
# - {pkl_dir}/{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-{model_str}_ses_relmat.pkl
# - data/{dataset}_relative_movement.csv
#
# **Outputs:**
# - results/proc-{proc_type}_model-{model_str}_reliability-edges.csv

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
from glob import glob 
import pickle
import pandas as pd
import numpy as np

# %%
date_str = '2026-10-04' 


# %%
def delete_full(dictionary):
    for sub in dictionary.keys():
        try: 
             del dictionary[sub]['ses-full']
        except KeyError: 
            print(sub)
    return dictionary


# %%
movement_df = pd.read_csv(op.join(base_dir, 'data', '2026_relative_movement.csv'), 
                         index_col = 0) 
movement_df.index = movement_df.index.astype(str) 


# %%
def make_icc_df(fc_dict, movement_df, dataset='hcp'):
    temp_list = []
    icc_df = pd.DataFrame(columns=['values', 'ses', 'sub', 'pos'] )
    for sub in list(fc_dict.keys()):
        # print(sub)
        for ses in fc_dict[sub].keys():
            if dataset == 'hcp':
                movement = np.mean(movement_df.loc[sub.split('-')[-1], [f'rfMRI_REST{ses[-1]}_RL', f'rfMRI_REST{ses[-1]}_LR']]) 
            elif dataset == 'pnc': 
                try: 
                  movement = movement_df.loc[sub.split('-')[-1]][f'mean_relative_rms_{ses.split('-')[-1]}']
                #   print(movement)
                except KeyError: 
                    movement = np.nan
            temp_list.append(pd.DataFrame(data = {'values': fc_dict[sub][ses].ravel(), 

                               'sub':  sub, 
                              'ses': ses, 
                              'pos': list(range(0,10000,1)),
                                'movement': movement }))
    icc_df = pd.concat(temp_list)
    return icc_df


# %%
proc_type = 'MSMAll_FIX'
date_str = '2026-10-04'
results_path = op.join(pkl_dir, proc_type)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_ses_relmat.pkl'), 'rb') as l:
    fix_lasso_dict = pickle.load(l)
fix_lasso_dict = delete_full(fix_lasso_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_ses_relmat.pkl'), 'rb') as l:
    fix_uoi_dict = pickle.load(l)
fix_uoi_dict = delete_full(fix_uoi_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_ses_relmat.pkl'), 'rb') as l:
    fix_pearson_dict = pickle.load(l)
fix_pearson_dict = delete_full(fix_pearson_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_ses_relmat.pkl'), 'rb') as l:
    fix_pcorr_dict = pickle.load(l)  
fix_pcorr_dict = delete_full(fix_pcorr_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_ses_relmat.pkl'), 'rb') as l:
    fix_ridge_dict = pickle.load(l)  
fix_ridge_dict = delete_full(fix_ridge_dict)


# %%
lasso_fix_icc_df = make_icc_df(fix_lasso_dict, movement_df)
lasso_fix_icc_df.to_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-lassoBIC_reliability-edges.csv')) 

# %%
ridge_fix_icc_df = make_icc_df(fix_ridge_dict, movement_df)
ridge_fix_icc_df.to_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_reliability-edges.csv')) 

# %%
uoi_fix_icc_df = make_icc_df(fix_uoi_dict, movement_df)
uoi_fix_icc_df.to_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_reliability-edges.csv')) 

# %%
pearson_fix_icc_df = make_icc_df(fix_pearson_dict, movement_df)
pearson_fix_icc_df.to_csv(op.join(results_dir,'MSMAll-FIX',  'proc-FIX_model-Pearson_reliability-edges.csv')) 

# %%
pcorr_fix_icc_df = make_icc_df(fix_pcorr_dict, movement_df)
pcorr_fix_icc_df.to_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-partial-correlation_reliability-edges.csv')) 

# %% [markdown]
# ### XCPD

# %%
proc_type = 'xcpd'
results_path = op.join(pkl_dir, proc_type)


# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_ses_relmat.pkl'), 'rb') as l:
    xcpd_pearson_dict = pickle.load(l)
xcpd_pearson_dict = delete_full(xcpd_pearson_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_ses_relmat.pkl'), 'rb') as l:
    xcpd_pcorr_dict = pickle.load(l)
xcpd_pcorr_dict = delete_full(xcpd_pcorr_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_ses_relmat.pkl'), 'rb') as l:
    xcpd_ridge_dict = pickle.load(l)
xcpd_ridge_dict = delete_full(xcpd_ridge_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_ses_relmat.pkl'), 'rb') as l:
    xcpd_lasso_dict = pickle.load(l)
xcpd_lasso_dict = delete_full(xcpd_lasso_dict)

with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_ses_relmat.pkl'), 'rb') as l:
    xcpd_uoi_dict = pickle.load(l)
xcpd_uoi_dict = delete_full(xcpd_uoi_dict)

# %%
pearson_xcpd_icc_df = make_icc_df(xcpd_pearson_dict, movement_df)
pearson_xcpd_icc_df.to_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-Pearson_reliability-edges.csv')) 

# %%
pcorr_xcpd_icc_df = make_icc_df(xcpd_pcorr_dict, movement_df)
pcorr_xcpd_icc_df.to_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-partial-correlation_reliability-edges.csv')) 

# %%
ridge_xcpd_icc_df = make_icc_df(xcpd_ridge_dict, movement_df)
ridge_xcpd_icc_df.to_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-ridgeCV_reliability-edges.csv')) 

# %%
lasso_xcpd_icc_df = make_icc_df(xcpd_lasso_dict, movement_df)
lasso_xcpd_icc_df.to_csv(op.join(results_dir,'XCPD',  'proc-XCPD_model-lassoBIC_reliability-edges.csv')) 

# %%
uoi_xcpd_icc_df = make_icc_df(xcpd_uoi_dict, movement_df)
uoi_xcpd_icc_df.to_csv(op.join(results_dir,'XCPD', 'proc-XCPD_model-uoiLasso_reliability-edges.csv')) 

# %% [markdown]
# #### PNC XCPD

# %%
proc_type = 'pnc_xcpd'
results_path = op.join(pkl_dir, proc_type)

# %%
movement_df = pd.read_csv(op.join(base_dir, 'data', 'pnc_relative_movement.csv'), index_col=0)
movement_df.index = movement_df.index.astype(str)

# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_task_relmat.pkl'), 'rb') as l:
    pnc_pearson_dict = pickle.load(l)

# %%
pnc_pearson_dict = delete_full(pnc_pearson_dict)

for sub in pnc_pearson_dict.keys(): 
    pnc_pearson_dict[sub]['ses-rest'] = pnc_pearson_dict[sub].pop('ses-2')
    pnc_pearson_dict[sub]['ses-idemo'] = pnc_pearson_dict[sub].pop('ses-1')


# %%
pearson_pnc_icc_df = make_icc_df(pnc_pearson_dict, movement_df, dataset='pnc')

# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_task_relmat.pkl'), 'rb') as l:
    pnc_lasso_dict = pickle.load(l)

# %%
pnc_lasso_dict = delete_full(pnc_lasso_dict)

for sub in pnc_lasso_dict.keys(): 
    pnc_lasso_dict[sub]['ses-rest'] = pnc_lasso_dict[sub].pop('ses-2')
    pnc_lasso_dict[sub]['ses-idemo'] = pnc_lasso_dict[sub].pop('ses-1')


# %%
lasso_pnc_icc_df = make_icc_df(pnc_lasso_dict, movement_df, dataset='pnc')

# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_task_relmat.pkl'), 'rb') as l:
    pnc_ridge_dict = pickle.load(l)

# %%
pnc_ridge_dict = delete_full(pnc_ridge_dict)

for sub in pnc_ridge_dict.keys(): 
    pnc_ridge_dict[sub]['ses-rest'] = pnc_ridge_dict[sub].pop('ses-2')
    pnc_ridge_dict[sub]['ses-idemo'] = pnc_ridge_dict[sub].pop('ses-1')


# %%
ridge_pnc_icc_df = make_icc_df(pnc_ridge_dict, movement_df, dataset='pnc')

# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_task_relmat.pkl'), 'rb') as l:
    pnc_pcorr_dict = pickle.load(l)

# %%
pnc_pcorr_dict = delete_full(pnc_pcorr_dict)

for sub in pnc_pcorr_dict.keys(): 
    pnc_pcorr_dict[sub]['ses-rest'] = pnc_pcorr_dict[sub].pop('ses-2')
    pnc_pcorr_dict[sub]['ses-idemo'] = pnc_pcorr_dict[sub].pop('ses-1')


# %%
pcorr_pnc_icc_df = make_icc_df(pnc_pcorr_dict, movement_df, dataset='pnc')

# %%
with open(op.join(results_path, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_task_relmat.pkl'), 'rb') as l:
    pnc_uoi_dict = pickle.load(l)

# %%
pnc_uoi_dict = delete_full(pnc_uoi_dict)

for sub in pnc_uoi_dict.keys(): 
    pnc_uoi_dict[sub]['ses-rest'] = pnc_uoi_dict[sub].pop('ses-2')
    pnc_uoi_dict[sub]['ses-idemo'] = pnc_uoi_dict[sub].pop('ses-1')


# %%
uoi_pnc_icc_df = make_icc_df(pnc_uoi_dict, movement_df, dataset='pnc')

# %%
pearson_pnc_icc_df.to_csv(op.join(results_dir, 'PNC_XCPD', 'proc-PNC_model-Pearson_reliability-edges.csv')) 

# %%
lasso_pnc_icc_df.to_csv(op.join(results_dir, 'PNC_XCPD', 'proc-PNC_model-lassoBIC_reliability-edges.csv')) 

# %%
ridge_pnc_icc_df.to_csv(op.join(results_dir, 'PNC_XCPD', 'proc-PNC_model-ridgeCV_reliability-edges.csv')) 

# %%
pcorr_pnc_icc_df.to_csv(op.join(results_dir, 'PNC_XCPD', 'proc-PNC_model-partial-correlation_reliability-edges.csv')) 

# %%
uoi_pnc_icc_df.to_csv(op.join(results_dir, 'PNC_XCPD', 'proc-PNC_model-uoiLasso_reliability-edges.csv')) 
