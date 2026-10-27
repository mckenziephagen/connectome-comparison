# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python [conda env:fc]
#     language: python
#     name: conda-env-fc-py
# ---

# %%
import os 
import os.path as op
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir , 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'notebooks', '02-Analysis')

config = configparser.ConfigParser()
config.read(op.join(base_dir,  'config.ini') )
pkl_dir = config['Paths']['pkls']

# %%
from glob import glob 
import pickle
import pandas as pd
import numpy as np
import datetime

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
def make_icc_df(fc_dict, movement_df):
    temp_list = []
    icc_df = pd.DataFrame(columns=['values', 'ses', 'sub', 'pos'] )
    for sub in list(fc_dict.keys()):
        # print(sub)
        for ses in fc_dict[sub].keys():
            movement = np.mean(movement_df.loc[sub.split('-')[-1], [f'rfMRI_REST{ses[-1]}_RL', f'rfMRI_REST{ses[-1]}_LR']]) 
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

# with open(glob(op.join(results_path, f'*{date_str}*corr-uoiThresh*pkl'))[0], 'rb') as l:
#     fix_pearson_uoiThresh_dict = pickle.load(l)   
# fix_pearson_uoiThresh_dict = delete_full(fix_pearson_uoiThresh_dict)

# with open(glob(op.join(results_path, f'*{date_str}*corr-lassoThresh*pkl'))[0], 'rb') as l:
#     fix_pearson_lassoThresh_dict = pickle.load(l)   
# fix_pearson_lassoThresh_dict = delete_full(fix_pearson_lassoThresh_dict)


# %%
lasso_fix_icc_df = make_icc_df(fix_lasso_dict, movement_df)
lasso_fix_icc_df.to_csv(op.join(results_path, f'{date_str}_lasso_fix_icc_df.csv')) 

# %%
ridge_fix_icc_df = make_icc_df(fix_ridge_dict, movement_df)
ridge_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_ridge_fix_icc_df.csv')) 

# %%
uoi_fix_icc_df = make_icc_df(fix_uoi_dict, movement_df)
uoi_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_uoi_fix_icc_df.csv')) 

# %%
pearson_fix_icc_df = make_icc_df(fix_pearson_dict, movement_df)
pearson_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_pearson_fix_icc_df.csv')) 

# %%
pcorr_fix_icc_df = make_icc_df(fix_pcorr_dict, movement_df)
pcorr_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_pcorr_fix_icc_df.csv')) 

# %%
pearson_uoiThresh_fix_icc_df = make_icc_df(fix_pearson_uoiThresh_dict, movement_df)
pearson_uoiThresh_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_uoiThresh_fix_icc_df.csv')) 

# %%
pearson_lassoThresh_fix_icc_df = make_icc_df(fix_pearson_lassoThresh_dict, movement_df)
pearson_lassoThresh_fix_icc_df.to_csv(op.join(results_path,  f'{date_str}_lassoThresh_fix_icc_df.csv')) 

# %% [markdown]
# ### XCPD

# %%
proc_type = 'xcpd'
results_path = op.join(pkl_dir, proc_type)


# %%
with open(glob(op.join(results_path, f'*model-correlation*pkl'))[0], 'rb') as l:
    xcpd_pearson_dict = pickle.load(l)
xcpd_pearson_dict = delete_full(xcpd_pearson_dict)

with open(glob(op.join(results_path, f'*partial*pkl'))[0], 'rb') as l:
    xcpd_pcorr_dict = pickle.load(l)
xcpd_pcorr_dict = delete_full(xcpd_pcorr_dict)

with open(glob(op.join(results_path, f'*ridgeCV*pkl'))[0], 'rb') as l:
    xcpd_ridge_dict = pickle.load(l)
xcpd_ridge_dict = delete_full(xcpd_ridge_dict)

with open(glob(op.join(results_path, f'*lassoBIC*pkl'))[0], 'rb') as l:
    xcpd_lasso_dict = pickle.load(l)
xcpd_lasso_dict = delete_full(xcpd_lasso_dict)

with open(glob(op.join(results_path, f'*uoiLasso*pkl'))[0], 'rb') as l:
    xcpd_uoi_dict = pickle.load(l)
xcpd_uoi_dict = delete_full(xcpd_uoi_dict)

# %%
pearson_xcpd_icc_df = make_icc_df(xcpd_pearson_dict, movement_df)
pearson_xcpd_icc_df.to_csv(op.join(results_path, f'{date_str}_pearson_xcpd_icc_df.csv')) 

# %%
pcorr_xcpd_icc_df = make_icc_df(xcpd_pcorr_dict, movement_df)
pcorr_xcpd_icc_df.to_csv(op.join(results_path, f'{date_str}_pcorr_xcpd_icc_df.csv')) 

# %%
ridge_xcpd_icc_df = make_icc_df(xcpd_ridge_dict, movement_df)
ridge_xcpd_icc_df.to_csv(op.join(results_path, f'{date_str}_ridge_xcpd_icc_df.csv')) 

# %%
lasso_xcpd_icc_df = make_icc_df(xcpd_lasso_dict, movement_df)
lasso_xcpd_icc_df.to_csv(op.join(results_path, f'{date_str}_lasso_xcpd_icc_df.csv')) 

# %%
uoi_xcpd_icc_df = make_icc_df(xcpd_uoi_dict, movement_df)
uoi_xcpd_icc_df.to_csv(op.join(results_path, f'{date_str}_uoi_xcpd_icc_df.csv')) 

# %% [markdown]
# #### PNC XCPD

# %%
proc_type = 'pnc_xcpd'
results_path = f'/pscratch/sd/m/mphagen/storage/results/{proc_type}'

# %%
with open(glob(op.join(results_path, f'*-correlation*pkl'))[0], 'rb') as l:
    pnc_pearson_dict = pickle.load(l)

# %%
for sub in pnc_pearson_dict.keys():
    pnc_pearson_dict[sub].pop('ses-full') 

# %%
with open(glob(op.join(results_path, f'*-lassoBIC*pkl'))[0], 'rb') as l:
    pnc_lasso_dict = pickle.load(l)

pnc_lasso_dict = delete_full(pnc_lasso_dict) 

# %%
with open(glob(op.join(results_path, f'*-ridge*pkl'))[0], 'rb') as l:
    pnc_ridge_dict = pickle.load(l)

pnc_ridge_dict = delete_full(pnc_ridge_dict)

# %%
with open(glob(op.join(results_path, f'*-partial_correlation*pkl'))[0], 'rb') as l:
    pnc_pcorr_dict = pickle.load(l)

pnc_pcorr_dict = delete_full(pnc_pcorr_dict)

# %%
with open(glob(op.join(results_path, f'*-uoiLasso*pkl'))[0], 'rb') as l:
    pnc_uoi_dict = pickle.load(l)

pnc_uoi_dict = delete_full(pnc_uoi_dict)

# %%
pnc_pearson_df = make_icc_df(pnc_pearson_dict, movement_df) 
pnc_pearson_df.to_csv(op.join(results_path,  f'{date_str}_pearson_pnc_icc_df.csv')) 

# %%
pnc_lasso_df = make_icc_df(pnc_lasso_dict, movement_df) 
pnc_lasso_df.to_csv(op.join(results_path,  f'{date_str}_lasso_pnc_icc_df.csv')) 

# %%
pnc_ridge_df = make_icc_df(pnc_ridge_dict, movement_df) 
pnc_ridge_df.to_csv(op.join(results_path,  f'{date_str}_ridge_pnc_icc_df.csv')) 

# %%
pnc_pcorr_df = make_icc_df(pnc_pcorr_dict, movement_df) 
pnc_pcorr_df.to_csv(op.join(results_path,  f'{date_str}_pcorr_pnc_icc_df.csv')) 

# %%
pnc_uoi_df = make_icc_df(pnc_uoi_dict, movement_df) 
pnc_uoi_df.to_csv(op.join(results_path,  f'{date_str}_uoi_pnc_icc_df.csv')) 

# %%
