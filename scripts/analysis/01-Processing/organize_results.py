# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python [conda env:base]
#     language: python
#     name: conda-base-py
# ---

# +
import os.path as op
import os 
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir , 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'notebooks', '02-Analysis')
config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )
# -

pkl_dir = config['Paths']['pkls']

from glob import glob
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns


def bids_entities(file): 
    base = op.basename(file) 
    sub_id = re.search(r'(?<=sub-)\d+', base).group()
    task_id = re.search(r'(?<=task-)[^_]+', base).group()
    ses_id = re.search(r'(?<=ses-)[^_]+', base).group()

    return sub_id, task_id, ses_id


def unpack_and_save_results(result_files, atlas_spec, model_id, organized_data_path):     
    results_dict = {} 
    for file in result_files: 
        sub_id, task_id, ses_id = bids_entities(file)  
        # print(sub_id, task_id, ses_id)
        
        with open(file, 'rb') as f:
             mat = pickle.load(f) 

        conn_mat = mat['full_fc_matrix']
        if f'sub-{sub_id}' not in results_dict.keys(): 
            results_dict[f'sub-{sub_id}'] = {} 
            results_dict[f'sub-{sub_id}'][f'ses-1'] = mat['fold_0']['fc_matrix']
            results_dict[f'sub-{sub_id}'][f'ses-2'] = mat['fold_1']['fc_matrix']
            results_dict[f'sub-{sub_id}'][f'ses-full'] = conn_mat
    
    with open(op.join(organized_data_path, 
                      f'{str(datetime.date.today())}_task-{task_id}_space-{atlas_spec}_model-{model_id}_relmat.pkl'), 'wb') as f:
        pickle.dump(results_dict, f) 
    return results_dict


def get_files(derivatives, proc_type, model_id): 
    conn_path = op.join(derivatives, 'connectivity-matrices', proc_type, model_id) 
    result_files = glob(op.join(conn_path, 'sub-*', '*.pkl'))
    return result_files


# ## Human Connectome Project 

atlas_spec = 'fsLR_seg-4S156Parcels_den-91k'
fc_bids = config['Paths']['hcp']
derivatives = op.join(fc_bids, 'derivatives')

# #### MSMAll FIX

# +
proc_type = 'MSMAll_FIX' 
model_id = 'correlation_ses'
fix_corr_results = get_files(derivatives, proc_type, model_id)
print(len(fix_corr_results)) 
organized_data_path = op.join(pkl_dir, proc_type)

os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_corr_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'partial_correlation_ses'
fix_pcorr_results = get_files(derivatives, proc_type, model_id)
print(len(fix_pcorr_results)) 
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_pcorr_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'ridgeCV_ses'

fix_ridge_results = get_files(derivatives, proc_type, model_id)
print(len(fix_ridge_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_ridge_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'lassoBIC_ses'
fix_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(fix_lasso_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_lasso_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'uoiLasso_ses'
fix_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(fix_uoi_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

uoi_fix_results = unpack_and_save_results(fix_uoi_results, atlas_spec, model_id, organized_data_path)
# -

# #### XCPD 

# +
proc_type = 'xcpd' 
model_id = 'correlation_ses'
xcpd_corr_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_corr_results)) 

organized_data_path = op.join(pkl_dir, proc_type)
os.makedirs(organized_data_path, exist_ok=True) 

xcpd_corr_dict = unpack_and_save_results(xcpd_corr_results, atlas_spec, model_id, organized_data_path)
# +
model_id = 'partial_correlation_ses'
xcpd_pcorr_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_pcorr_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

xcpd_pcorr_dict = unpack_and_save_results(xcpd_pcorr_results, atlas_spec, model_id, organized_data_path)
# +
model_id = 'lassoBIC_ses'
xcpd_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_lasso_results))

os.makedirs(organized_data_path, exist_ok=True) 

xcpd_lasso_results_dict = unpack_and_save_results(xcpd_lasso_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'ridgeCV_ses'
xcpd_ridge_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_ridge_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

ridge_xcpd_results = unpack_and_save_results(xcpd_ridge_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'uoiLasso_ses'
xcpd_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_uoi_results)) 

os.makedirs(organized_data_path, exist_ok=True) 

uoi_xcpd_results = unpack_and_save_results(xcpd_uoi_results, atlas_spec, model_id, organized_data_path)
# -

# ## PNC 

fc_bids = config['Paths']['pnc']
derivatives = op.join(fc_bids, 'derivatives')

# +
proc_type = 'xcpd' 
model_id = 'lassoBIC_task'
pnc_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(pnc_lasso_results))

organized_data_path = op.join(pkl_path, proc_type)
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(pnc_lasso_results, atlas_spec, model_id, organized_data_path)
# +
model_id = 'correlation_task'
pnc_corr_results = get_files(derivatives, proc_type, model_id)
print(len(pnc_corr_results))
        
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(pnc_corr_results, atlas_spec, model_id, organized_data_path)
# +
model_id = 'partial_correlation_task'
pnc_pcorr_results = get_files(derivatives, proc_type, model_id)
print(len(pnc_corr_results))

os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(pnc_pcorr_results, atlas_spec, model_id, organized_data_path)
# +
model_id = 'ridgeCV_task'
pnc_ridge_results = get_files(derivatives, proc_type, model_id)
print(len(pnc_ridge_results))

os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(pnc_ridge_results, atlas_spec, model_id, organized_data_path)

# +
model_id = 'uoiLasso_task'
pnc_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(pnc_uoi_results))
        
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(pnc_uoi_results, atlas_spec, model_id, organized_data_path)
# -


