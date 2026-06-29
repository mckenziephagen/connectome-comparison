# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: fc_311
#     language: python
#     name: fc_311
# ---

import os
import os.path as op
from glob import glob
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
atlas_spec = 'fsLR_seg-4S156Parcels_den-91k'

test_subjects = pd.read_csv('/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/test_subjects.txt',
                           header=None)
# test_subjects = list(test_subjects) 
test_subjects = list(test_subjects[0]) 


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
        
        with open(file, 'rb') as f:
             mat = pickle.load(f) 

        if model_id in ['correlation_ses', 'partial_correlation_ses']:
            print('model id == corr') 
            conn_mat = mat['full_fc_matrix']
            if f'sub-{sub_id}' not in results_dict.keys(): 
            # print(f"adding sub_id, {sub_id}, {ses_id}") 
                results_dict[f'sub-{sub_id}'] = {} 
                results_dict[f'sub-{sub_id}'][f'ses-1'] = mat['fold_0']
                results_dict[f'sub-{sub_id}'][f'ses-2'] = mat['fold_1']
                results_dict[f'sub-{sub_id}'][f'ses-full'] = conn_mat
    
        elif model_id not in ['correlation_ses', 'partial_correlation_ses']:
            print('model id != corr') 
            try: 
                conn_mat = mat['full_fc_matrix']
            except KeyError: 
                print('key error') 
                conn_mat = np.nan
            if f'sub-{sub_id}' not in results_dict.keys(): 
            # print(f"adding sub_id, {sub_id}, {ses_id}") 
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


# ## HCP 

fc_bids = '/pscratch/sd/m/mphagen/human-connectome-project-openaccess/HCP1200'
derivatives = op.join(fc_bids, 'derivatives')

# +
# fc_bids = '/pscratch/sd/m/mphagen/hcp-functional-connectivity/'
# derivatives = op.join(fc_bids, 'derivatives') 
# -

# #### MSMAll FIX

# +
proc_type = 'MSMAll_FIX' 
model_id = 'correlation_ses'
fix_corr_results = get_files(derivatives, proc_type, model_id)
print(len(fix_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_corr_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll_FIX' 
model_id = 'partial_correlation_ses'
msmall_corr_results = get_files(derivatives, proc_type, model_id)
print(len(msmall_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(msmall_corr_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll_FIX' 
model_id = 'ridgeCV_ses'

fc_bids = '/pscratch/sd/m/mphagen/human-connectome-project-openaccess/HCP1200'
derivatives = op.join(fc_bids, 'derivatives') 

fix_ridge_results = get_files(derivatives, proc_type, model_id)
print(len(fix_ridge_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

results_dict = unpack_and_save_results(fix_ridge_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll_FIX' 
model_id = 'lassoBIC_ses'
fix_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(fix_lasso_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(fix_lasso_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll_FIX' 
model_id = 'uoiLasso_ses'
fix_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(fix_uoi_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(fix_uoi_results, atlas_spec, model_id, organized_data_path)
# -

# #### XCPD 

# +
proc_type = 'xcpd' 
model_id = 'correlation_ses'
xcpd_corr_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_corr_results, atlas_spec, model_id, organized_data_path)
# +
proc_type = 'xcpd' 
model_id = 'partial_correlation_ses'
xcpd_pcorr_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_pcorr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_pcorr_results, atlas_spec, model_id, organized_data_path)
# +
proc_type = 'xcpd' 
model_id = 'lassoBIC_ses'
result_files = get_files(derivatives, proc_type, model_id)
xcpd_lasso_results = []
for ii in result_files: 
    sub = int(ii.split('/')[-2].split('-')[-1])
    if sub in test_subjects: 
        xcpd_lasso_results.append(ii) 
        
organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_lasso_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'xcpd' 
model_id = 'ridgeCV_ses'
xcpd_ridge_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_ridge_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_ridge_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'xcpd' 
model_id = 'uoiLasso_ses'
xcpd_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_uoi_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_uoi_results, atlas_spec, model_id, organized_data_path)
# -

# ### MSMAll 

# +
proc_type = 'MSMAll' 
model_id = 'correlation_ses'
msmall_corr_results = get_files(derivatives, proc_type, model_id)
print(len(msmall_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(msmall_corr_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll' 
model_id = 'lassoBIC_ses'
msmall_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(msmall_lasso_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(msmall_lasso_results, atlas_spec, model_id, organized_data_path)


# +
proc_type = 'MSMAll' 
model_id = 'uoiLasso_ses'
msmall_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(msmall_uoi_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(msmall_uoi_results, atlas_spec, model_id, organized_data_path)
# -


# ## PNC 

fc_bids = '/pscratch/sd/m/mphagen/PNC'
derivatives = op.join(fc_bids, 'derivatives')

# +
proc_type = 'xcpd' 
model_id = 'lassoBIC_task'
result_files = get_files(derivatives, proc_type, model_id)
pnc_xcpd_results = []
for ii in result_files: 
    sub = int(ii.split('/')[-2].split('-')[-1])
    # if sub in test_subjects: 
    pnc_xcpd_results.append(ii) 
        
organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/pnc_{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(pnc_xcpd_results, atlas_spec, model_id, organized_data_path)
# -


