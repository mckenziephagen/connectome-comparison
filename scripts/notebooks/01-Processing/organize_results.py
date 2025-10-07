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

# +
import os
import os.path as op
from glob import glob
import pickle
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re

atlas_spec = 'fsLR_seg-4S156Parcels_den-91k'


# -

def bids_entities(file): 
    base = op.basename(file) 
    sub_id = re.search(r'(?<=sub-)\d+', base).group()
    task_id = re.search(r'(?<=task-)[^_]+', base).group()
    ses_id = re.search(r'(?<=ses-)[^_]+', base).group()

    return sub_id, task_id, ses_id


def unpack_and_save_results(result_files, atlas_spec, model_id, organized_data_path): 

    #ask chatgpt for the regex for this
    
    results_dict = {} 
    for file in result_files: 
        sub_id, task_id, ses_id = bids_entities(file)    
        file_string = op.join(organized_data_path, 
                              f'sub-{sub_id}_ses-{ses_id}_task-{task_id}_space-{atlas_spec}_model-{model_id}_stat-median_relmat.dense.tsv')
        #if not op.exists(file_string): 
        
        with open(file, 'rb') as f:
             mat = pickle.load(f) 
        #take median of all folds
        if model_id != 'correlation': 
            conn_mat = np.median([mat[key]['fc_matrix'] for key in mat.keys()], axis=0)
            std_mat = np.std([mat[key]['fc_matrix'] for key in mat.keys()], axis=0)

        elif model_id == 'correlation': 
            conn_mat = mat['fc_matrix']

        pd.DataFrame(conn_mat).to_csv(file_string, sep='\t')
        pd.DataFrame(std_mat).to_csv(file_string, sep='\t')

        if f'sub-{sub_id}' not in results_dict.keys(): 
            # print(f"adding sub_id, {sub_id}, {ses_id}") 
            results_dict[f'sub-{sub_id}'] = {} 
        results_dict[f'sub-{sub_id}'][f'ses-{ses_id}'] = conn_mat
    
    with open(op.join(organized_data_path, 
                      f'{str(datetime.date.today())}_task-{task_id}_space-{atlas_spec}_model-{model_id}_stat-median_relmat.pkl'), 'wb') as f:
        pickle.dump(results_dict, f) 


def get_files(derivatives, proc_type, model_id): 
    conn_path = op.join(derivatives, 'connectivity-matrices', proc_type, '*') 
    result_files = glob(op.join(conn_path, 'sub-*', f'*{model_id}*.pkl'))
    return result_files


fc_bids = '/pscratch/sd/m/mphagen/hcp-functional-connectivity/'
derivatives = op.join(fc_bids, 'derivatives') 

# +
proc_type = 'xcpd' 
model_id = 'lassoBIC'
xcpd_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(xcpd_lasso_results)) 
xcpd_lasso_results.sort() 
organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(xcpd_lasso_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll' 
model_id = 'lassoBIC'
minProc_lasso_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_lasso_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_lasso_results, atlas_spec, model_id, organized_data_path)


# +
proc_type = 'MSMAll' 
model_id = 'uoiLasso'
minProc_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_uoi_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_uoi_results, atlas_spec, model_id, organized_data_path)


# +
proc_type = 'xcpd' 
model_id = 'uoiLasso'
minProc_uoi_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_uoi_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_uoi_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'MSMAll' 
model_id = 'correlation'
minProc_corr_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_corr_results, atlas_spec, model_id, organized_data_path)

# +
proc_type = 'minProc' 
model_id = 'correlation'
minProc_corr_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_corr_results, atlas_spec, model_id, organized_data_path)


# +
proc_type = 'xcpd' 
model_id = 'correlation'
minProc_corr_results = get_files(derivatives, proc_type, model_id)
print(len(minProc_corr_results)) 

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'
os.makedirs(organized_data_path, exist_ok=True) 

unpack_and_save_results(minProc_corr_results, atlas_spec, model_id, organized_data_path)
# -

r2_mean_list = []
for i in pyuoi_r2_files: 
    with open(i, 'rb') as f:
         mat = pickle.load(f)  
    r2_df = pd.json_normalize(mat).filter(like='test')
    
    r2_mean_list.append(np.mean(r2_df.explode(list(r2_df.columns))))


