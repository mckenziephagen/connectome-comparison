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


def bids_entities(file): 
    base = op.basename(file) 
    sub_id = re.search(r'(?<=sub-)\d+', base).group()
    task_id = re.search(r'(?<=task-)[^_]+', base).group()
    ses_id = re.search(r'(?<=ses-)[^_]+', base).group()

    return sub_id, task_id, ses_id


proc_type = 'minProc'

model_id = 'connectivity_blocks'

organized_data_path = f'/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results/{proc_type}'

# +
fc_bids = '/pscratch/sd/m/mphagen/hcp-functional-connectivity/'
derivatives = op.join(fc_bids, 'derivatives') 
conn_path = op.join(derivatives, 'connectivity-matrices', proc_type, '*') 

if model_id == 'lasso-bic-blocks': 
    model_id = 'lassoBic'
    

# +
fc_bids = '/pscratch/sd/m/mphagen/hcp-functional-connectivity/'
derivatives = op.join(fc_bids, 'derivatives') 
conn_path = op.join(derivatives, 'connectivity-matrices', proc_type, '*') 

if model_id == 'lasso-bic-blocks': 
    model_id = 'lassoBic'
    
results = glob(op.join(conn_path, 'sub-*', '*.pkl'))

#ask chatgpt for the regex for this
atlas_spec = 'fsLR_seg-4S156Parcels_den-91k'

lasso_dict = {} 
for file in lasso_results: 
    sub_id, task_id, ses_id = bids_entities(file)
    print(sub_id, ses_id) 

    file_string = op.join(organized_data_path, 
                          f'sub-{sub_id}_ses-{ses_id}_task-{task_id}_space-{atlas_spec}_model-{model_id}_stat-median_relmat.dense.tsv')

    #if not op.exists(file_string): 
    with open(file, 'rb') as f:
         mat = pickle.load(f) 
    #take median of all folds
    conn_mat = np.median([mat[key]['fc_matrix'] for key in mat.keys()], axis=0)
    pd.DataFrame(conn_mat).to_csv(file_string, sep='\t')
    
    if sub_id not in lasso_dict.keys(): 
        lasso_dict[f'sub-{sub_id}'] = {} 
    lasso_dict[f'sub-{sub_id}'][f'sub-{ses_id}'] = conn_mat

with open(op.join(organized_data_path, 
                  f'{str(datetime.date.today())}_task-{task_id}_ses-{ses_id}_space-{atlas_spec}_model-{model_id}_stat-median_relmat.pkl'), 'wb') as f:
    pickle.dump(lasso_dict, f)
# -

lasso_dict.keys() 

lasso_dict['sub-174841']

for ii in lasso_dict.keys(): 
    print(lasso_dict[ii].keys())

pearson_dict = {} 
for file in pearson_results: 
    with open(file, 'rb') as f:
         mat = pickle.load(f) 
    conn_mat = mat['fc_matrix']
    sub_id = file.split('/')[-1].split('_')[0]
    ses_id = file.split('/')[-1].split('_')[-2]
    task_id = pearson_results[0].split('/')[-1].split('_')[2]
    file_string = op.join(org_data_path, 'pearson_connectome', 
                          '_'.join([sub_id, ses_id, task_id, 
                                    'meas-pearson', 'desc-Schaefer100'])) 
    pd.DataFrame(conn_mat).to_csv(f'{file_string}_relmat.dense.tsv', sep='\t')

    
    if sub_id not in pearson_dict.keys(): 
        pearson_dict[sub_id] = {} 
    #average five folds together
    pearson_dict[sub_id].update({ses_id: conn_mat})
    with open(op.join(org_dat_path, 
                      f'{str(datetime.date.today())}_pearson_dict.pkl'), 'wb') as f:
        pickle.dump(pearson_dict, f)

pearson_dict = {} 
for idx, file in enumerate(pearson_results_files): 
    with open(file, 'rb') as f:
         mat = pickle.load(f)    
    ses_id = '_'.join(file.split('/')[-1].split('_')[-3:-1])
    sub_id = file.split('/')[-2]

    if sub_id not in pearson_dict.keys(): 
        pearson_dict[sub_id] = {} 
    #average five folds together
    pearson_dict[sub_id].update({ses_id: mat})
with open(op.join('results', f'{str(datetime.date.today())}_pearson_dict.pkl'), 'wb') as f:
            pickle.dump(pearson_dict, f)


# +
uoi_dict = {} 
for idx, file in enumerate(pyuoi_result_files): 
    with open(file, 'rb') as f:
         mat = pickle.load(f)    
    ses_id = '_'.join(file.split('/')[-1].split('_')[-4:-2])
    sub_id = file.split('/')[-2]

    if sub_id not in uoi_dict.keys(): 
        uoi_dict[sub_id] = {} 
    #average five folds together
    uoi_dict[sub_id].update({ses_id: np.median(np.array([*mat.values()]), axis=0)})
    
with open(op.join('results', f'{str(datetime.date.today())}_uoi_dict.pkl'), 'wb') as f:
        pickle.dump(uoi_dict, f)
# -

pearson_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub'] )
for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': pearson_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner, 
                            'pos': list(range(1,10001,1))})
        pearson_icc_df = pd.concat([pearson_icc_df, temp_df])
pearson_icc_df.to_csv('pearson_icc_df.csv')

lasso_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub', 'pos'] )
for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': lasso_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner, 
                              'pos': list(range(1,10001,1))})
        lasso_icc_df = pd.concat([lasso_icc_df, temp_df])
lasso_icc_df.to_csv('lasso_icc_df.csv')

# +
uoi_icc_df = pd.DataFrame(columns=['values', 'ses', 'sub'] )
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        temp_df = pd.DataFrame(data = {'values': uoi_dict[outer][inner].ravel(), 
                               'sub':  outer, 
                              'ses': inner})
        uoi_icc_df = pd.concat([uoi_icc_df, temp_df])
        
uoi_icc_df.to_csv('uoi_icc_df.csv')
# -

r2_mean_list = []
for i in pyuoi_r2_files: 
    with open(i, 'rb') as f:
         mat = pickle.load(f)  
    r2_df = pd.json_normalize(mat).filter(like='test')
    
    r2_mean_list.append(np.mean(r2_df.explode(list(r2_df.columns))))

#I shoudl investigate those lower ones and cut them
plt.hist(r2_mean_list)
plt.title('Average Union of Intersections Model Accuracy Per Scan') 

plt.hist(np.mean(r2_df.explode(list(r2_df.columns)), axis=1))


