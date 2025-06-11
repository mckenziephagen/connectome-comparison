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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as op
import pickle
import random

# +
results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'
date_string='2023-11-07'
op.join
with open(op.join(results_path, f'{date_string}_lasso_dict.pkl'), 'rb') as l:
        lasso_dict = pickle.load(l)
        
with open(op.join(results_path, f'{date_string}_uoi_dict.pkl'), 'rb') as u:
        uoi_dict = pickle.load(u)
        
with open(op.join(results_path, f'{date_string}_pearson_dict.pkl'), 'rb') as f:
        pearson_dict = pickle.load(f)


# -

#the data I'm currently working with is done by run, not session
#this won't be necessary for future analyses
def average_runs(result_dict):
    ses_dict = {} 
    for sub_id in result_dict.keys(): 
        if sub_id not in ses_dict.keys(): 
            ses_dict[sub_id] = {} 
        try: 
            ses_dict[sub_id]['ses-1'] = np.median(np.array([result_dict[sub_id]['ses-1_run-1'].ravel(),
                                                        result_dict[sub_id]['ses-1_run-2'].ravel()]), axis=0)
        except KeyError: 
            pass
        
        try:
            ses_dict[sub_id]['ses-2'] = np.median(np.array([result_dict[sub_id]['ses-2_run-1'].ravel(),
                                                        result_dict[sub_id]['ses-2_run-2'].ravel()]), axis=0)
        except KeyError:
            pass

    return ses_dict


def fingerprint(target_dict, database_dict): 
    corr_dict = {}
    for target_sub in target_dict.keys(): 
        if target_sub not in corr_dict.keys(): 
            corr_dict[target_sub] = {}
    
        for db_sub in database_dict.keys():
         #   print(db_sub)
            try: 
                corr_dict[target_sub][db_sub] = (np.corrcoef(target_dict[target_sub]['ses-1'],
                                                          database_dict[db_sub]['ses-2'])[0,1])
            except KeyError: 
                pass
    
    fp_list = []
    for key, value in corr_dict.items():
        # print("target: ", key)
        
        # print("match: ", max(corr_dict[key],key=corr_dict[key].get))
        fp_list.append(key.split('_')[0] in (max(corr_dict[key],key=corr_dict[key].get)))
    acc = sum(fp_list) / len(corr_dict)
    # print(acc)
    return corr_dict, fp_list, acc


# +
#one way to implement bootsrapping
ses_dict = average_runs(lasso_dict) 
sub_list = list(ses_dict.keys())
_, fp_list, _ = fingerprint(ses_dict, ses_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / 98) 
    
acc_list.sort()
print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
ses_dict = average_runs(pearson_dict) 
sub_list = list(ses_dict.keys())
_, fp_list, _ = fingerprint(ses_dict, ses_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()
print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
ses_dict = average_runs(uoi_dict) 
sub_list = list(ses_dict.keys())
_, fp_list, _ = fingerprint(ses_dict, ses_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()
print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 
# -

acc_list = [] 
for i in range(1000): 
    boot_dict = {}
    
    for ii in range(len(sub_list)): 
        sub_id = random.choice(sub_list)
        boot_dict[f'{sub_id}_{random.randint(1, 10000)}'] = ses_dict[sub_id]
    
    og_list = []
    _ = [og_list.append(i.split('_')[0]) for i in list(boot_dict.keys())]
    # print(len(set(og_list))) 
    
    corr_dict, _, _ = fingerprint(boot_dict, ses_dict) 
    fp_list = []
    
    for key, value in corr_dict.items(): 
        # print("target: ", key)
        # print(key.split('_')[0])
        # print("match: ", max(corr_dict[key],key=corr_dict[key].get))
        fp_list.append( key.split('_')[0] in (max(corr_dict[key],key=corr_dict[key].get)))
    
    acc_list.append(np.mean(fp_list))

acc_list.sort() 

acc_list[50]

acc_list[-50]

np.mean(acc_list) 

for key, value in corr_dict.items(): 
    #print(key, ',', value) 
    
    for sub_id, corr in corr_dict[key].items(): 
        if corr == max(corr_dict[key].values()): 
            print(sub_id in key) 
            break #stop after found

max(corr_dict['sub-129533_7828'].values()) 


