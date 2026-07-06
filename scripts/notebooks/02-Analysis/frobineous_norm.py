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

import pickle 
import os.path as op
from glob import glob 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# +
results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'

date_string = '2026-05-22'


with open(glob(op.join(results_path, 'xcpd', f'{date_string}*lassoBIC*pkl'))[0],   'rb') as f:
    xcpd_lasso_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'{date_string}*lassoBIC*pkl'))[0],   'rb') as f:
    fix_lasso_dict = pickle.load(f)

with open(glob(op.join(results_path, 'xcpd', f'{date_string}*uoiLasso*pkl'))[0], 'rb') as f:
    xcpd_uoi_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'{date_string}*uoiLasso*pkl'))[0], 'rb') as f:
    fix_uoi_dict = pickle.load(f)

with open(glob(op.join(results_path,'MSMAll_FIX', f'{date_string}*-corr*pkl'))[0], 'rb') as l:
    fix_pearson_dict = pickle.load(l)

with open(glob(op.join(results_path,'xcpd', f'{date_string}*corr*pkl'))[0], 'rb') as l:
    xcpd_pearson_dict = pickle.load(l)

with open(glob(op.join(results_path,'MSMAll_FIX', f'{date_string}*partial*pkl'))[0], 'rb') as l:
    fix_pcorr_dict = pickle.load(l)

with open(glob(op.join(results_path,'MSMAll_FIX', f'2026-05-23*ridge*pkl'))[0], 'rb') as l:
    fix_ridge_dict = pickle.load(l)
# -

xcpd_lasso_norm_arr = np.empty((0,0))
for sub in xcpd_lasso_dict.keys(): 
    A = xcpd_lasso_dict[sub]['ses-full']
    xcpd_lasso_norm_arr = np.append(xcpd_lasso_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

xcpd_uoi_norm_arr = np.empty((0,0))
for sub in xcpd_uoi_dict.keys(): 
    A = xcpd_uoi_dict[sub]['ses-full']
    xcpd_uoi_norm_arr = np.append(xcpd_uoi_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

xcpd_pearson_norm_arr = np.empty((0,0))
for sub in xcpd_pearson_dict.keys(): 
    A = xcpd_pearson_dict[sub]['ses-full']
    xcpd_pearson_norm_arr = np.append(xcpd_pearson_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

fix_lasso_norm_arr = np.empty((0,0))
for sub in fix_lasso_dict.keys(): 
    A = fix_lasso_dict[sub]['ses-full']
    fix_lasso_norm_arr = np.append(fix_lasso_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

fix_uoi_norm_arr = np.empty((0,0))
for sub in fix_uoi_dict.keys(): 
    A = fix_uoi_dict[sub]['ses-full']
    fix_uoi_norm_arr = np.append(fix_uoi_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

fix_pearson_norm_arr = np.empty((0,0))
for sub in fix_pearson_dict.keys(): 
    A = fix_pearson_dict[sub]['ses-full']
    fix_pearson_norm_arr = np.append(fix_pearson_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

fix_pcorr_norm_arr = np.empty((0,0))
for sub in fix_pcorr_dict.keys(): 
    A = fix_pcorr_dict[sub]['ses-full']
    fix_pcorr_norm_arr = np.append(fix_pcorr_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

fix_ridge_norm_arr = np.empty((0,0))
for sub in fix_ridge_dict.keys(): 
    A = fix_ridge_dict[sub]['ses-full']
    fix_ridge_norm_arr = np.append(fix_ridge_norm_arr, np.linalg.norm(A@A.T - A.T@A) ) 

import seaborn as sns

plot_df = pd.concat([
    pd.DataFrame({'norm': fix_uoi_norm_arr.ravel(), 'proc_type': 'fix', 'model': 'uoi'}),
    pd.DataFrame({'norm': fix_lasso_norm_arr.ravel(), 'proc_type': 'fix', 'model': 'lasso'}), 
     pd.DataFrame({'norm': fix_ridge_norm_arr.ravel(), 'proc_type': 'fix', 'model': 'ridge'}),
    pd.DataFrame({'norm': fix_pcorr_norm_arr.ravel(), 'proc_type': 'fix', 'model': 'pcorr'}) ,
    pd.DataFrame({'norm': fix_pearson_norm_arr.ravel(), 'proc_type': 'fix', 'model': 'pearson'})], 
    axis=0).dropna()


plot_df.to_csv('results/forb_norm_2026-07-06.csv') 

plot_df

plot_df['norm'] = np.round(plot_df['norm'], 3) 

ax = sns.violinplot(data=plot_df, x="model", y="norm") 
ax.invert_xaxis()
plt.savefig('frobineous_norm.png') 

# +
bins = np.linspace(0, 5, 15)


plt.hist(msmall_lasso_norm_arr, bins=bins, label = 'lasso')
plt.hist(msmall_uoi_norm_arr, bins=bins, alpha=.5, label = 'uoi')
plt.legend() 
# -

len(xcpd_norm_arr) 

len(msmall_norm_arr) 










