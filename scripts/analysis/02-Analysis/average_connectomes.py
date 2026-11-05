# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# Visualize and compare sparsities for different models. 
#
# **Inputs:**

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

import sys
sys.path.append(op.join(base_dir, 'scripts', 'analysis'))
from fc_utils import read_pickles, make_mat


# %%
import numpy as np

# %%
date_str = '2026-10-04'

fix_pearson_dict, fix_uoi_dict, fix_lasso_dict, fix_pcorr_dict, fix_ridge_dict = read_pickles(date_str, 'MSMAll_FIX', pkl_dir)

xcpd_pearson_dict, xcpd_uoi_dict, xcpd_lasso_dict, xcpd_pcorr_dict, xcpd_ridge_dict = read_pickles(date_str, 'xcpd', pkl_dir)


# %% [markdown]
# ### PNC 

# %%
# for sub in pnc_lasso_dict.keys(): 
#     pnc_lasso_dict[sub]['ses-median'] = np.median([pnc_lasso_dict[sub]['ses-1'], pnc_lasso_dict[sub]['ses-2']], axis=0) 
# pnc_lasso_dict = delete_ses(pnc_lasso_dict)


# %% [markdown]
# ### HCP

# %%
fix_lasso_mat = make_mat(fix_lasso_dict)

fix_pearson_mat = make_mat(fix_pearson_dict)

fix_ridge_mat = make_mat(fix_ridge_dict)

fix_pcorr_mat = make_mat(fix_pcorr_dict)

fix_uoi_mat = make_mat(fix_uoi_dict)

# %%
xcpd_lasso_mat = make_mat(xcpd_lasso_dict)

xcpd_pearson_mat = make_mat(xcpd_pearson_dict)

xcpd_ridge_mat = make_mat(xcpd_ridge_dict)

xcpd_pcorr_mat = make_mat(xcpd_pcorr_dict)

xcpd_uoi_mat = make_mat(xcpd_uoi_dict)


# %%
xcpd_lasso_mean = np.mean(xcpd_lasso_mat, axis=2)
fix_lasso_mean = np.mean(fix_lasso_mat, axis=2)

xcpd_ridge_mean = np.mean(xcpd_ridge_mat, axis=2)
fix_ridge_mean = np.mean(fix_ridge_mat, axis=2)

xcpd_uoi_mean = np.mean(xcpd_uoi_mat, axis=2)
fix_uoi_mean = np.mean(fix_uoi_mat, axis=2)


fix_pcorr_mean = np.mean(fix_pcorr_mat, axis=2)
xcpd_pcorr_mean = np.mean(xcpd_pcorr_mat, axis=2)

xcpd_pearson_mean = np.mean(xcpd_pearson_mat, axis=2)
fix_pearson_mean = np.mean(fix_pearson_mat, axis=2)


# %%
np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-lassoBIC_stat-mean_connectome.txt'), fix_lasso_mean)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-lassoBIC_stat-mean_connectome.txt'), xcpd_lasso_mean)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_stat-mean_connectome.txt'), fix_ridge_mean)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-ridgeCV_stat-mean_connectome.txt'), xcpd_ridge_mean) 

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_stat-mean_connectome.txt'), fix_uoi_mean)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-uoiLasso_stat-mean_connectome.txt'), xcpd_uoi_mean)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-Pearson_stat-mean_connectome.txt'), fix_pearson_mean)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-Pearson_stat-mean_connectome.txt'), xcpd_pearson_mean)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-partial-correlation_stat-mean_connectome.txt'), fix_pcorr_mean)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-partial-correlation_stat-mean_connectome.txt'), xcpd_pcorr_mean)

# %%
xcpd_lasso_median = np.median(xcpd_lasso_mat, axis=2)
fix_lasso_median = np.median(fix_lasso_mat, axis=2)

xcpd_uoi_median = np.median(xcpd_uoi_mat, axis=2)
fix_uoi_median = np.median(fix_uoi_mat, axis=2)

xcpd_ridge_median = np.median(xcpd_ridge_mat, axis=2)
fix_ridge_median = np.median(fix_ridge_mat, axis=2)

xcpd_pearson_median = np.median(xcpd_pearson_mat, axis=2)
fix_pearson_median = np.median(fix_pearson_mat, axis=2)

xcpd_pcorr_median = np.median(xcpd_pcorr_mat, axis=2)
fix_pcorr_median = np.median(fix_pcorr_mat, axis=2)

# %%
np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-lassoBIC_stat-median_connectome.txt'), fix_lasso_median)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-lassoBIC_stat-median_connectome.txt'), xcpd_lasso_median)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_stat-median_connectome.txt'), fix_ridge_median)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-ridgeCV_stat-median_connectome.txt'), xcpd_ridge_median)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_stat-median_connectome.txt'), fix_uoi_median)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-uoiLasso_stat-median_connectome.txt'), xcpd_uoi_median)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-Pearson_stat-median_connectome.txt'), fix_pearson_median)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-Pearson_stat-median_connectome.txt'), xcpd_pearson_median)

np.savetxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-partial-correlation_stat-median_connectome.txt'), fix_pcorr_median)
np.savetxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-partial-correlation_stat-median_connectome.txt'), xcpd_pcorr_median)

# %% [markdown]
# Save csvs for Talita. 

# %%
participant_dir = op.join(results_dir, 'MSMAll-FIX', 'participant_connectomes')
for model_dir in ['lasso', 'pearson', 'ridge', 'partial_corr', 'uoi']: 
    os.makedirs(op.join(participant_dir, model_dir), exist_ok=True)

# %%
for sub in fix_lasso_dict.keys(): 
    for ses in ['ses-1', 'ses-2']:
        np.savetxt(op.join(participant_dir, 'lasso', f'{sub}_{ses}_proc-FIX_model-lassoBIC_connectome.csv'), 
                   fix_lasso_dict[sub][ses],delimiter=',' )

# %%
for sub in fix_pearson_dict.keys(): 
    for ses in ['ses-1', 'ses-2']:
        np.savetxt(op.join(participant_dir, 'pearson', f'{sub}_{ses}_proc-FIX_model-Pearson_connectome.csv'), 
                   fix_pearson_dict[sub][ses],delimiter=',' )

# %%
for sub in fix_ridge_dict.keys(): 
    for ses in ['ses-1', 'ses-2']:
        np.savetxt(op.join(participant_dir, 'ridge', f'{sub}_{ses}_proc-FIX_model-ridgeCV_connectome.csv'), 
                   fix_ridge_dict[sub][ses],delimiter=',' )

# %%
for sub in fix_pcorr_dict.keys(): 
    for ses in ['ses-1', 'ses-2']:
        np.savetxt(op.join(participant_dir, 'partial_corr', f'{sub}_{ses}_proc-FIX_model-partial-correlation_connectome.csv'), 
                   fix_pcorr_dict[sub][ses],delimiter=',' )

# %%
for sub in fix_uoi_dict.keys(): 
    for ses in ['ses-1', 'ses-2']:
        np.savetxt(op.join(participant_dir, 'uoi', f'{sub}_{ses}_proc-FIX_model-uoiLasso_connectome.csv'), 
                   fix_uoi_dict[sub][ses],delimiter=',' )
