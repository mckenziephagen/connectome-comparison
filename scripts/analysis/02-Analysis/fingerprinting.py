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
from fc_utils import read_pickles

# %%
import numpy as np
import random
import pandas as pd

random.seed(42)

# %%
date_str = '2026-10-04'
models = ['lasso', 'pearson', 'uoi', 'ridge', 'pcorr']


# %%
def load_fc_dicts(proc_type, cv_type='ses'): 
    corr_dict, uoi_dict, lasso_dict, pcorr_dict, ridge_dict = read_pickles(date_str, proc_type, pkl_dir, cv_type=cv_type)
    fc_dicts = {'lasso': lasso_dict, 'pearson': corr_dict, 'uoi': uoi_dict, 
                'ridge': ridge_dict, 'pcorr': pcorr_dict}
    for fc_dict in fc_dicts.values(): 
        for sub in fc_dict.keys(): 
            fc_dict[sub].pop('ses-full', None)
    return fc_dicts


# %%
def fingerprint(target_dict, database_dict, target_ses='ses-1', db_ses='ses-2'): 
    corr_dict = {}
    for target_sub in target_dict.keys(): 
        if target_ses not in target_dict[target_sub]: 
            continue
        corr_dict[target_sub] = {}
    
        for db_sub in database_dict.keys():
            try: 
                corr = np.corrcoef(target_dict[target_sub][target_ses].ravel(),
                                    database_dict[db_sub][db_ses].ravel())[0,1]
                corr_dict[target_sub][db_sub] = corr
            except KeyError: 
                pass
    
    fp_dict = {}
    for sub in corr_dict.keys():
        ## max(corr_dict ...) returns the keys associated with the highest key.value(). 
        fp_dict[sub] = sub.split('_')[0] == max(corr_dict[sub],key=corr_dict[sub].get).split('_')[0]
    acc = sum(fp_dict.values()) / len(fp_dict)
    return corr_dict, fp_dict, acc


# %%
def bootstrap_fingerprint(fc_dict, n_boot=1000): 
    ## identify in both directions (ses-1 -> ses-2 and ses-2 -> ses-1)
    _, fp_12, acc_12 = fingerprint(fc_dict, fc_dict, target_ses='ses-1', db_ses='ses-2')
    _, fp_21, acc_21 = fingerprint(fc_dict, fc_dict, target_ses='ses-2', db_ses='ses-1')

    ## average the two directions within subject, then resample subjects 
    subs = [sub for sub in fp_12.keys() if sub in fp_21]
    sub_scores = [(fp_12[sub] + fp_21[sub]) / 2 for sub in subs]

    acc_list = [] 
    for i in range(n_boot): 
        acc_list.append(np.mean(random.choices(sub_scores, k=len(sub_scores)))) 
    acc_list.sort()

    result = {'acc_12': acc_12, 
              'acc_21': acc_21, 
              'avg': np.mean(acc_list), 
              'lower': np.percentile(acc_list, 2.5), 
              'upper': np.percentile(acc_list, 97.5), 
              'boot_vals': acc_list}

    print("1->2: ", acc_12, " 2->1: ", acc_21)
    print("average: ", result['avg'])
    print("CI: ", result['lower'], result['upper']) 
    return result


# %%
def run_fingerprinting(fc_dicts, out_file): 
    fp_dict = {}
    for model in models: 
        print(model)
        fp_dict[model] = bootstrap_fingerprint(fc_dicts[model])

    boot_df = pd.DataFrame({model: fp_dict[model]['boot_vals'] for model in models})
    boot_df.to_csv(out_file) 
    return fp_dict


# %% [markdown]
# ### HCP 

# %%
fix_dicts = load_fc_dicts('MSMAll_FIX')
fix_dict = run_fingerprinting(fix_dicts, 
                              op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_stat-fingerprinting.csv'))

# %%
xcpd_dicts = load_fc_dicts('xcpd')
xcpd_dict = run_fingerprinting(xcpd_dicts, 
                               op.join(results_dir, 'XCPD', 'proc-XCPD_stat-fingerprinting.csv'))

# %% [markdown]
# ### PNC 

# %%
pnc_dicts = load_fc_dicts('pnsc_xcpd', cv_type='task')
pnc_dict = run_fingerprinting(pnc_dicts, 
                              op.join(results_dir, 'PNC_XCPD', 'proc-PNC_stat-fingerprinting.csv'))
