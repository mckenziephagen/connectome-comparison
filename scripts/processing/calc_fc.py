# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: fc_311
#     language: python
#     name: fc_311
# ---

# %% [markdown]
# This code expects to find parcellated timeseries data in `f'{deriv_dir}/timeseries/{proc_type}/sub-{sub_id}/sub-{sub_id}*ses-1*timeseries.ptseries.nii'`. 
#
#

# %%
import sys
import os
import numpy.matlib
from numpy.matlib import repmat
import nibabel as nib
import json
from sklearn.exceptions import ConvergenceWarning
import warnings
from copy import deepcopy
import argparse

# %%
#work around until I install fc_comparison as an actual package
sys.path.append(os.path.dirname('/global/homes/m/mphagen/functional-connectivity/model-fc/src/model_fc'))

# %%
import time

import nilearn

from pyuoi.utils import log_likelihood_glm, AIC, BIC

import numpy as np
import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import sklearn 

import pickle

from model_fc.models import init_model, run_model

# %%
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--sub_id',default='sub-979984')
parser.add_argument('--ses_id', default='full')
parser.add_argument('--run_id', default=None)
parser.add_argument('--task_id', default='rest') 

parser.add_argument('--atlas_spec', default='fsLR_seg-4S156Parcels_den-91k')
parser.add_argument('--n_rois', default=100, type=int) #default for hcp; 
parser.add_argument('--n_trs', default=1200, type=int) #default for hcp;
parser.add_argument('--n_folds', default=5)
#must be larger than 6 for PNC
parser.add_argument('--model', default='correlation') 
parser.add_argument('--cv', default='ses') 
parser.add_argument('--cv_task', default=None) 

parser.add_argument('--proc_type', default='MSMAll_FIX') 

parser.add_argument('--profile', action='store_true', 
                   help="Run single fold for CPU/mem profiling") 

parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/human-connectome-project-openaccess/HCP1200') 
parser.add_argument('--max_iter', default=1000) 

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
    notebook = False
except KeyError: 
    args = parser.parse_args([])
    notebook = True
    results_path = 'notebook_results'

args.sub_id = str(args.sub_id).replace('sub-', '')
sub_id = args.sub_id
args.ses_id = str(args.ses_id).replace('ses-', '')
ses_id = args.ses_id
task_id = args.task_id

args.run_id = str(args.run_id).replace('run-', '')
run_id = args.run_id

atlas_spec = args.atlas_spec 
n_rois = args.n_rois
n_trs = args.n_trs
n_folds = args.n_folds
model_str = args.model
cv = args.cv
 
fc_data_path = args.fc_data_path
profile = args.profile
proc_type = args.proc_type

max_iter = args.max_iter
random_state = 1
print(args)
# -

args_dict = vars(args) 


# %%
args_dict = vars(args) 


# %%
def get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id): 
    if 'HCP' in fc_data_path: 
        if ses_id == 'full': 
            print('here')
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}*ses-*ptseries.nii'))
        elif run_id == None: 
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}*ses-{ses_id}*ptseries.nii'))
        
        elif run_id == '1' and ses_id != 'full':
            print(run_id)
            ts_file = False

        return ts_file


# %%
ts_files = get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id)

if proc_type == 'MSMAll_FIX': 
    ts_files[:] = [x for x in ts_files if 'NORMED' in x]

if notebook == False:
    results_path = op.join(fc_data_path, 
                    'derivatives',
                    'connectivity-matrices',
                    proc_type, 
                    f"{model_str.replace('-', '')}_{cv}",
                    f'sub-{sub_id}')
                       
os.makedirs(results_path, exist_ok=True)

assert len(ts_files) >= 1 

print(f"Found {len(ts_files)} rest scans for subject {sub_id}.") 

print(f"Saving results to {results_path}.")

# %%
model = init_model(model_str, max_iter, random_state)

# %%
model

# %%
print(model)


# %%
def set_warnings_filters():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

set_warnings_filters()


# %%
def split_kfold(cv, time_series, n_folds): 
    if cv == 'random': 
        
        kfolds = KFold(n_splits=n_folds,
                      shuffle=True)
        
        splits = kfolds.split(X=time_series)
        
    if cv == 'blocks':

        group =  repmat(np.arange(1, n_folds+1), 
                        int(time_series.shape[0]/n_folds), 1).T.ravel()
        
        kfold = GroupKFold(n_splits=n_folds)   
        splits = kfold.split(X=time_series, groups=group) 
        
    if cv == 'timeseries': 
        tscv = TimeSeriesSplit()
        splits = tscv(tscv.split(X=time_series))

    if cv == 'ses': 
        ses_len = int(len(time_series)/2)
        ses_idx = np.concat([repmat(0, 1, ses_len),
                             repmat(1, 1, ses_len)],axis=-1).ravel()
        ses = sklearn.model_selection.PredefinedSplit(ses_idx)
        splits = ses.split(time_series)

    if cv == 'run': 
        run_len = int(len(time_series)/4)
        run_idx = np.concat([repmat(0, 1, run_len), 
                             repmat(1, 1, run_len), 
                             repmat(2, 1, run_len), 
                             repmat(3, 1, run_len)
                            ],axis=-1).ravel()
        run = sklearn.model_selection.PredefinedSplit(run_idx)
        splits = run.split(time_series)

    if cv == 'task': 
        task_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        'xcpd',
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{cv_task}_space*ptseries.nii')) 
        task_scan =  nib.load(task_file).get_fdata(dtype=np.float32)
        time_series = time_series.append(task_scan)

        ses_len = int(len(time_series)/2)
        ses_idx = np.concat([repmat(0, 1, ses_len),
                             repmat(1, 1, ses_len)],axis=-1).ravel()
        ses = sklearn.model_selection.PredefinedSplit(ses_idx)
        splits = ses.split(time_series)
    

    return splits


# %%
def read_cifti(file): 
    time_series = nib.load(file).get_fdata(dtype=np.float32)
    if n_rois != time_series.shape[1]: 
        time_series = time_series[:, :n_rois]
    return time_series


# %%
if len(ts_files) > 1:
    time_series = np.append(read_cifti(ts_files[0]) , read_cifti(ts_files[1]) , axis = 0) 
else: 
    time_series = read_cifti(ts_files[0]) 
#create copy of time_series to norm and fit entire model on


# %%
time_series

# %%
results_dict = {}    
print(f"Calculating {model_str} FC for {sub_id} {ses_id}")

scaler = StandardScaler()
file = f"sub-{sub_id}_task-{task_id}_ses-{ses_id}_{atlas_spec}_model-{model_str.replace('-', '')}_cv-{cv}_results.pkl"


# %%
##fit full timeseries - this can be a func
ts = time_series
full_fc_mat = np.empty((n_rois, n_rois))
alpha_dict = {}
ts = scaler.fit_transform(ts)
if model_str in ['lassoBIC', "uoiLasso", "pearsonRegressor", "ridgeCV"]: 
    for target_idx in range(ts.shape[1]):
        y_ts = np.array(ts[:, target_idx])
        X_ts = np.delete(ts, target_idx, axis=1)
        model.fit(X_ts, y_ts)
        # alpha_dict[f'node_{target_idx}'] = deepcopy(model)
        full_fc_mat[target_idx, :] = np.insert(model.coef_, target_idx, 1)

elif model_str in ['correlation', "partial_correlation"]: 
    full_fc_mat = model.fit_transform(np.array(ts))[0]
    np.fill_diagonal(full_fc_mat, 1)

results_dict['full_fc_matrix'] = full_fc_mat

# %%
splits = split_kfold(cv, time_series, n_folds)

for fold_idx, (train_idx, test_idx) in enumerate( splits ): 
    if model_str in ["lassoCV", "uoiLasso", "enet", "lassoBIC", "ridgeCV", "pearsonRegressor"] :        
        print(model_str) 
   
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: range={train_idx}")
        print(f"  Test:  index={test_idx}")

        train_ts = time_series[train_idx, :]
        test_ts = time_series[test_idx, :]
        
        scaler = StandardScaler()
        train_ts = scaler.fit_transform(train_ts)
        test_ts = scaler.transform(test_ts)

        start_time = time.time()
        results_dict[f'fold_{fold_idx}'] = run_model(train_ts, test_ts, n_rois, model)       
        print(time.time() - start_time, ' seconds')     

        if profile == True: 
            break 

    elif model_str in ['correlation', 'tangent', 'partial_correlation']: 
        #we're not really using "train" and "test" 
        #but this seperates by session, and keeps the 
        #terminology clear
        train_ts = time_series[train_idx, :]
        # test_ts = time_series[test_idx, :]
    
        corr_mat_train = model.fit_transform(np.array(train_ts))[0]
        np.fill_diagonal(corr_mat_train, 1)
    
        # corr_mat_test = model.fit_transform(np.array(test_ts))[0]
        # np.fill_diagonal(corr_mat_test, 1)
        
        results_dict[f'fold_{fold_idx}'] = corr_mat_train
        # results_dict['fold_1'] = corr_mat_test
        
with open(op.join(results_path,file), 'wb') as f:
    pickle.dump(results_dict, f) 

config_file = file.replace('_results.pkl', '_results.json')
with open(op.join(results_path, config_file), "w") as f:
    json.dump(args_dict, f, indent=4)

# %%
