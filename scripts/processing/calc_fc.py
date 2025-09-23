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

# This code expects to find parcellated timeseries data in `f'deriv_dir/timeseries/{proc_type}/sub-{sub_id}/sub-{sub_id}*ses-1*timeseries.ptseries.nii'`. 
#
#

import sys
import os
import numpy.matlib
from numpy.matlib import repmat
import nibabel as nib
import json
from sklearn.exceptions import ConvergenceWarning
import warnings


#work around until I install fc_comparison as an actual package
sys.path.append(os.path.dirname('/global/homes/m/mphagen/functional-connectivity/model-fc/src/model_fc'))

# +
import time

import nilearn

from pyuoi.utils import log_likelihood_glm, AIC, BIC

import numpy as np

import matplotlib.pyplot as plt

import argparse

import os
import os.path as op

from glob import glob

from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import pickle

from model_fc.models import init_model, run_model

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--sub_id',default='sub-102109')
parser.add_argument('--ses_id', default=1)
parser.add_argument('--run_id', default=None)
parser.add_argument('--task_id', default='rest') 

parser.add_argument('--atlas_spec', default='fsLR_seg-4S156Parcels_den-91k')
parser.add_argument('--n_rois', default=100, type=int) #default for hcp; 
parser.add_argument('--n_trs', default=1200, type=int) #default for hcp;
parser.add_argument('--n_folds', default=5) 
parser.add_argument('--model', default='ridgeCV') 
parser.add_argument('--cv', default='blocks') 
parser.add_argument('--proc_type', default='MSMAll') 

parser.add_argument('--profile', action='store_true') 

parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/hcp-functional-connectivity') 
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
run_id = args.run_id
random_state = 1
print(args)
# -

args_dict = vars(args) 


def get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id): 
    if 'hcp' in fc_data_path: 
        if run_id == None: 
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}*ses-{ses_id}*ptseries.nii'))
            
        elif run_id == '1': 
            #YES, future self, it is supposed to be run-{ses_id}, 
            #not run-{run_id}. 
            #RL scan was always ran first.
            #This is consistent with XCPDs HCP renaming code.
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}_dir-RL_*run-{ses_id}**ptseries.nii'))
            
        elif run_id == '2': 
            ts_file = glob(op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}',
                        f'sub-{sub_id}*task-{task_id}_dir-LR_*run-{ses_id}**ptseries.nii'))

        return ts_file


# +
ts_files = get_ts_files(fc_data_path, proc_type, sub_id, task_id, ses_id, run_id)

#
if notebook == False:
    results_path = op.join(fc_data_path, 
                    'derivatives',
                    'connectivity-matrices',
                    proc_type, 
                    f"{model_str.replace('-', '')}_{cv}",
                    f'sub-{sub_id}')
                       
os.makedirs(results_path, exist_ok=True)

assert len(ts_files) == 1 

print(f"Found {len(ts_files)} rest scans for subject {sub_id}.") 

print(f"Saving results to {results_path}.")
# -

model = init_model(model_str, max_iter, random_state)

print(model)


# +
def set_warnings_filters():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

set_warnings_filters()


# -

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
        
    return splits


def read_cifti(file): 
    time_series = nib.load(file).get_fdata(dtype=np.float32)
    if n_rois != time_series.shape[1]: 
        time_series = time_series[:, :n_rois]
    return time_series


time_series = read_cifti(ts_files[0]) 

# +
results_dict = {}    
print(f"Calculating {model_str} FC for {sub_id} {ses_id}")

scaler = StandardScaler()
file = f"sub-{sub_id}_task-{task_id}_ses-{ses_id}_{atlas_spec}_model-{model_str.replace('-', '')}_results.pkl"

if model_str in ["lassoCV", "uoiLasso", "enet", "lassoBIC", "ridgeCV"] :
    splits = split_kfold(cv, time_series, n_folds)
    print(model_str) 

    for fold_idx, (train_idx, test_idx) in enumerate( splits ): 
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: range={train_idx[0]}, {train_idx[-1]}")
        print(f"  Test:  index={test_idx[0]}, {test_idx[-1]}")

        train_ts = time_series[train_idx, :]
        test_ts = time_series[test_idx, :]
        
        scaler = StandardScaler()
        scaler.fit_transform(train_ts)
        scaler.transform(test_ts)

        start_time = time.time()
        results_dict[f'fold_{fold_idx}'] = run_model(train_ts, test_ts, n_rois, model=model)
        results_dict[f'fold_{fold_idx}']['model'] = model
        print(time.time() - start_time, ' seconds')     

        if profile == True: 
            break 
    
elif model_str in ['correlation', 'tangent']: 

    corr_mat = model.fit_transform(np.array(time_series))[0]
    np.fill_diagonal(corr_mat, 1)

    results_dict['fc_matrix'] = corr_mat
    results_dict['model'] = model
    
with open(op.join(results_path,file), 'wb') as f:
    pickle.dump(results_dict, f) 

config_file = file.replace('_results.pkl', '_results.json')
with open(op.join(results_path, config_file), "w") as f:
    json.dump(args_dict, f, indent=4)
# -



