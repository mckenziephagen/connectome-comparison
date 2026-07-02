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

# This code expects to find parcellated timeseries data from `extract_timeseries.py` and `concat_timeseries.py` in `{bids_dir}/derivatives/timeseries/{proc_type}/sub-{sub_id}/sub-{sub_id}*ses-*timeseries.ptseries.nii`. 
#
# It outputs data into `{bids_dir}/derivatives/connectome-matrices/{proc_type}/sub-{sub-id}`. Each subject has their own results saved to a pickle file for Analysis. 

import os
import os.path as op
import configparser
config = configparser.ConfigParser()
config.read(op.join(os.getcwd(), 'config.ini')) 

# +
import sys
from copy import deepcopy
from numpy.matlib import repmat
import nibabel as nib
import json
from sklearn.exceptions import ConvergenceWarning
import sklearn
import warnings
import time
import nilearn
from pyuoi.utils import log_likelihood_glm, AIC, BIC
import numpy as np
import argparse
from glob import glob
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(os.path.dirname(f"{config['General']['script_dir']}/model-fc/src/model_fc"))
from model_fc.models import init_model, run_model


# -


def glob_ts_file(fc_data_path, dataset, proc_type, sub_id, 
                 **kwargs): 
    ts_file = []
    sub_dir = op.join(fc_data_path, 
                        'derivatives', 
                        'timeseries',
                        proc_type,
                        f'sub-{sub_id}') 
    if dataset == 'hcp':                   
        if ses_id is None and run_id is None: 
            # if you want to run on runs concatonated into sessions
            # and both sessions
            file_str =  [f'sub-{sub_id}*task-{task_id}*ses-*ptseries.nii']
        
        elif run_id is not None: 
            #YES, future self, it is supposed to be run-{ses_id}, 
            #not run-{run_id}. 
            #This is consistent with XCPDs HCP renaming code.
            file_str = [f'sub-{sub_id}*task-{task_id}_dir-RL_*run-{ses_id}**ptseries.nii', 
                        f'sub-{sub_id}*task-{task_id}_dir-RL_*run-{ses_id}**ptseries.nii']
        
    elif dataset == 'pnc': 
        #the PNC xcpd data has ses- and task- swapped. 
        # and some participants have an extra, shorter rest scan we want to filter out
        file_str =  [f'sub-{sub_id}*ses-*task-{task_id}*singleband*ptseries.nii']

    if cv == 'task': 
        file_str.append(f'sub-{sub_id}*task-{cv_task}_*space*ptseries.nii')
    print(file_str)                      
    for file in file_str: 
        globbed_files = glob(op.join(sub_dir, file))
        for ii in globbed_files: 
            ts_file.append(ii)
    return ts_file


def filter_singleband(ts_files): 
    """ Some PNC participants have multple rest scans. This selects
    the singleband rest. """
    singlebands = [file for file in ts_files if 'singleband' in file]
    if singlebands:
        ts_files = [singlebands[0]]
    else:
        ts_files = [ts_files[0]]
    assert len(ts_files) == 1 
    return ts_files


def filter_ts_files(ts_files, dataset, proc_type):
    # if dataset == 'pnc': 
    #     if len(ts_files) > 1:
    #       ts_files = filter_singleband(ts_files)
            
    if proc_type == 'MSMAll_FIX': 
        ts_files[:] = [x for x in ts_files if 'NORMED' in x]
   
    return ts_files


def set_warnings_filters():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def split_kfold(cv, n_trs, time_series, **kwargs):
    if cv == 'random': 
        
        kfolds = KFold(n_splits=n_folds,
                      shuffle=False)
        splits = kfolds.split(X=time_series)
        
    if cv == 'blocks':
        group =  repmat(np.arange(1, n_folds+1), 
                        int(time_series.shape[0]/n_folds), 1).T.ravel()
        
        group = np.concatenate([group, np.ones(time_series.shape[0] - group.shape[0]) * 
                                np.max(group)]) 

        
        kfold = GroupKFold(n_splits=n_folds)
        
        splits = kfold.split(X=time_series, groups=group) 

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
        
    if cv == 'timeseries': 
        tscv = TimeSeriesSplit()
        splits = tscv(tscv.split(X=time_series))
    
    if cv == 'task': 
        ses_idx = np.concat([repmat(0, 1, n_trs[0]),
                             repmat(1, 1, n_trs[1])],axis=-1).ravel()
        ses = sklearn.model_selection.PredefinedSplit(ses_idx)
        splits = ses.split(time_series)
        
    return splits


def fit_model_full(model, model_str, sub_id, ses_id, time_series, n_rois, cv): 
    results_dict = {}  
    results_dict['full_models'] = {}
    print(f"Calculating {model_str} FC for {sub_id} {ses_id}")
    
    scaler = StandardScaler()
    print(model)
    
    ts = time_series
    full_fc_mat = np.empty((n_rois, n_rois))
    ts = scaler.fit_transform(ts)
    if model_str in ['lassoBIC', "uoiLasso", "pearsonRegressor", 
                     "partialCorrelationRegressor",  "ridgeCV"]: 
        for target_idx in range(ts.shape[1]):
            y_ts = np.array(ts[:, target_idx])
            X_ts = np.delete(ts, target_idx, axis=1)
            cloned_model = deepcopy(model)
            cloned_model.fit(X=X_ts, y=y_ts)
            full_fc_mat[target_idx, :] = np.insert(cloned_model.coef_, target_idx, 1)           
            results_dict['full_models'][f'node_{target_idx}'] = cloned_model
    
    elif model_str in ['correlation', "partial_correlation"]: 
        full_fc_mat = model.fit_transform(np.array(ts))[0]
        np.fill_diagonal(full_fc_mat, 1)

    else: 
        print('model not valid') 

    #duplicating for ease
    results_dict['full_fc_matrix'] = full_fc_mat
    results_dict['full_models']['full_fc_matrix'] = full_fc_mat

    return results_dict


def fit_model_cv(model, model_str, splits, time_series, 
                 max_iter, n_rois, random_state, sub_id, ses_id):
    
    results_dict = {}    
    print(f"Calculating {model_str} FC for {sub_id} {ses_id}")
    scaler = StandardScaler()

    for fold_idx, (train_idx, test_idx) in enumerate(splits): 
        if model_str in ["lassoCV", "uoiLasso", "enet", "lassoBIC", "ridgeCV",
                    "partialCorrelationRegressor", "pearsonRegressor"]:
            print(model_str) 
    
            
            print(f"Fold {fold_idx}:")
            print(f"  Train: range={train_idx[0]}, {train_idx[-1]}")
            print(f"  Test:  index={test_idx[0]}, {test_idx[-1]}")
    
            train_ts = time_series[train_idx, :]
            test_ts = time_series[test_idx, :]
            
            train_ts = scaler.fit_transform(train_ts)
            test_ts = scaler.transform(test_ts)
    
            start_time = time.time()
            results_dict[f'fold_{fold_idx}'] = run_model(train_ts, test_ts,
                                                         n_rois, model=model)
            print(time.time() - start_time, ' seconds')    
            print(model)
                        
        elif model_str in ['correlation', 'tangent', 'partial_correlation']: 
            #we're not really using "train" and "test" 
            #but this seperates by session, and keeps the 
            #terminology consistent

            #nilearn.ConnectivityMeasure norms, 
            #no need for scaler()
            train_ts = time_series[train_idx, :]
    
            corr_mat_train = model.fit_transform(np.array(train_ts))[0]
            np.fill_diagonal(corr_mat_train, 1)
    
            results_dict[f'fold_{fold_idx}']['fc_matrix'] = corr_mat_train        
            results_dict[f'fold_{fold_idx}']['model'] = deepcopy(model)        

    return results_dict

# +
args = argparse.Namespace()
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--dataset',default='pnc')
parser.add_argument('--sub_id',default='sub-1057423710')
parser.add_argument('--ses_id', default=None)
parser.add_argument('--run_id', default=None)
parser.add_argument('--task_id', default='rest') 

parser.add_argument('--atlas_spec', default='fsLR_seg-4S156Parcels_den-91k')
parser.add_argument('--n_rois', default=100, type=int) #default 100-schaefer
parser.add_argument('--n_folds', default=None)
parser.add_argument('--model', default='lassoBIC') 

parser.add_argument('--cv', default='task') #rest 
parser.add_argument('--cv_task', default='idemo') #None

parser.add_argument('--proc_type', default='xcpd') #MSMAll_FIX
parser.add_argument('--max_iter', default=1000) 
parser.add_argument('--test', default=False) 

set_warnings_filters()
# -

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
    notebook = False
except KeyError: 
    args = parser.parse_args([])
    notebook = True
    results_path = op.join(os.getcwd(), 'notebook_results') 

# +
args.sub_id = str(args.sub_id).replace('sub-', '')
sub_id = args.sub_id


if args.ses_id is not None: 
    args.ses_id = str(args.ses_id).replace('ses-', '')
ses_id = args.ses_id

dataset = args.dataset
task_id = args.task_id
atlas_spec = args.atlas_spec 
n_rois = args.n_rois
# n_trs = args.n_trs
n_folds = args.n_folds
model_str = args.model
cv = args.cv
proc_type = args.proc_type
cv_task = args.cv_task

max_iter = args.max_iter
run_id = args.run_id
test = args.test
random_state = 1 

fc_data_path = config['Paths'][dataset]

if not notebook:
    results_path = op.join(fc_data_path, 
                    'derivatives',
                    'connectivity-matrices',
                    proc_type,
                    f"{model_str.replace('-', '')}_{cv}",
                    f'sub-{sub_id}')
if test: 
    results_path = 'test_results'
    
print(args)
# -

args_dict = vars(args) 
os.makedirs(results_path, exist_ok=True)

# +
ts_files = glob_ts_file(fc_data_path, dataset, proc_type, sub_id,
                        cv_task=cv_task, cv=cv)

ts_files = filter_ts_files(ts_files, dataset, proc_type)
# -

print(f"Found {len(ts_files)} scans for subject {sub_id}.") 
print(f"Saving results to {results_path}.")


def trim_ts(file): 
    """ Trims the 56 subcortical regions off of '4S' Schaefer"""
    time_series = nib.load(file).get_fdata(dtype=np.float32)
    if n_rois != time_series.shape[1]: 
        time_series = time_series[:, :n_rois]
    return time_series


def read_ts(ts_files): 
    ts_list  = [] 
    n_trs = []
    for file in ts_files: 
        ts = trim_ts(file)
        n_trs.append(ts.shape[0]) 
        ts_list.append(ts) 
    time_series = np.concat([*ts_list], axis=0)

    return time_series, n_trs


# +
time_series, n_trs = read_ts(ts_files) 

model = init_model(model_str, max_iter, random_state)
    
full_model_dict = fit_model_full(model, model_str, sub_id, ses_id, time_series,
                                 n_rois, cv)

splits = split_kfold(cv, n_trs, time_series)

results_dict = fit_model_cv(model, model_str, splits, time_series, 
                            max_iter, n_rois, random_state, sub_id, ses_id)

file = f"sub-{sub_id}_task-{task_id}_ses-{ses_id}_{atlas_spec}_model-{model_str.replace('-', '')}_results.pkl"

results_dict['full_fc_matrix'] = full_model_dict['full_fc_matrix']
results_dict['full_models'] = full_model_dict['full_models']

with open(op.join(results_path,file), 'wb') as f:
    pickle.dump(results_dict, f) 

config_file = file.replace('_results.pkl', '_results.json')
with open(op.join(results_path, config_file), "w") as f:
    json.dump(args_dict, f, indent=4)
