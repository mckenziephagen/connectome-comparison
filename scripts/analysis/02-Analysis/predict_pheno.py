# ---
# jupyter:
#   jupytext:
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
import os.path as op
import os 
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir , 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')
config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )



# %%
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle
import random
import sys

# %%
import pyuoi.utils as putils

sys.path.append(op.join(base_dir, 'scripts', 'analysis'))
from fc_utils._utils import PHENO_LIST


# %%
import argparse
args = argparse.Namespace()

parser = argparse.ArgumentParser()
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--seed',default=1)
parser.add_argument('--date_str', default='2026-10-04')
parser.add_argument('--proc_type', default='MSMAll-FIX')
parser.add_argument('--connectome_type', default='lassoBIC')
parser.add_argument('--ses', default='full')
parser.add_argument('--dataset', default='hcp')
parser.add_argument('--predictors', default='edges')
parser.add_argument('--pca', default=True, type=bool)


try: 
    os.environ['_']
    args, _ = parser.parse_known_args()
    notebook = False
except KeyError: 
    args, _ = parser.parse_known_args()
    notebook = True

seed = args.seed
date_str = args.date_str
proc_type = args.proc_type

dataset = args.dataset
predictors = args.predictors
pca = args.pca
connectome_type = args.connectome_type
ses = args.ses

assert dataset in ['hcp', 'pnc']
assert proc_type in ['MSMAll-FIX', 'xcpd']
assert predictors in ['edges', 'ave_cont', 'degree_weighted', 'degree_unweighted']

n_folds = 5

# %%
with open(op.join(base_dir, 'data', f'documented_subs.txt'), 'r') as file:
    sub_list = file.read().splitlines()
print(len(sub_list))

cog_df = pd.read_csv(op.join(base_dir, 'data', 'restricted',
                              'hcp_phenotype_cleaned.csv'), 
                              index_col=0)


# %%
def calc_bic(pipeline, X_train, y_train):
    # ravel both so (n, 1) - (n,) doesn't broadcast to an (n, n) residual matrix
    y_true = np.asarray(y_train).ravel()
    y_pred = np.asarray(pipeline.predict(X_train)).ravel()
    ll = putils.log_likelihood_glm(model='normal',
                        y_true=y_true,
                        y_pred=y_pred
                       )
    n_features = np.count_nonzero(pipeline['model'].coef_)
    n_samples = X_train.shape[0]
    bic = putils.BIC(ll, n_features, n_samples)
    return ll, bic


# %%
if dataset == 'hcp': 
    fc_df = pd.read_csv(op.join(
                     results_dir, proc_type, 
    f'proc-{proc_type.split('-')[-1]}_model-{connectome_type}_ses-full_connectome-edges.csv'),
    index_col=0)
elif dataset == 'pnc': 
    fc_df = pd.read_csv(op.join(
                     results_dir, 'pnc_xcpd', 
    f'{date_str}_model-{connectome_type}_ses-{ses}.csv'), index_col=0  ) 
fc_df = fc_df.dropna(axis=1)

# %%
model = LassoCV() 
if pca == True: 
    pl = Pipeline([ ('scaler', StandardScaler()), ('PCA', PCA()), ('model', model)])
else: 
    pl = Pipeline([ ('scaler', StandardScaler()), ('model', model)])


# %%
def predict_pheno(data, pl, pheno): 
    results_dict = {} 
    X = np.array(data.drop(pheno, axis=1) )
    y = np.array(data[pheno] ) 
    groups = np.array(data.index)
    kf = GroupKFold(n_folds)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y, groups)):
        results_dict[f'fold_{fold}'] = {}

        target_scaler = StandardScaler() 
        train_X = X[train_index]
        train_y = target_scaler.fit_transform(y[train_index].reshape(-1, 1)).ravel()
        test_X = X[test_index]
        test_y = target_scaler.transform(y[test_index].reshape(-1, 1)).ravel()

        pl.fit(X= train_X, y=train_y)      
        r2 = pl.score(test_X, test_y)
        ll, bic = calc_bic(pl, train_X, train_y)

        error = (test_y.ravel() - pl.predict(test_X).ravel())
        sub_order.extend(groups[test_index])
        results_dict[f'fold_{fold}'] = {"model": deepcopy(pl),
                                        "r2": r2,
                                        "ll": ll,
                                        "bic": bic, 
                                        "error": error, 
                                        "sub_order": groups[test_index]
                                                        }
            # print(acc)
    return results_dict


# %%
results_dict = {}  
sub_order = [] 
#if resample: 
for pheno in PHENO_LIST:
    random.seed(seed) 
    data = pd.concat([fc_df, cog_df[pheno]], join='inner', axis=1).dropna(axis=0)
    results_dict[pheno] = predict_pheno(data, pl, pheno)

# %%
with open(f'results/{connectome_type}-bb_dict.pkl', 'wb') as f: 
    pickle.dump(results_dict, f)
