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
from sklearn.model_selection import GroupKFold, KFold
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

def str2bool(v):
    # type=bool treats any non-empty string (including 'False') as True
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't', '1', 'yes'):
        return True
    if v.lower() in ('false', 'f', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'expected a boolean, got {v!r}')

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--proc_type', default='MSMAll-FIX')
parser.add_argument('--connectome_type', default='correlation')
parser.add_argument('--ses', default='full')
parser.add_argument('--dataset', default='hcp')
parser.add_argument('--predictors', default='edges')
parser.add_argument('--pca', default=True, type=str2bool)
parser.add_argument('--n_boot', default=1, type=int,
                    help='number of bootstrap resamples of subjects (0 = no bootstrap)')


try: 
    os.environ['_']
    args, _ = parser.parse_known_args()
    notebook = False
except KeyError: 
    args, _ = parser.parse_known_args()
    notebook = True

seed = args.seed
proc_type = args.proc_type

dataset = args.dataset
predictors = args.predictors
pca = args.pca
connectome_type = args.connectome_type
ses = args.ses
n_boot = args.n_boot

assert dataset in ['hcp', 'pnc']
assert proc_type in ['MSMAll-FIX', 'xcpd', 'PNC_XCPD']
assert predictors in ['edges', 'ave_cont', 'degree_weighted', 'degree_unweighted']

n_folds = 5

print(n_boot)

# %%
if dataset == 'hcp': 
    with open(op.join(base_dir, 'data', f'documented_subs.txt'), 'r') as file:
        sub_list = file.read().splitlines()
    print(len(sub_list))

    cog_df = pd.read_csv(op.join(base_dir, 'data', 'restricted',
                              'hcp_phenotype_cleaned.csv'), 
                              index_col=0)
elif dataset == 'pnc':
    cog_df = pd.read_csv(op.join(base_dir, 'data', 
                               'study-PNC_desc-participants.tsv'), sep='\t')
    cog_df.index = [f'sub-{sub}' for sub in cog_df['participant_id']]

    PHENO_LIST = ['age', 'p_factor_mcelroy_harmonized_all_samples']


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
                     results_dir, 'PNC_XCPD', 
    f'proc-PNC_model-{connectome_type}_ses-full_connectome-edges.csv'), index_col=0  ) 
    
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
    # grouping by subject ID keeps bootstrap duplicates of a subject in the same fold
    groups = np.array(data.index)
    kf = KFold(n_folds)

    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        results_dict[f'fold_{fold}'] = {}

      #  target_scaler = StandardScaler() 
        train_X = X[train_index]
        # train_y = target_scaler.fit_transform(y[train_index].reshape(-1, 1)).ravel()
        train_y = y[train_index]
        test_X = X[test_index]
        test_y = y[test_index]
        # test_y = target_scaler.transform(y[test_index].reshape(-1, 1)).ravel()

        pl.fit(X= train_X, y=train_y)
        r2 = pl.score(test_X, test_y)
        ll, bic = calc_bic(pl, train_X, train_y)

        error = (test_y.ravel() - pl.predict(test_X).ravel())
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
for pheno in PHENO_LIST:
    results_dict[pheno] = {}
    random.seed(seed)

    for boot in range(n_boot): 
        if n_boot > 1: 
            print('resampling')
            data = pd.concat([fc_df, cog_df[pheno]], 
                            join='inner', axis=1).dropna(axis=0).sample(frac=1, replace=True)
        else: 
            data = pd.concat([fc_df, cog_df[pheno]], 
                            join='inner', axis=1).dropna(axis=0)
        results_dict[pheno][boot] = predict_pheno(data, pl, pheno)

# %%
results_dict

# %%
# with open(f'results/predictions/{proc_type}_{dataset}_{connectome_type}_pca-{pca}-nboots_{n_boot}_dict_NO-GROUP.pkl', 'wb') as f: 
    # pickle.dump(results_dict, f)
