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
from sklearn.linear_model import LassoLarsIC, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from joblib import Parallel, delayed
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

parser.add_argument('--seed',default=1)
parser.add_argument('--date_str', default='2026-10-04')
parser.add_argument('--proc_type', default='MSMAll-FIX')
parser.add_argument('--connectome_type', default='lassoBIC')
parser.add_argument('--ses', default='full')
parser.add_argument('--dataset', default='hcp')
parser.add_argument('--predictors', default='edges')
parser.add_argument('--pca', default=True, type=str2bool)
parser.add_argument('--n_boot', default=1, type=int,
                    help='number of bootstrap resamples to run in this job')
parser.add_argument('--boot_start', default=0, type=int,
                    help='index of the first resample in this job; resample i is always '
                         'seeded from (seed, i), so chunks split across jobs match one long run')
parser.add_argument('--resample', default=None, type=str2bool,
                    help='resample subjects with replacement (default: True if n_boot > 1)')
parser.add_argument('--save_models', default=None, type=str2bool,
                    help='store fitted pipelines in the output (default: only when not resampling, '
                         'since a PCA fit per fold x resample gets very large)')
parser.add_argument('--out_dir', default=None,
                    help='output directory (default: results/predictions[/resamples])')
parser.add_argument('--suffix', default='',
                    help='string appended to the non-resampled output filename, e.g. _NO-GROUP')
parser.add_argument('--cv_group', default=None, choices=['family', 'subject'],
                    help='unit kept together in CV folds (default: family for hcp, subject for pnc). '
                         'HCP is a twin/sibling sample, so grouping by subject lets relatives '
                         'land in both train and test')
parser.add_argument('--cv', default='kfold', choices=['kfold', 'logo'],
                    help="'kfold': 5-fold GroupKFold. 'logo': leave one group out, i.e. "
                         "leave-one-family-out with --cv_group family")
parser.add_argument('--pheno', default=None,
                    help='run one phenotype, by index into PHENO_LIST or by name (default: all)')
parser.add_argument('--n_jobs', default=1, type=int,
                    help='number of CV folds fitted in parallel')
parser.add_argument('--scale_y', default=True, type=str2bool,
                    help='z-score the target within each training fold (default: True); '
                         'if False, the model is fit on y in its original units')

parser.add_argument('--n_folds', default=5,
                    help='')


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
n_boot = args.n_boot
boot_start = args.boot_start
resample = args.resample if args.resample is not None else n_boot > 1
cv = args.cv
# a fitted PCA pipeline per family (~430 folds) is too large to keep, so models
# are only saved by default for 5-fold runs without resampling
save_models = (args.save_models if args.save_models is not None
               else not resample and cv == 'kfold')
cv_group = args.cv_group or ('family' if dataset == 'hcp' else 'subject')
n_jobs = args.n_jobs
scale_y = args.scale_y
n_folds = args.n_folds
if args.pheno is None:
    pheno_list = PHENO_LIST
elif str(args.pheno).isdigit():
    pheno_list = [PHENO_LIST[int(args.pheno)]]
else:
    assert args.pheno in PHENO_LIST, f'{args.pheno} is not in PHENO_LIST'
    pheno_list = [args.pheno]

assert dataset in ['hcp', 'pnc']
assert not (cv_group == 'family' and dataset != 'hcp'), 'family grouping is only defined for hcp'
assert proc_type in ['MSMAll-FIX', 'xcpd']
assert predictors in ['edges', 'ave_cont', 'degree_weighted', 'degree_unweighted']


# seed the global RNGs too, as a backstop for anything that doesn't take a random_state
random.seed(seed)
np.random.seed(seed)

print(f'{connectome_type}: resample={resample}, boots {boot_start}-{boot_start + n_boot - 1}, '
      f'seed={seed}, save_models={save_models}, cv_group={cv_group}')

# %%
with open(op.join(base_dir, 'data', f'documented_subs.txt'), 'r') as file:
    sub_list = file.read().splitlines()
print(len(sub_list))

cog_df = pd.read_csv(op.join(base_dir, 'data', 'restricted',
                              'hcp_phenotype_cleaned.csv'), 
                              index_col=0)

# %%
# CV group label for each subject. With 'family', twins/siblings always fall in
# the same fold; with 'subject', each subject is its own group (bootstrap
# duplicates still stay together either way)
if cv_group == 'family':
    restricted_df = pd.read_csv(op.join(base_dir, 'data', 'restricted',
                                        'RESTRICTED_1_31_2022_23_26_45.csv'),
                                usecols=['Subject', 'Family_ID'])
    group_map = pd.Series(restricted_df['Family_ID'].values,
                          index=[f'sub-{s}' for s in restricted_df['Subject']])
else:
    group_map = None


# %% [markdown]
#

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
# random_state is only consumed by LassoCV when selection='random' and by PCA
# when the randomized/arpack solver is used, but pinning it makes the pipeline
# deterministic regardless of which solver sklearn picks for a given data shape
model = LassoCV()
if pca == True: 
    pl = Pipeline([ ('scaler', StandardScaler()), ('PCA', PCA(random_state=seed)), ('model', model)])
else: 
    pl = Pipeline([ ('scaler', StandardScaler()), ('model', model)])


# %%
def predict_pheno(data, pl, pheno): 
    results_dict = {} 
    X = np.array(data.drop(pheno, axis=1) )
    y = np.array(data[pheno] ) 
    sub_ids = np.array(data.index)
    if group_map is not None:
        missing = data.index.difference(group_map.index)
        assert len(missing) == 0, f'no Family_ID for {len(missing)} subjects, e.g. {list(missing[:3])}'
        groups = group_map.loc[data.index].values
    else:
        groups = sub_ids
    kf = GroupKFold(n_folds) if cv == 'kfold' else LeaveOneGroupOut()

    # folds are independent, so they can be fitted in parallel (--n_jobs); with
    # n_jobs > 1 each worker gets its own copy of pl
    def fit_fold(fold, train_index, test_index):
        results_dict = {}
        results_dict[f'fold_{fold}'] = {}

        # with scale_y=False this is an identity transform, so y stays in original units
        target_scaler = StandardScaler(with_mean=scale_y, with_std=scale_y)
        train_X = X[train_index]
        train_y = target_scaler.fit_transform(y[train_index].reshape(-1, 1)).ravel()
        test_X = X[test_index]
        test_y = target_scaler.transform(y[test_index].reshape(-1, 1)).ravel()

        # LassoCV chooses alpha with its own internal CV; by default that's plain
        # KFold, which splits relatives. Give it splits grouped like the outer CV.
        inner_splits = list(GroupKFold(n_folds).split(train_X, groups=groups[train_index]))
        pl.set_params(model__cv=inner_splits)

        pl.fit(X= train_X, y=train_y)      
        r2 = pl.score(test_X, test_y)
        ll, bic = calc_bic(pl, train_X, train_y)

        error = (test_y.ravel() - pl.predict(test_X).ravel())
        results_dict[f'fold_{fold}'] = {"r2": r2,
                                        "ll": ll,
                                        "bic": bic, 
                                        "error": error, 
                                        "sub_order": sub_ids[test_index],
                                        "cv_group": cv_group,
                                        # original units, so predictions can be pooled across folds
                                        # (per-fold r2 is undefined for 1-4 person LOFO test folds)
                                        "y_true": y[test_index],
                                        "y_pred": target_scaler.inverse_transform(
                                            pl.predict(test_X).reshape(-1, 1)).ravel(),
                                        # lightweight model summary, kept even when models aren't saved
                                        "alpha": pl['model'].alpha_,
                                        "coef": pl['model'].coef_.copy(),
                                        }
        if save_models:
            results_dict[f'fold_{fold}']["model"] = deepcopy(pl)
            # print(acc)
        return results_dict

    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(fit_fold)(fold, train_index, test_index)
        for fold, (train_index, test_index) in enumerate(kf.split(X, y, groups)))
    for fold_result in fold_results:
        results_dict.update(fold_result)
    return results_dict


# %%
results_dict = {}  
sub_order = [] 
#if resample: 
for pheno in pheno_list:
    results_dict[pheno] = {}
    pheno_data = pd.concat([fc_df, cog_df[pheno]], join='inner', axis=1).dropna(axis=0)

    for boot in range(boot_start, boot_start + n_boot): 
        if resample: 
            # pandas .sample() ignores python's `random` module, so pass a numpy
            # Generator seeded from (seed, boot): each resample is identical no
            # matter which job runs it or which phenotypes/resamples ran before it
            rng = np.random.default_rng([seed, boot])
            data = pheno_data.sample(frac=1, replace=True, random_state=rng)
        else: 
            data = pheno_data
        results_dict[pheno][boot] = predict_pheno(data, pl, pheno)
        print(f'{pheno} boot {boot} done', flush=True)

# %%

out_dir = args.out_dir or op.join(analysis_dir, 'results', 'predictions')
# 5-fold keeps its original name; leave-one-group-out runs are tagged so they don't overwrite it
cv_part = '' if cv == 'kfold' else {'family': '_cv-lofo', 'subject': '_cv-loso'}[cv_group]
# a single-phenotype run (--pheno) is tagged so array tasks don't overwrite each other;
# combine them with merge_predictions.py --mode lofo
pheno_part = f'_pheno-{pheno_list[0]}' if args.pheno is not None else ''
# unscaled-y runs are tagged so they don't overwrite the default (scaled) output
scale_part = '' if scale_y else '_scaleY-False'
out_name = f'{proc_type}_{dataset}_{connectome_type}_pca-{pca}-nboots_{n_boot}{cv_part}{scale_part}{pheno_part}_dict{args.suffix}.pkl'

# only skip saving inside an interactive kernel; checking ipykernel directly is
# more robust than the os.environ['_'] test, which isn't guaranteed under Slurm
if 'ipykernel' not in sys.modules:
    os.makedirs(out_dir, exist_ok=True)
    out_path = op.join(out_dir, out_name)
    with open(out_path, 'wb') as f: 
        pickle.dump(results_dict, f)
    print(f'saved {out_path}')
