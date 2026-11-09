# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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


# %%
import pickle
import numpy as np
import pandas as pd

# %%
connectome_type = 'uoiLasso'

with open(f'results/{connectome_type}-bb_dict.pkl', 'rb') as f:
    uoi_results_dict = pickle.load(f)

    connectome_type = 'ridgeCV'

with open(f'results/{connectome_type}-bb_dict.pkl', 'rb') as f:
    ridge_results_dict = pickle.load(f)

    connectome_type = 'correlation'

with open(f'results/{connectome_type}-bb_dict.pkl', 'rb') as f:
    corr_results_dict = pickle.load(f)


# %%
def get_errors(results_dict):
    """Collect out-of-fold errors into a DataFrame (rows = subjects, columns = phenotypes)."""
    errors = {}
    for pheno, folds in results_dict.items():
        pheno_errors = [pd.Series(np.asarray(fold['error']).ravel(),
                                  index=fold['sub_order'])
                        for fold in folds.values()]
        errors[pheno] = pd.concat(pheno_errors)
    # phenotypes can have different subjects (dropna), so missing ones are NaN
    error_df = pd.DataFrame(errors)
    error_df.index.name = 'subject_id'
    return error_df


# %%
# Align by (subject, phenotype)
error_long = pd.concat(
    {name: get_errors(results_dict).stack()      # stack() drops NaNs
     for name, results_dict in [('correlation', corr_results_dict),
                       ('uoiLasso', uoi_results_dict),
                       ('ridgeCV', ridge_results_dict)]},
    axis=1, join='inner')
error_long.index.names = ['subject_id', 'pheno']

ortho_array = error_long.to_numpy()     # shape (n_subjects * n_phenos, 3)

# %%
error_long.to_csv(op.join(results_dir, 'pred_error_df.csv'))

# %%
error_long.describe()

# %%
import seaborn as sns

# %%
sns.histplot(error_long, alpha=.5)

# %%
from sklearn.decomposition import PCA

pca = PCA()
ortho_scores = pca.fit_transform(ortho_array)
ortho_df = pd.DataFrame(ortho_scores, index=error_long.index,
                        columns=[f'PC{i+1}' for i in range(ortho_scores.shape[1])])

print('explained variance ratio:', pca.explained_variance_ratio_.round(3))
print('loadings:')
print(pd.DataFrame(pca.components_, columns=error_long.columns,
                   index=ortho_df.columns).round(3))


# %%
# Project ortho_array onto PC1
pc1 = pca.components_[0]                          

# 1) PC1 scores
pc1_scores = ortho_df['PC1']                  

# 2) The same projection expressed in the original 3 error columns
#    shared error component for each connectome type)
pc1_proj = pca.mean_ + np.outer(pc1_scores, pc1)  # shape (n, 3)
pc1_proj_df = pd.DataFrame(pc1_proj, index=error_long.index,
                           columns=error_long.columns)

# each connectome type's non-shared error
pc1_resid_df = error_long - pc1_proj_df

# %%
from scipy.stats import friedmanchisquare as fried

# %%
fried(pc1_proj_df['correlation'], 
      pc1_proj_df['uoiLasso'], pc1_proj_df['ridgeCV'])

# %% [markdown]
# Okay, let's try again: 

# %%
pca = PCA()

# %%
corr_df = get_errors(corr_results_dict).dropna()

uoi_df = get_errors(uoi_results_dict).dropna()

# %%
corr_scores = pca.fit_transform(corr_df)

# %%
corr_scores[0]

# %%
pca.components_.shape

# %%
np.outer(uoi_df,pca.components_[0] )

# %%
np.outer(uoi_df,pca.components_[0])

# %% [markdown]
# Ariel's second idea: 

# %%
error_long

# %%
error_long

# %%
error_long.reset_index().groupby('subject_id').mean

# %%
error_long.mean(axis=1)

# %% [markdown]
#
