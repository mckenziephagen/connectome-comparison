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
config.read(op.join(base_dir,  'config.ini') )

# %%
import pandas as pd 
import numpy as np 


# %%
def residualize(group, add_intercept=True):
    y = np.asarray(group['values'], dtype=float).ravel()
    X = np.asarray(group['movement'], dtype=float).reshape(len(y), -1)
    resid = np.full(len(y), np.nan)

    mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))  # drop missing rows
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)

    resid[mask] = y[mask] - X[mask] @ beta
    return pd.Series(resid, index=group.index)


# %%
lasso_fix_df = pd.read_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-lassoBIC_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
lasso_fix_df['resid'] = lasso_fix_df.groupby('pos', 
                                             group_keys=False).apply(residualize, include_groups=False)


# %%
lasso_fix_df[['sub', 'ses', 'pos', 
              'movement', 'resid']].to_csv(op.join(results_dir, 'MSMAll-FIX', 
                                  'proc-FIX_model-lassoBIC_residual-edges.csv'))

# %%
uoi_fix_df = pd.read_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
uoi_fix_df['resid'] = uoi_fix_df.groupby('pos', 
                                         group_keys=False).apply(residualize, include_groups=False)


# %%
uoi_fix_df[['sub', 'ses', 'pos', 
            'movement', 'resid']].to_csv(op.join(results_dir, 'MSMAll-FIX', 
                                  'proc-FIX_model-uoiLasso_residual-edges.csv'))

# %%
ridge_fix_df = pd.read_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
ridge_fix_df['resid'] = ridge_fix_df.groupby('pos', 
                                             group_keys=False).apply(residualize, include_groups=False)


# %%
ridge_fix_df[['sub', 'ses', 'pos', 
              'movement', 'resid']].to_csv(op.join(results_dir, 'MSMAll-FIX', 
                                  'proc-FIX_model-ridgeCV_residual-edges.csv'))

# %%
pearson_fix_df = pd.read_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-Pearson_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
pearson_fix_df['resid'] = pearson_fix_df.groupby('pos', 
                                                 group_keys=False).apply(residualize, include_groups=False)


# %%
pearson_fix_df[['sub', 'ses', 'pos', 
                'movement', 'resid']].to_csv(op.join(results_dir, 'MSMAll-FIX', 
                                  'proc-FIX_model-Pearson_residual-edges.csv'))

# %%
pcorr_fix_df = pd.read_csv(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-partial-correlation_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
pcorr_fix_df['resid'] = pcorr_fix_df.groupby('pos', 
                                             group_keys=False).apply(residualize, include_groups=False)


# %%
pcorr_fix_df[['sub', 'ses', 'pos', 
              'movement', 'resid']].to_csv(op.join(results_dir, 'MSMAll-FIX', 
                                  'proc-FIX_model-partial-correlation_residual-edges.csv'))

# %% [markdown]
# ### XCPD

# %%
lasso_xcpd_df = pd.read_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-lassoBIC_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
lasso_xcpd_df['resid'] = lasso_xcpd_df.groupby('pos', 
                                               group_keys=False).apply(residualize, include_groups=False)


# %%
lasso_xcpd_df[['sub', 'ses', 'pos', 
               'movement', 'resid']].to_csv(op.join(results_dir, 'XCPD', 
                                  'proc-XCPD_model-lassoBIC_residual-edges.csv'))

# %%
uoi_xcpd_df = pd.read_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-uoiLasso_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
uoi_xcpd_df['resid'] = uoi_xcpd_df.groupby('pos', 
                                           group_keys=False).apply(residualize, include_groups=False)


# %%
uoi_xcpd_df[['sub', 'ses', 'pos', 
             'movement', 'resid']].to_csv(op.join(results_dir, 'XCPD', 
                                  'proc-XCPD_model-uoiLasso_residual-edges.csv'))

# %%
ridge_xcpd_df = pd.read_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-ridgeCV_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
ridge_xcpd_df['resid'] = ridge_xcpd_df.groupby('pos', 
                                               group_keys=False).apply(residualize, include_groups=False)


# %%
ridge_xcpd_df[['sub', 'ses', 'pos', 
               'movement', 'resid']].to_csv(op.join(results_dir, 'XCPD', 
                                  'proc-XCPD_model-ridgeCV_residual-edges.csv'))

# %%
pearson_xcpd_df = pd.read_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-Pearson_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
pearson_xcpd_df['resid'] = pearson_xcpd_df.groupby('pos', 
                                                   group_keys=False).apply(residualize, include_groups=False)


# %%
pearson_xcpd_df[['sub', 'ses', 'pos', 
                 'movement', 'resid']].to_csv(op.join(results_dir, 'XCPD', 
                                  'proc-XCPD_model-Pearson_residual-edges.csv'))

# %%
pcorr_xcpd_df = pd.read_csv(op.join(results_dir, 'XCPD', 'proc-XCPD_model-partial-correlation_reliability-edges.csv'), 
                   index_col=0).reset_index(drop=True)
pcorr_xcpd_df['resid'] = pcorr_xcpd_df.groupby('pos', 
                                               group_keys=False).apply(residualize, include_groups=False)


# %%
pcorr_xcpd_df[['sub', 'ses', 'pos', 
               'movement', 'resid']].to_csv(op.join(results_dir, 'XCPD', 
                                  'proc-XCPD_model-partial-correlation_residual-edges.csv'))
