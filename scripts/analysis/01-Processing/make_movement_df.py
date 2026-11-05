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
#movement files were downloaded from S3 and put in derivatives/movement

# %%
config['Paths']['deriv_dir']

# %% [markdown]
# ### HCP

# %%
run_strs = ['rfMRI_REST1_RL',  'rfMRI_REST1_LR', 
            'rfMRI_REST2_RL',  'rfMRI_REST2_LR']

# %%
doc_subs = list(pd.read_csv(op.join(base_dir, 'data', 'documented_subs.txt'), header=None)[0] )
mv_dir = op.join(config['Paths']['deriv_dir'], 'movement') 

# %%
qa_df = pd.DataFrame(index=doc_subs, columns=['rfMRI_REST1_RL',  'rfMRI_REST1_LR', 
            'rfMRI_REST2_RL',  'rfMRI_REST2_LR'])

# %%
for sub_id in doc_subs:
    for run in run_strs: 
        rms = np.loadtxt(op.join(mv_dir, f'sub-{sub_id}', f'{run}_Movement_RelativeRMS_mean.txt'))
        qa_df.loc[sub_id, run] = rms

# %%
qa_df.shape

# %%
qa_df.to_csv(op.join(base_dir, 'data', 'hcp_relative_movement.csv' ))

# %% [markdown]
# ### PNC

# %%
pnc_df = pd.read_csv(op.join(base_dir, 'data', 'study-PNC_desc-func_qc.tsv'), sep='\t')
pnc_df = pnc_df.set_index('sub', drop=True)

# %%
idemo_df =pnc_df.loc[pnc_df['task'] == 'idemo'][['mean_relative_rms']]
rest_df = pnc_df.loc[(pnc_df['task'] == 'rest') & (pnc_df['acq'] == 'singleband')][['mean_relative_rms']]

# %%
mv_df = rest_df.join(idemo_df, lsuffix = '_rest', rsuffix='_idemo')

# %%
mv_df.to_csv(op.join(base_dir, 'data', 'pnc_relative_movement.csv'))
