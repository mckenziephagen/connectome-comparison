# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (my_env)
#     language: python
#     name: fc_analysis
# ---

# %% [markdown]
# Note: takes about two hours 

# %%
import boto3
import os 
import os.path as op
from glob import glob
import numpy as np
import pandas as pd

# %%
test_subjects = np.array(pd.read_csv('test_subjects.txt', 
                                     index_col=False, header=None)[0] )  

# %%
aws_session = boto3.session.Session(profile_name='nrdg')
s3 = aws_session.resource('s3')

# %%
s3_keys = []

# %%
bucket_name = 'xcpd.nrdg.uw.edu'

# %%
my_bucket = s3.Bucket(bucket_name)

# %%
s3_keys = list(my_bucket.objects.all()) 

# %%
cp_keys = []
for key_obj in s3_keys: 
    if 'pnc' in key_obj.key: 
        if 'rest' in  key_obj.key: 
            cp_keys.append(key_obj.key)

# %%
cp_keys

# %%
data_path = '/gscratch/scrubbed/mphagen/xcpd_output'

# %%
for key in test_keys: 
    if not op.exists(op.join(data_path, key)):
        my_bucket.download_file(key, op.join(data_path, key))

# %%
