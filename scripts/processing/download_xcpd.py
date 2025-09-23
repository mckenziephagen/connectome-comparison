# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: xcpd_dl
#     language: python
#     name: xcpd_dl
# ---

# Note: takes about two hours 

import boto3
import os 
import os.path as op
from glob import glob
import numpy as np
import pandas as pd

test_subjects = np.array(pd.read_csv('../../data/test_subjects.txt', 
                                     index_col=False, header=None)[0] )  

aws_session = boto3.session.Session(profile_name='nrdg')
s3 = aws_session.resource('s3')

s3_keys = []

bucket_name = 'xcpd.nrdg.uw.edu'

# +
my_bucket = s3.Bucket(bucket_name)

s3_keys = list(my_bucket.objects.all()) 
# -

cp_keys = []
for key_obj in s3_keys: 
    if 'rest' in key_obj.key: 
        if 'timeseries' in key_obj.key:
            if 'hcpya' in key_obj.key: 
                cp_keys.append(key_obj.key)

test_keys = [] 
for ii in test_subjects: 
    for jj in cp_keys: 
        if str(ii) in jj:
            test_keys.append(jj)

data_path = '/pscratch/sd/m/mphagen/hcp-functional-connectivity/derivatives/xcpd_output'


for key in test_keys: 
    if not op.exists(op.join(data_path, key)):
        my_bucket.download_file(key, op.join(data_path, key))

