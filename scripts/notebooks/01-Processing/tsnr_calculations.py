# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: fc
#     language: python
#     name: fc_py
# ---

# %%
import nibabel as nb 
import nilearn
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import datalad.api as dl
import os
import pandas as pd
os.environ["PATH"] = "/mmfs1/gscratch/escience/mphagen/miniforge/envs/fc_py/bin:" + os.environ["PATH"]

import argparse
import pandas as pd

# %%
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser("tsnr_calculations.py")
parser.add_argument('--sub_id',  default='103010') 
parser.add_argument('--dataset', default='human-connectome-project-openaccess') 
parser.add_argument('--proc', default='Unproc' ) 

#hack argparse to be jupyter friendly AND cmdline compatible
# try: 
#     os.environ['_']
#     args = parser.parse_args()
# except KeyError: 
args = parser.parse_args([])

sub_id = args.sub_id
dataset = args.dataset

proc = args.proc


print(args)

sub_id = sub_id.replace('sub-', '') #just strip out 

# %%
bids_dir = '/gscratch/scrubbed/mphagen/data/human-connectome-project-openaccess' 
dl_dir = op.join(bids_dir, 'HCP1200') 

# %%
if proc_type == 'raw': 
    file_strs = ['rfMRI_REST1_RL.nii.gz',  'rfMRI_REST1_LR.nii.gz', 
            'rfMRI_REST2_RL.nii.gz',  'rfMRI_REST2_LR.nii.gz']

# %%
sub_dir = op.join(dl_dir, f'{sub_id}', 
                  'MNINonLinear/Results/rfMRI_REST*/*.nii.gz')
dl.get(path = op.join(dl_dir,  sub_id, 'MNINonLinear/Results'), 
        dataset=bids_dir,
        recursive=True,
        get_data=False)

sub_niftis = [] 
_ = [sub_niftis.append(ii) for ii in glob(sub_dir) if any(x in ii for x in file_strs)] 
_ = [sub_niftis.drop(ii) for ii in sub_niftis if '7T' in ii]  


for nifti_file in sub_niftis: 
    dl.get(path = nifti_file, dataset=bids_dir)
    nifti = nb.load(nifti_file)
    nifti_data = nifti.get_fdata()
    
    tsnr_array = np.mean(nifti_data,axis=3) / np.std(nifti_data,axis=3) 
    tsnr_array[np.isnan(tsnr_array)] = 0 
    np.save(f'tsnr_results/{proc_type}/{sub_id}_{op.basename(nifti_file).replace('.nii.gz', '.npy')}', tsnr_array)
    dl.drop(nifti_file, dataset=bids_dir)

# %%
dl.uninstall(path = op.join(dl_dir,  sub_id, 'MNINonLinear/Results'), 
        dataset=bids_dir,
        recursive=True) 
