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
#     display_name: xcpd_dl
#     language: python
#     name: xcpd_dl
# ---

# combine resting state scans from HCP https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# RL then LR 

# +
import nibabel as nib
import os 
import os.path as op
import numpy as np
from glob import glob
from xcp_d.utils.concatenation import concatenate_niimgs
import numpy as np
import argparse


os.environ["PATH"] = "/global/homes/m/mphagen/software/workbench/bin_linux64:" + os.environ["PATH"]

# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--sub_id',default='100307')
parser.add_argument('--run_id', default='2') 
parser.add_argument('--proc_type', default='parcellated-timeseries') 
parser.add_argument('--atlas_spec', default='space-fsLR_seg-4S156Parcels_den-91k') 

try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
   
   
sub_id = args.sub_id
run_id = args.run_id
proc_type = args.proc_type
atlas_spec = args.atlas_spec
# -

sub_id = sub_id.replace('sub-', '') #strip sub- if presentv

# +
pscratch = '/pscratch/sd/m/mphagen'
derivatives = f'hcp-functional-connectivity/derivatives/{proc_type}'

lr_file = glob(op.join(pscratch, 
                       derivatives, 
                       f'sub-{sub_id}',
                       f'sub-{sub_id}_task-rest_dir-LR_*run-{run_id}*.ptseries.nii') )

rl_file = glob(op.join(pscratch, 
                       derivatives, 
                       f'sub-{sub_id}',
                       f'sub-{sub_id}_task-rest_dir-RL_*run-{run_id}*.ptseries.nii') ) 

out_file = op.join(pscratch, 
                   derivatives, 
                   f'sub-{sub_id}',
                   f'sub-{sub_id}_task-rest_ses-{run_id}_{atlas_spec}_stat-mean_timeseries.ptseries.nii')
# -


assert len(rl_file) == 1
assert len(lr_file) == 1

if op.exists(out_file): 
    print("File already exits!" ) 
else: 
    merge_files = list([rl_file[0], lr_file[0] ]) 
    concatenate_niimgs(merge_files, out_file) 
