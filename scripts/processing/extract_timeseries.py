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

# This script takes in pre-processed niftis, and outputs a CSV with the timeseries by parcel. 

# [793465, 160729, 173233, 126931]

# +
import numpy as np

import argparse

from glob import glob
import re

import sys
import os 
import os.path as op
import pandas as pd

sys.path.append(os.path.dirname('/global/homes/m/mphagen/functional-connectivity/model-fc/src/model_fc'))

import datalad.api as dl
import nibabel as nib

from xcp_d.interfaces.workbench import CiftiMath, CiftiParcellateWorkbench

# -


#to access git-annex, add env bin to $PATH
#add to jupyter kernel spec to get rid of this line
os.environ["PATH"] = "/global/homes/m/mphagen/miniconda3/envs/fc_w_datalad/bin:" + os.environ["PATH"]
#add wb_command to path
os.environ["PATH"] = "/global/homes/m/mphagen/software/workbench/bin_linux64:" + os.environ["PATH"]


# +
args = argparse.Namespace(verbose=False, verbose_1=False)

parser = argparse.ArgumentParser("extract_timeseries.py")
parser.add_argument('--sub_id',  default='126931') 
parser.add_argument('--atlas_file', default='tpl-fsLR_atlas-4S156Parcels_den-91k_dseg.dlabel.nii')
parser.add_argument('--dataset', default='HCP') 

#hack argparse to be jupyter friendly AND cmdline compatible
try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])

sub_id = args.sub_id
atlas_file = args.atlas_file #from atlaspack
dataset = args.dataset

print(args)

sub_id = sub_id.replace('sub-', '') #just strip out 


# -
def hcp_param_extract(hcp_path): 
    """
    taken from xcp_d - thanks!! 
    """
    
    base_task_name = hcp_path.split('/')[-2]
    _, base_task_id, dir_id = base_task_name.split('_')
    print(base_task_id) 
    match = re.match(r'([A-Za-z0-9]+[a-zA-Z]+)(\d+)$', base_task_id)
    print(match)
    if match:
        task_id = match.group(1).lower()
        run_id = int(match.group(2))
    else:
        print('no match') 
        pass
    
    return(task_id, run_id, dir_id)


def define_paths(path_prefix, dataset, results_str, atlas_file): 
    """
    DOCSTRING
    """
    if dataset == 'HCP': 
        dataset_path = op.join(path_prefix, 'hcp-functional-connectivity') 
   
    derivatives_path = op.join(dataset_path, 'derivatives') 
    results_dir = op.join(derivatives_path, results_str,)
    atlas_path = op.join(path_prefix, 'AtlasPack', atlas_file)
    
    path_dict = {
                'dataset_path': dataset_path, 
                 'derivatives_path': derivatives_path,
                 'results_dir': results_dir, 
                 'atlas_path': atlas_path
                }
        
    return path_dict


def parcellate_data(in_file, out_file, dataset_path, ciftiparcel):
    """
    Takes a minimally preprocessed HCP fMRI scan, 
    downloads with datalad and uses 
    wb_command -cifti-parcellate to parcellate.

    in_file: /path/to/file.dtseries.nii
    out_file: /derivatives/
    
    dataset_path: for datalad dataset parameter
    template: atlas file from AtlasPack

    """
    os.system(ciftiparcel.cmdline) 
    
    try: 
        dl.drop(file, dataset=dataset_path)
    except: 
        pass


# +
path_dict = define_paths('/pscratch/sd/m/mphagen', 'HCP', 'timeseries/min-proc', atlas_file)
                       
os.makedirs(path_dict['results_dir'], exist_ok=True)

rest_scans = glob(op.join(path_dict['dataset_path'], 
                          sub_id, 
                          'MNINonLinear/Results/rfMRI*', 
                          'rfMRI_REST*Atlas_MSMAll*clean*.dtseries.nii'))

rest_scans = list(filter(lambda x: '_7T_' not in x , rest_scans))

assert len(rest_scans) > 0 

print(f"Found {len(rest_scans)} rest scans for subject {sub_id}") 
# -

#hacky, but matches xcp-d output now
atlas_spec = atlas_file.split('.')[0].replace('_dseg', '').replace('tpl-', '').replace('atlas', 'seg')

for file in rest_scans: 
    
  #  print(tsv_path)                  
    task_id, run_id, dir_id = hcp_param_extract(file)
    
    file_str =  f'sub-{sub_id}_task-{task_id}_dir-{dir_id}_run-{run_id}_space-{atlas_spec}_stat-mean_timeseries.ptseries.nii'
    out_file = op.join(path_dict['results_dir'], f'sub-{sub_id}', file_str)
    if not op.exists(op.join(out_file, file_str)): 

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
        if not op.exists(file): 
            print("datalad getting") 
            dl.get(file, dataset=path_dict['dataset_path'])


        ciftiparcel = CiftiParcellateWorkbench()
        ciftiparcel.inputs.in_file = file
        ciftiparcel.inputs.out_file = out_file
        ciftiparcel.inputs.atlas_label =  path_dict['atlas_path']
        ciftiparcel.inputs.direction = 'COLUMN'
        ciftiparcel.cmdline  
        print(ciftiparcel.cmdline)

        ts = parcellate_data(file, out_file, path_dict['dataset_path'], ciftiparcel)  
        
    #double check BIDS nroi (does this need to be desc instead?
