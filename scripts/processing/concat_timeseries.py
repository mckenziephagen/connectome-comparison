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
from xcp_d.interfaces.workbench import CiftiMath
from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, CommandLineInputSpec, traits

os.environ["PATH"] = "/global/homes/m/mphagen/software/workbench/bin_linux64:" + os.environ["PATH"]

# -

os.environ["PATH"] = "/global/homes/m/mphagen/software/workbench/bin_linux64/wb_shortcuts:" + os.environ["PATH"]


# +
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument('--sub_id',default='856463')
parser.add_argument('--run_id', default='2') 
parser.add_argument('--proc_type', default='MSMAll') 
parser.add_argument('--atlas_spec', default='space-fsLR_seg-4S156Parcels_den-91k') 
parser.add_argument('--normalize', default=True) 
parser.add_argument('--dataset', default='human-connectome-project-openaccess/HCP1200') 


try: 
    os.environ['_']
    args = parser.parse_args()
except KeyError: 
    args = parser.parse_args([])
   
   
sub_id = args.sub_id
run_id = args.run_id
proc_type = args.proc_type
atlas_spec = args.atlas_spec
normalize = args.normalize
dataset = args.dataset
# -

sub_id = sub_id.replace('sub-', '') #strip sub- if presentv

# +
pscratch = '/pscratch/sd/m/mphagen'
derivatives = f'{dataset}/derivatives/timeseries/{proc_type}'

lr_file = glob(op.join(pscratch, 
                       derivatives, 
                       f'sub-{sub_id}',
                       f'sub-{sub_id}_task-rest_dir-LR_*run-{run_id}*mean_timeseries*.ptseries.nii') )

rl_file = glob(op.join(pscratch, 
                       derivatives, 
                       f'sub-{sub_id}',
                       f'sub-{sub_id}_task-rest_dir-RL_*run-{run_id}*mean_timeseries*.ptseries.nii') ) 

out_file = op.join(pscratch, 
                   derivatives, 
                   f'sub-{sub_id}',
                   f'sub-{sub_id}_task-rest_ses-{run_id}_{atlas_spec}_stat-mean_timeseries.ptseries.nii')
# -


op.join(pscratch, 
                       derivatives, 
                       f'sub-{sub_id}',
                       f'sub-{sub_id}_task-rest_dir-RL_*run-{run_id}*mean_timeseries*.ptseries.nii') 

assert len(rl_file) == 1
assert len(lr_file) == 1


class _CiftiDemeanOutputSpec(TraitedSpec):
    """Output specification for the CiftiParcellateWorkbench command."""

    out_file = File(exists=True, desc="output file")


class _CiftiDemeanInputSpec(CommandLineInputSpec):
    """Output specification for ConvertAffine."""

    in_file = File(
        position=2,
        argstr='%s',
        exists=True,
        mandatory=True,
        desc='input file')

    out_file = File(
        name_source=['in_file'],
        name_template='smoothed_%s.nii',
        keep_extension=True,
        argstr='%s',
        position=1,
        desc='The output CIFTI'
    )


class CiftiDemean(WBCommand):
    """Interface for wb_command's -convert-affine command."""

    input_spec = _CiftiDemeanInputSpec
    output_spec = _CiftiDemeanOutputSpec

    _cmd = "wb_shortcuts -cifti-demean -normalize"


if normalize: 
    ciftidemean_rl = CiftiDemean()
    ciftidemean_rl.inputs.in_file = rl_file[0]
    ciftidemean_rl.inputs.out_file = rl_file[0].replace('mean', 'meanNORMED') 
    ciftidemean_rl.cmdline
    
    os.system(ciftidemean_rl.cmdline)
    
    ciftidemean_lr = CiftiDemean()
    ciftidemean_lr.inputs.in_file = lr_file[0]
    ciftidemean_lr.inputs.out_file = lr_file[0].replace('mean', 'meanNORMED') 
    ciftidemean_lr.cmdline
    
    os.system(ciftidemean_lr.cmdline)

    out_file = out_file.replace('mean', 'meanNORMED')

    assert ciftidemean_lr.inputs.out_file != ciftidemean_rl.inputs.out_file

if op.exists(out_file): 
    print("File already exits!" ) 
else: 
    merge_files = list([ciftidemean_rl.inputs.out_file, ciftidemean_lr.inputs.out_file ]) 
    concatenate_niimgs(merge_files, out_file) 


















