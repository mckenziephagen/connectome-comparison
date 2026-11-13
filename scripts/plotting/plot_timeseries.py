# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python [conda env:base]
#     language: python
#     name: conda-base-py
# ---

from glob import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb

import nilearn
import nilearn.datasets
from nilearn.maskers import NiftiLabelsMasker

sub_path = '/pscratch/sd/m/mphagen/human-connectome-project-openaccess/HCP1200/derivatives/timeseries/MSMAll_FIX/sub-793465' 

ses_1_msmall_file = glob(op.join(sub_path, '*ses-1*NORMED*.nii') ) 
ses_2_file = glob(op.join(sub_path, '*ses-2*.nii') ) 

ses_1_msmall_nii = nb.load(ses_2_file[0]).get_fdata() 

import seaborn as sns

fs=12

# +
bbox_params = {'facecolor': 'lightblue', 'alpha': 0.7, 'pad': 4}
n_trs = 200
l_w = 1
ax1 = plt.subplot(511)
ax1.plot(ses_1_msmall_nii[:n_trs,30], lw=l_w,color='black')

ax2 = plt.subplot(512, sharex=ax1)
ax2.plot(ses_1_msmall_nii[:n_trs,21], lw=l_w,color='black')
ax3 = plt.subplot(513, sharex=ax1)
ax3.plot(ses_1_msmall_nii[:n_trs,41], lw=l_w,color='black')

ax4 = plt.subplot(514, sharex=ax1)
ax4.plot(ses_1_msmall_nii[:n_trs,81], lw=l_w,color='black')

ax5 = plt.subplot(515, sharex=ax1)
ax5.plot(ses_1_msmall_nii[:n_trs,90], lw=l_w,color='black')

for axii in [ax1, ax2, ax3, ax4, ax5]: 
    axii.set_xticklabels([])
    axii.set_yticklabels([]) 
    axii.set_xticks([])
    axii.set_yticks([])
    axii.patch.set_facecolor('none')

ax1.set_ylabel('a', rotation=0, bbox=bbox_params, fontsize=fs,  fontname="Liberation Serif")
ax2.set_ylabel('b', rotation=0, bbox=bbox_params, fontsize=fs,   fontname="Liberation Serif") 
ax3.set_ylabel('c', rotation=0, bbox=bbox_params, fontsize=fs,   fontname="Liberation Serif") 
ax4.set_ylabel('d', rotation=0, bbox=bbox_params, fontsize=fs,   fontname="Liberation Serif") 
ax5.set_ylabel('e', rotation=0, bbox=bbox_params, fontsize=fs,   fontname="Liberation Serif") 

ax1.yaxis.set_label_coords(.01,.6,  transform=None)
ax2.yaxis.set_label_coords(.01, .6, transform=None)
ax3.yaxis.set_label_coords(.01,.6, transform=None)
ax4.yaxis.set_label_coords(.01,.6, transform=None)
ax5.yaxis.set_label_coords(.01,.6, transform=None)

# ax1.set_label_coords(0, 0, transform=None)
plt.subplots_adjust(hspace=-.3, wspace=.1)
# plt.legend(frameon=False)
sns.despine(left=True, bottom=True, right=True)
plt.savefig('fix_timeseries.svg', transparent=True, format="svg") 
