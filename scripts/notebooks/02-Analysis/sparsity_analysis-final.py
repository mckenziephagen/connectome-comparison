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

# Visualize and compare sparsities for different models. 

# +
import matplotlib as mpl
import matplotlib.patches as patches

from cmap import Colormap
import scipy


# +
import pickle
import os
import os.path as op

from glob import glob

from nilearn import datasets, plotting

import scipy
import numpy as np

import pandas as pd

from itertools import chain

import matplotlib.pyplot as plt
  
import networkx as nx

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import datetime
# -

results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'

from nilearn.plotting import (
    find_parcellation_cut_coords,
    plot_connectome,
    show,
)


def delete_ses(dictionary):
    for sub in dictionary.keys(): 
        del dictionary[sub]['ses-1']
        del dictionary[sub]['ses-2']
    return dictionary


# +
date_str = '2026-05-01'

with open(glob(op.join(results_path, 'MSMAll', f'{date_str}*lassoBIC*pkl'))[0], 'rb') as f:
        msmall_lasso_dict = pickle.load(f)
len(msmall_lasso_dict) 

with open(glob(op.join(results_path, 'MSMAll', f'{date_str}*corr*pkl'))[0], 'rb') as f:
        msmall_corr_dict = pickle.load(f)
len(msmall_corr_dict) 

date_str = '2026-05-22'

with open(glob(op.join(results_path, 'xcpd', f'{date_str}*lassoBIC*pkl'))[0],   'rb') as f:
        xcpd_lasso_dict = pickle.load(f)
len(xcpd_lasso_dict) 

with open(glob(op.join(results_path, 'xcpd', f'{date_str}*-correlation*pkl'))[0], 'rb') as f:
    xcpd_corr_dict = pickle.load(f)
    
with open(glob(op.join(results_path, 'xcpd', f'{date_str}*partial*pkl'))[0], 'rb') as f:
    xcpd_pcorr_dict = pickle.load(f)
    
with open(glob(op.join(results_path, 'xcpd', f'{date_str}*uoiLasso*pkl'))[0], 'rb') as f:
    xcpd_uoi_dict = pickle.load(f)

    
with open(glob(op.join(results_path, 'xcpd', f'*05-23*ridgeCV*pkl'))[0], 'rb') as f:
    xcpd_ridge_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'{date_str}*-correlation*pkl'))[0], 'rb') as f:
    fix_corr_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'{date_str}*uoiLasso*pkl'))[0], 'rb') as f:
    fix_uoi_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'*{date_str}*lassoBIC*pkl'))[0], 'rb') as f:
    fix_lasso_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll_FIX', f'*{date_str}*partial*pkl'))[0], 'rb') as f:
    fix_pcorr_dict = pickle.load(f)


with open(glob(op.join(results_path, 'MSMAll_FIX', f'*05-23*ridgeCV*pkl'))[0], 'rb') as f:
    fix_ridge_dict = pickle.load(f)

# fix_lasso_dict = delete_ses(fix_lasso_dict)
# fix_uoi_dict = delete_ses(fix_uoi_dict)
# xcpd_lasso_dict = delete_ses(xcpd_lasso_dict)
# xcpd_uoi_dict = delete_ses(xcpd_uoi_dict)
# xcpd_corr_dict = delete_ses(xcpd_corr_dict)
# fix_corr_dict = delete_ses(fix_corr_dict)
# fix_ridge_dict = delete_ses(fix_ridge_dict)
# -
# ### PNC 

# +
# with open(glob(op.join(results_path, 'xcpdPNC', f'*lasso*pkl'))[0], 'rb') as f:
#     pnc_lasso_dict = pickle.load(f)

# for sub in pnc_lasso_dict.keys(): 
#     del pnc_lasso_dict[sub]['ses-2']
#     del pnc_lasso_dict[sub]['ses-full']


# +
# for sub in pnc_lasso_dict.keys(): 
#     pnc_lasso_dict[sub]['ses-median'] = np.median([pnc_lasso_dict[sub]['ses-1'], pnc_lasso_dict[sub]['ses-2']], axis=0) 
# pnc_lasso_dict = delete_ses(pnc_lasso_dict)


# +
# pnc_lasso_mat = np.zeros((100,100, 0))

# for participant in pnc_lasso_dict.keys():
#     # for ses in msmall_lasso_dict[participant].keys():
#         temp_mat = pnc_lasso_dict[participant]['ses-1']
#         np.fill_diagonal(temp_mat, 0)
#         pnc_lasso_mat = np.dstack((pnc_lasso_mat, temp_mat)  ) 

# -

# ### HCP

# +
msmall_lasso_mat = np.zeros((100,100, 0))

for participant in msmall_lasso_dict.keys():
    temp_mat = msmall_lasso_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    msmall_lasso_mat = np.dstack((msmall_lasso_mat, temp_mat)  ) 

msmall_corr_mat = np.zeros((100,100, 0))

for participant in msmall_corr_dict.keys():
    temp_mat = msmall_corr_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    msmall_corr_mat = np.dstack((msmall_corr_mat, temp_mat)  ) 

# +
fix_ridge_mat = np.zeros((100,100, 0))

for participant in fix_ridge_dict.keys():
    temp_mat = fix_ridge_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    fix_ridge_mat = np.dstack((fix_ridge_mat, temp_mat)  ) 

xcpd_ridge_mat = np.zeros((100,100, 0))

for participant in xcpd_ridge_dict.keys():
    temp_mat = xcpd_ridge_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    xcpd_ridge_mat = np.dstack((xcpd_ridge_mat, temp_mat)  ) 

# +
fix_lasso_mat = np.zeros((100,100, 0))

for participant in fix_lasso_dict.keys():
    temp_mat = fix_lasso_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    fix_lasso_mat = np.dstack((fix_lasso_mat, temp_mat)  ) 

xcpd_lasso_mat = np.zeros((100,100, 0))

for participant in xcpd_lasso_dict.keys():
    temp_mat = xcpd_lasso_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    xcpd_lasso_mat = np.dstack((xcpd_lasso_mat, temp_mat)  ) 

# +
xcpd_uoi_mat = np.zeros((100,100, 0))

for participant in xcpd_uoi_dict.keys():
    temp_mat = xcpd_uoi_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    xcpd_uoi_mat = np.dstack((xcpd_uoi_mat, temp_mat)  ) 

fix_uoi_mat = np.zeros((100,100, 0))

for participant in fix_uoi_dict.keys():
    temp_mat = fix_uoi_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    fix_uoi_mat = np.dstack((fix_uoi_mat, temp_mat)  ) 

# +
xcpd_corr_mat = np.zeros((100,100, 0))

for participant in xcpd_corr_dict.keys():
    temp_mat = xcpd_corr_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    xcpd_corr_mat = np.dstack((xcpd_corr_mat, temp_mat)  ) 

fix_corr_mat = np.zeros((100,100, 0))
for participant in fix_corr_dict.keys():
    temp_mat = fix_corr_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    fix_corr_mat = np.dstack((fix_corr_mat, temp_mat)  )  
# -

fix_pcorr_mat = np.zeros((100,100, 0))
for participant in fix_pcorr_dict.keys():
    temp_mat = fix_pcorr_dict[participant]['ses-full']
    np.fill_diagonal(temp_mat, 0)
    fix_pcorr_mat = np.dstack((fix_pcorr_mat, temp_mat)  )  

# +
xcpd_lasso_mean = np.mean(xcpd_lasso_mat, axis=2)
fix_lasso_mean = np.mean(fix_lasso_mat, axis=2)

xcpd_ridge_mean = np.mean(xcpd_ridge_mat, axis=2)
fix_ridge_mean = np.mean(fix_ridge_mat, axis=2)

xcpd_uoi_mean = np.mean(xcpd_uoi_mat, axis=2)
fix_uoi_mean = np.mean(fix_uoi_mat, axis=2)

msmall_lasso_mean = np.mean(msmall_lasso_mat, axis=2)
msmall_corr_mean = np.mean(msmall_corr_mat, axis=2)


fix_pcorr_mean = np.mean(fix_pcorr_mat, axis=2)

# -

fix_pcorr_mean = np.mean(fix_pcorr_mat, axis=2)


np.savetxt(f'results/{str(datetime.date.today())}_fix_pcorr_mean.txt', fix_pcorr_mean)


xcpd_corr_mean = np.mean(xcpd_corr_mat, axis=2)
fix_corr_mean = np.mean(fix_corr_mat, axis=2)

# +
np.savetxt(f'results/{str(datetime.date.today())}_fix_lasso_mean.txt', fix_lasso_mean)
np.savetxt(f'results/{str(datetime.date.today())}_msmall_lasso_mean.txt', msmall_lasso_mean)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_lasso_mean.txt', xcpd_lasso_mean)

np.savetxt(f'results/{str(datetime.date.today())}_fix_ridge_mean.txt', fix_ridge_mean)
# np.savetxt('msmall_ridge_mean.txt', msmall_ridge_mean)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_ridge_mean.txt', xcpd_ridge_mean)

np.savetxt(f'results/{str(datetime.date.today())}_fix_uoi_mean.txt', fix_uoi_mean)
# np.savetxt('msmall_uoi_mean.txt', msmall_uoi_mean)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_uoi_mean.txt', xcpd_uoi_mean)

np.savetxt(f'results/{str(datetime.date.today())}_fix_corr_mean.txt', fix_corr_mean)
np.savetxt(f'results/{str(datetime.date.today())}_msmall_corr_mean.txt', msmall_corr_mean)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_corr_mean.txt', xcpd_corr_mean)

# +
xcpd_lasso_median = np.median(xcpd_lasso_mat, axis=2)
fix_lasso_median = np.median(fix_lasso_mat, axis=2)

xcpd_uoi_median = np.median(xcpd_uoi_mat, axis=2)
fix_uoi_median = np.median(fix_uoi_mat, axis=2)

xcpd_ridge_median = np.median(xcpd_ridge_mat, axis=2)
fix_ridge_median = np.median(fix_ridge_mat, axis=2)

xcpd_corr_median = np.median(xcpd_corr_mat, axis=2)
fix_corr_median = np.median(fix_corr_mat, axis=2)

# +
np.savetxt(f'results/{str(datetime.date.today())}_fix_lasso_median.txt', fix_lasso_median)
# np.savetxt(f'results/{str(datetime.date.today())}_msmall_lasso_median.txt', msmall_lasso_median)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_lasso_median.txt', xcpd_lasso_median)

np.savetxt(f'results/{str(datetime.date.today())}_fix_ridge_median.txt', fix_ridge_median)
# np.savetxt('msmall_ridge_median.txt', msmall_ridge_median)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_ridge_median.txt', xcpd_ridge_median)

np.savetxt(f'results/{str(datetime.date.today())}_fix_uoi_median.txt', fix_uoi_median)
# np.savetxt('msmall_uoi_median.txt', msmall_uoi_median)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_uoi_median.txt', xcpd_uoi_median)

np.savetxt(f'results/{str(datetime.date.today())}_fix_corr_median.txt', fix_corr_median)
# np.savetxt(f'results/{str(datetime.date.today())}_msmall_corr_median.txt', msmall_corr_median)
np.savetxt(f'results/{str(datetime.date.today())}_xcpd_corr_median.txt', xcpd_corr_median)
# -

xcpd_lasso_stdev = np.std(xcpd_lasso_mat, axis=2)
xcpd_uoi_stdev = np.std(xcpd_uoi_mat, axis=2)
xcpd_corr_stdev = np.std(xcpd_corr_mat, axis=2)

# +
fix_lasso_stdev = np.std(fix_lasso_mat, axis=2)

fix_uoi_stdev = np.std(fix_uoi_mat, axis=2)
fix_corr_stdev = np.std(fix_corr_mat, axis=2)
# -
stdev_dict = {'LASSO': fix_lasso_stdev, 
             'UoI': fix_uoi_stdev, 
             'Pearson': fix_corr_stdev}

with open('results/stdev_pickle.pkl', 'wb') as f: 
    pickle.dump(stdev_dict, f )

np.mean(fix_lasso_stdev) 

np.mean(xcpd_lasso_stdev) 

np.mean(fix_corr_stdev) 

np.mean(xcpd_corr_stdev) 

np.mean(fix_uoi_stdev)

np.mean(xcpd_uoi_stdev)

# Max edge standard deviations: 

np.max(fix_lasso_stdev) 

np.max(xcpd_lasso_stdev) 

np.max(fix_corr_stdev) 

np.max(xcpd_corr_stdev) 

# Mean edge standard deviations: 

np.mean(fix_lasso_stdev) 

np.mean(xcpd_lasso_stdev) 

np.mean(fix_corr_stdev) 

np.mean(xcpd_corr_stdev) 

np.min(fix_lasso_stdev) 


np.min(xcpd_lasso_stdev) 

np.max(xcpd_lasso_stdev) 

np.max(xcpd_corr_stdev) 

np.corrcoef([fix_corr_stdev.ravel(), fix_lasso_stdev.ravel()]) 

np.corrcoef(fix_corr_mean.ravel(), fix_lasso_mean.ravel())

np.corrcoef(fix_corr_mean.ravel(), fix_uoi_mean.ravel())

np.corrcoef(xcpd_corr_mean.ravel(), xcpd_lasso_mean.ravel())

np.corrcoef(xcpd_corr_mean.ravel(), xcpd_uoi_mean.ravel())

plt.scatter(fix_corr_stdev.ravel()[:4950], fix_lasso_stdev.ravel()[:4950]) 
plt.ylabel('LASSO')
plt.xlabel('Pearson') 
plt.ylim(0,.23) 
plt.xlim(0,.23) 

np.corrcoef([xcpd_corr_stdev.ravel(), xcpd_lasso_stdev.ravel()]) 

# +
plt.scatter(xcpd_corr_stdev.ravel()[:4950], xcpd_lasso_stdev.ravel()[:4950]) 

plt.ylabel('LASSO')
plt.xlabel('Pearson') 
plt.ylim(0,.23) 
plt.xlim(0,.23) 

# +
fig, ax = plt.subplots(2, 2, figsize=[20,20]) 

plotting.plot_matrix(xcpd_uoi_mean.reshape(100,100) - fix_uoi_mean.reshape(100,100), 
    vmin=-.2, 
    vmax=.2,
    reorder=False, 
    axes=ax[0,0])
    # labels = output_list)

plotting.plot_matrix(xcpd_lasso_mean.reshape(100,100) - fix_lasso_mean.reshape(100,100), 
    vmin=-.2, 
    vmax=.2,
    reorder=False, 
    axes=ax[0,1])
    # labels = output_list)

plotting.plot_matrix(xcpd_uoi_mean.reshape(100,100) - xcpd_lasso_mean.reshape(100,100), 
    # vmin=-.2, 
    # vmax=.2,
    reorder=False, 
    axes=ax[1,0])
    # labels = output_list)

plotting.plot_matrix(fix_uoi_mean.reshape(100,100) - fix_lasso_mean.reshape(100,100), 
    # vmin=-.2, 
    # vmax=.2,
    reorder=False, 
    axes=ax[1,1])

# plotting.plot_matrix(msmall_lasso_mean.reshape(100,100) , 
#     vmax=1,
#     vmin=-1, 
#     axes=ax[1])
# plotting.plot_matrix(xcpd_uoi_mean.reshape(100,100) , 
#     vmax=1,
#     vmin=-1, 
#     axes=ax[2], 
#     title='UoI XCPD')


# plotting.plot_matrix(msmall_uoi_mean.reshape(100,100) , 
#     vmax=1,
#     vmin=-1, 
#     axes=ax[3], 
#     title='UoI MSMALL')
plt.savefig('matrix.pdf') 
#add in 7 network 
# -

def selection_ratio(model_dict):
    ratio_list = [] 
    for sub in model_dict.keys():
        # for ses in model_dict[sub].keys():
            np.fill_diagonal(model_dict[sub]['ses-full'], 1)
            sel_ratio = sum(sum(model_dict[sub]['ses-full'] != 0 )) / 10000
            ratio_list.append(sel_ratio)  
    return ratio_list


xcpd_lasso_ratio = selection_ratio(xcpd_lasso_dict) 
xcpd_uoi_ratio = selection_ratio(xcpd_uoi_dict) 

# +
fix_lasso_ratio = selection_ratio(fix_lasso_dict) 
# pnc_lasso_ratio = selection_ratio(pnc_lasso_dict) 

xcpd_uoi_ratio = selection_ratio(xcpd_uoi_dict) 
fix_uoi_ratio = selection_ratio(fix_uoi_dict) 

xcpd_ridge_ratio = selection_ratio(xcpd_ridge_dict) 
fix_ridge_ratio = selection_ratio(fix_ridge_dict) 
# -

fix_pcorr_ratio = selection_ratio(fix_pcorr_dict) 
xcpd_pcorr_ratio = selection_ratio(xcpd_pcorr_dict) 

fix_pearson_ratio = selection_ratio(fix_corr_dict) 
xcpd_pearson_ratio = selection_ratio(xcpd_corr_dict) 

plt.hist(xcpd_lasso_ratio) 

np.mean(xcpd_lasso_ratio) 

np.mean(fix_lasso_ratio) 

np.mean(xcpd_lasso_ratio)

np.mean(xcpd_uoi_ratio) 

np.mean(fix_ridge_ratio) 

# +
xcpd_lasso_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "lasso",
                          "sparsity": xcpd_lasso_ratio}) 
fix_lasso_plot_df = pd.DataFrame({"proc": "fix", 
                            "model": "lasso",
                          "sparsity": fix_lasso_ratio})

xcpd_uoi_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "uoi",

                          "sparsity": xcpd_uoi_ratio}) 

fix_uoi_plot_df = pd.DataFrame({"proc": "fix", 
                            "model": "uoi",
                          "sparsity": fix_uoi_ratio})

fix_ridge_plot_df = pd.DataFrame({"proc": "fix", 
                            "model": "Ridge",
                          "sparsity": fix_ridge_ratio})


fix_pcorr_plot_df = pd.DataFrame({"proc": "fix", 
                            "model": "PCorr",
                          "sparsity": fix_pcorr_ratio})

xcpd_pcorr_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "PCorr",
                          "sparsity": xcpd_pcorr_ratio})

xcpd_ridge_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "Ridge",
                          "sparsity": xcpd_ridge_ratio})

fix_corr_plot_df = pd.DataFrame({"proc": "fix", 
                            "model": "Pearson",
                          "sparsity": fix_pearson_ratio})

xcpd_corr_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "Pearson",
                          "sparsity": xcpd_pearson_ratio})
# pnc_lasso_plot_df = pd.DataFrame({"proc": "pnc", 
#                             "model": "lasso",
#                           "sparsity": pnc_lasso_ratio})

hcp_sparsity_plot_df = pd.concat([fix_lasso_plot_df, 
                              fix_uoi_plot_df, 
                              xcpd_lasso_plot_df, 
                              xcpd_uoi_plot_df,
                                fix_ridge_plot_df, 
                                 xcpd_ridge_plot_df, 
                                 fix_pcorr_plot_df, 
                                 xcpd_corr_plot_df, 
                                 fix_corr_plot_df]) 

# +
hcp_sparsity_plot_df = hcp_sparsity_plot_df.replace(to_replace='msmall', value='MinProc')
hcp_sparsity_plot_df = hcp_sparsity_plot_df.replace(to_replace='xcpd', value='XCPD')


hcp_sparsity_plot_df = hcp_sparsity_plot_df.replace(to_replace='uoi', value='UoI')
hcp_sparsity_plot_df = hcp_sparsity_plot_df.replace(to_replace='lasso', value='LASSO')

hcp_sparsity_plot_df = hcp_sparsity_plot_df.replace(to_replace='pnc', value='PNC')
# -

import datetime

hcp_sparsity_plot_df.to_csv(f'results/{str(datetime.date.today())}_sparsity_plot_df.csv') 

# +
import tol_colors as tc
cm =  tc.muted

models = ['Pearson',  'LASSO', 'UoI', ]
pipelines = ['MinProc', 'XCPD']
labels = models + pipelines
palette_colors = cm
palette_dict = {labels: color for labels, color in zip(labels, palette_colors)}
models.remove('Pearson')
pipelines.remove('XCPD')

# +
# palette_dict = {'Pearson': '#4477AA',
#                  'LASSO': '#EE6677',
#                  'UoI': '#228833',
#                  'MinProc': '#CCBB44',
#                  'XCPD': '#66CCEE'}
# -

sparsity_plot_df = sparsity_plot_df[sparsity_plot_df.loc[:,'proc'] == 'MinProc']

# +
plt.figure(figsize=(3, 5))

ax =  sns.violinplot(data = sparsity_plot_df,
                          x="proc",
                          y="sparsity", 
                          hue="model", 
                          hue_order= ['LASSO', 'UoI'],
                          legend=False,
                           order = pipelines, 
                          palette = [palette_dict.get(key) for key in models], 
                          inner='box',
                    linecolor='black', 
                    alpha=.9) 

ax.set_ylim(0, 1) 
# ax.legend(title='') 

ax.set_xticks([])
plt.grid(axis='y', linewidth=.25)
# ax.set_yticks(ticks=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]) 
# ax.fig.set_size_inches(15,15)
ax.set_xlabel('')
# ax.get_xticks().set_xticklables(labels=['XCPD', 'MinProc']) 
ax.set_ylabel('Selection Ratio', weight='semibold')
sns.despine()
# plt.title('Sparsity', weight = 'bold') 
plt.savefig('../03-Plotting/plots/2026-04-27_sparsity.png',
             bbox_inches='tight') 
plt.show() 
# +
##### BOTH BOTH BOTH
plt.figure(figsize=(5, 5))

ax =  sns.violinplot(data = sparsity_plot_df,
                          x="proc",
                          y="sparsity", 
                          hue="model", 
                          hue_order= ['LASSO', 'UoI'],
                          legend=False,
                           order = pipelines, 
                          palette = [palette_dict.get(key) for key in models], 
                          inner='box') 

# ax = sns.swarmplot(data = sparsity_plot_df,
#                           x="proc",
#                           y="sparsity", 
#                           hue="model", 
#                           hue_order= ['LASSO', 'UoI'],
#                                         order = pipelines,
#                    size=3.7, 
#                             dodge=True,
#                                        palette = [palette_dict.get(key) for key in models], 
# )

# ax = sns.boxplot(data = sparsity_plot_df,
#                           x="proc",
#                           y="sparsity", 
#                           hue="model", 
#                           hue_order= ['LASSO', 'UoI'],
#                                         order = pipelines,
#                             dodge=True,
#                                        palette = [palette_dict.get(key) for key in models], 
#                  ax=ax
# )

ax.set_ylim(0, 1) 



plt.grid(axis='y', linewidth=.25)
# ax.set_yticks(ticks=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]) 
# ax.fig.set_size_inches(15,15)
ax.set_xlabel('Processing Pipeline', weight='semibold')
# ax.get_xticks().set_xticklables(labels=['XCPD', 'MinProc']) 
ax.set_ylabel('Selection Ratio', weight='semibold')
sns.despine()
plt.title('Sparsity of Model-based Connectomes', weight = 'bold') 
plt.savefig('plots/2025-12-12_sparsity.png',
             bbox_inches='tight', 
            transparent=True)
plt.show() 
# -






































































