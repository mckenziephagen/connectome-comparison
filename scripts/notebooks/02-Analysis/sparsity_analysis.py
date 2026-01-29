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

import matplotlib
import infomap

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
# -

results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'

date_string = '2025-12-08'
with open(glob(op.join(results_path, 'xcpd', f'{date_string}*lassoBIC*pkl'))[0],   'rb') as f:
        xcpd_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll', f'{date_string}*lasso*pkl'))[0], 'rb') as f:
        msmall_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll', '*9-15*correlation*pkl'))[0], 'rb') as f:
    pearson_dict = pickle.load(f)

# +
# pearson_mat = np.zeros((100,100,0))
# for participant in pearson_dict.keys():
#     for ses in pearson_dict[participant].keys():
#         temp_mat = pearson_dict[participant][ses]
#         np.fill_diagonal(temp_mat, 0)
#         pearson_mat = np.dstack((pearson_mat, temp_mat)  ) 

# +
msmall_mat = np.zeros((100,100, 0))

for participant in msmall_dict.keys():
        temp_mat = msmall_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        msmall_mat = np.dstack((msmall_mat, temp_mat)  ) 

# +
xcpd_mat = np.zeros((100,100, 0))

for participant in msmall_dict.keys():
    for ses in msmall_dict[participant].keys():
        temp_mat = xcpd_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        xcpd_mat = np.dstack((xcpd_mat, temp_mat)  ) 
# -

df = pd.DataFrame.from_dict(msmall_dict, orient='index')


xcpd_mean = np.mean(xcpd_mat, axis=2)
msmall_mean = np.mean(msmall_mat, axis=2)
# pearson_mean = np.mean(pearson_mat, axis=2) 

xcpd_std = np.std(xcpd_mat, axis=2)


# +
fix, ax = plt.subplots(1,2) 

plotting.plot_matrix(msmall_mean.reshape(100,100) , 
    vmin=-1, 
    vmax=1,
    reorder=False, 
    axes=ax[0], 
    title='MSMAll Average') 

plotting.plot_matrix(xcpd_mean.reshape(100,100) , 
    vmax=1,
    vmin=-1, 
    axes=ax[1], 
    title='XCPD Average')
plt.savefig('matrix.png') 
#add in 7 network 

# +
fig, ax = plt.subplots(1,2, figsize=(10, 5)) 

plotting.plot_matrix(msmall_mean , 
    vmin=-1, 
    vmax=1,
    reorder=False, 
    axes=ax[0], 
    title='Average LASSO Connectome')

plotting.plot_matrix(pearson_mean, 
    vmin=-1, 
    vmax=1,
    axes=ax[1], 
    title='Average Pearson Connectome')

plt.tight_layout()

plt.savefig('LASSO_Pearson_Matrices.png') 
plt.show() 


# -

def selection_ratio(model_dict):
    ratio_list = [] 
    for sub in model_dict.keys():
        sel_ratio = sum(sum(model_dict[sub]['ses-full'] != 0) )/ 10000
            ratio_list.append(sel_ratio)  
    return ratio_list


xcpd_sel_ratio = selection_ratio(xcpd_dict) 

msmall_sel_ratio = selection_ratio(msmall_dict) 

np.mean(msmall_sel_ratio)

np.mean(xcpd_sel_ratio) 

import seaborn as sns

import pandas as pd

# +
xcpd_df = pd.DataFrame({"proc": "xcpd", 
                          "sparsity": xcpd_sel_ratio}) 
msmall_df = pd.DataFrame({"proc": "msmall", 
                          "sparsity": msmall_sel_ratio})

plot_df = pd.concat([xcpd_df, msmall_df]) 
# -

np.mean(xcpd_sel_ratio) 

ax = sns.violinplot(data=plot_df, x="proc", y="sparsity") 
ax.set_ylim(0, 1.1) 
ax.set_xticklabels(labels=['XCPD', 'Minimally Processed']) 
plt.title('LASSO Connectome Sparsity') 
plt.ylabel('Selection Ratio') 
plt.xlabel('Post-processing Pipeline') 
plt.savefig('lasso_sparsity.png')
plt.tight_layout()
plt.show() 


# +
#### Everything under here is old 
# -







# +
bins = np.linspace(0, .7, 50)

plt.hist(uoi_edges_list, bins, alpha=.75, label = 'UoI', color=matplotlib.colors.to_rgba('#00313C')) 
plt.hist(lasso_edges_list, bins, alpha=.75, label = 'LASSO', color=matplotlib.colors.to_rgba('#007681')) 

# plt.vlines(np.mean(lasso_edges_list), ymin=0, ymax=100, color='black')
# plt.vlines(np.mean(uoi_edges_list), ymin=0, ymax=100, color='black')
plt.gca().set_aspect(1/250)
plt.legend(prop={'size': 16})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Number of Participants', size=14) 
plt.xlabel('Selection Ratio', size=14) 
plt.savefig('selection_ratio.png', bbox_inches="tight") 

# +
bins = np.linspace(0, .7, 50)

plt.hist(uoi_edges_list, bins, alpha=.75, label = 'UoI', color=matplotlib.colors.to_rgba('#00313C')) 
plt.hist(lasso_edges_list, bins, alpha=.75, label = 'LASSO', color=matplotlib.colors.to_rgba('#007681')) 
plt.hist(pearson_edges_list, alpha=.75, label = 'pearson') 

# plt.vlines(np.mean(lasso_edges_list), ymin=0, ymax=100, color='black')
# plt.vlines(np.mean(uoi_edges_list), ymin=0, ymax=100, color='black')
plt.gca().set_aspect(1/250)
plt.legend()
plt.savefig('selection_ratio.png', bbox_inches="tight") 
# -

plt.hist2d(pearson_edges_list, uoi_edges_list, bins=(100,100) ) 


np.mean(uoi_edges_list)

np.mean(lasso_edges_list)

plt.scatter(lasso_edges_list, uoi_edges_list) 
plt.axline((0, 0), slope=1)
plt.ylim(0,1)
plt.xlim(0,1)
#less variability across subjects

mean_out_lasso = list()
mod_lasso = list()
for outer in lasso_dict.keys():
    for inner in lasso_dict[outer].keys():
        graph = nx.DiGraph(lasso_dict[outer][inner])
        mean_out_lasso.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_lasso.append(nx.community.modularity(graph, c))

mean_out_uoi = list()
mod_uoi = list()
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        graph = nx.DiGraph(uoi_dict[outer][inner])
        mean_out_uoi.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_uoi.append(nx.community.modularity(graph, c))

mean_out_uoi = list()
mod_uoi = list()
for outer in uoi_dict.keys():
    for inner in uoi_dict[outer].keys():
        graph = nx.DiGraph(uoi_dict[outer][inner])
        mean_out_uoi.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_uoi.append(nx.community.modularity(graph, c))

mean_out_pearson = list()
mod_pearson = list()
for outer in pearson_dict.keys():
    for inner in pearson_dict[outer].keys():
        graph = nx.DiGraph(pearson_dict[outer][inner])
        mean_out_pearson.append(np.mean([val for (node, val) in graph.out_degree()]))
        c = nx.community.greedy_modularity_communities(graph)
        mod_pearson.append(nx.community.modularity(graph, c))

# +
plt.hist(mean_out_lasso, color='red', label='LASSO')
plt.hist(mean_out_uoi, label = 'UoI')

plt.legend()
plt.title('Average Out Degree Across Nodes') 
plt.ylabel('Count')
plt.xlabel('Out Degree')

# +
plt.hist(mod_lasso, color='red', label='LASSO')
plt.hist(mod_uoi, label='UoI')

plt.legend()
plt.title('Modularity Index') 
# -





# +








