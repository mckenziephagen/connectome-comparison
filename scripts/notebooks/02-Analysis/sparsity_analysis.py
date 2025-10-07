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

date_string = '2025-09-10'
with open(glob(op.join(results_path, 'xcpd', f'{date_string}*lassoBIC*pkl'))[0],   'rb') as f:
        xcpd_dict = pickle.load(f)

len(xcpd_dict) 

date_string='2025-08-26'
with open(glob(op.join(results_path, 'MSMAll', f'{date_string}*lasso*pkl'))[0], 'rb') as f:
        msmall_dict = pickle.load(f)

len(msmall_dict) 

# +
msmall_mat = np.zeros((100,100, 0))

for participant in msmall_dict.keys():
    for ses in msmall_dict[participant].keys():
        temp_mat = msmall_dict[participant][ses]
        np.fill_diagonal(temp_mat, 0)
        msmall_mat = np.dstack((msmall_mat, temp_mat)  ) 

# +
xcpd_mat = np.zeros((100,100, 0))

for participant in msmall_dict.keys():
    for ses in msmall_dict[participant].keys():
        temp_mat = xcpd_dict[participant][ses]
        np.fill_diagonal(temp_mat, 0)
        xcpd_mat = np.dstack((xcpd_mat, temp_mat)  ) 
# -

xcpd_mat.shape

msmall_mat.shape

xcpd_mean = np.mean(xcpd_mat, axis=2)
msmall_mean = np.mean(msmall_mat, axis=2)

xcpd_mtd = np.std(xcpd_mat, axis=2)


# +
ax = plotting.plot_matrix(xcpd_mtd.reshape(100,100) , 
    reorder=False) 

#ax.set_xticklabels([])
# plotting.plot_matrix(lasso_median_list.reshape(100,100) , 
#     vmax=.5,
#     vmin=-0.5, 
#     reorder=False)
plt.savefig('matrix.png') 
#add in 7 network 

# + jupyter={"source_hidden": true}
xcpd_edges_list = []
xcpd_ratio = []
xcpd_sparsity = [] 
for outer in msmall_dict.keys():
    for inner in msmall_dict[outer].keys():
        xcpd_edges_list.append(xcpd_dict[outer][inner].ravel()) 

        xcpd_sparsity.append(sum(xcpd_dict[outer][inner].ravel()  != 0 )  / 10000) 
        
        xcpd_ratio.append(sum(sum(xcpd_dict[outer][inner] < 0)) / sum(sum(xcpd_dict[outer][inner] > 0 )))
# -

np.mean(xcpd_sparsity)

msmall_edges_list = []
msmall_ratio = []
msmall_sparsity = [] 
for outer in msmall_dict.keys():
    for inner in msmall_dict[outer].keys():
        msmall_edges_list.append(msmall_dict[outer][inner].ravel()) 

        msmall_sparsity.append(sum(msmall_dict[outer][inner].ravel()  != 0 )  / 10000) 

        msmall_ratio.append(sum(sum(msmall_dict[outer][inner] < 0)) / sum(sum(msmall_dict[outer][inner] > 0 )))

np.mean(msmall_sparsity)

len(msmall_sparsity) 

len(xcpd_sparsity) 

import seaborn as sns

np.corrcoef(xcpd_sparsity, msmall_sparsity)

plt.scatter(xcpd_sparsity, msmall_sparsity, alpha=.1) 
#correlate participant sparsity and participant qcfc
plt.xlim([.2,.9])
plt.ylim([.2,.9]) 
#put ridge and uoi here too? 

import pandas as pd

# +
xcpd_df = pd.DataFrame({"proc": "xcpd", 
                          "sparsity": xcpd_sparsity}) 
msmall_df = pd.DataFrame({"proc": "msmall", 
                          "sparsity": msmall_sparsity})

plot_df = pd.concat([xcpd_df, msmall_df]) 
# -

ax = sns.swarmplot(data=plot_df, x="proc", y="sparsity", size=2) 
ax.set_ylim(0, 1.1) 

sns.swarmplot(data=xcpd_df) 

ax = sns.boxplot(data=plot_df, x="proc", y="sparsity") 
ax.set_ylim(0, 1.1)

import networkx

from infomap import Infomap

xcpd_G = networkx.Graph(xcpd_mean)

im_xcpd = Infomap(flow_model="directed") 

_ = im_xcpd.add_networkx_graph(xcpd_G)


im_xcpd.run()

print(f"Found {im_xcpd.num_top_modules} modules with codelength: {im_xcpd.codelength}")


msmall_G = networkx.Graph(msmall_mean)

im_msmall = Infomap(flow_model="directed") 

_ = im_msmall.add_networkx_graph(msmall_G)


im_msmall.run()

print(f"Found {im_msmall.num_top_modules} modules with codelength: {im_msmall.codelength}")












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




