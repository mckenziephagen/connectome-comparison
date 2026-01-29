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

import pandas as pd

import seaborn as sns
# -

results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'


def delete_ses(dictionary):
    for sub in dictionary.keys(): 
        del dictionary[sub]['ses-1']
        del dictionary[sub]['ses-2']
    return dictionary


# +
date_string = '2025-12-10'
with open(glob(op.join(results_path, 'xcpd', f'{date_string}*lassoBIC*pkl'))[0],   'rb') as f:
        xcpd_lasso_dict = pickle.load(f)
len(xcpd_lasso_dict) 

with open(glob(op.join(results_path, 'MSMAll', f'{date_string}*lassoBIC*pkl'))[0], 'rb') as f:
        msmall_lasso_dict = pickle.load(f)
len(msmall_lasso_dict) 

with open(glob(op.join(results_path, 'xcpd', f'{date_string}*uoiLasso*pkl'))[0], 'rb') as f:
    xcpd_uoi_dict = pickle.load(f)

with open(glob(op.join(results_path, 'MSMAll', f'{date_string}*uoiLasso*pkl'))[0], 'rb') as f:
    msmall_uoi_dict = pickle.load(f)

msmall_lasso_dict = delete_ses(msmall_lasso_dict)
msmall_uoi_dict = delete_ses(msmall_uoi_dict)
xcpd_lasso_dict = delete_ses(xcpd_lasso_dict)
xcpd_uoi_dict = delete_ses(xcpd_uoi_dict)
# +
msmall_lasso_mat = np.zeros((100,100, 0))

for participant in msmall_lasso_dict.keys():
    # for ses in msmall_lasso_dict[participant].keys():
        temp_mat = msmall_lasso_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        msmall_lasso_mat = np.dstack((msmall_lasso_mat, temp_mat)  ) 
# -

msmall_lasso_mat.shape

# +
xcpd_lasso_mat = np.zeros((100,100, 0))

for participant in xcpd_lasso_dict.keys():
    # for ses in msmall_dict[participant].keys():
        temp_mat = xcpd_lasso_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        xcpd_lasso_mat = np.dstack((xcpd_lasso_mat, temp_mat)  ) 

# +
xcpd_uoi_mat = np.zeros((100,100, 0))

for participant in xcpd_uoi_dict.keys():
    # for ses in msmall_dict[participant].keys():
        temp_mat = xcpd_uoi_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        xcpd_uoi_mat = np.dstack((xcpd_uoi_mat, temp_mat)  ) 

msmall_uoi_mat = np.zeros((100,100, 0))

for participant in msmall_uoi_dict.keys():
    # for ses in msmall_dict[participant].keys():
        temp_mat = msmall_uoi_dict[participant]['ses-full']
        np.fill_diagonal(temp_mat, 0)
        msmall_uoi_mat = np.dstack((msmall_uoi_mat, temp_mat)  ) 
# -

xcpd_lasso_mean = np.mean(xcpd_lasso_mat, axis=2)
msmall_lasso_mean = np.mean(msmall_lasso_mat, axis=2)
xcpd_uoi_mean = np.mean(xcpd_uoi_mat, axis=2)
msmall_uoi_mean = np.mean(xcpd_uoi_mat, axis=2)

np.savetxt('msmall_lasso_mean.txt', msmall_lasso_mean)

xcpd_lasso_median = np.median(xcpd_lasso_mat, axis=2)
msmall_lasso_median = np.median(msmall_lasso_mat, axis=2)
xcpd_uoi_median = np.median(xcpd_uoi_mat, axis=2)
msmall_uoi_median = np.median(xcpd_uoi_mat, axis=2)

# +
_ = plt.hist(xcpd_lasso_median.ravel(), label='xcpd_lasso')
_ = plt.hist(xcpd_uoi_median.ravel(), label='xcpd_uoi')

_ = plt.hist(msmall_lasso_median.ravel(), label='msmall_lasso')
_ = plt.hist(msmall_uoi_median.ravel(), label='msmall_uoi')
plt.legend()

plt.savefig('Edge weights')

# +
_ = plt.hist(xcpd_lasso_mean.ravel(), label='xcpd_lasso')
_ = plt.hist(xcpd_uoi_mean.ravel(), label='xcpd_uoi')

_ = plt.hist(msmall_lasso_mean.ravel(), label='msmall_lasso')
_ = plt.hist(msmall_uoi_mean.ravel(), label='msmall_uoi')
plt.legend()

plt.savefig('Edge weights') 
# -

schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100) 
trunc_labels = ['_'.join(ii.split('_')[1:3]) for ii in schaefer['labels'][1:]]

seen = set()
output_list = []
for item in trunc_labels:
    if item not in seen:
        output_list.append(item)
        seen.add(item)
    else:
        output_list.append("")




# +
fig, ax = plt.subplots(1, 1, figsize=[10,40]) 

plotting.plot_matrix(xcpd_lasso_mean.reshape(100,100) , 
    vmin=-1, 
    vmax=1,
    reorder=False, 
    axes=ax, 
    labels = output_list)


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

# +
from nilearn.plotting import (
    find_parcellation_cut_coords,
    plot_connectome,
    show,
)

# grab center coordinates for atlas labels
coordinates = find_parcellation_cut_coords(labels_img=schaefer["maps"])


# +

plot_connectome(
    msmall_lasso_mean.reshape(100,100), coordinates, edge_threshold="99%"
)


# +
# plotting.plot_matrix(xcpd_lasso_mean.reshape(100,100) , 
#     vmin=-1, 
#     vmax=1,
#     figure=(10, 8), 
#     labels=trunc_labels)

# plt.suptitle('Model-based Connectome', fontweight='bold') 
# plt.savefig('plots/exemplar_connectome.png')
# -

def selection_ratio(model_dict):
    ratio_list = [] 
    for sub in model_dict.keys():
        for ses in model_dict[sub].keys():
            sel_ratio = sum(sum(model_dict[sub][ses] != 0) )/ 10000
            ratio_list.append(sel_ratio)  
    return ratio_list


xcpd_lasso_ratio = selection_ratio(xcpd_lasso_dict) 

msmall_lasso_ratio = selection_ratio(msmall_lasso_dict) 
xcpd_uoi_ratio = selection_ratio(xcpd_uoi_dict) 
msmall_uoi_ratio = selection_ratio(msmall_uoi_dict) 

len(msmall_lasso_ratio) 

np.mean(xcpd_lasso_ratio)

np.mean(xcpd_uoi_ratio) 

np.mean(msmall_uoi_ratio)

np.mean(msmall_lasso_ratio) 

# +
xcpd_lasso_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "lasso",
                          "sparsity": xcpd_lasso_ratio}) 
msmall_lasso_plot_df = pd.DataFrame({"proc": "msmall", 
                            "model": "lasso",
                          "sparsity": msmall_lasso_ratio})

xcpd_uoi_plot_df = pd.DataFrame({"proc": "xcpd", 
                            "model": "uoi",

                          "sparsity": xcpd_uoi_ratio}) 

msmall_uoi_plot_df = pd.DataFrame({"proc": "msmall", 
                            "model": "uoi",
                          "sparsity": msmall_uoi_ratio})

sparsity_plot_df = pd.concat([xcpd_lasso_plot_df, msmall_lasso_plot_df, 
                    xcpd_uoi_plot_df, msmall_uoi_plot_df]) 
# -

sparsity_plot_df

palette_dict= {'Pearson': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 'UoI': (1.0, 0.4980392156862745, 0.054901960784313725),
 'LASSO': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)}

# +
sparsity_plot_df = sparsity_plot_df.replace(to_replace='msmall', value='MinProc')
sparsity_plot_df = sparsity_plot_df.replace(to_replace='xcpd', value='XCPD')

sparsity_plot_df = sparsity_plot_df.replace(to_replace='uoi', value='UoI')
sparsity_plot_df = sparsity_plot_df.replace(to_replace='lasso', value='LASSO')
# -

# ?ax.set_xlabel

matplotlib.font_manager.get_font_names()


plt.rcParams["font.size"] = 10

# +
fig,  ax = plt.subplots()


g = sns.violinplot(data = sparsity_plot_df,
                          x="proc",
                          y="sparsity", 
                          hue="model", 
                          # legend = 'full',
                          palette = palette_dict,
                          ax=ax, 
                          gap = -.5)

ax.invert_xaxis()

ax.set_ylim(0, 1.1) 
ax.figure.set_size_inches(4,5) 
ax.legend(
    fontsize=12, 
    labelspacing=.6,
    # Increase vertical space between items
)
ax.legend(loc='upper left', 
           title="Model")


# ax.fig.set_size_inches(15,15)
ax.set_xlabel('Processing Pipeline', weight='semibold')
# ax.get_xticks().set_xticklables(labels=['XCPD', 'MinProc']) 
ax.set_ylabel('Selection Ratio', weight='semibold')
plt.title('Sparsity of Model-based Connectomes', weight = 'bold') 
plt.savefig('plots/2025-12-12_sparsity.png',
             bbox_inches='tight', 
            transparent=True)
# -



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




















