# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: fc
#     language: python
#     name: python3
# ---

# %% [markdown]
# Plots figure 2A, and unordered supplementary figure

# %%
import os.path as op
import os 
import configparser

current_dir = os.getcwd()
base_dir = current_dir.split('scripts')[0]
script_dir = op.join(base_dir , 'scripts')
results_dir = op.join(base_dir, 'results') 
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')
config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )

# %%
from _plot_utils import models, cm, whiten, palette_dict, wrap_labels, name_dict


# %%
import matplotlib as mpl
import matplotlib.patches as patches

# %%
from nilearn import datasets, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tol_colors as tc

import datetime

# %%
from nilearn.plotting import (
    find_parcellation_cut_coords,
    plot_connectome,
    show,
)

# %%
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, ) 
trunc_labels = ['_'.join(ii.split('_')[1:3]) for ii in schaefer['labels'][1:]]

seen = set()
seen_list = []
output_list = []
for item in trunc_labels:
    if item not in seen:
        output_list.append(item)
        seen.add(item)
        seen_list.append(item)
    else:
        output_list.append("")

unique_labels = np.unique(trunc_labels, return_index=True, return_counts=True)

label_df = pd.DataFrame(unique_labels, index=['labels', 'unique_index', 'count']).T

colors = schaefer['lut'].iloc[label_df['unique_index'].values +1].sort_index()
colors.index = seen_list
col_df = colors

full_col_df = pd.concat([label_df.set_index('labels'), col_df], axis=1).drop('unique_index',axis=1)

# %%

fix_pearson_mean = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-Pearson_stat-mean_connectome.txt')) 
fix_pcorr_mean = np.loadtxt(op.join(results_dir, 'MSMAll-FIX','proc-FIX_model-partial-correlation_stat-mean_connectome.txt')) 

fix_lasso_mean = np.loadtxt(op.join(results_dir, 'MSMAll-FIX','proc-FIX_model-lassoBIC_stat-mean_connectome.txt')) 
fix_ridge_mean = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_stat-mean_connectome.txt')) 
fix_uoi_mean = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_stat-mean_connectome.txt')) 


# %%
fix_pearson_median = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-Pearson_stat-median_connectome.txt')) 
fix_pcorr_median = np.loadtxt(op.join(results_dir, 'MSMAll-FIX','proc-FIX_model-partial-correlation_stat-median_connectome.txt')) 

fix_lasso_median = np.loadtxt(op.join(results_dir, 'MSMAll-FIX','proc-FIX_model-lassoBIC_stat-median_connectome.txt')) 
fix_ridge_median = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-ridgeCV_stat-median_connectome.txt')) 
fix_uoi_median = np.loadtxt(op.join(results_dir, 'MSMAll-FIX', 'proc-FIX_model-uoiLasso_stat-median_connectome.txt')) 


# %%
xcpd_pearson_median = np.loadtxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-Pearson_stat-median_connectome.txt')) 
xcpd_pcorr_median = np.loadtxt(op.join(results_dir, 'XCPD','proc-XCPD_model-partial-correlation_stat-median_connectome.txt')) 

xcpd_lasso_median = np.loadtxt(op.join(results_dir, 'XCPD','proc-XCPD_model-lassoBIC_stat-median_connectome.txt')) 
xcpd_ridge_median = np.loadtxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-ridgeCV_stat-median_connectome.txt')) 
xcpd_uoi_median = np.loadtxt(op.join(results_dir, 'XCPD', 'proc-XCPD_model-uoiLasso_stat-median_connectome.txt')) 

# %%
fig, ax = plt.subplots(1,1, figsize=(1,1))

for ii, loc in enumerate(full_col_df['index'][:7]): 
    height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
    col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
    loc = loc - 1
    
    ax.add_patch(patches.Rectangle((0,ii/4), .5, .11,
                          clip_on=False, facecolor=col))

    ax.text(x=.55, y=(ii/4)+.01, s=name_dict[full_col_df.index[ii].split('_')[-1]])
    # ax.add_patch(patches.Rectangle((loc,0), height, patch_width,
    #                           clip_on=False, facecolor=col)) 


# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
plt.axis('off') 
plt.savefig('plots/Schaefer_legend.png', bbox_inches='tight', transparent=True)


# %%
def add_network_patches(full_col_df, axes, patch_width=1, xloc=-5, yloc=102, offset=1): 
    for loc in full_col_df['index']: 
        height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
        col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
        loc = loc - offset
        for axx in axes:
            axx.add_patch(patches.Rectangle((xloc,loc), patch_width, height,
                                      clip_on=False, facecolor=col))
        
        
            axx.add_patch(patches.Rectangle((loc,yloc), height, patch_width,
                                      clip_on=False, facecolor=col)) 
    
            if loc != 0: 
                axx.axhline(loc, color='black',  linestyle='--', 
                      linewidth=.25)
                axx.axvline(loc, color='black',  linestyle='--', 
                      linewidth=.25)

# %%
import scipy.stats

# %%
np.std(fix_ridge_median.ravel()) 

# %%
# norm = mpl.colors.Normalize(vmin=-np.max([fix_pearson_median, fix_ridge_median, fix_uoi_median]),
#                             vmax=np.max([fix_pearson_median, fix_ridge_median, fix_uoi_median]), 
#                             clip=False)


norm = mpl.colors.Normalize(vmin=-.2,
                            vmax=.2, 
                            clip=False)
cmap = sns.color_palette("vlag", as_cmap=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.subplots_adjust(wspace=.10)

axes[0].set_title('Pearson', weight=weight)
axes[1].set_title('Ridge', weight=weight)
axes[2].set_title('Union of Intersections', weight=weight)

sns.heatmap(fix_pearson_median,
           cmap=cmap,
            vmin=-.8, 
            vmax=.8,
           ax=axes[0], 
           yticklabels=output_list, 
           cbar=True,
            cbar_kws={ "shrink":.9})

# cax = fig.add_axes([.3, 0.12, .01, .7])

sns.heatmap(fix_ridge_median,
           cmap=cmap,
           ax=axes[1], 
           norm=norm, 
           cbar=True, 
            cbar_kws={ "shrink":.9})



sns.heatmap(fix_uoi_median,
           cmap=cmap,
           ax=axes[2], 
           norm=norm,
           cbar=True, 
                        cbar_kws={ "shrink":.9})

             # cbar_kws={ "orientation": "horizontal","anchor": (-.3,-1.2) , "pad":-.1,
             #         "shrink":.9})

add_network_patches(full_col_df, axes) 

axes[0].set_ylabel('Right                             Left', labelpad=15)

axes[0].set_xlabel('Left                            Right', labelpad=15)
axes[1].set_xlabel('Left                            Right', labelpad=15)
axes[2].set_xlabel('Left                            Right', labelpad=15)

for axx in axes: 
    axx.set_xticks([]) 
    axx.set_yticks([]) 

    axx.axhline(50, color='black',  linestyle='--', 
                  linewidth=.5) 
    axx.axvline(50, color='black',  linestyle='--', 
                  linewidth=.5) 

# cax = fig.add_axes([0, -.1, .3, .04])

# fig.colorbar(mpl.cm.ScalarMappable( cmap=cmap, norm=norm),
#              cax=cax, orientation='horizontal')
# cax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15)) 
plt.savefig(f'../03-Plotting/plots/{str(datetime.date.today())}_fix-model_edge_matrices.png',  bbox_inches='tight') 


# %%
network_list = [ii.split('_')[-1] for ii in seen_list[:7]]

# %%
# LH within hemisphere: 
fig, axes =  plt.subplots(1, 3, figsize=(12, 4))

patch_width=1

sns.heatmap(fix_pearson_median[:50, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[0], 
            # yticklabels=trunc_labels[:50],
            yticklabels=output_list[:50],
            xticklabels=output_list[:50], 
           cbar=False)

# #need this for centering network labels
orig_ytick_loc = axes[0].get_yticks()[np.array(output_list[:50]) != '']
offset_arr = np.diff(np.append(orig_ytick_loc, 50.5)) / 2
ytick_loc = orig_ytick_loc + offset_arr - .5

sns.heatmap(fix_ridge_median[:50, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[1], 
           cbar=False)

sns.heatmap(fix_uoi_median[:50, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[2], 
           cbar=False)

add_network_patches(full_col_df[:7], axes, xloc=-2, yloc=51, offset=1)

for axx in axes: 
    axx.set_xticks([])
    axx.set_yticks([])
    
axes[0].set_yticks(labels = [name_dict.get(ii) for ii in network_list], ticks = ytick_loc, weight='book', fontsize=8)

for axx in axes: 
    axx.set_xticks(labels = [name_dict.get(ii) for ii in network_list], ticks = ytick_loc, 
                   weight='book', fontsize=8, 
                   rotation=45, ha='right' , rotation_mode='anchor')
    axx.tick_params(length=0, pad=10, axis='y') 
    axx.tick_params(length=0, pad=10, axis='x',) 



plt.suptitle('Within - Left')
plt.savefig('hemi-l-l.png', bbox_inches='tight')

# %%
# RH within hemisphere: 
fig, axes =  plt.subplots(1, 3, figsize=(12, 4))

patch_width=1

sns.heatmap(fix_pearson_median[50:, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[0], 
            yticklabels=output_list[50:],
            xticklabels=output_list[50:], 
           cbar=False)

#need this for centering network labels
orig_ytick_loc = axes[0].get_yticks()[np.array(output_list[50:]) != '']
offset_arr = np.diff(np.append(orig_ytick_loc, 50.5)) / 2
ytick_loc = orig_ytick_loc + offset_arr - .5

sns.heatmap(fix_ridge_median[50:, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[1], 
           cbar=False)

sns.heatmap(fix_uoi_median[50:, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[2], 
           cbar=False)

add_network_patches(full_col_df[7:], axes, xloc=-2, yloc=51, offset=51)

for axx in axes: 
    axx.set_xticks([])
    axx.set_yticks([])
    
axes[0].set_yticks(labels = [name_dict.get(ii) for ii in network_list], ticks = ytick_loc, weight='book', fontsize=8)

for axx in axes: 
    axx.set_xticks(labels =[name_dict.get(ii) for ii in network_list], ticks = ytick_loc, 
                   weight='book', fontsize=8, 
                   rotation=45, ha='right' , rotation_mode='anchor')
    axx.tick_params(length=0, pad=10, axis='y') 
    axx.tick_params(length=0, pad=10, axis='x',) 

plt.suptitle('Right hemisphere to right hemisphere connectomes')
plt.savefig('hemi-r-r.png', bbox_inches='tight')

# %%
### Rh -> LEFT
fig, axes =  plt.subplots(1, 3, figsize=(12, 4))

patch_width=1

sns.heatmap(fix_pearson_median[:50, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[0], 
            yticklabels=output_list[:50],
            xticklabels=output_list[50:], 
           cbar=False)

#need this for centering network labels
orig_ytick_loc = axes[0].get_yticks()[np.array(output_list[:50]) != '']
offset_arr = np.diff(np.append(orig_ytick_loc, 50.5)) / 2
ytick_loc = orig_ytick_loc + offset_arr -.5

orig_xtick_loc = axes[0].get_xticks()[np.array(output_list[50:]) != '']
xoffset_arr = np.diff(np.append(orig_xtick_loc, 50.5)) / 2
xtick_loc = orig_xtick_loc + xoffset_arr -.5

sns.heatmap(fix_ridge_median[:50, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[1], 
           cbar=False)

sns.heatmap(fix_uoi_median[:50, 50:],
           cmap=cmap,
           norm=norm,
            ax=axes[2], 
           cbar=False)

##Y PATCHES
for loc in full_col_df['index'][:7]: 
    height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
    col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
    loc = loc - 1
    for axx in axes:
        axx.add_patch(patches.Rectangle((-2,loc), patch_width, height,
                                  clip_on=False, facecolor=col))
        if loc != 0: 
            axx.axhline(loc, color='black',  linestyle='--', 
                  linewidth=.5)
    
## XPATCHES
for loc in full_col_df['index'][7:]: 
    height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
    col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
    loc = loc - 51
    for axx in axes:

        axx.add_patch(patches.Rectangle((loc,51), height, patch_width,
                                  clip_on=False, facecolor=col)) 

        if loc != 0: 
#          
            axx.axvline(loc, color='black',  linestyle='--', 
                  linewidth=.5)


for axx in axes: 
    axx.set_xticks([])
    axx.set_yticks([])
    
axes[0].set_yticks(labels = [name_dict.get(ii) for ii in network_list], ticks = ytick_loc, weight='book', fontsize=8)

for axx in axes: 
    axx.set_xticks(labels = [name_dict.get(ii) for ii in network_list], ticks = xtick_loc, 
                   weight='book', fontsize=8, 
                   rotation=45, ha='right' , rotation_mode='anchor')
    axx.tick_params(length=0, pad=10, axis='y') 
    axx.tick_params(length=0, pad=10, axis='x',) 

plt.suptitle('Between - R -> L')
plt.savefig('hemi-r-l.png', bbox_inches='tight')

# %%
### LH - > R 

fig, axes =  plt.subplots(1, 3, figsize=(12, 4))

patch_width=1

sns.heatmap(fix_pearson_median[50:, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[0], 
            yticklabels=output_list[50:],
            xticklabels=output_list[:50], 
           cbar=False)

#need this for centering network labels
orig_ytick_loc = axes[0].get_yticks()[np.array(output_list[50:]) != '']
offset_arr = np.diff(np.append(orig_ytick_loc, 50.5)) / 2
ytick_loc = orig_ytick_loc + offset_arr -.5

orig_xtick_loc = axes[0].get_xticks()[np.array(output_list[:50]) != '']
xoffset_arr = np.diff(np.append(orig_xtick_loc, 50.5)) / 2
xtick_loc = orig_xtick_loc + xoffset_arr -.5

sns.heatmap(fix_ridge_median[50:, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[1], 
           cbar=False)

sns.heatmap(fix_uoi_median[50:, :50],
           cmap=cmap,
           norm=norm,
            ax=axes[2], 
           cbar=False)

#Y PATCHES
for loc in full_col_df['index'][7:]: 
    height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
    col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
    loc = loc - 51
    for axx in axes:
        axx.add_patch(patches.Rectangle((-2,loc), patch_width, height,
                                  clip_on=False, facecolor=col))
        if loc != 0: 
            axx.axhline(loc, color='black',  linestyle='--', 
                  linewidth=.5)
    
## XPATCHES
for loc in full_col_df['index'][:7]: 
    height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
    col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
    loc = loc - 1
    for axx in axes:

        axx.add_patch(patches.Rectangle((loc,51), height, patch_width,
                                  clip_on=False, facecolor=col)) 

        if loc != 0: 
#          
            axx.axvline(loc, color='black',  linestyle='--', 
                  linewidth=.5)

for axx in axes: 
    axx.set_xticks([])
    axx.set_yticks([])
    
axes[0].set_yticks(labels = [name_dict.get(ii) for ii in network_list], ticks = ytick_loc, weight='book', fontsize=8)

for axx in axes: 
    axx.set_xticks(labels = [name_dict.get(ii) for ii in network_list], ticks = xtick_loc, 
                   weight='book', fontsize=8, 
                   rotation=45, ha='right' , rotation_mode='anchor')
    axx.tick_params(length=0, pad=10, axis='y') 
    axx.tick_params(length=0, pad=10, axis='x',) 

plt.suptitle('Left hemisphere to right hemisphere connections')

plt.savefig('hemi-l-r.png', bbox_inches='tight')

# %%
pnc_pearson_median = np.loadtxt(op.join(analysis_dir, 'interim', 'connectome_csvs', f'2026-10-04_pnc_pearson_median.txt')) 
pnc_ridge_median = np.loadtxt(op.join(analysis_dir, 'interim', 'connectome_csvs', f'2026-10-04_pnc_ridge_median.txt')) 
pnc_uoi_median = np.loadtxt(op.join(analysis_dir, 'interim', 'connectome_csvs', f'2026-10-04_pnc_uoi_median.txt')) 
pnc_lasso_median = np.loadtxt(op.join(analysis_dir, 'interim', 'connectome_csvs', f'2026-10-04_pnc_lasso_median.txt')) 
pnc_pcorr_median = np.loadtxt(op.join(analysis_dir, 'interim', 'connectome_csvs', f'2026-10-04_pnc_pcorr_median.txt')) 


# %%
norm = mpl.colors.Normalize(vmin=-.2,
                            vmax=.2, 
                            clip=False)

cmap = sns.color_palette("vlag", as_cmap=True)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
fig.subplots_adjust(wspace=.10)

axes[0].set_title('Pearson', weight=weight)
axes[1].set_title('Partial Correlation', weight=weight)

axes[2].set_title('Ridge', weight=weight)
axes[3].set_title('LASSO', weight=weight)

axes[4].set_title('Union of Intersections', weight=weight)

sns.heatmap(pnc_pearson_median,
           cmap=cmap,
           ax=axes[0],
             vmin=-.8, 
            vmax=.8,
           yticklabels=output_list, 
           cbar=True, 
 cbar_kws={ "shrink":.9, 'location':'bottom', 
               "ticks": [-.8, -.5, 0, .5, .8]})


sns.heatmap(pnc_pcorr_median,
           cmap=cmap,
           ax=axes[1], 
           norm=norm,
           yticklabels=output_list, 
           cbar=True, 
         cbar_kws={ "shrink":.8, 'location':'bottom',
             })


sns.heatmap(pnc_ridge_median,
           cmap=cmap,
           ax=axes[2], 
           norm=norm, 
           cbar=True, 
 cbar_kws={ "shrink":.8, 'location':'bottom'})


sns.heatmap(pnc_ridge_median,
           cmap=cmap,
           ax=axes[3], 
           norm=norm, 
          cbar=True, 
              cbar_kws={ "shrink":.8, 'location':'bottom'})

sns.heatmap(pnc_uoi_median,
           cmap=cmap,
           ax=axes[4], 
           norm=norm,
         cbar=True, 
 cbar_kws={ "shrink":.8, 'location':'bottom'})

add_network_patches(full_col_df, axes) 
label_padding = 6

axes[0].set_ylabel('Right                     Left', labelpad=label_padding+2)

axes[0].set_xlabel('Left                       Right', labelpad=label_padding)
axes[1].set_xlabel('Left                       Right', labelpad=label_padding)
axes[2].set_xlabel('Left                       Right', labelpad=label_padding)
axes[3].set_xlabel('Left                       Right', labelpad=label_padding)
axes[4].set_xlabel('Left                       Right', labelpad=label_padding)

for axx in axes: 
    axx.set_xticks([]) 
    axx.set_yticks([]) 

    axx.axhline(50, color='black',  linestyle='--', 
                  linewidth=.5) 
    axx.axvline(50, color='black',  linestyle='--', 
                  linewidth=.5) 

# cax = fig.add_axes([.91, 0.12, .01, .7])

# fig.colorbar(mpl.cm.ScalarMappable( cmap=cmap, norm=norm),
#              cax=cax, orientation='vertical')
# cax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15)) 

plt.savefig(f'../03-Plotting/plots/{str(datetime.date.today())}_pnc-model_edge_matrices.png',  bbox_inches='tight', transparent=True) 
plt.show() 
# %%
norm = mpl.colors.Normalize(vmin=-.3,
                            vmax=.3, 
                            clip=False)

cmap = sns.color_palette("vlag", as_cmap=True)

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
fig.subplots_adjust(wspace=.10)

axes[0].set_title('Pearson', weight=weight)
axes[1].set_title('Partial Correlation', weight=weight)

axes[2].set_title('Ridge', weight=weight)
axes[3].set_title('LASSO', weight=weight)
axes[4].set_title('UoI', weight=weight)

sns.heatmap(xcpd_pearson_median,
           cmap=cmap,
           ax=axes[0],
             vmin=-.8, 
            vmax=.8,
           yticklabels=output_list, 
           cbar=True, 
 cbar_kws={ "shrink":.9, 'location':'bottom', 
               "ticks": [-.8, -.5, 0, .5, .8]})


sns.heatmap(xcpd_pcorr_median,
           cmap=cmap,
           ax=axes[1], 
           norm=norm,
           yticklabels=output_list, 
           cbar=True, 
         cbar_kws={ "shrink":.8, 'location':'bottom',
             })


sns.heatmap(xcpd_ridge_median,
           cmap=cmap,
           ax=axes[2], 
           norm=norm, 
           cbar=True, 
 cbar_kws={ "shrink":.8, 'location':'bottom'})


sns.heatmap(xcpd_ridge_median,
           cmap=cmap,
           ax=axes[3], 
           norm=norm, 
          cbar=True, 
              cbar_kws={ "shrink":.8, 'location':'bottom'})

sns.heatmap(xcpd_uoi_median,
           cmap=cmap,
           ax=axes[4], 
           norm=norm,
         cbar=True, 
 cbar_kws={ "shrink":.8, 'location':'bottom'})
add_network_patches(full_col_df, axes) 
label_padding = 6

axes[0].set_ylabel('Right                  Left', labelpad=label_padding+2)

axes[0].set_xlabel('Left                       Right', labelpad=label_padding)
axes[1].set_xlabel('Left                       Right', labelpad=label_padding)
axes[2].set_xlabel('Left                       Right', labelpad=label_padding)
axes[3].set_xlabel('Left                       Right', labelpad=label_padding)
axes[4].set_xlabel('Left                       Right', labelpad=label_padding)

for axx in axes: 
    axx.set_xticks([]) 
    axx.set_yticks([]) 

    axx.axhline(50, color='black',  linestyle='--', 
                  linewidth=.5) 
    axx.axvline(50, color='black',  linestyle='--', 
                  linewidth=.5) 

# cax = fig.add_axes([.91, 0.12, .01, .7])

# fig.colorbar(mpl.cm.ScalarMappable( cmap=cmap, norm=norm),
#              cax=cax, orientation='vertical')
# cax.yaxis.set_major_locator(plt.MaxNLocator(nbins=15)) 

plt.savefig(f'../03-Plotting/plots/{str(datetime.date.today())}_xcpd-model_edge_matrices.png',  bbox_inches='tight', transparent=True) 
plt.show() 
# %%

# %%

# %%
