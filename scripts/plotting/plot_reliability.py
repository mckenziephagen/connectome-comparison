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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

import numpy as np
import seaborn as sns
import pickle
import datetime
from glob import glob

# %%
from _plot_utils import (models, cm, wrap_labels,
 palette_dict, output_list, full_col_df) 
# from model_fc.utils import set_config
import os.path as op
import os 
import configparser
import pandas as pd

# %%
current_dir = os.getcwd()
script_dir = op.join(current_dir.split('scripts')[0], 'scripts')
analysis_dir = op.join(script_dir, 'analysis', '02-Analysis')
config = configparser.ConfigParser()
config.read(op.join(current_dir.split('scripts')[0],  'config.ini') )

# %%
plot_models = ['Pearson', 'Ridge', 'UoI']

# %%
col_name_dict = {'lasso': 'LASSO', 
               'pearson': 'Pearson', 
                'uoi': 'UoI', 
              'pcorr': 'Partial Correlation', 
              'ridge': 'Ridge'}

# %%
fix_fp_df = pd.read_csv(op.join(results_dir, 'plotting', 'MSMAll-FIX',
                                 'proc-FIX_stat-fingerprinting.csv'),
            index_col=0)

xcpd_fp_df = pd.read_csv(op.join(results_dir, 'plotting', 
                                 'XCPD', 'proc-XCPD_stat-fingerprinting.csv'),
            index_col=0)


pnc_fp_df = pd.read_csv(op.join(results_dir, 'plotting',
                                'PNC_XCPD', 'proc-PNC_stat-fingerprinting.csv'),
            index_col=0)

for df in fix_fp_df, xcpd_fp_df, pnc_fp_df: 
    df.rename(columns=col_name_dict, inplace=True) 


# %%
discrim_df = pd.read_csv(op.join(results_dir, 'plotting',
                                 'stat-discrim.csv'), index_col=0)

# %%
fix_lasso_icc = pd.read_csv(op.join(results_dir, 'plotting', 
                                'MSMAll-FIX',
                               'proc-FIX_model-lassoBIC_stat-ICC.csv')) 
fix_lasso_icc_vals = fix_lasso_icc.iloc[0].values.copy()

fix_pearson_icc = pd.read_csv(op.join(results_dir, 'plotting', 
                                  'MSMAll-FIX',

                               'proc-FIX_model-Pearson_stat-ICC.csv')) 
    
fix_pearson_icc_vals = fix_pearson_icc.iloc[0].values.copy()

fix_uoi_icc =  pd.read_csv(op.join(results_dir, 'plotting',
                                'MSMAll-FIX',
 
                               'proc-FIX_model-uoiLasso_stat-ICC.csv')) 
fix_uoi_icc_vals = fix_uoi_icc.iloc[0].values.copy()

fix_ridge_icc =  pd.read_csv(op.join(results_dir, 'plotting', 
                                     'MSMAll-FIX',
                               'proc-FIX_model-ridgeCV_stat-ICC.csv')) 
                       
fix_ridge_icc_vals = fix_ridge_icc.iloc[0].values.copy()

fix_pcorr_icc =  pd.read_csv(op.join(results_dir, 'plotting', 
                                     'MSMAll-FIX',
                               'proc-FIX_model-partial-correlation_stat-ICC.csv')) 
                
fix_pcorr_icc_vals = fix_pcorr_icc.iloc[0].values.copy()


# %%
# xcpd_lasso_icc = pd.read_csv(op.join(results_dir, 'plotting', 
#                                'proc-XCPD_model-lassoBIC_stat-ICC.csv')) 
# xcpd_lasso_icc_vals = xcpd_lasso_icc.iloc[0].values.copy()

# xcpd_pearson_icc = pd.read_csv(op.join(results_dir, 'plotting', 
#                                'proc-XCPD_model-Pearson_stat-ICC.csv')) 
    
# xcpd_pearson_icc_vals = xcpd_pearson_icc.iloc[0].values.copy()

# xcpd_uoi_icc =  pd.read_csv(op.join(results_dir, 'plotting', 
#                                'proc-XCPD_model-uoiLasso_stat-ICC.csv')) 
# xcpd_uoi_icc_vals = xcpd_uoi_icc.iloc[0].values.copy()

# xcpd_ridge_icc =  pd.read_csv(op.join(results_dir, 'plotting', 
#                                'proc-XCPD_model-ridgeCV_stat-ICC.csv')) 
                       
# xcpd_ridge_icc_vals = xcpd_ridge_icc.iloc[0].values.copy()

# xcpd_pcorr_icc =  pd.read_csv(op.join(results_dir, 'plotting', 
#                                'proc-XCPD_model-partial-correlation_stat-ICC.csv')) 
                
# xcpd_pcorr_icc_vals = xcpd_pcorr_icc.iloc[0].values.copy()

# %%
from statsmodels.stats.descriptivestats import describe


# %%
fix_fp_stat = describe(fix_fp_df, stats=['mean', 'percentiles'] , percentiles=[97.5, 2.5])

# %%
xcpd_fp_stat = describe(xcpd_fp_df, stats=['mean', 'percentiles'] , percentiles=[97.5, 2.5])

# %%
pnc_fp_stat = describe(pnc_fp_df, stats=['mean', 'percentiles'] , percentiles=[97.5, 2.5])

# %%
fig, axes = plt.subplots(1,1, figsize=(2,4), sharey=False) 

fp = sns.violinplot(data=fix_fp_df[plot_models].melt()
, 
                    y='value', 
                    x='variable', 
                   ax=axes,
                   palette= [palette_dict.get(key) for key in plot_models]  ,       
                    order = plot_models                  ) 

axes.grid(axis='y', linewidth=.25)
axes.set_ylabel('Fingerprinting') 
axes.set_xlabel('') 
sns.despine() 
axes.set_axisbelow(True)
wrap_labels(fp, 5) 
plt.savefig(f'fig3/{str(datetime.date.today())}_fix_fingerprinting.png', bbox_inches='tight')


# %%
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 

fp = sns.violinplot(data=fix_fp_df.melt()
, 
                    y='value', 
                    x='variable', 
                   ax=axes,
                   palette= [palette_dict.get(key) for key in models]  ,       
                    order = models                  ) 
axes.set_title('HCP Minimal Processing') 

axes.grid(axis='y', linewidth=.25)
axes.set_ylabel('Fingerprinting') 
axes.set_xlabel('') 
sns.despine() 
axes.set_axisbelow(True)
wrap_labels(fp, 5) 
plt.savefig(f'fig3/{str(datetime.date.today())}_fix_fingerprinting_all.png', bbox_inches='tight')

# %%
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 

fp = sns.violinplot(data=xcpd_fp_df.melt()
, 
                    y='value', 
                    x='variable', 
                   ax=axes,
                   palette= [palette_dict.get(key) for key in models]  ,       
                    order = models                  ) 
axes.set_title('XCPD') 

axes.grid(axis='y', linewidth=.25)
axes.set_ylabel('Fingerprinting') 
axes.set_xlabel('') 
sns.despine() 
axes.set_axisbelow(True)
wrap_labels(fp, 5) 
plt.savefig(f'fig3/{str(datetime.date.today())}_xcpd_fingerprinting_all.png', bbox_inches='tight')

# %%
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 

fp = sns.violinplot(data=pnc_fp_df.melt()
, 
                    y='value', 
                    x='variable', 
                   ax=axes,
                   palette= [palette_dict.get(key) for key in models]  ,       
                    order = models                  ) 
axes.set_title('PNC') 
axes.grid(axis='y', linewidth=.25)
axes.set_ylabel('Fingerprinting') 
axes.set_xlabel('') 
sns.despine() 
axes.set_axisbelow(True)
wrap_labels(fp, 5) 
plt.savefig(f'fig3/{str(datetime.date.today())}_pnc_fingerprinting_all.png', bbox_inches='tight')

# %%
fix_i2c2_df = pd.read_csv(op.join(results_dir, 
                                  'i2c2_fix_df.csv'),index_col=0) 

fix_i2c2_df.rename(columns={'Pcorr': 'Partial Correlation'}, inplace=True)
fix_i2c2_df = fix_i2c2_df[models]
i2c2_fix_upper = fix_i2c2_df.loc['97.5'] - fix_i2c2_df.loc['lambda']
i2c2_fix_lower = fix_i2c2_df.loc['lambda'] - fix_i2c2_df.loc['2.5']

# %%
xcpd_i2c2_df = pd.read_csv(op.join(results_dir, 
                                  'i2c2_xcpd_df.csv'),index_col=0) 
xcpd_i2c2_df.rename(columns={'Pcorr': 'Partial Correlation'}, inplace=True)

xcpd_i2c2_df = xcpd_i2c2_df[models]
i2c2_xcpd_upper = xcpd_i2c2_df.loc['97.5'] - xcpd_i2c2_df.loc['lambda']
i2c2_xcpd_lower = xcpd_i2c2_df.loc['lambda'] - xcpd_i2c2_df.loc['2.5']

# %%
fig, axes = plt.subplots(1,1, figsize=(2,4), sharey=False) 
weight='normal'


axes.grid(axis='y', linewidth=.25)

axes.bar(x=xcpd_i2c2_df.columns, 
          height=xcpd_i2c2_df.T['lambda'], 
          color = [palette_dict.get(key) for key in models], 
          # alpha=.9, label=['Pearson', 'LASSO', 'UoI']
          )

axes.errorbar(x=xcpd_i2c2_df.columns,
               y=xcpd_i2c2_df.T['lambda'], 
               yerr=[i2c2_xcpd_lower, i2c2_xcpd_upper],
               color='black', 
               linestyle='', 
               elinewidth=2, 
               barsabove=True, 
               capsize=3, 
                alpha=.9)

axes.set_ylabel('I2C2',
                   weight=weight, fontsize=12) 

axes.grid(axis='y', linewidth=.25)
# axes.legend() 

axes.set_axisbelow(True)
wrap_labels(axes, 5)

sns.despine()

# plt.savefig(f'plots/{str(datetime.date.today())}_xcpd_full_i2c2.png', bbox_inches='tight')
# plt.show() 

# %%
fig, axes = plt.subplots(1,1, figsize=(4,4), sharey=False) 
weight='normal'


axes.grid(axis='y', linewidth=.25)
# axes[1].grid(axis='y', linewidth=.25)
# axes[2].grid(axis='y', linewidth=.25)

###
## MVAR Reliability
### 
axes.bar(x=fix_i2c2_df.columns, 
          height=fix_i2c2_df.T['lambda'], 
          color = [palette_dict.get(key) for key in models], 
          # alpha=.9, label=['Pearson', 'LASSO', 'UoI']
          )

axes.errorbar(x=fix_i2c2_df.columns,
               y=fix_i2c2_df.T['lambda'], 
               yerr=[i2c2_fix_lower, i2c2_fix_upper],
               color='black', 
               linestyle='', 
               elinewidth=2, 
               barsabove=True, 
               capsize=3, 
                alpha=.9)

axes.set_ylabel('I2C2',
                   weight=weight, fontsize=12) 

axes.grid(axis='y', linewidth=.25)
# axes.legend() 

axes.set_axisbelow(True)
wrap_labels(axes, 5)

sns.despine()

plt.savefig(f'fig3/{str(datetime.date.today())}_fix_full_i2c2.png', bbox_inches='tight')
plt.show() 

# %%
fig, axes = plt.subplots(1,1, figsize=(4,4), sharey=False) 
weight='normal'


axes.grid(axis='y', linewidth=.25)
# axes[1].grid(axis='y', linewidth=.25)
# axes[2].grid(axis='y', linewidth=.25)

###
## MVAR Reliability
### 
axes.bar(x=fix_i2c2_df.columns, 
          height=fix_i2c2_df.T['lambda'], 
          color = [palette_dict.get(key) for key in models], 
          # alpha=.9, label=['Pearson', 'LASSO', 'UoI']
          )

axes.errorbar(x=fix_i2c2_df.columns,
               y=fix_i2c2_df.T['lambda'], 
               yerr=[i2c2_fix_lower, i2c2_fix_upper],
               color='black', 
               linestyle='', 
               elinewidth=2, 
               barsabove=True, 
               capsize=3, 
                alpha=.9)

axes.set_ylabel('I2C2',
                   weight=weight, fontsize=12) 

axes.grid(axis='y', linewidth=.25)
# axes.legend() 

axes.set_axisbelow(True)
wrap_labels(axes, 5)

sns.despine()

plt.savefig(f'fig3/{str(datetime.date.today())}_fix_full_i2c2.png', bbox_inches='tight')
plt.show() 

# %%
discrim_df = discrim_df.T

# %%
discrim_df.loc['MSMAll_FIX'].index

# %%
discrim_df

# %%
# ### 
# # Discriminability
# ### 
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 
weight='normal'

sns.barplot(x=discrim_df.loc['MSMAll_FIX'].index, 
        y=discrim_df.loc['MSMAll_FIX'], 
        order = plot_models   ,               
        palette= [palette_dict.get(key) for key in plot_models]  ,       
           )

axes.plot([0,1,2], [1.06, 1.06, 1.06], lw=.75, c='black')
axes.plot([0,1], [1.025,1.025], lw=.75, c='black')

axes.text(1, 1.06, s='*', fontsize=7, ha='center')
axes.text(.5, 1.02, s='*', fontsize=7, ha='center')


axes.set_ylabel('Discriminability', 
                 weight='normal') 
sns.despine()
axes.grid(axis='y', linewidth=.25)
axes.set_axisbelow(True)
wrap_labels(axes, 5)

plt.savefig(f'fig3/{str(datetime.date.today())}_fix_discrim.png', bbox_inches='tight')
plt.show() 

# %%
# ### 
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 
weight='normal'

sns.barplot(x=discrim_df.loc['XCPD'].index, 
        y=discrim_df.loc['XCPD'], 
        order = models   ,               
        palette= [palette_dict.get(key) for key in models]  ,       
           )

# axes.plot([0,1,2], [1.05, 1.05, 1.05], lw=.75, c='black')
# axes.plot([0,1], [1,1], lw=.75, c='black')

# axes.text(.5, .975, s='*', fontsize=7, ha='center')

axes.set_ylabel('Discriminability', 
                 weight='normal') 

axes.set_xlabel('') 
sns.despine()
axes.grid(axis='y', linewidth=.25)
wrap_labels(axes, 5)

plt.savefig(f'fig3/{str(datetime.date.today())}_xcpd_discrim.png', bbox_inches='tight')
plt.show() 

# %%
# ### 
fig, axes = plt.subplots(1,1, figsize=(2.5,4), sharey=False) 
weight='normal'

sns.barplot(x=discrim_df.loc['PNC'].index, 
        y=discrim_df.loc['PNC'], 
        order = models   ,               
        palette= [palette_dict.get(key) for key in models]  ,       
           )

# axes.plot([0,1,2], [1.05, 1.05, 1.05], lw=.75, c='black')
# axes.plot([0,1], [1,1], lw=.75, c='black')

# axes.text(.5, .975, s='*', fontsize=7, ha='center')

axes.set_ylabel('Discriminability', 
                 weight='normal') 
axes.set_xlabel('') 

sns.despine()
axes.grid(axis='y', linewidth=.25)
axes.set_axisbelow(True)
wrap_labels(axes, 5)

plt.savefig(f'fig3/{str(datetime.date.today())}_pnc_discrim.png', bbox_inches='tight')
plt.show() 

# %%
fig, axes = plt.subplots(1,3, figsize=(7, 2), sharex=True, sharey=True)

fig.subplots_adjust(wspace=.1)
sns.histplot(fix_pearson_icc_vals, ax=axes[0] , element='step',
            color=palette_dict['Pearson']) 

sns.histplot(fix_lasso_icc_vals, ax=axes[1] , element='step',
                        color=palette_dict['LASSO']) 

sns.histplot(fix_uoi_icc_vals, ax=axes[2] , element='step',
                                    color=palette_dict['UoI']) 

fig.tight_layout() 
# plt.legend() 
sns.despine()
plt.show() 

# %%
fix_icc_data = pd.DataFrame({'Pearson': fix_pearson_icc_vals, 
                          'Partial Correlation': fix_pcorr_icc_vals,

                          'Ridge': fix_ridge_icc_vals,
                          'LASSO': fix_lasso_icc_vals,

              'UoI': fix_uoi_icc_vals}) 


# %%
alpha=1
fig, axes = plt.subplots(1,1, figsize=(2.5,4))

# sns.histplot(icc_data, element='step', 
#             palette=[palette_dict['Pearson'], palette_dict['LASSO'], palette_dict['UoI']], 
#              ax=axes) 
axes.set_ylabel('ICC') 

# axes[0].text(x=-.2, y=1.05,s='a', weight='semibold', transform=axes[0].transAxes)
axes.grid(axis='y', linewidth=.25)

sns.violinplot(fix_icc_data,
               order=models, 
                       palette= [palette_dict.get(key) for key in models]  ,       

             ax=axes, alpha=alpha) 

# axes[1].grid(axis='y', linewidth=.25)
# axes[1].set_ylabel('')
# axes[1].set_xlabel('ICC') 

# axes[1].text(x=-.1, y=1.05,s='b', weight='semibold', transform=axes[1].transAxes)

# axes[0].set_axisbelow(True)
# axes[1].set_axisbelow(True)
wrap_labels(axes, 5)
sns.despine() 
plt.savefig(f'fig3/{str(datetime.date.today())}_fix__full_icc.png', bbox_inches='tight')
plt.show() 


# %%
def plot_patches(full_col_df, axes): 
    for loc in full_col_df['index']: 
        height = full_col_df.loc[full_col_df['index'] == loc]['count'].item()
        col = full_col_df.loc[full_col_df['index'] == loc]['color'].item()
        loc = loc - 1
        for axx in axes: 
            axx.add_patch(patches.Rectangle((-2.5, loc), 1.5, height,
                                  clip_on=False, facecolor=col))
    
    
            axx.add_patch(patches.Rectangle((loc,101.5), height, 1.5,
                                  clip_on=False, facecolor=col))
    
    
            if loc != 0: 
                axx.axhline(loc, color='black',  linestyle='--', 
                      linewidth=.25)
                axx.axvline(loc, color='black',  linestyle='--', 
                      linewidth=.25)


# %%
full_cmap = plt.get_cmap("vlag")
from matplotlib.colors import LinearSegmentedColormap

# vlag goes blue -> white -> red, so the "non-blue" half is the upper half (0.5 to 1.0)
colors = full_cmap(np.linspace(0.5, 1.0, 256))
red_half = LinearSegmentedColormap.from_list("vlag_red_half", colors)

# %%
# plot_patches = True
add_lines = False 

norm = mpl.colors.Normalize(vmin=0, vmax=1  ) 
# cmap = sns.color_palette(red_half, as_cmap=True)
cmap = red_half
fig, axes = plt.subplots(1,3, figsize=(7, 2.4))
fig.subplots_adjust(wspace=.1)


sns.heatmap(fix_pearson_icc_vals.reshape(100,100), ax=axes[0], cbar=False, 
            yticklabels=output_list, 
            xticklabels=output_list, 
           norm=norm, 
           cmap=cmap) 

sns.heatmap(fix_ridge_icc_vals.reshape(100,100), ax=axes[1], cbar=False,
            norm=norm,
                       cmap=cmap) 


sns.heatmap(fix_uoi_icc_vals.reshape(100,100), ax=axes[2], cbar=False, 
           norm=norm,
                       cmap=cmap) 

plot_patches(full_col_df, axes) 

axes[0].set_title('Pearson', weight=weight)
axes[1].set_title('Ridge', weight=weight)
axes[2].set_title('Union of Intersections', weight=weight)

axes[0].set_yticks([]) 

axes[0].set_xticks([]) 
axes[1].set_xticks([]) 
axes[2].set_xticks([]) 

axes[1].set_yticks([]) 
axes[2].set_yticks([]) 

cax = fig.add_axes([.91, 0.12, .01, .7])

fig.colorbar(mpl.cm.ScalarMappable( norm=norm, cmap=cmap),
             cax=cax, orientation='vertical')

plt.savefig('fig3/fix_icc_heatmaps_main.png', bbox_inches='tight') 
plt.show() 
