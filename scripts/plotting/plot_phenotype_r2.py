# ---
# jupyter:
#   jupytext:
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
config.read(op.join(base_dir,  'config.ini') )

# %%
cog_dict = {'Age_in_Yrs': 'Age',
 'CogCrystalComp_Unadj': 'Crystalized Intelligence',
 'CogFluidComp_Unadj': 'Fluid Intelligence',
 'CogTotalComp_Unadj': 'Global Intelligence',
 'DDisc_AUC_200': 'Impulsivity',
 'Endurance_Unadj': 'Endurance',
 'IWRD_TOT': 'Verbal Memory',
 'ReadEng_Unadj': 'Reading Ability',
 'SCPT_SEN': 'Attention',
 'VSPLOT_TC': 'Spatial Orientation'}

# %%
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# %%
r2_plot_df = pd.read_csv(op.join(results_dir, 'plotting', 
                                  'proc-FIX_desc-phenopredictions_stat-r2.csv')) 

# %%
#add boxes and whiskers 
fig, ax = plt.subplots(1,1) 
sns.boxplot(r2_plot_df, y = 'variable', 
            x='value', hue= 'Model')
ax.set_xlim(-1, .5 ) 
ax.axvline(0, linestyle='--', c='black', lw=.75) 
locs = ax.get_yticks()


ax.set_yticklabels([cog_dict[t.get_text()] for t in ax.get_yticklabels()])ax.set_ylabel('Phenotype') 
ax.set_xlabel('$R^2$') 
plt.savefig('fig4/pheno_r2_boxplot.svg',  format="svg", bbox_inches="tight")

# %%
