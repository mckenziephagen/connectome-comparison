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
#     display_name: fc_311
#     language: python
#     name: fc_311
# ---

import sys

# +
#janky, but ok
sys.path.append("/global/homes/m/mphagen/software/fmriprep-denoise-benchmark/fmriprep_denoise")

import os.path as op
import seaborn as sns
# -


import pandas as pd
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from scipy import stats, linalg
from features.quality_control_connectivity import qcfc, partial_correlation

#I'm 80% sure we should be using relative
qa_df = pd.read_csv('/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/relative_movement.csv', 
                    index_col=0).sort_index()
qa_df.index = [f'sub-{i}' for i in qa_df.index] 
print(qa_df.shape, qa_df.columns)

unrestricted_df = pd.read_csv('../../../data/unrestricted_mphagen_1_27_2022_20_50_7.csv')
unrestricted_df.index = [f'sub-{i}' for i in unrestricted_df['Subject']]
unrestricted_df[['Gender']] = unrestricted_df[['Gender']].replace({'M':0, 'F':1}).infer_objects(copy=False)

restricted_df = pd.read_csv('../../../data/RESTRICTED_arokem_1_31_2022_23_26_45.csv')
restricted_df.index = [f'sub-{i}' for i in restricted_df['Subject']]

full_cov_df = unrestricted_df[['Gender']].join(restricted_df[['Age_in_Yrs']]) 
covariates = ['Gender', 'Age_in_Yrs']
qa_df = qa_df.join(full_cov_df[covariates]) #select only subjects that have QA values
covariate_df = qa_df[covariates]


def flatten_mat(mat_dict, ses_string, rms='absolute'): 
    df = pd.DataFrame()
    for sub_id in mat_dict.keys(): 
        try: 
            df = df.join(pd.DataFrame({sub_id: mat_dict[sub_id][ses_string].flatten()} ),
                how='outer')
        except KeyError: 
            pass
        
    return df


# +
def wrap_qcfc(connectome_df, covariate_df): 
    
    movement = connectome_df[['mean_framewise_displacement']]
    connectomes = connectome_df.drop(['mean_framewise_displacement'], axis=1)
    qcfc_results = qcfc(movement=movement, connectomes=connectomes, covarates=covariate_df)

    return(qcfc_results) 

def unpack_qcfc(results_obj): 
    pval_list = []
    sig_corr_list = [] 
    all_corr_list = []
    _ =[pval_list.append(i['pvalue']) for i in results_obj]
    _ =[ sig_corr_list.append(i['correlation']) for i in results_obj if i['pvalue'] <.05]
    _ =[all_corr_list.append(i['correlation']) for i in results_obj]
    return(pval_list, sig_corr_list, all_corr_list) 


# -

proc_list = ['MSMAll', 'minProc', 'xcpd']
model_list = ['lassoBIC', 'uoi', 'correlation']

results_path = op.join('/global/u1/m/mphagen',
                       'functional-connectivity/connectome-comparison/results')


def prep_qcfc(pkl_file, qa_df):
    """
    qa: pd dataframe with qa metric as col
    """


# +
results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results/MSMAll'
with open(glob(op.join(results_path, '*9-15*lassoBIC*relmat.pkl'))[0], 'rb') as l:
        lasso_dict = pickle.load(l)

    
lasso_df = flatten_mat(lasso_dict, ses_string='ses-1').T
lasso_df = lasso_df.join(qa_df[['RL_1', 'LR_1']].mean(axis=1).rename('mean_framewise_displacement'))
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
#has to be renamed like this for fmriprep_denoise benchmark 
msmall_lasso_pval, msmall_lasso_sig_list, msmall_lasso_corr_list  = unpack_qcfc(lasso_results)
print(covariate_df.columns, len(msmall_lasso_sig_list) ) 

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
#has to be renamed like this for fmriprep_denoise benchmark 
lasso_pval, lasso_sig_list, lasso_corr_list = unpack_qcfc(lasso_results)
print("No Covariates", len(lasso_sig_list) ) 


with open(glob(op.join(results_path, '*9-15*correlation*pkl'))[0], 'rb') as f:
        pearson_dict = pickle.load(f)

pearson_df = flatten_mat(pearson_dict, ses_string='ses-1').T
pearson_df = pearson_df.join(qa_df[['RL_1', 'LR_1']].mean(axis=1).rename('mean_framewise_displacement')) 

pearson_results = wrap_qcfc(pearson_df, covariate_df=covariate_df) 
msmall_pearson_pval, msmall_pearson_sig_list, msmall_pearson_corr_list = unpack_qcfc(pearson_results)

print(covariate_df.columns, len(msmall_pearson_sig_list) ) 


pearson_results = wrap_qcfc(pearson_df, covariate_df=None) 
pearson_pval, pearson_sig_list, pearson_corr_list = unpack_qcfc(pearson_results)

print("No covariates", len(pearson_sig_list) ) 

# +
msmall_lasso_df = pd.DataFrame({'model': 'LASSO' , 
                                'sig_corr': msmall_lasso_corr_list, 
                                'proc_type': 'MSMAll', 
                                 'covariates': 'True'})

msmall_lasso_df_cov = pd.DataFrame({'model': 'LASSO' , 
                                'sig_corr': lasso_corr_list , 
                                'proc_type': 'MSMAll', 
                                 'covariates': 'False'})

msmall_pearson_df = pd.DataFrame({'model': 'Pearson' , 
                                  'sig_corr': msmall_pearson_corr_list, 
                                  'proc_type': 'MSMAll', 
                                 'covariates': 'True'})


msmall_pearson_df_cov = pd.DataFrame({'model': 'Pearson' , 
                                  'sig_corr': pearson_corr_list, 
                                  'proc_type': 'MSMAll', 
                                 'covariates': 'False'})


# +
# results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results/MSMAll'

# with open(glob(op.join(results_path, '*09-15*uoi*pkl'))[0], 'rb') as l:
#         uoi_dict = pickle.load(l)


# uoi_df = flatten_mat(uoi_dict, ses_string='ses-1').T
# uoi_df = uoi_df.join(qa_df[['RL_1', 'LR_1']].mean(axis=1).rename('mean_framewise_displacement'))
# uoi_results = wrap_qcfc(uoi_df, covariate_df=covariate_df) 
# uoi_pval, uoi_corr_list = unpack_qcfc(uoi_results)

# print(covariate_df.columns, len(uoi_corr_list))

# uoi_results = wrap_qcfc(uoi_df, covariate_df=None) 
# uoi_pval, uoi_corr_list = unpack_qcfc(uoi_results)

# print("No covariates", len(uoi_corr_list))

# +
results_path = '/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results/xcpd'
with open(glob(op.join(results_path, '*9-15*lassoBIC*relmat.pkl'))[0], 'rb') as l:
        lasso_dict = pickle.load(l)

    
lasso_df = flatten_mat(lasso_dict, ses_string='ses-1').T
lasso_df = lasso_df.join(qa_df[['RL_1', 'LR_1']].mean(axis=1).rename('mean_framewise_displacement'))
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
#has to be renamed like this for fmriprep_denoise benchmark 
xcpd_lasso_pval, xcpd_lasso_sig_list, xcpd_lasso_corr_list = unpack_qcfc(lasso_results)
print(covariate_df.columns, len(xcpd_lasso_sig_list) ) 

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
#has to be renamed like this for fmriprep_denoise benchmark 
lasso_pval, lasso_sig_list, lasso_corr_list = unpack_qcfc(lasso_results)
print("No Covariates", len(lasso_sig_list) ) 


with open(glob(op.join(results_path, '*9-16*correlation*pkl'))[0], 'rb') as f:
        pearson_dict = pickle.load(f)

pearson_df = flatten_mat(pearson_dict, ses_string='ses-1').T
pearson_df = pearson_df.join(qa_df[['RL_1', 'LR_1']].mean(axis=1).rename('mean_framewise_displacement')) 

pearson_results = wrap_qcfc(pearson_df, covariate_df=covariate_df) 
xcpd_pearson_pval, xcpd_pearson_sig_list, xcpd_pearson_corr_list = unpack_qcfc(pearson_results)

print(covariate_df.columns, len(xcpd_pearson_sig_list) ) 


pearson_results = wrap_qcfc(pearson_df, covariate_df=None) 
pearson_pval, pearson_sig_list, pearson_corr_list = unpack_qcfc(pearson_results)

print("No covariates", len(pearson_sig_list) ) 

# +
xcpd_lasso_df = pd.DataFrame({'model': 'LASSO' , 
                                'sig_corr': xcpd_lasso_corr_list, 
                                'proc_type': 'xcpd', 
                                 'covariates': 'True'})

xcpd_lasso_df_cov = pd.DataFrame({'model': 'LASSO' , 
                                'sig_corr': lasso_corr_list , 
                                'proc_type': 'xcpd', 
                                 'covariates': 'False'})

xcpd_pearson_df = pd.DataFrame({'model': 'Pearson' , 
                                  'sig_corr': xcpd_pearson_corr_list, 
                                  'proc_type': 'xcpd', 
                                 'covariates': 'True'})


xcpd_pearson_df_cov = pd.DataFrame({'model': 'Pearson' , 
                                  'sig_corr': pearson_corr_list, 
                                  'proc_type': 'xcpd', 
                                 'covariates': 'False'})
# -

plot_df = pd.concat([msmall_lasso_df, msmall_pearson_df, 
            xcpd_lasso_df, xcpd_pearson_df, msmall_lasso_df_cov, msmall_pearson_df_cov, 
            xcpd_lasso_df_cov, xcpd_pearson_df_cov])

plot_df = plot_df.dropna() 

ax = sns.swarmplot(data=plot_df.loc[plot_df['covariates'] == 'True'], x="model", y="sig_corr", hue="proc_type")
ax.set(xlabel="Connectome Model", ylabel="Structure-Function Coupling")
ax.get_legend().set_title("Processing Pipeline")
ax.set_title("SFC by Model and Processing Type")  
plt.show()
plt.savefig('SFC.png') 

ax = sns.boxplot(data=plot_df.loc[plot_df['covariates'] == 'False'], x="model", y="sig_corr", hue="proc_type", width=.5)
ax.set(xlabel="Connectome Model", ylabel="Structure-Function Coupling")
ax.get_legend().set_title("Processing Pipeline")
ax.set_title("SFC by Model and Processing Type")  
plt.show()
plt.savefig('SFC.png') 

plt.hist(pearson_corr_list)
plt.hist(lasso_corr_list) 






# +
##EVERYTHING UNDER HERE IS OLD
# -

plt.figure(figsize=[6,3])
plt.hist(uoi_corr_list, alpha=.75, 
         label=f'UoI ({len(uoi_corr_list) })', 
         color='#00313C') 
plt.hist(lasso_corr_list, alpha=.75, 
         label = f'LASSO ({len(lasso_corr_list) }) ', 
         color='#007681') 
plt.hist(pearson_corr_list, alpha=.75, 
         label = f'Pearson ({len(pearson_corr_list) })', 
         color='#B1B3B3')
plt.legend() 
plt.savefig('qcfc.png',  bbox_inches="tight", )

results_path = '/pscratch/sd/m/mphagen/old_home/old_results/results/2024-04-29_pearson_dict.pkl'
with open(results_path, 'rb') as f:
        pearson_dict = pickle.load(f)


# +

pearson_df = flatten_mat(pearson_dict, ses_string='ses-1_run-1').T
pearson_df = pearson_df.join(qa_df[['RL_1']]) 
pearson_df['mean_framewise_displacement'] = pearson_df['RL_1']
pearson_df = pearson_df.drop(['RL_1'], axis=1)
pearson_results = wrap_qcfc(pearson_df, covariate_df=covariate_df) 
pearson_pval, pearson_corr_list = unpack_qcfc(pearson_results)


print(len(pearson_corr_list) )

pearson_results = wrap_qcfc(pearson_df, covariate_df=None) 
pearson_pval, pearson_corr_list = unpack_qcfc(pearson_results)


print(len(pearson_corr_list) )

# +
pearson_df = flatten_mat(pearson_dict, ses_string='ses-1_run-2').T
pearson_df = pearson_df.join(qa_df[['LR_1']]) 
pearson_df['mean_framewise_displacement'] = pearson_df['LR_1']
pearson_df = pearson_df.drop(['LR_1'], axis=1)
pearson_results = wrap_qcfc(pearson_df, covariate_df=covariate_df) 
pearson_pval, pearson_corr_list = unpack_qcfc(pearson_results)


print(len(pearson_corr_list) )

pearson_df = flatten_mat(pearson_dict, ses_string='ses-1_run-2').T
pearson_df = pearson_df.join(qa_df[['LR_1']]) 
pearson_df['mean_framewise_displacement'] = pearson_df['LR_1']
pearson_df = pearson_df.drop(['LR_1'], axis=1)
pearson_results = wrap_qcfc(pearson_df, covariate_df=None) 
pearson_pval, pearson_corr_list = unpack_qcfc(pearson_results)


print(len(pearson_corr_list) )

# +
results_path = '/pscratch/sd/m/mphagen/old_home/old_results/results/2024-04-29_lasso-bic_dict.pkl'
with open(results_path, 'rb') as f:
        lasso_dict = pickle.load(f)

lasso_df = flatten_mat(lasso_dict, ses_string='ses-1_run-1').T
lasso_df = lasso_df.join(qa_df[['RL_1']]) 
lasso_df['mean_framewise_displacement'] = lasso_df['RL_1']
lasso_df = lasso_df.drop(['RL_1'], axis=1)
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

# +
lasso_df = flatten_mat(lasso_dict, ses_string='ses-1_run-2').T
lasso_df = lasso_df.join(qa_df[['LR_1']]) 
lasso_df['mean_framewise_displacement'] = lasso_df['LR_1']
lasso_df = lasso_df.drop(['LR_1'], axis=1)
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

# +
results_path = '/pscratch/sd/m/mphagen/old_home/old_results/results/2024-04-29_uoi_dict.pkl'
with open(results_path, 'rb') as f:
        lasso_dict = pickle.load(f)

lasso_df = flatten_mat(lasso_dict, ses_string='ses-1_run-1').T
lasso_df = lasso_df.join(qa_df[['RL_1']]) 
lasso_df['mean_framewise_displacement'] = lasso_df['RL_1']
lasso_df = lasso_df.drop(['RL_1'], axis=1)
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

# +
lasso_df = flatten_mat(lasso_dict, ses_string='ses-1_run-2').T
lasso_df = lasso_df.join(qa_df[['LR_1']]) 
lasso_df['mean_framewise_displacement'] = lasso_df['LR_1']
lasso_df = lasso_df.drop(['LR_1'], axis=1)
lasso_results = wrap_qcfc(lasso_df, covariate_df=covariate_df) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )

lasso_results = wrap_qcfc(lasso_df, covariate_df=None) 
lasso_pval, lasso_corr_list = unpack_qcfc(lasso_results)


print(len(lasso_corr_list) )
# -






