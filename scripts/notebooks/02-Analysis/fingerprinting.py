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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as op
import pickle
import random
from glob import glob

# ### HCP 

# +
results_path = f'/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'
with open(glob(op.join(results_path, 'MSMAll', f'*12-10*lassoBIC*pkl'))[0], 'rb') as l:
    old_fix_lasso_dict = pickle.load(l)

_ = [old_fix_lasso_dict[sub].pop("ses-full") for sub in old_fix_lasso_dict.keys()] 

# +
with open(glob(op.join(results_path,'xcpd', '*12-10*lassoBIC*pkl'))[0], 'rb') as l:
    old_xcpd_lasso_dict = pickle.load(l)

_= [old_xcpd_lasso_dict[sub].pop("ses-full") for sub in old_xcpd_lasso_dict.keys()] 


# +
date_str = '2026-05-22'

results_path = f'/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'
proc_type = 'MSMAll_FIX'

with open(glob(op.join(results_path,proc_type, f'*{date_str}*lassoBIC*pkl'))[0], 'rb') as l:
    fix_lasso_dict = pickle.load(l)

_ = [fix_lasso_dict[sub].pop("ses-full") for sub in fix_lasso_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'*{date_str}*ridgeCV*pkl'))[0], 'rb') as l:
    fix_ridge_dict = pickle.load(l)

_ = [fix_ridge_dict[sub].pop("ses-full") for sub in fix_ridge_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'*{date_str}*-correlation*pkl'))[0], 'rb') as l:
    fix_pearson_dict = pickle.load(l)

_= [fix_pearson_dict[sub].pop("ses-full") for sub in fix_pearson_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'*{date_str}*partial_corr*pkl'))[0], 'rb') as l:
    fix_pcorr_dict = pickle.load(l)

_= [fix_pcorr_dict[sub].pop("ses-full") for sub in fix_pcorr_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'*{date_str}*uoi*pkl'))[0], 'rb') as l:
    fix_uoi_dict = pickle.load(l)

_ = [fix_uoi_dict[sub].pop("ses-full") for sub in fix_uoi_dict.keys()] 

# ###### 
proc_type = 'xcpd'
with open(glob(op.join(results_path,proc_type, f'*{date_str}*-corr*pkl'))[0], 'rb') as l:
    xcpd_pearson_dict = pickle.load(l)

_= [xcpd_pearson_dict[sub].pop("ses-full") for sub in xcpd_pearson_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'{date_str}*lassoBIC*pkl'))[0], 'rb') as l:
    xcpd_lasso_dict = pickle.load(l)

_= [xcpd_lasso_dict[sub].pop("ses-full") for sub in xcpd_lasso_dict.keys()] 

with open(glob(op.join(results_path,proc_type, f'*{date_str}*ridgeCV*pkl'))[0], 'rb') as l:
    xcpd_ridge_dict = pickle.load(l)

_= [xcpd_ridge_dict[sub].pop("ses-full") for sub in xcpd_ridge_dict.keys()] 

with open(glob(op.join(results_path,'xcpd', f'*{date_str}*uoi*pkl'))[0], 'rb') as l:
    xcpd_uoi_dict = pickle.load(l)

_= [xcpd_uoi_dict[sub].pop("ses-full") for sub in xcpd_uoi_dict.keys()] 


# +
#### PNC 
# with open(glob(op.join(results_path,'PNC', f'*{date_str}*lassoBIC*pkl'))[0], 'rb') as l:
#     pnc_lasso_dict = pickle.load(l)

# _ = [pnc_lasso_dict[sub].pop("ses-full") for sub in pnc_lasso_dict.keys()] 
# -

def fingerprint(target_dict, database_dict): 
    corr_dict = {}
    for target_sub in target_dict.keys(): 
        if target_sub not in corr_dict.keys(): 
            corr_dict[target_sub] = {}
    
        for db_sub in database_dict.keys():
         #   print(db_sub)
            try: 
                corr = np.corrcoef(target_dict[target_sub]['ses-1'].ravel(),
                                    database_dict[db_sub]['ses-2'].ravel())[0,1]
                corr_dict[target_sub][db_sub] = corr
            except KeyError: 
                pass
    
    fp_list = []
    rank_list = [] 
    for sub, value in corr_dict.items():

        fp_list.append(sub.split('_')[0] in (max(corr_dict[sub],key=corr_dict[sub].get)))
    acc = sum(fp_list) / len(corr_dict)
    # print(acc)
    return corr_dict, fp_list, acc


# note for future self: max(corr_dict ...) returns the keys associated with the highest key.value(). 

# +
fix_dict = {} 
fix_dict['lasso'] = {} 
fix_dict['pearson'] = {} 
fix_dict['uoi'] = {} 
fix_dict['pcorr'] = {} 

fix_dict['thresh_lasso'] = {} 
fix_dict['thresh_uoi'] = {} 
fix_dict['ridge'] = {} 

# +
lasso_dict, fp_list, _ = fingerprint(fix_lasso_dict, fix_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()

fix_dict['lasso']['avg'] = np.mean(acc_list)
fix_dict['lasso']['lower'] =    acc_list[50]
fix_dict['lasso']['upper'] =    acc_list[-50]
fix_dict['lasso']['boot_vals'] = acc_list


print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(fix_ridge_dict, fix_ridge_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()

fix_dict['ridge']['avg'] = np.mean(acc_list)
fix_dict['ridge']['lower'] =    acc_list[50]
fix_dict['ridge']['upper'] =    acc_list[-50]
fix_dict['ridge']['boot_vals'] = acc_list


print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(fix_pcorr_dict, fix_pcorr_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()

fix_dict['pcorr']['avg'] = np.mean(acc_list)
fix_dict['pcorr']['lower'] =    acc_list[50]
fix_dict['pcorr']['upper'] =    acc_list[-50]
fix_dict['pcorr']['boot_vals'] = acc_list


print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
old_lasso_dict, fp_list, _ = fingerprint(old_fix_lasso_dict, old_fix_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
    
acc_list.sort()

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
res_fix_pearson, fp_list, _ = fingerprint(fix_pearson_dict, fix_pearson_dict)

acc_list = [] 
for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
acc_list.sort()


fix_dict['pearson']['avg'] = np.mean(acc_list)
fix_dict['pearson']['lower'] =  acc_list[50]
fix_dict['pearson']['upper'] =   acc_list[-50]
fix_dict['pearson']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(fix_uoi_dict, fix_uoi_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
acc_list.sort()

fix_dict['uoi']['avg'] = np.mean(acc_list)
fix_dict['uoi']['lower'] =    acc_list[50]
fix_dict['uoi']['upper'] =    acc_list[-50]
fix_dict['uoi']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
# pnc_dict = {} 
# pnc_dict['lasso'] = {} 
# pnc_dict['pearson'] = {} 

# +
# _, fp_list, _ = fingerprint(pnc_lasso_dict, pnc_lasso_dict)

# acc_list = [] 

# for i in range(1000): 
#     acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
# acc_list.sort()

# pnc_dict['lasso']['avg'] = np.mean(acc_list)
# pnc_dict['lasso']['lower'] =    acc_list[50]
# pnc_dict['lasso']['upper'] =    acc_list[-50]
# pnc_dict['lasso']['boot_vals'] = acc_list

# print("average: ", np.mean(acc_list))
# print("CI: ", acc_list[50], acc_list[-50]) 

# +
xcpd_dict = {} 
xcpd_dict['lasso'] = {} 
xcpd_dict['pearson'] = {} 
xcpd_dict['uoi'] = {} 

xcpd_dict['thresh_lasso'] = {} 
xcpd_dict['thresh_uoi'] = {} 

# +
_, fp_list, _ = fingerprint(xcpd_pearson_dict, xcpd_pearson_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
acc_list.sort()

xcpd_dict['pearson']['avg'] = np.mean(acc_list)
xcpd_dict['pearson']['lower'] =  acc_list[50]

xcpd_dict['pearson']['upper'] =   acc_list[-50]
xcpd_dict['pearson']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(xcpd_lasso_dict, xcpd_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 

acc_list.sort()

xcpd_dict['lasso']['avg'] = np.mean(acc_list)
xcpd_dict['lasso']['lower'] =  acc_list[50]

xcpd_dict['lasso']['upper'] =  acc_list[-50]
xcpd_dict['lasso']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(xcpd_uoi_dict, xcpd_uoi_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 

acc_list.sort()

xcpd_dict['uoi']['avg'] = np.mean(acc_list)
xcpd_dict['uoi']['lower'] =  acc_list[50]
xcpd_dict['uoi']['upper'] =  acc_list[-50]
xcpd_dict['uoi']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 
# -

fix_boot_df = pd.DataFrame.from_dict([fix_dict['lasso']['boot_vals'], 
                                         fix_dict['pearson']['boot_vals'],
                                         fix_dict['uoi']['boot_vals'] , 
                                        fix_dict['ridge']['boot_vals'] , 
                                        fix_dict['pcorr']['boot_vals']] ) 

                                         # fix_dict['thresh_lasso']['boot_vals'] , 
                                         # fix_dict['thresh_uoi']['boot_vals']])
fix_boot_df.index = ['lasso', 'pearson', 'uoi', 'ridge', 'pcorr'] 
fix_boot_df = fix_boot_df.T
fix_boot_df.to_csv('results/msmall_fp_boot-vals_2026-05-25.csv')

assert 1 == 0 

# ### Thresholded Analyses

# +
proc_type = 'MSMAll_FIX'
with open(glob(op.join(results_path, proc_type,  '*12-10**corr-uoiThresh*pkl'))[0], 'rb') as l:
    thresh_fix_uoi_dict = pickle.load(l)

_ = [thresh_fix_uoi_dict[sub].pop("ses-full") for sub in thresh_fix_uoi_dict.keys()] 

with open(glob(op.join(results_path, proc_type, '*12-10**corr-lassoThresh*pkl'))[0], 'rb') as l:
    thresh_msmall_lasso_dict = pickle.load(l)

_ = [thresh_msmall_lasso_dict[sub].pop("ses-full") for sub in thresh_msmall_lasso_dict.keys()] 


# +
corr_dict, fp_list, _ = fingerprint(thresh_msmall_lasso_dict, thresh_msmall_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(thresh_msmall_lasso_dict)) 
    
acc_list.sort()
np.mean(acc_list) 

msmall_dict['thresh_lasso']['avg'] = np.mean(acc_list)
msmall_dict['thresh_lasso']['lower'] =  acc_list[50]
msmall_dict['thresh_lasso']['upper'] =  acc_list[-50]
msmall_dict['thresh_lasso']['boot_vals'] = acc_list

# +
corr_dict, fp_list, _ = fingerprint(thresh_msmall_uoi_dict, thresh_msmall_uoi_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(thresh_msmall_uoi_dict)) 
    
acc_list.sort()
np.mean(acc_list) 

msmall_dict['thresh_uoi']['avg'] = np.mean(acc_list)
msmall_dict['thresh_uoi']['lower'] =  acc_list[50]
msmall_dict['thresh_uoi']['upper'] =  acc_list[-50]
msmall_dict['thresh_uoi']['boot_vals'] = acc_list

# +
proc_type = 'xcpd'
with open(glob(op.join(results_path, proc_type,  '*12-10**corr-uoiThresh*pkl'))[0], 'rb') as l:
    thresh_xcpd_uoi_dict = pickle.load(l)

_ = [thresh_xcpd_uoi_dict[sub].pop("ses-full") for sub in thresh_xcpd_uoi_dict.keys()] 

with open(glob(op.join(results_path, proc_type, '*12-10**corr-lassoThresh*pkl'))[0], 'rb') as l:
    thresh_xcpd_lasso_dict = pickle.load(l)

_ = [thresh_xcpd_lasso_dict[sub].pop("ses-full") for sub in thresh_xcpd_lasso_dict.keys()] 


# +
corr_dict, fp_list, _ = fingerprint(thresh_xcpd_uoi_dict, thresh_xcpd_uoi_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(thresh_xcpd_uoi_dict)) 
    
acc_list.sort()
np.mean(acc_list) 


xcpd_dict['thresh_uoi']['avg'] = np.mean(acc_list)
xcpd_dict['thresh_uoi']['lower'] =  acc_list[50]
xcpd_dict['thresh_uoi']['upper'] =  acc_list[-50]
xcpd_dict['thresh_uoi']['boot_vals'] = acc_list

# +
corr_dict, fp_list, _ = fingerprint(thresh_xcpd_lasso_dict, thresh_xcpd_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(thresh_xcpd_lasso_dict)) 
    
acc_list.sort()
np.mean(acc_list) 


xcpd_dict['thresh_lasso']['avg'] = np.mean(acc_list)
xcpd_dict['thresh_lasso']['lower'] =  acc_list[50]
xcpd_dict['thresh_lasso']['upper'] =  acc_list[-50]
xcpd_dict['thresh_lasso']['boot_vals'] = acc_list
# -

xcpd_boot_df = pd.DataFrame.from_dict([xcpd_dict['lasso']['boot_vals'], 
                                       xcpd_dict['pearson']['boot_vals'],
                                       xcpd_dict['uoi']['boot_vals'], 
                                       xcpd_dict['thresh_lasso']['boot_vals'] , 
                                       xcpd_dict['thresh_uoi']['boot_vals']])
xcpd_boot_df.index = ['lasso', 'pearson', 'uoi', 'thresh_lasso', 'thresh_uoi']
xcpd_boot_df = xcpd_boot_df.T
xcpd_boot_df.to_csv('xcpd_fp_boot-vals_2025-12-10.csv')

msmall_boot_df = pd.DataFrame.from_dict([msmall_dict['lasso']['boot_vals'], 
                                         msmall_dict['pearson']['boot_vals'],
                                         msmall_dict['uoi']['boot_vals'] , 
                                         msmall_dict['thresh_lasso']['boot_vals'] , 
                                         msmall_dict['thresh_uoi']['boot_vals']])
msmall_boot_df.index = ['lasso', 'pearson', 'uoi', 'thresh_lasso', 'thresh_uoi']
msmall_boot_df = msmall_boot_df.T
msmall_boot_df.to_csv('results/msmall_fp_boot-vals_2026_04_27.csv')

assert 0 == 1

for ii in xcpd_dict.keys(): 
    del xcpd_dict[ii]['boot_vals']

for ii in msmall_dict.keys(): 
    del msmall_dict[ii]['boot_vals']

pd.DataFrame.from_dict(xcpd_dict).to_csv(op.join(results_path,
    'xcpd_ci_2025_12-10.csv'))

pd.DataFrame.from_dict(msmall_dict).to_csv(op.join(results_path,
    'msmall_ci_2025_12-10.csv'))

msmall_boot_df['proc_type'] = 'msmall'

xcpd_boot_df['proc_type'] = 'xcpd'

plot_df = pd.concat([msmall_boot_df, xcpd_boot_df]) 


msmall_df = pd.DataFrame.from_dict(msmall_dict) 
xcpd_df = pd.DataFrame.from_dict(xcpd_dict) 

plot_df = plot_df.melt(id_vars=['proc_type'])


plot_df = plot_df.replace({'msmall': 'MinProc',
                'xcpd': 'XCPD',
                'lasso': 'LASSO',
                'uoi': 'UoI', 
                 'pearson': 'Pearson'})

plot_df = plot_df.rename(columns={'variable':'model'}) 

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette()


models = ['Pearson',  'LASSO', 'UoI', ]
pipelines = ['MinProc', 'XCPD']
labels = models + pipelines
palette_colors = sns.color_palette("deep")
palette_dict = {labels: color for labels, color in zip(labels, palette_colors)}


palette_dict

# +
# plt.figure(figsize=(5, 5))

g = sns.violinplot(data=plot_df, 
            x='proc_type', 
            y='value',
           hue='model', 
               order=pipelines,
               hue_order=models,
              palette=[palette_dict.get(key) for key in models], 
             legend=False
) 
g.set_xlabel('') 
# g.legend(loc="upper right", ncol=len(models))
g.set_ylabel('Accuracy', weight='semibold')
plt.title('Fingerprinting Identification', weight='bold')
##for consistency
plt.grid(axis='y', linewidth=.25)
sns.despine()
plt.tight_layout() 

plt.savefig('plots/final_fingerprinting.png', 
            dpi=400,
            bbox_inches='tight',)

# +
# plt.figure(figsize=(6,4))

g = sns.violinplot(data=plot_df, 
            x='model', 
            y='value',
           hue='proc_type', 
               order=models,
               hue_order=pipelines,
              palette=[palette_dict.get(key) for key in pipelines], 
                   legend=False
) 
g.set_ylabel('Accuracy', weight='semibold')
plt.title('Fingerprinting Identification', weight='bold')
g.set_xlabel('') 

##for consistency
plt.grid(axis='y', linewidth=.25)
sns.despine()
plt.savefig('plots/final_fingerprinting_2.png', 
            dpi=400,
            bbox_inches='tight',)

# +
sns.set_palette('deep') 

xcpd_df.columns == msmall_df.columns

x = np.array([1,2, 3]) 
plt.bar(x-.2, msmall_df.loc['avg'], width=.4, color=sns.color_palette()[0])
plt.bar(x+.2, xcpd_df.loc['avg'], width=.4, color=sns.color_palette()[1]) 
plt.legend(['MinProc', 'XCPD'], title='Processing Pipeline')
plt.xticks(x, ['LASSO', 'Pearson', 'UoI']) 
plt.ylim([0, 1])
msmall_upper = msmall_df.loc['upper']  - msmall_df.loc['avg']
msmall_lower = msmall_df.loc['avg'] - msmall_df.loc['lower']

xcpd_upper = xcpd_df.loc['upper']  - xcpd_df.loc['avg']
xcpd_lower = xcpd_df.loc['avg'] - xcpd_df.loc['lower']

plt.errorbar(x-.2, y=msmall_df.loc['avg'], 
             yerr=[msmall_lower, msmall_upper],
                 color='black', linestyle='', 
            elinewidth=2, barsabove=True, capsize=3)

plt.errorbar(x+.2, y=xcpd_df.loc['avg'], 
             yerr=[xcpd_lower, xcpd_upper],
                 color='black', linestyle='', 
            elinewidth=2, barsabove=True, capsize=3)

plt.title('Fingerprinting Accuracy', weight='bold') 
plt.ylabel('Accuracy of Identification', weight='semibold') 
plt.xlabel('Model', weight='semibold') 
plt.savefig('plots/2025-12-10_fingerprinting_acc.png', 
            transparent=True) 
# -

















































