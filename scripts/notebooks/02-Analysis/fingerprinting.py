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

# +
results_path = f'/global/homes/m/mphagen/functional-connectivity/connectome-comparison/results'

with open(glob(op.join(results_path,'MSMAll', '*12-10*lassoBIC*pkl'))[0], 'rb') as l:
    msmall_lasso_dict = pickle.load(l)

_ = [msmall_lasso_dict[sub].pop("ses-full") for sub in msmall_lasso_dict.keys()] 

with open(glob(op.join(results_path,'MSMAll', '*12-10*corr*pkl'))[0], 'rb') as l:
    msmall_pearson_dict = pickle.load(l)

_= [msmall_pearson_dict[sub].pop("ses-full") for sub in msmall_pearson_dict.keys()] 

with open(glob(op.join(results_path,'MSMAll', '*12-10*uoi*pkl'))[0], 'rb') as l:
    msmall_uoi_dict = pickle.load(l)

_ = [msmall_uoi_dict[sub].pop("ses-full") for sub in msmall_uoi_dict.keys()] 

### 
with open(glob(op.join(results_path,'xcpd', '*12-10*corr*pkl'))[0], 'rb') as l:
    xcpd_pearson_dict = pickle.load(l)

_= [xcpd_pearson_dict[sub].pop("ses-full") for sub in xcpd_pearson_dict.keys()] 

with open(glob(op.join(results_path,'xcpd', '*12-10*lassoBIC*pkl'))[0], 'rb') as l:
    xcpd_lasso_dict = pickle.load(l)

_= [xcpd_lasso_dict[sub].pop("ses-full") for sub in xcpd_lasso_dict.keys()] 



with open(glob(op.join(results_path,'xcpd', '*12-10*uoi*pkl'))[0], 'rb') as l:
    xcpd_uoi_dict = pickle.load(l)

_= [xcpd_uoi_dict[sub].pop("ses-full") for sub in xcpd_uoi_dict.keys()] 

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

msmall_dict = {} 
msmall_dict['lasso'] = {} 
msmall_dict['pearson'] = {} 
msmall_dict['uoi'] = {} 

# +
corr_dict, fp_list, _ = fingerprint(msmall_lasso_dict, msmall_lasso_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(msmall_lasso_dict)) 
    
acc_list.sort()

msmall_dict['lasso']['avg'] = np.mean(acc_list)
msmall_dict['lasso']['lower'] =    acc_list[50]
msmall_dict['lasso']['upper'] =    acc_list[-50]
msmall_dict['lasso']['boot_vals'] = acc_list


print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 
# -

np.array(corr_dict[key].values())

corr_dict[key].get() 

key

corr_dict[key]['sub-979984']

key

max(corr_dict[key]) 

for key in corr_dict.keys():
    print(key) 
    # print(key,value)
    print( key.split('_')[0]) 
    print(key,  max(corr_dict[key],key=corr_dict[key].get), key in (max(corr_dict[key],key=corr_dict[key].get)))

pd.DataFrame(corr_dict['sub-793465']) 

# +
_, fp_list, _ = fingerprint(msmall_pearson_dict, msmall_pearson_dict)

acc_list = [] 
for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
acc_list.sort()


msmall_dict['pearson']['avg'] = np.mean(acc_list)
msmall_dict['pearson']['lower'] =  acc_list[50]
msmall_dict['pearson']['upper'] =   acc_list[-50]
msmall_dict['pearson']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 

# +
_, fp_list, _ = fingerprint(msmall_uoi_dict, msmall_uoi_dict)

acc_list = [] 

for i in range(1000): 
    acc_list.append(sum(random.choices(fp_list, k=len(fp_list))) / len(fp_list)) 
acc_list.sort()

msmall_dict['uoi']['avg'] = np.mean(acc_list)
msmall_dict['uoi']['lower'] =    acc_list[50]
msmall_dict['uoi']['upper'] =    acc_list[-50]
msmall_dict['uoi']['boot_vals'] = acc_list

print("average: ", np.mean(acc_list))
print("CI: ", acc_list[50], acc_list[-50]) 
# -

xcpd_dict = {} 
xcpd_dict['lasso'] = {} 
xcpd_dict['pearson'] = {} 
xcpd_dict['uoi'] = {} 

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

xcpd_boot_df = pd.DataFrame.from_dict([xcpd_dict['lasso']['boot_vals'], 
                                 xcpd_dict['pearson']['boot_vals'],
                                  xcpd_dict['uoi']['boot_vals'] ])
xcpd_boot_df.index = ['lasso', 'pearson', 'uoi']
xcpd_boot_df = xcpd_boot_df.T
xcpd_boot_df.to_csv('xcpd_boot-vals_2025-12-10.csv')

msmall_boot_df = pd.DataFrame.from_dict([msmall_dict['lasso']['boot_vals'], 
                                 msmall_dict['pearson']['boot_vals'],
                                  msmall_dict['uoi']['boot_vals'] ])
msmall_boot_df.index = ['lasso', 'pearson', 'uoi']
msmall_boot_df = msmall_boot_df.T
msmall_boot_df.to_csv('msmall_boot-vals_2025-12-10.csv')

for ii in xcpd_dict.keys(): 
    del xcpd_dict[ii]['boot_vals']

for ii in msmall_dict.keys(): 
    del msmall_dict[ii]['boot_vals']

pd.DataFrame.from_dict(xcpd_dict).to_csv(op.join(results_path,
    'xcpd_ci_2025_12-10.csv'))

pd.DataFrame.from_dict(msmall_dict).to_csv(op.join(results_path,
    'msmall_ci_2025_12-10.csv'))

msmall_boot_df['model'] = 'msmall'

xcpd_boot_df['model'] = 'xcpd'

plot_df = pd.concat([msmall_boot_df, xcpd_boot_df]) 

msmall_df = pd.DataFrame.from_dict(msmall_dict) 
xcpd_df = pd.DataFrame.from_dict(xcpd_dict) 

import matplotlib.pyplot as plt
import seaborn as sns

palette_dict= {'Pearson': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
 'UoI': (1.0, 0.4980392156862745, 0.054901960784313725),
 'LASSO': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)}

colors = sns.color_palette()


colors

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





