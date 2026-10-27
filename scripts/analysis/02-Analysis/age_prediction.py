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

# +
import os 
import os.path as op
import nilearn
from nilearn import datasets
import pandas as pd
from glob import glob
import pickle
from _functions import fit_model
import numpy as np

from sklearn.linear_model import LassoCV, RidgeCV

# +
import argparse
args = argparse.Namespace()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
#https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter

parser.add_argument('--seed',default='1')
parser.add_argument('--pred', default='Age_in_Yrs')
parser.add_argument('--cpm', default=False)

parser.add_argument('--fc_data_path', 
                    default='/pscratch/sd/m/mphagen/human-connectome-project-openaccess/HCP1200')

parser.add_argument('--pheno_data_path', default='/global/u1/m/mphagen/functional-connectivity/connectome-comparison/data')
parser.add_argument('--results_path', default='/global/u1/m/mphagen/functional-connectivity/connectome-comparison/results')


try: 
    os.environ['_']
    args = parser.parse_args()
    notebook = False
except KeyError: 
    args = parser.parse_args([])
    notebook = True

seed = args.seed
pred = args.pred

fc_data_path = args.fc_data_path
pheno_data_path = args.pheno_data_path
results_path = args.results_path


# -

def make_fc_df(results_dict, ses): 
    fc_df = pd.DataFrame(columns=results_dict.keys()) 
    
    for key in results_dict.keys(): 
        fc_df[key] = results_dict[key][f'ses-{ses}'].flatten() 
    fc_df = fc_df.T
    # if model == 'correlation': 
    #      fc_df
    # fc_df = fc_df.drop(fc_df.columns[fc_df.nunique() <= 1],axis=1) 
    return fc_df


with open(op.join(pheno_data_path, 'test_subjects.txt'), 'r') as file:
    sub_list = file.read().splitlines()
print(len(sub_list))

# +
atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_filename = atlas.maps
#plot_roi(atlas_filename, title="Schaefer_2018 atlas", view_type="contours")

atlas_17 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
atlas17_filename = atlas.maps


# +
full_labels = atlas.labels[1:]
labels_7 = [label.split("_")[2] for label in full_labels]

full_labels_17 = atlas_17.labels[1:]
labels_17 = [label.split("_")[2] for label in full_labels_17]

# +
pheno_df = pd.read_csv(op.join(pheno_data_path, 'RESTRICTED_arokem_1_31_2022_23_26_45.csv'), 
                        header=0)
pheno_df.set_index('Subject', inplace=True, drop=False)
pheno_df.rename(columns={'Subject': 'id'}, inplace=True)
pheno_df.index = [str(ii) for ii in pheno_df.index]

pheno_df = pheno_df.loc[sub_list]

pheno_df.index = [f'sub-{ii}' for ii in pheno_df.index]
# -

proc_type = 'MSMAll' 
with open(glob(op.join(results_path, proc_type, f'*12-10*lassoBIC*relmat.pkl'))[0], 'rb') as l:
    connectome_dict = pickle.load(l)

fc_df = make_fc_df(connectome_dict, ses='full') 
fc_df.rename(columns={num: f'edge_{num}' for num in range(10000)}, inplace=True)

all_model_names = [ 'lassoCV', 'PCLasso']
data_names = ['lasso', 'corr']

results_dict = {model:{data:{} for data in data_names} for model in all_model_names}

num_edges = 10000

length_of_connectomes = [[f'edge_{num}' for num in range(num_edges)]]

lasso_data = pd.concat([fc_df, pheno_df['Age_in_Yrs']], join='inner', axis=1)

folds = 5
component = None



pca_r2_dict = {}
for model in [LassoCV()]:
    for ii in range(1,500): 
        resample = lasso_data.sample(frac=1, replace=True)
    
        output = fit_model(resample[length_of_connectomes[0]], 
               resample['Age_in_Yrs'], 
               model, 
               component='PCA',
               folds=folds, 
               verbose=False)
        r2_list.append(output['full_r2'])
    print(np.median(r2_list) ) 

for model in [LassoCV(), RidgeCV()]:
    for ii in range(1,100): 
        resample = lasso_data.sample(frac=1, replace=True)
    
        output = fit_model(resample[length_of_connectomes[0]], 
               resample['Age_in_Yrs'], 
               model, 
               component=None,
               folds=folds, 
               verbose=False)
        r2_list.append(output['full_r2'])
    print(np.median(r2_list) ) 

0.02531620831379522
0.032417455027790254

0.02474805116951473
0.025211122041595424






