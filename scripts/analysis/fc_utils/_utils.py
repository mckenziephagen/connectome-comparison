import os.path as op
import pickle
import numpy as np

def read_pickles(date_str, proc_type, results_dir, cv_type='ses'): 

    with open(op.join(results_dir, proc_type, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-correlation_{cv_type}_relmat.pkl'), 'rb') as f:
        corr_dict = pickle.load(f)

    with open(op.join(results_dir, proc_type, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-uoiLasso_{cv_type}_relmat.pkl'), 'rb') as f:
        uoi_dict = pickle.load(f)

    with open(op.join(results_dir, proc_type, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-lassoBIC_{cv_type}_relmat.pkl'), 'rb') as f:
        lasso_dict = pickle.load(f)

    with open(op.join(results_dir, proc_type, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-partial_correlation_{cv_type}_relmat.pkl'), 'rb') as f:
        pcorr_dict = pickle.load(f)

    with open(op.join(results_dir, proc_type, f'{date_str}_task-rest_space-fsLR_seg-4S156Parcels_den-91k_model-ridgeCV_{cv_type}_relmat.pkl'), 'rb') as f:
        ridge_dict = pickle.load(f)

    return corr_dict, uoi_dict, lasso_dict, pcorr_dict, ridge_dict


def make_mat(mat_dict, ses='full'): 
    mat = np.zeros((100,100, 0))
    for sub in mat_dict.keys():
        temp_mat = mat_dict[sub][f'ses-{ses}']
        np.fill_diagonal(temp_mat, 0)
        mat = np.dstack((mat, temp_mat)  ) 
    return mat