import os.path as op
import pandas as pd
from functools import reduce
from operator import getitem
import pickle

def extract_r2(file, r2_df): 
    sub = op.basename(file).split('_')[0]

    with open(file, 'rb') as l:
        result_dict = pickle.load(l)

    for ii in range(0,5): 
        for jj in range(0,100):
            r2_df.loc[(f'fold_{ii}', f'node_{jj}'), sub] = reduce(getitem, 
                                                                  (f'fold_{ii}', f'node_{jj}', 'test_r2'), 
                                                                  result_dict) 
    return r2_df

def extract_r2_ses(file, r2_df): 
    sub = op.basename(file).split('_')[0]
    
    with open(file, 'rb') as l:
        result_dict = pickle.load(l)
    
    for ii in range(0,2): 
        for jj in range(0,100):
            r2_df.loc[(f'fold_{ii}', f'node_{jj}'), sub] = reduce(getitem, 
                                                                  (f'fold_{ii}', f'node_{jj}', 'test_r2'), 
                                                                  result_dict) 
    return r2_df