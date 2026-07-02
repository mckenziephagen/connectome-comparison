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

import pytest
import subprocess
import os.path as op
import configparser
import os
import sys
from glob import glob 
config = configparser.ConfigParser()
config.read('../scripts/config.ini')


def run_calc_fc(python, script, flags): 
    
    output = subprocess.run([python, script, *flags], 
             capture_output=True, 
             env=os.environ) 

    return output


def test_hcp_ses():
    """ Test whether calc_fc.pu runs for HCP. """ 
    script = op.join(config['General']['script_dir'], 
                     'connectome-comparison', 'scripts', 'processing', 'calc_fc.py')
    python=sys.executable
    
    
    test_id = '101309'  
    [os.remove(ii) for ii in glob(op.join('test_results', f'*{test_id}*'))]
       
    
    output = run_calc_fc(python, script, flags = ['--sub_id', test_id,
                                              '--proc_type', 'MSMAll_FIX',
                                              '--dataset', 'hcp',
                                              '--cv', 'ses',
                                              '--test', 'True'] )
    
    output_files = glob(op.join('test_results', f'*{test_id}*'))
    
    assert output.returncode == 0
    assert len(output_files) == 2 


def test_hcp_ses_fail():
    """ Test whether calc_fc.py fails for nonexistent 
    sub_id. """ 
    script = op.join(config['General']['script_dir'], 
                     'connectome-comparison', 'scripts', 'processing', 'calc_fc.py')
    python=sys.executable

    test_id = '111111'
    [os.remove(ii) for ii in glob(op.join('test_results', '*{test_id}*'))]
    
    output = run_calc_fc(python, script, flags = ['--sub_id', test_id,
                                                   '--proc_type', 'MSMAll_FIX',
                                                  '--dataset', 'hcp',
                                                  '--cv', 'ses',
                                                  '--test', 'True'] )

    assert output.returncode == 1


def test_pnc_task():
    """ Test whether calc_fc.pu runs for HCP. """ 
    script = op.join(config['General']['script_dir'], 
                     'connectome-comparison', 'scripts', 'processing', 'calc_fc.py')
    python=sys.executable
    
    
    test_id = '97005004'  
    [os.remove(ii) for ii in glob(op.join('test_results', f'*{test_id}*'))]
       
    
    output = run_calc_fc(python, script, flags = ['--sub_id', test_id,
                                              '--proc_type', 'xcpd',
                                              '--dataset', 'pnc',
                                              '--cv', 'task',
                                              '--test', 'True'] )
    
    output_files = glob(op.join('test_results', f'*{test_id}*'))
    
    assert output.returncode == 0
    assert len(output_files) == 2 
