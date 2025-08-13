#### About

Calculate and compare functional connectivity matrices from minimally pre-processed functional MRI data. 


#### Installation

TODO: Conda env here

installing mpi4py on nersc:
https://docs.nersc.gov/development/languages/python/parallel-python/

Downloading HCP data using datalad: 
`datalad clone https://github.com/datalad-datasets/hcp-functional-connectivity.git` 


#### Basic useage

## Timeseries extraction
To get parcellated timeseries from a minimally processed fMRI scan, run `processing/extract_timeseries.py`. This accepts an HCP subject ID, and will output a nii from connectome workbench

To calculate a functional connectivity matrix from those timeseries, run `processing/calc_fc.py`. 

TODO: Other analysis script documentation. 

#### Other notes

While the HCP data isn't organized according to the BIDS standard, the derivatives I generate follow BEP-17 ("Relationship Matrices"). 

Parcellated timeseries created by `01-Processing/extract_timeseries.py` are saved to `$SCRATCH/derivatives/timeseries/...` according to their atlas and model

Generated connectivity matrices live in `$SCRATCH/hcp-functional-connectivity/derivatives/connectivity-matrices/...` according to their atlas and model. 

Order to run scripts in 
1. extract_timeseries.py for minProc
2. combine_timeseries.py for minProc and xcpd
3. calc_fc.py for minProc and xcpd













