#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=10:00
#SBATCH --constraint=cpu
#SBATCH --array=1
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m3513
#SBATCH --mail-user=mphagen@uw.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1

source ${HOME}/.bashrc

script_path="${HOME}/functional-connectivity/connectome-comparison/scripts/processing/extract_timeseries.py" 
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/all_subjects.txt"

conda activate fc_w_datalad

srun parallel python $script_path --subject_id '{}' ::: $( head -n 10 $subject_file )

