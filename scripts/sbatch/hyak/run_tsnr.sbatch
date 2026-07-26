#!/bin/bash
#SBATCH --job-name=run_fc_preg
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --time=30:00 
#SBATCH --array=1-100
#SBATCH --mem=20GB

#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out

source ${HOME}/.bash_profile
conda activate fc_py

#variable definition
script="/gscratch/escience/mphagen/functional-connectivity/connectome-comparison/scripts/notebooks/01-Processing/tsnr_calculations.py"
subject_file="/gscratch/escience/mphagen/functional-connectivity/connectome-comparison/data/documented_subs.txt"

cat $0

subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID))p $subject_file )
echo $subject_id

srun python $script --sub_id $subject_id --proc_type 'raw' & 

wait



