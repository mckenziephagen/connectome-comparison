#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --job-name=single
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m1266
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mphagen@uw.edu

#SBATCH --array=25

#activate conda envs
source /global/homes/m/mphagen/.bashrc
conda activate fc_py311

#variable definition
script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/all_subjects.txt"

cat $0

subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID))p $subject_file )
echo $subject_id
srun --cpu-bind=cores python $script --sub_id $subject_id --model 'uoi-lasso' --ses_id 'ses-1' --profile 


