#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=10:00:00
#SBATCH --constraint=cpu
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --job-name=shared
#SBATCH --array=1-100
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m1266

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mphagen@uw.edu

#activate conda envs
source /global/homes/m/mphagen/.bashrc
conda activate fc_py311

#variable definition
script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/test_subjects.txt"

cat $0

subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID))p $subject_file )
echo $subject_id

srun --ntasks 1 --cpus-per-task 2 --exclusive python $script --sub_id $subject_id --model 'uoiLasso' --ses_id 'full' --proc_type 'xcpd' & 

wait


