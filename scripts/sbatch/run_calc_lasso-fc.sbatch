#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=1 #1-100%150
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m1266
#SBATCH --nodes=1
#SBATCH --ntasks=100

source /global/homes/m/mphagen/.bash_profile
conda activate fc

script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/all_subjects.txt"

echo $PATH

for i in {0..100}; do

    subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID+$i))p $subject_file )
    echo $subject_id
    srun -n 1 -c 2 python $script --subject_id $subject_id --model 'lasso-bic' &    
done
wait



