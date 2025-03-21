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

echo $PATH

for i in {0..100}; do

    subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID+$i))p ../test_subjects.txt )
    echo $subject_id
    srun -n 1 -c 2 python ../01-Processing/calc_fc.py --subject_id $subject_id --model 'lasso-bic' &    
done
wait



