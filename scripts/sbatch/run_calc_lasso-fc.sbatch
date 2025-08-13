#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=1 #1-100%150
#SBATCH --error logs/lasso_%A_%a.error
#SBATCH --out logs/lasso_%A_%a.out
#SBATCH --account=m1266
#SBATCH --nodes=1
#SBATCH --ntasks=200

source /global/homes/m/mphagen/.bash_profile
conda activate fc_py311

script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/test_subjects.txt"

echo $PATH

cat $0 

idx=$(( $SLURM_ARRAY_TASK_ID )) 


for i in {0..100}; do

    sub_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $sub_id

    srun -n 1 -c 2 python $script --sub_id $sub_id --ses_id 1 --model 'lassoBic' --proc_type 'minProc' &  
    srun -n 1 -c 2 python $script --sub_id $sub_id --ses_id 2 --model 'lassoBic' --proc_type 'minProc' &  

done
wait



