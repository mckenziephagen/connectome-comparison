#!/bin/bash
#SBATCH --qos=debug
#SBATCH --job-name=corr

#SBATCH --time=20:00
#SBATCH --constraint=cpu
##SBATCH --array=1 #1-100%150
#SBATCH --array=0 #1-100%150

#SBATCH --error logs/%A_%a_corr.error
#SBATCH --out logs/%A_%a_corr.out
#SBATCH --account=m1266

#SBATCH --nodes=1
#SBATCH --ntasks=200 #can go up to 

#activate conda envs
source /global/homes/m/mphagen/.bash_profile
conda activate fc_py311

script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/test_subjects.txt"
echo $PATH
cat $0 

idx=$(( $SLURM_ARRAY_TASK_ID )) 

for i in {0..100}; do

    #parse subject ID from text file
    sub_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $sub_id

    srun -n 1 -c 1 --exclusive python $script --sub_id $sub_id --ses_id 'full' --model 'correlation' --proc_type 'xcpd' --cv 'ses' &

    # srun -n 1 -c 1 --exclusive python $script --sub_id $sub_id  --ses_id 'full' --model 'correlation' --proc_type 'MSMAll' --cv 'ses'&

wait

done





