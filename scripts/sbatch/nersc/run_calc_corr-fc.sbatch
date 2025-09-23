#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=15:00qq
#SBATCH --constraint=cpu
#SBATCH --array=1 #1-100%150
#SBATCH --error logs/%A_%a_corr.error
#SBATCH --out logs/%A_%a_corr.out
#SBATCH --account=m1266

#SBATCH --nodes=1
#SBATCH --ntasks=100 #can go up to 100

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
    subject_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $subject_id

    srun -n 1 -c 1 --exclusive python $script --sub_id $subject_id --ses_id 1 --model 'correlation' --proc_type 'minProc' &
    srun -n 1 -c 1 --exclusive python $script --sub_id $subject_id --ses_id 2 --model 'correlation' --proc_type 'minProc' &
  
done

for i in {0..100}; do

    #parse subject ID from text file
    subject_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $subject_id

    srun -n 1 -c 1 --exclusive python $script --sub_id $subject_id --ses_id 1 --model 'correlation' --proc_type 'xcpd' &
    srun -n 1 -c 1 --exclusive python $script --sub_id $subject_id --ses_id 2 --model 'correlation' --proc_type 'xcpd' &
  
done

wait



