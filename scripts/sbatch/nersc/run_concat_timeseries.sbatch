#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=15:00
#SBATCH --constraint=cpu
#SBATCH --array=0
#SBATCH --error logs/concat_%A_%a.error
#SBATCH --out logs/concat_%A_%a.out
#SBATCH --account=m1266
#SBATCH --mail-user=mphagen@uw.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1


source ${HOME}/.bashrc

script_path="${HOME}/functional-connectivity/connectome-comparison/scripts/processing/concat_timeseries.py" 
#subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/notebooks/exploratory/msmall_1.txt"

subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/2026_04_27_test_subs.txt"

conda activate xcpd

idx=$(( $SLURM_ARRAY_TASK_ID*100 )) 

## srun parallel python $script_path --run_id 1 --proc_type 'MSMAll_FIX' --sub_id '{}' ::: $( head -n $idx | tail -n 100 $subject_file  )

## srun parallel python $script_path --run_id 2 --proc_type 'MSMAll_FIX' --sub_id '{}' ::: $( head -n $idx | tail -n 100 $subject_file  )

srun parallel python $script_path --run_id 1 --proc_type 'MSMAll_FIX' --sub_id '{}' ::: $( cat $subject_file  )

srun parallel python $script_path --run_id 2 --proc_type 'MSMAll_FIX' --sub_id '{}' ::: $( cat $subject_file  )

