#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=1
#SBATCH --error logs/parc_%A_%a.error
#SBATCH --out logs/parc_%A_%a.out
#SBATCH --account=m1266
#SBATCH --mail-user=mphagen@uw.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1


source ${HOME}/.bashrc
conda activate xcpd

script_path="${HOME}/functional-connectivity/connectome-comparison/scripts/processing/extract_timeseries.py" 

subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/unproc_subs.txt"

idx=$(( $SLURM_ARRAY_TASK_ID*200 )) 

## srun parallel python $script_path --sub_id '{}' --MSMAll True --FIX True ::: $( head -n $idx $subject_file | tail -n 200  )

srun parallel python $script_path --sub_id '{}' --proc 'MSMAll_FIX' ::: $( head -n $idx $subject_file | tail -n 200  )
