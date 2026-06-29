#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --job-name=parallel
#SBATCH --time=4:30:00
#SBATCH --ntasks=100
#SBATCH --array=1
#SBATCH --nodes=1

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

for i in {0..127}; do
    subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID+$i))p $subject_file )
    echo $subject_id

    srun --ntasks 1 --cpus-per-task 2 --cpu-bind=cores --exclusive \
        python $script --sub_id $subject_id --model 'uoiLasso' \
        --ses_id 'ses-1'--proc_type 'MSMAll' --cv 'ses' &
done 
wait


