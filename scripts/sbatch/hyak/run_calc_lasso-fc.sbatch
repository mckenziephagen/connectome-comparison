#!/bin/bash
#SBATCH --job-name=run_fc_lasso
#SBATCH --account=nrdg
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --time=30:00 
#SBATCH --array=1-10 
#SBATCH -o logs/%j_%a.out
#SBATCH -e logs/%j_%a.err


source ${HOME}/.bash_profile
conda activate fc

#GRIFFYN: EDIT
script="${HOME}/mphagen/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="${HOME}/mphagen/connectome-comparison/data/sub_list.txt"

echo $PATH

cat $0 

idx=$(( $SLURM_ARRAY_TASK_ID )) 

sub_id=$( sed -n $(($idx))p $subject_file )
echo $sub_id

python $script --sub_id $sub_id --model 'lassoBIC' --proc_type 'xcpd' --fc_data_path '/gscratch/scrubbed/mphagen/pnc'




