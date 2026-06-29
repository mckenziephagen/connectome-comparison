#!/bin/bash
#SBATCH --qos=debug
#### SBATCH --qos=regular

#SBATCH --time=20:00
#SBATCH --constraint=cpu
#SBATCH --array=0 
### SBATCH --array=0-10

#SBATCH --error logs/lasso_%A_%a.error
#SBATCH --out logs/lasso_%A_%a.out
#SBATCH --account=m1266
#SBATCH --nodes=1
#SBATCH --ntasks=200

source /global/homes/m/mphagen/.bash_profile
conda activate fc_py311

script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/2026_04_27_test_subs.txt"

### subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/all_subjects.txt"

echo $PATH

cat $0 

idx=$(( $SLURM_ARRAY_TASK_ID*100 ))

for i in {0..100}; do
    
    sub_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $sub_id

   srun -n 1 -c 1 python $script --sub_id $sub_id --ses_id 'full' --model 'lassoBIC' --proc_type 'xcpd' --cv 'ses' &  


 #  srun -n 1 -c 1 python $script --sub_id $sub_id --ses_id 'full' --model 'lassoBIC' --proc_type 'MSMAll_FIX' --cv 'ses' &  

  #  srun -n 1 -c 2 python $script --sub_id $sub_id --ses_id 'full' --model 'lassoBIC' --proc_type 'MSMAll' --cv 'blocks' &  

 #   srun -n 1 -c 1 python $script --sub_id $sub_id --ses_id 'full' --model 'lassoBIC' --proc_type 'xcpd' --cv 'ses' &  

  #  srun -n 1 -c 1 python $script --sub_id $sub_id --ses_id 'full' --model 'lassoBIC' --proc_type 'xcpd' --cv 'blocks' &  

done
wait



