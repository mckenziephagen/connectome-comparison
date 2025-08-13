#!/bin/bash

source /global/homes/m/mphagen/.bash_profile
conda activate fc_py311

script="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/scripts/processing/calc_fc.py"
subject_file="/global/homes/m/mphagen/functional-connectivity/connectome-comparison/data/test_subjects.txt"

idx=1

for i in {0..20}; do

    sub_id=$( sed -n $(($idx+$i))p $subject_file )
    echo $sub_id

    srun -n 1 -c 2 python $script --sub_id $sub_id --ses_id 1 --model 'lassoBIC' --proc_type 'xcpd' &
    srun -n 1 -c 2 python $script --sub_id $sub_id --ses_id 2 --model 'lassoBIC' --proc_type 'xcpd' &

done
wait


