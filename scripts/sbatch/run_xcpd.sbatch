#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=2
#SBATCH --error logs/xcpd_%A_%a.error
#SBATCH --out logs/xcpd_%A_%a.out
#SBATCH --account=m1266
#SBATCH --mail-user=mphagen@uw.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1

source ${HOME}/.bashrc

conda activate fc_w_datalad

sub_id=110007

bids_dataset="/pscratch/sd/m/mphagen/hcp_aw" 

podman-hpc run --volume=$bids_dataset:/tmp/data pennlinc/xcp_d:latest \
     data derivatives/xcp participant \
    --participant_label  $sub_id  \
    --mode linc \
    -v  \
    -t rest \
    --input-type hcp \
    -w  $SCRATCH/tmp/workdir  

