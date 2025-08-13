#!/bin/bash
#SBATCH --account=m1266
#SBATCH --error debug_logs/%A_%a.error
#SBATCH --out debug_logs/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mphagen@uw.edu

#SBATCH --qos=debug
#SBATCH --time=1:00
#SBATCH --constraint=cpu
#SBATCH --ntasks=10
#SBATCH --nodes=1

echo "cpus per node:" 
echo $SLURM_JOB_CPUS_PER_NODE

echo "tasks per node:"
echo $SLURM_TASKS_PER_NODE


env | grep -e SBATCH -e SLURM



