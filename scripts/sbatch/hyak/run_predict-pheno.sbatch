#!/bin/bash
# Bootstrap resampling for predict_pheno.py as a Slurm job array.
#
# Each array task runs CHUNK resamples for one connectome type. Resample i is
# seeded from (SEED, i), so results don't depend on how resamples are split
# across tasks, and a failed task can be rerun on its own:
#   sbatch --array=7 run_predict-pheno.sbatch
#
# Submit from this directory (predict_pheno.py finds the repo root from the cwd):
#   mkdir -p logs && sbatch run_predict-pheno.sbatch
# Then combine the chunks:
#   python merge_resamples.py --connectome_type uoiLasso   (etc.)
#
# The array size must equal len(TYPES) * N_CHUNKS - 1 (0-indexed).
#SBATCH --job-name=pheno-boot
#SBATCH --account=escience          # check: `hyakalloc` lists your accounts/partitions
#SBATCH --partition=ckpt-all        # preemptible; tasks are short, so requeue is cheap
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

TYPES=(uoiLasso correlation ridgeCV)
N_CHUNKS=10          # chunks per connectome type
CHUNK=10             # resamples per chunk -> N_CHUNKS * CHUNK = 100 resamples per type
SEED=1
PROC_TYPE=MSMAll-FIX
DATASET=hcp
PCA=True

if (( SLURM_ARRAY_TASK_ID >= ${#TYPES[@]} * N_CHUNKS )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} out of range" >&2
    exit 1
fi

CT=${TYPES[$(( SLURM_ARRAY_TASK_ID / N_CHUNKS ))]}
START=$(( (SLURM_ARRAY_TASK_ID % N_CHUNKS) * CHUNK ))

source ~/.bashrc
conda activate fc

# match BLAS threads to the allocation so tasks don't oversubscribe a node
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"
echo "task ${SLURM_ARRAY_TASK_ID}: ${CT} boots ${START}-$(( START + CHUNK - 1 )) on $(hostname)"

python predict_pheno.py \
    --connectome_type "${CT}" \
    --proc_type "${PROC_TYPE}" \
    --dataset "${DATASET}" \
    --pca "${PCA}" \
    --resample True \
    --n_boot "${CHUNK}" \
    --boot_start "${START}" \
    --seed "${SEED}"
