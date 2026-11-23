#!/bin/bash
# Leave-one-family-out phenotype prediction for predict_pheno.py as a Slurm job array.
#
# One array task per (connectome type, phenotype): task ID = type_index * N_PHENO + pheno_index.
# Within a task, the LOFO folds (one per HCP family) are fitted in parallel across
# --cpus-per-task. LOFO has no randomness, so any task can be rerun on its own:
#   sbatch --array=7 run_predict-pheno.sbatch
#
# Submit from scripts/analysis/02-Analysis (predict_pheno.py finds the repo root
# from the cwd, and logs/ is relative to it):
#   cd scripts/analysis/02-Analysis
#   mkdir -p logs && sbatch ../../sbatch/hyak/run_predict-pheno.sbatch
# Then combine the per-phenotype files:
#   python merge_predictions.py --mode lofo --connectome_type uoiLasso   (etc.)
#
# The array size must equal len(TYPES) * N_PHENO - 1 (0-indexed). N_PHENO is the
# length of PHENO_LIST in fc_utils/_utils.py (10 for HCP).
#SBATCH --job-name=pheno-lofo
#SBATCH --account=escience          # check: `hyakalloc` lists your accounts/partitions
#SBATCH --partition=ckpt-all        # preemptible; a requeued task restarts from scratch
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

TYPES=(uoiLasso correlation ridgeCV)
N_PHENO=10
PROC_TYPE=MSMAll-FIX
DATASET=hcp
PCA=True

if (( SLURM_ARRAY_TASK_ID >= ${#TYPES[@]} * N_PHENO )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} out of range" >&2
    exit 1
fi

CT=${TYPES[$(( SLURM_ARRAY_TASK_ID / N_PHENO ))]}
PHENO=$(( SLURM_ARRAY_TASK_ID % N_PHENO ))

source ~/.bashrc
conda activate fc

# parallelism comes from fitting folds in separate processes (--n_jobs), so keep
# each process's math libraries single-threaded to avoid oversubscribing the CPUs
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
echo "task ${SLURM_ARRAY_TASK_ID}: ${CT} pheno ${PHENO} on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"

python predict_pheno.py \
    --connectome_type "${CT}" \
    --proc_type "${PROC_TYPE}" \
    --dataset "${DATASET}" \
    --pca "${PCA}" \
    --cv logo \
    --pheno "${PHENO}" \
    --n_jobs "${SLURM_CPUS_PER_TASK}"
