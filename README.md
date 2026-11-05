# connectome-comparison

Calculate and compare functional connectivity (FC) matrices estimated with different models from resting-state fMRI, and evaluate them on test-retest reliability, fingerprinting, and prediction accuracy.

TODO: one-paragraph summary of the question and main finding; link to the manuscript/preprint once available.

## Overview

- **Datasets:** Human Connectome Project (HCP 1200; MSMAll FIX and XCP-D processing) and the Philadelphia Neurodevelopmental Cohort (PNC; XCP-D processing).
- **Atlas:** 4S156Parcels (fsLR, 91k), with 100 cortical ROIs used for the connectomes.
- **FC models:** Pearson correlation, partial correlation, ridge (`ridgeCV`), LASSO (`lassoBIC`), and UoI-LASSO (`uoiLasso`).
- **Evaluations:**
  - reliability
    - edge-level test-retest reliability (ICC)
    - connectome fingerprinting (identification accuracy)
  - validity
    - out-of-sample prediction accuracy
    - mean and median group connectomes

## Repository layout

```
config.ini               # machine-specific paths (copy from config_files/)

data/                    # movement and QC tables, subject lists (restricted/ is not shared)

results/                 # analysis outputs, organized by processing type
  MSMAll-FIX/  XCPD/  PNC_XCPD/  plotting/

scripts/
  processing/            # timeseries extraction and FC estimation (run on HPC)
  sbatch/                # SLURM job scripts for hyak and nersc
  analysis/
    fc_utils/            # shared helpers (read_pickles, make_mat)
    01-Processing/       # organize FC results into tables
    02-Analysis/         # reliability, fingerprinting, accuracy, average connectomes
  plotting/              # manuscript figures
pytest/                  # tests for FC calculation
renv/, renv.lock         # R environment
```

## Installation

### Python

TODO: add an `environment.yml` or `requirements.txt` (the notebooks use the `fc` / `fc_311` kernels). Main dependencies: numpy, pandas, scikit-learn, pyuoi, nilearn, matplotlib, seaborn, statsmodels, jupytext.

FC estimation depends on [`model-fc`](TODO: link), which is expected to be checked out next to this repo.

### R

The R environment is managed with renv:

TODO: add renv install instructions and test

```r
renv::restore()
```

```r
install.packages("devtools")
devtools::install_github("muschellij2/I2C2")
```

### Notebooks

Notebooks are paired with `.py` / `.R` scripts using jupytext (see `jupytext.toml`). Edit either one and sync with `jupytext --sync <file>`.

## Configuration

Copy the template for your machine to the repo root and fill in the paths:


| Key | Description |
|---|---|
| `[Paths] deriv_dir` | HCP derivatives directory |
| `[Paths] hcp_bids` | HCP 1200 root |
| `[Paths] pnc_bids` | PNC BIDS root |
| `[Paths] pkl_dir` | organized FC pickles (`results_pkls`) |
| `[Paths] script_dir` | path to this repo |
| `[Paths] datalad_dir` | datalad environment `bin/` |

Scripts locate the repo root from the working directory, so run them from inside `scripts/`.

## Data

- **HCP MSMAll-FIX:** download with datalad: `datalad clone https://github.com/datalad-datasets/hcp-functional-connectivity.git`.
- **HCP XCPD**: TODO: are we sharing the source data here? 
- **PNC:** TODO: source and access (dbGaP).
- **Not shared:** restricted behavioral data belongs in `data/restricted/` and is not included in the repo.

HCP is not organized according to BIDS, but the derivatives generated here follow BEP-17 (Relationship Matrices).

## Pipeline

Run the steps in this order. Steps 1–2 run on HPC; the rest can run locally once the organized pickles are available.

### 1. Timeseries extraction (`scripts/processing/`)
1. `extract_timeseries.py --sub_id <HCP id> --proc MSMAll`: parcellated timeseries from minimally preprocessed data, written to `$SCRATCH/derivatives/timeseries/...`
2. `combine_timeseries.py` / `concat_timeseries.py`: combine runs for minimally preprocessed and XCP-D data

### 2. FC estimation
- `calc_fc.py --dataset <hcp|pnc> --sub_id <id> --model <model> --cv <ses|task>`: fit FC models with cross-validation across sessions (HCP) or tasks (PNC). Matrices go to `.../derivatives/connectivity-matrices/<proc>/<model>_<cv>/`
- SLURM scripts are in `scripts/sbatch/{hyak,nersc}/`.

### 3. Organize results (`scripts/analysis/01-Processing/`)
| Script | Output |
|---|---|
| `organize_results.py` | one pickle per proc × model in `pkl_dir/<proc>/` |
| `make_movement_df.py` | `data/<dataset>_relative_movement.csv` |
| `make_icc_df.py` | `results/<proc>/proc-<PROC>_model-<model>_reliability-edges.csv` |
| `make_fc_dfs.py` | `results/<proc>/proc-<PROC>_model-<model>_ses-full_connectome-edges.csv` |
| `residual.py` | `results/<proc>/proc-<PROC>_model-<model>_residual-edges.csv` |
| `clean_behavioral_data.py` | TODO |

### 4. Analysis (`scripts/analysis/02-Analysis/`)
| Script | Output |
|---|---|
| `ICC_reliability.R` | edge-wise ICC in `results/plotting/<proc>/proc-<PROC>_model-<model>_stat-ICC.csv` |
| `fingerprinting.py` | bootstrapped identification accuracy in `results/<proc>/proc-<PROC>_stat-fingerprinting.csv` |
| `average_connectomes.py` | mean and median connectomes in `results/<proc>/`; per-participant CSVs |
| `accuracy-mean.py` | TODO |

### 5. Figures (`scripts/plotting/`)
TODO: map each manuscript figure to its plotting notebook.

## Output naming

Outputs use BIDS-like entities:

```
proc-<FIX|XCPD|PNC>_model-<lassoBIC|uoiLasso|ridgeCV|Pearson|partial-correlation>_<stat-…|…-edges>.csv
```

## Tests

```
pytest pytest/
```

## Citation

TODO

## License

TODO

## Contact

McKenzie Hagen (mphagen@uw.edu)
