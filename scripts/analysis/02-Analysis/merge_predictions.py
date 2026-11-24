"""Combine per-task prediction pickles from scripts/sbatch/hyak/run_predict-pheno.sbatch.

Output has the same layout as a single predict_pheno.py run,
{pheno: {boot: {fold_k: {...}}}}, so downstream code can read it unchanged.

    # leave-one-out, one file per phenotype -> ..._cv-loo_dict.pkl
    python merge_predictions.py --mode loo --connectome_type uoiLasso

    # leave-one-family-out, one file per phenotype -> ..._nboots_1_cv-lofo_dict.pkl
    python merge_predictions.py --mode lofo --connectome_type uoiLasso

    # bootstrap chunks -> ..._seed-1_nboots-100_dict.pkl
    python merge_predictions.py --mode resample --connectome_type uoiLasso --expected 100
"""
import argparse
import glob
import os
import os.path as op
import pickle
import sys

import pandas as pd

analysis_dir = op.dirname(op.abspath(__file__))
pred_dir = op.join(analysis_dir, 'results', 'predictions')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, choices=['loo', 'lofo', 'resample'])
parser.add_argument('--proc_type', default='MSMAll-FIX')
parser.add_argument('--dataset', default='hcp')
parser.add_argument('--connectome_type', required=True)
parser.add_argument('--pca', default='True')
parser.add_argument('--seed', default=1, type=int, help='resample mode only')
parser.add_argument('--expected', default=None, type=int,
                    help='resample mode: total resamples expected; errors if any are missing')
parser.add_argument('--in_dir', default=None)
parser.add_argument('--out_dir', default=pred_dir)
args = parser.parse_args()

base = f'{args.proc_type}_{args.dataset}_{args.connectome_type}_pca-{args.pca}'


def load(files):
    for fname in files:
        with open(fname, 'rb') as f:
            yield fname, pickle.load(f)


if args.mode in ('loo', 'lofo'):
    if args.mode == 'loo':
        in_dir = args.in_dir or op.join(pred_dir, 'loo')
        stem = f'{base}_cv-loo'
    else:
        # predict_pheno.py --cv logo --pheno <i> writes these straight to results/predictions
        in_dir = args.in_dir or pred_dir
        stem = f'{base}-nboots_1_cv-lofo'
    files = sorted(glob.glob(op.join(in_dir, f'{stem}_pheno-*_dict.pkl')))
    if not files:
        raise SystemExit(f'no files matching {stem}_pheno-*_dict.pkl in {in_dir}')

    merged = {}
    for fname, chunk in load(files):
        for pheno, boots in chunk.items():
            if pheno in merged:
                print(f'warning: {pheno} appears in more than one file, keeping first')
                continue
            merged[pheno] = boots

    # every phenotype the run was meant to cover should be present
    if args.dataset == 'hcp':
        sys.path.append(op.join(analysis_dir, '..'))
        from fc_utils._utils import PHENO_LIST
    else:
        PHENO_LIST = ['age', 'p_factor_mcelroy_harmonized_all_samples']
    missing = [p for p in PHENO_LIST if p not in merged]
    if missing:
        print(f'missing phenotypes: {missing}')
        raise SystemExit('rerun those array tasks (task ID = phenotype index) before merging')
    merged = {p: merged[p] for p in PHENO_LIST}
    out_name = f'{stem}_dict.pkl'
    summary_glob = op.join(in_dir, f'{stem}_pheno-*_dict_summary.csv')

else:
    prefix = f'{base}_seed-{args.seed}'
    in_dir = args.in_dir or op.join(pred_dir, 'resamples')
    files = sorted(glob.glob(op.join(in_dir, f'{prefix}_boot-*.pkl')))
    if not files:
        raise SystemExit(f'no files matching {prefix}_boot-*.pkl in {in_dir}')

    merged = {}
    for fname, chunk in load(files):
        for pheno, boots in chunk.items():
            merged.setdefault(pheno, {})
            for boot, res in boots.items():
                if boot in merged[pheno]:
                    # a requeued task can leave a duplicate; same seed means same result
                    print(f'warning: {pheno} boot {boot} appears more than once, keeping first')
                    continue
                merged[pheno][boot] = res

    # every phenotype should have the same, gap-free set of resample indices
    boot_sets = {pheno: set(b) for pheno, b in merged.items()}
    all_boots = set().union(*boot_sets.values())
    n_total = args.expected if args.expected is not None else max(all_boots) + 1
    expected = set(range(n_total))
    problems = {p: sorted(expected - b) for p, b in boot_sets.items() if expected - b}
    if problems:
        for p, missing in problems.items():
            print(f'{p}: missing boots {missing}')
        raise SystemExit('some resamples are missing; rerun those array tasks before merging')

    merged = {p: dict(sorted(b.items())) for p, b in merged.items()}
    out_name = f'{prefix}_nboots-{n_total}_dict.pkl'
    summary_glob = op.join(in_dir, f'{prefix}_boot-*_summary.csv')

os.makedirs(args.out_dir, exist_ok=True)
out_path = op.join(args.out_dir, out_name)
with open(out_path, 'wb') as f:
    pickle.dump(merged, f)

summaries = sorted(glob.glob(summary_glob))
if summaries:
    summary = (pd.concat([pd.read_csv(s) for s in summaries])
               .drop_duplicates(['pheno', 'boot'])
               .sort_values(['pheno', 'boot']))
    summary.to_csv(out_path.replace('.pkl', '_summary.csv'), index=False)
    print(summary.to_string(index=False))

print(f'merged {len(files)} files, {len(merged)} phenotypes -> {out_path}')
