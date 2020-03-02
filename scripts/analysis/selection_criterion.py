"""
Generates the selection criterion results
-- requires data file with embedding distance and disagreement
between pairs of embeddings as input.
"""

import argparse
import csv
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_metrics', type=str, nargs='+', required=True,
        help='List of embedding metrics')
    parser.add_argument('--ds_metrics', type=str, nargs='+', required=True,
        help='List of downstream metrics')
    parser.add_argument('--csv-file', type=str, required=True,
        help='CSV file with pair results for emb metrics and DS results')
    parser.add_argument('--acc-file', type=str, required=True,
        help='File to write accuracy results to')
    parser.add_argument('--rob-file', type=str, required=True,
        help='File to write robustness results to')
    parser.add_argument('--same-space', action='store_true',
        help='Restrict selection to the same space budget')
    parser.add_argument('--verbose', action='store_true',
        help='Print information about selection')
    args = parser.parse_args()
    return args

def get_selection_error(emb_metric, ds_metric, df, space_budget=None,
    verbose=False, seed=None, same_space=False):
    """
    Returns the selection error when using emb_metric to select the more
    stable pair on ds_metric and the maximum difference to the oracle when
    a mistake is made.
    """
    # Only compare pairs of a specific seed
    df = df.loc[df['seed'] == seed].reset_index()
    n_rows = len(df.index)
    count = 0.
    total = 0.
    max_diff = 0
    idx = None
    # Iterate over all pairs of pairs
    for i in range(n_rows-1):
        for j in range(i+1, n_rows):
            row1 = df.loc[i][emb_metric]
            row2 = df.loc[j][emb_metric]
            ds_row1 = df.loc[i][ds_metric]
            ds_row2 = df.loc[j][ds_metric]

            # Skip pairs of pairs where either pair exceed space budget
            if space_budget is not None:
                if (df.loc[i]['space'] > space_budget
                    or df.loc[j]['space'] > space_budget):
                    continue

            # If same_space, only compare pairs of pairs of the same space budget
            if same_space and (df.loc[i]['space'] != df.loc[j]['space']):
                continue

            # most stable emb depends on metric
            # for knn and eigen_overlap, higher value is more stable
            # for other metrics, lower value is more stable
            if 'knn' in emb_metric or emb_metric == 'eigen_overlap_top_10000':
                emb_vote = np.argmax([row1, row2])
            else:
                emb_vote = np.argmin([row1, row2])

            # most stable downstream is smallest %
            ds_vote = np.argmin([ds_row1, ds_row2])

            # incorrect vote
            if emb_vote != ds_vote:
                count += 1
                # keep track to compute the max. difference to oracle
                diff = np.abs(ds_row1 - ds_row2)
                if diff > max_diff:
                    max_diff = diff
                    idx = (i, j)
            total += 1

    error = count / total
    if verbose:
        print(f'Maximum difference {max_diff} @ {idx}')
        print(f'Compared {total} pairs')
    return error, max_diff

def compute_sel_results(df, acc_file, rob_file, emb_metrics, ds_metrics,
    space_budgets=[None], seeds=[1234, 1235, 1236], verbose=False,
    same_space=False):
    """
    Write the selection error and max. error results to acc_file and rob_file,
    respectively. Iterate over emb_metrics and ds_metrics, computing these
    values for each combination and reporting the result as the average over
    seeds.
    """
    with open(acc_file, 'w') as f1, open(rob_file, 'w') as f2:
        writer1 = csv.writer(f1)
        writer1.writerow(['metric'] + ds_metrics)
        writer2 = csv.writer(f2)
        writer2.writerow(['metric'] + ds_metrics)
        for budget in space_budgets:
            for emb_metric in emb_metrics:
                emb_results_acc = []
                emb_results_robust = []
                for ds_metric in ds_metrics:
                    seed_error = []
                    seed_diff = []
                    for seed in seeds:
                        error, max_diff = get_selection_error(emb_metric,
                            ds_metric, df, space_budget=budget, verbose=verbose,
                            seed=seed, same_space=same_space)
                        if verbose:
                            print(emb_metric, ds_metric, error, max_diff)
                        seed_error.append(error)
                        seed_diff.append(max_diff)
                    # take average and max over seed
                    emb_results_acc.append(np.mean(seed_error))
                    emb_results_robust.append(np.max(seed_diff))
                writer1.writerow([emb_metric] + emb_results_acc)
                writer2.writerow([emb_metric] + emb_results_robust)

def main():
    args = parse_args()
    # Read in pandas dataframe
    df = pd.read_csv(args.csv_file)
    compute_sel_results(df=df, acc_file=args.acc_file, rob_file=args.rob_file,
        emb_metrics=args.emb_metrics, ds_metrics=args.ds_metrics,
        same_space=args.same_space, verbose=args.verbose)

if __name__ == '__main__':
    main()
