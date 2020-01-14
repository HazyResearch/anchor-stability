import pandas as pd
import argparse
import csv
import numpy as np

# Computes the distance to the oracle when given a selection of pairs at
# the same memory budget and must select the pair which attains the lowest
# embedding distance measure

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True,
        help='CSV file with pair results for emb metrics and DS results')
    parser.add_argument('--out-file', type=str, required=True,
        help='File to write results to')
    parser.add_argument('--emb_metrics', type=str, nargs='+', required=True,
        help='List of embedding metrics')
    parser.add_argument('--ds_metrics', type=str, nargs='+', required=True,
        help='List of downstream metrics')
    return parser.parse_args()

def compute_diff_to_oracle(df, summary_file, emb_metrics, ds_metrics):
    """
    Compute the average difference to the oracle across space budgets and
    seeds for each emb_metric and ds_metric combination and write to
    the summary_file.
    """
    space_vals = set(df['space'])
    seeds = set(df['seed'])
    space_vals = sorted(space_vals)
    all_results = {}
    for metric in emb_metrics:
        metric_dict = {}
        for space in space_vals:
            # average over seeds
            for seed in seeds:
                vals = {}
                # get rows for space
                # need to match over seed as well
                subset = df.loc[(df['space'] == space) & (df['seed'] == seed)]
                # there is a vote to make (at least two things)
                if len(subset.index) > 1:
                    for dist in ds_metrics:
                        oracle = subset.loc[[subset[(dist)].idxmin()]]
                        oracle_val = oracle[(dist)].values[0]
                        if 'baseline' not in metric:
                            # high means more stable
                            if 'knn' in metric or metric == 'eigen_overlap_top_10000':
                                predicted = subset.loc[[subset[(metric)].idxmax()]]
                            # low means more stable
                            else:
                                predicted = subset.loc[[subset[(metric)].idxmin()]]
                        else:
                            if 'fp' in metric:
                                # select highest bitrate
                                predicted = subset.loc[[subset['bitrate'].idxmax()]]
                            else:
                                # select lowest bitrate
                                predicted = subset.loc[[subset['bitrate'].idxmin()]]
                        predicted_val = predicted[(dist)].values[0]
                        diff = predicted_val-oracle_val
                        if dist in metric_dict:
                            metric_dict[dist].append(diff)
                        else:
                            metric_dict[dist] = [diff]
        all_results[metric] = metric_dict

    # write averages
    with open(summary_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric'] + ds_metrics)
        # write average values
        for metric in emb_metrics:
            writer.writerow([metric] + [str(np.mean(all_results[metric][ds_metric]))
                for ds_metric in ds_metrics])
        writer.writerow([])
        # write max values
        for metric in emb_metrics:
            writer.writerow([metric] + [str(np.max(all_results[metric][ds_metric]))
                for ds_metric in ds_metrics])

def main():
    args = parse_args()
    # Read in pandas dataframe
    df = pd.read_csv(args.csv_file)
    compute_diff_to_oracle(df=df, summary_file=args.out_file,
        emb_metrics=args.emb_metrics, ds_metrics=args.ds_metrics)


if __name__ == '__main__':
    main()