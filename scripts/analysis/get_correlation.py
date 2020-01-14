import pandas as pd
import argparse
import csv
import numpy as np

# Generates the Spearman correlations between the embedding metrics
# and downstream metrics

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

def get_corr(df, emb_metrics, ds_metrics, summary_file):
    """
    Writes the Spearman correlations for all emb_metrics and ds_metrics pairs
    to summary_file.
    """
    with open(summary_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric'] + ds_metrics)
        for em in emb_metrics:
            corr = []
            for dm in ds_metrics:
                # higher is more stable for these metrics, reverse to maintain consistent
                # correlation value meaning
                if 'knn' in em or em == 'eigen_overlap_top_10000':
                    emb_val = 1-df[em]
                else:
                    emb_val = df[em]
                # get spearman corr of column (aggregates over seeds and space budgets)
                correlation = pd.Series.corr(emb_val, df[dm], method='spearman')
                corr.append(correlation)
            writer.writerow([em] + corr)

def main():
    args = parse_args()
    # Read in pandas dataframe
    df = pd.read_csv(args.csv_file)
    get_corr(df=df, summary_file=args.out_file,
        emb_metrics=args.emb_metrics, ds_metrics=args.ds_metrics)

if __name__ == '__main__':
    main()