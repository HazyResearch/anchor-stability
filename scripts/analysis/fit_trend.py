import argparse
import pandas as pd
import numpy as np
import functools
import scipy.stats

# Fits linear-log models to the instability v. memory, instability v. dimension,
# and instability v. precision trends.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-files', type=str, nargs='+', required=True,
        help='CSV file (list) with pair results for emb metrics and DS results')
    parser.add_argument('--tasks', type=str, nargs='+',
        default=[
            'la_sst_no_emb_norm',
            'la_mr_no_emb_norm',
            'la_subj_no_emb_norm',
            'la_mpqa_no_emb_norm',
            'rnn_no_crf_ner'],
        help='List of downstream metrics')
    parser.add_argument('--thresh', type=int, default=1000,
        help='Maximum memory budget')
    parser.add_argument('--dim', action='store_true',
        help='Fit the trend with respect to dimension')
    parser.add_argument('--prec', action='store_true',
        help='Fit the trend with respect to precision')
    return parser.parse_args()

def solve_lstsq_combine(dfs, thresh, tasks):
    """
    Fit a single trend to the downstream stability v. memory results across
    embedding algorithms and downstream tasks.
    """
    ncombos = len(tasks) * len(dfs)
    space_vals = np.log2(dfs[0].loc[dfs[0]['space'] < thresh]['space'].values)
    num_vals = len(space_vals)
    X = np.zeros((num_vals*len(tasks) * len(dfs), ncombos+1))
    y = np.zeros((num_vals*len(tasks) * len(dfs)))
    for i, df in enumerate(dfs):
        # Only consider those results less than thresh space budget
        df_subset = df.loc[df['space'] < thresh]
        for j, t in enumerate(tasks):
            idx = i*len(tasks) + j
            y[idx*num_vals:(idx+1)*num_vals] = df_subset[t].values
            # First column is the log2(m) results
            X[idx*num_vals:(idx+1)*num_vals][:,0] = space_vals
            # Append a 1-hot vector to learn a separate y-intercept per task
            X[idx*num_vals:(idx+1)*num_vals][:,idx+1] = np.ones(num_vals)
    # print(f'Dimensions of data matrix: {X.shape}')
    return np.linalg.inv(X.T @ X) @ X.T @ y

def solve_lstsq_combine_prec(dfs, thresh, tasks, dims=[25, 50, 100, 200, 400, 800]):
    """
    Fit a single trend to the downstream stability v. *precision* results across
    embedding algorithms and downstream tasks and *dimensions*.
    """
    ncombos = len(tasks) * len(dfs) * len(dims)
    num_vals = len(dfs[0].loc[(dfs[0]['space'] < thresh)]['space'].values)
    X = np.zeros((num_vals*len(tasks)*len(dfs), ncombos+1))
    y = np.zeros((num_vals*len(tasks)*len(dfs)))
    row_idx = 0
    col_idx = 0
    for i, df in enumerate(dfs):
        for j, t in enumerate(tasks):
            for k,dim in enumerate(dims):
                df_subset = df.loc[(df['space'] < thresh) & (df['dim'] == dim)]
                prec_vals = np.log2(df_subset['bitrate'].values)
                # Number of values diffs by dimension
                num_vals = len(prec_vals)
                y[row_idx:row_idx+num_vals] = df_subset[t].values
                X[row_idx:row_idx+num_vals][:,0] = prec_vals
                X[row_idx:row_idx+num_vals][:,col_idx+1] = np.ones(num_vals)
                row_idx += num_vals
                # Learn a different y-intercept for each algo/task/dim combination
                col_idx += 1
    # print(f'Dimensions of data matrix: {X.shape}')
    return np.linalg.inv(X.T @ X) @ X.T @ y

def solve_lstsq_combine_dim(dfs, thresh, tasks, bitrates=[1,2,4,8,16,32]):
    """
    Fit a single trend to the downstream stability v. *dimension* results across
    embedding algorithms and downstream tasks and *precisions*.
    """
    ncombos = len(tasks) * len(dfs) * len(bitrates)
    num_vals = len(dfs[0].loc[(dfs[0]['space'] < thresh)]['space'].values)
    X = np.zeros((num_vals*len(tasks)*len(dfs), ncombos+1))
    y = np.zeros((num_vals*len(tasks)*len(dfs)))
    row_idx = 0
    col_idx = 0
    for i, df in enumerate(dfs):
        for j, t in enumerate(tasks):
            for k, bitrate in enumerate(bitrates):
                df_subset = df.loc[(df['space'] < thresh) & (df['bitrate'] == bitrate)]
                space_vals = np.log2(df_subset['dim'].values)
                # Number of values diffs by precision
                num_vals = len(space_vals)
                y[row_idx:row_idx+num_vals] = df_subset[t].values
                X[row_idx:row_idx+num_vals][:,0] = space_vals
                X[row_idx:row_idx+num_vals][:,col_idx+1] = np.ones(num_vals)
                row_idx += num_vals
                # Learn a different y-intercept for each algo/task/prec combination
                col_idx += 1
    # print(f'Dimensions of data matrix: {X.shape}')
    return np.linalg.inv(X.T @ X) @ X.T @ y

def main():
    args = parse_args()
    dfs = []
    for file in args.csv_files:
        dfs.append(pd.read_csv(file))
    if args.dim:
        print('Instability v. dimension trend')
        m = solve_lstsq_combine_dim(dfs=dfs, thresh=args.thresh, tasks=args.tasks)
    elif args.prec:
        print('Instability v. precision trend')
        m = solve_lstsq_combine_prec(dfs=dfs, thresh=args.thresh, tasks=args.tasks)
    else:
        print('Instability v. memory trend')
        m = solve_lstsq_combine(dfs=dfs, thresh=args.thresh, tasks=args.tasks)
    print(f'Slope: {m[0]}')
    print(f'y-intercepts: {m[1:]}')

if __name__ == '__main__':
    main()