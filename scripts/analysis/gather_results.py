import argparse
import pandas as pd
import numpy as np

# ugly way to keep track of all the optimal parameters
optimal_wiki = {
        ('mc', 'la_sst_no_emb_norm'): 0.001,
        ('mc', 'la_subj_no_emb_norm'): 0.1,
        ('mc', 'la_mr_no_emb_norm'): 0.1,
        ('mc', 'la_mpqa_no_emb_norm'): 0.001,
        ('mc', 'rnn_no_crf_ner'): 1.0,
        ('w2v_cbow', 'la_sst_no_emb_norm'): 0.0001,
        ('w2v_cbow', 'la_subj_no_emb_norm'): 0.0001,
        ('w2v_cbow', 'la_mr_no_emb_norm'): 0.001,
        ('w2v_cbow', 'la_mpqa_no_emb_norm'): 0.001,
        ('w2v_cbow', 'rnn_no_crf_ner'): 0.1,
        ('mc', 'la_sst_no_emb_norm_val'): 0.001,
        ('mc', 'la_subj_no_emb_norm_val'): 0.1,
        ('mc', 'la_mr_no_emb_norm_val'): 0.1,
        ('mc', 'la_mpqa_no_emb_norm_val'): 0.001,
        ('mc', 'rnn_no_crf_ner_val'): 1.0,
        ('w2v_cbow', 'la_sst_no_emb_norm_val'): 0.0001,
        ('w2v_cbow', 'la_subj_no_emb_norm_val'): 0.0001,
        ('w2v_cbow', 'la_mr_no_emb_norm_val'): 0.001,
        ('w2v_cbow', 'la_mpqa_no_emb_norm_val'): 0.001,
        ('w2v_cbow', 'rnn_no_crf_ner_val'): 0.1,
        ('glove', 'la_sst_no_emb_norm'): 0.01,
        ('glove', 'la_subj_no_emb_norm'): 0.01,
        ('glove', 'la_mr_no_emb_norm'): 0.01,
        ('glove', 'la_mpqa_no_emb_norm'): 0.001,
        ('glove', 'rnn_no_crf_ner'): 1.0,
        ('ft_sg', 'la_sst_no_emb_norm'): 0.001,
        ('ft_sg', 'la_subj_no_emb_norm'): 100.0,
        ('ft_sg', 'la_mr_no_emb_norm'): 0.01,
        ('ft_sg', 'la_mpqa_no_emb_norm'): 0.01,
        ('ft_sg', 'rnn_no_crf_ner'): 1.0,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='Embedding algorithm', required=True)
    parser.add_argument('--default', action='store_true', help='Use default hyperparameters for DS models')
    parser.add_argument('--datadir', type=str, help='Data directory to read extracted dataframes', required=True)
    parser.add_argument('--resultdir', type=str, help='Result directory to write csv of results', required=True)
    parser.add_argument('--emb_metrics', type=str, nargs='+', required=True,
        help='List of embedding metrics')
    parser.add_argument('--ds_metrics', type=str, nargs='+', required=True,
        help='List of downstream metrics')
    return parser.parse_args()

def read(algo, dist, datadir):
    df = pd.read_pickle(f"{datadir}/{algo}_{dist}_results.pkl")
    return df

def gather_all_tasks(algo, optimal, emb_metrics, ds_metrics, dims, datadir):
    total_metrics = ds_metrics + emb_metrics
    merge_tags = ['space', 'bitrate', 'dim', 'seed', 'lr', 'compress_type', 'algo']
    # merge over metrics
    total_df = None
    for m in total_metrics:
        # concat dimensions
        dfs = []
        for d in dims:
            if m in ds_metrics:
                if optimal:
                    # hacky soln to avoid copying and pasting vals in dictionary
                    metric = m
                    if 'quality' in m:
                        metric = ''.join(m.split('_quality'))
                    full_metric = f'{m}_dim_{d}_uniform_lr_{optimal_wiki[(algo,metric)]}_wiki'
                else:
                    full_metric = f'{m}_dim_{d}_uniform_lr_0.001_wiki'
            else:
                full_metric = f'{m}_dim_{d}_uniform_wiki'
            single_df = read(algo, full_metric, datadir=datadir)
            single_df['space'] = single_df['bitrate'] * single_df['dim']
            dfs.append(single_df)
        df = pd.concat(dfs)
        # merge with other metrics
        if total_df is None:
            total_df = df
        else:
            total_df = total_df.merge(df, on=merge_tags)
    return total_df

def main():
    args = parse_args()
    dims = [25, 50, 100, 200, 400, 800]
    emb_metrics = args.emb_metrics
    ds_metrics = args.ds_metrics
    total_df = gather_all_tasks(algo=args.algo, optimal=not args.default,
        emb_metrics=emb_metrics, ds_metrics=ds_metrics, dims=dims, datadir=args.datadir)
    if not args.default:
        total_df.to_csv(f'{args.resultdir}/{args.algo}_optimal_no_emb_norm_top_10000.csv', index=False)
    else:
        total_df.to_csv(f'{args.resultdir}/{args.algo}_no_emb_norm_top_10000.csv', index=False)

if __name__ == '__main__':
    main()