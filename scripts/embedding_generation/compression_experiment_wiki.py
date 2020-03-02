"""
Generate commands for compressing embeddings.
"""

import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--resultdir1",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resultdir2",
        type=str,
        required=True,
    )
    parser.add_argument('--algo',
        type=str,
        required=True,
        help='Embedding algorithm')
    parser.add_argument('--bitrate',
        type=int,
        help='Run specific bitrate')
    return parser.parse_args()

def main():
    args = parse_args()
    algo = args.algo
    bitrates = [1,2,4,8,16,32]
    if args.bitrate:
        bitrates = [args.bitrate]
    seeds = range(1234, 1237)
    dims = [25, 50, 100, 200, 400, 800]
    lrs = {'mc': 0.2, 'w2v_cbow': 0.05, 'glove': 0.01}

    for seed in seeds:
        for dim in dims:
            for bitrate in bitrates:
                if algo != 'ft_sg':
                    lr = lrs[algo]
                    if algo == "w2v_cbow":
                        end_tag = "50.w.txt"
                    elif algo == 'mc':
                        end_tag = "50.txt"
                    elif algo == 'glove':
                        end_tag = "050.w.txt"

                    path1 = f"{args.resultdir1}/{algo}_wiki.en.txt_2017_seed_{seed}_dim_{dim}_lr_{lr}.{end_tag}"
                    path2 = f"{args.resultdir2}/{algo}_wiki.en.txt_2018_seed_{seed}_dim_{dim}_lr_{lr}.{end_tag}"
                else:
                    path1 = f"{args.resultdir1}/ft_sg_wiki.en.txt_2017_seed_{seed}_dim_{dim}.vec"
                    path2 = f"{args.resultdir2}/ft_sg_wiki.en.txt_2018_seed_{seed}_dim_{dim}.vec"
                # gen compressed 2017
                print(f'python scripts/embedding_generation/gen_compressed.py --emb_path {path1} --out {path1}_br_{bitrate} --bitrate {bitrate} --seed {seed}')

                # gen compressed 2018, align to 2017 embedding
                print(f'python scripts/embedding_generation/gen_compressed.py --emb_path {path2} --base_emb_path {path1} --out {path2}_br_{bitrate}_same_range --bitrate {bitrate} --seed {seed}')

if __name__ == '__main__':
    main()
