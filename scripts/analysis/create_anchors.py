
from anchor.embedding import Embedding
import argparse
import pickle

# Save the high-precision, full-dimensional embeddings needed for the eigenspace instability measure

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, help='Embedding algorithm', required=True)
	parser.add_argument('--n', type=int, help='Number to truncate', default=10000)
	parser.add_argument('--embdir', type=str, help='Location to load embs and store anchors', default='runs/embs')
	args = parser.parse_args()

	seeds = [1234, 1235, 1236]

	for seed in seeds:
		if args.algo == 'mc':
			path1 = f'{args.embdir}/wiki_2017/mc_wiki.en.txt_2017_seed_{seed}_dim_800_lr_0.2.50.txt_br_32'
			path2 = f'{args.embdir}/wiki_2018/mc_wiki.en.txt_2018_seed_{seed}_dim_800_lr_0.2.50.txt_br_32_same_range'
		elif args.algo == 'w2v_cbow':
			path1 = f'{args.embdir}/wiki_2017/w2v_cbow_wiki.en.txt_2017_seed_{seed}_dim_800_lr_0.05.50.w.txt_br_32'
			path2 = f'{args.embdir}/wiki_2018/w2v_cbow_wiki.en.txt_2018_seed_{seed}_dim_800_lr_0.05.50.w.txt_br_32_same_range'
		else:
			raise ValueError('Algorithm not supported')

		emb1 = Embedding(path1)
		emb2 = Embedding(path2)
		print(f'Loaded {path1} and {path2}')
		emb1_anchor, emb2_anchor, shared_vocab = emb2.get_subembeds_same_vocab(emb1, n=args.n, return_vocab=True)

		filename = f'{args.embdir}/wiki_2017/{args.algo}_anchor_seed_{seed}_top_{args.n}.pkl'
		with open(filename, 'wb') as f:
			pickle.dump((emb1_anchor, emb2_anchor, shared_vocab), f)

		print(f'Saved to {filename}')

if __name__ == '__main__':
    main()
