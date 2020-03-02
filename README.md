# Understanding the Downstream Instability of Word Embeddings

Code for "Understanding the Downstream Instability of Word Embeddings" in MLSys 2020.

## Install

We recommend using a virtualenv or a conda environment.

virtualenv:
```
virtualenv -p python3.6 anchor_venv
source anchor_venv/bin/activate
```

conda:
```
conda create -n anchor python=3.6
conda activate anchor
conda install --file requirements.txt
```

To install anchor, run
```
git clone --recursive https://github.com/HazyResearch/anchor-stability.git
cd anchor-stability
pip install -r requirements.txt
pip install -e .
```

If you just want to try out the embedding distance measures, you can skip to 'Basic Usage' below. If you are interested in reproducing the experiments in 'Experiments' below, then you will also need to run:
```
bash run_install.sh
```

## Basic Usage: Embedding Distance Measures

To compute the instability between a pair of word embeddings (without training downstream models):
1. Obtain sample word embeddings. These embeddings were trained on the Wikipedia'2017 and Wikipedia'2018 dumps.
```
wget  https://storage.googleapis.com/embstability/demo_embs.tar.gz
tar -xzvf demo_embs.tar.gz
```
2. Create Embedding objects for the word embeddings you want to compare. We lazily load the embeddings with the first embedding distance measure. For example, using sample word embeddings:
```
from anchor.embedding import Embedding

emb1 = Embedding('demo/glove_wiki_2017_dim_25.txt')
emb2 = Embedding('demo/glove_wiki_2018_dim_25.txt')
```
To use the eigenspace instability measure, we must also load "anchor" embeddings. We recommend using the largest precision, highest dimensional embeddings you have available:
```
emb1_anchor = Embedding('demo/glove_wiki_2017_dim_100.txt')
emb2_anchor = Embedding('demo/glove_wiki_2018_dim_100.txt')
```

1. Compare the embeddings using several metrics. We use `n=10000` to only compare the top-10K words for each measure. Increasing this will result in slower computation times, especially for the PIP loss and the k-NN measures. If you swap the order of the embeddings, the distance may change slightly as the top n words is based on the first embedding (emb2 in the examples below).

- Eigenspace instability (smaller is more stable)
```
emb2.eis(emb1, curr_anchor=emb2_anchor, other_anchor=emb1_anchor, n=10000)
```
- k-Nearest neighbors (larger is more stable)
```
emb2.knn(emb1, n=10000)
```
You can also set `nquery` to change the number of words that are sampled to compare their overlapping neighbors (default 1000) and `nneighbors` the number of neighbors compared for each query (default 5). For example, to compare the 10 nearest neighbors for 100 query words: 
```
emb2.knn(emb1, n=10000, nneighbors=10, nquery=100)
```
- Semantic displacement (smaller is more stable)
```
emb2.sem_disp(emb1, n=10000)
```
- PIP loss (smaller is more stable)
```
emb2.pip_loss(emb1, n=10000)
```
- Eigenspace overlap (larger is more stable)
```
emb2.eigen_overlap(emb1, n=10000)
```

If you repeat the above steps with another pair of word embeddings, you can compare the instability values for a relative notion of instability. Check out the [demo notebook](notebooks/demo_distance_measures.ipynb) for more details!

In our experiments, we found that our theoretically grounded eigenspace instability measure and the k-NN measure had the strongest correlations with downstream prediction disagreement. In other words, the embeddings that are more unstable with respect to these metrics, are likely to have greater prediction disagreement when downstream models are trained on top of them.

## Experiments

We include steps to reproduce the main results of the paper.

### 1. Prepare the embeddings

We provide the pre-trained word embeddings on two datasets (Wikipedia'17 and Wikipedia'18) and two embedding algorithms (word2vec CBOW and matrix completion). Download and extract them with the command below. Note, this takes some time as there are four tarballs of about ~7GB each.

```
bash run_get_embs.sh
```
You should have 36 embeddings each in `runs/embs/wiki_2017` and `runs/embs/wiki_2018` (there are 6 dimensions, 3 seeds, and 2 datasets).

Use uniform quantization to compress the embeddings to 6 different precisions (1, 2, 4, 8, 16, 32-bit). Generate the commands with:
```
bash run_compress.sh
```
You should now have a file `compress_cmds.sh` with 432 commands. As we need to read and write so many embedding files, this can take a long time (on the order of hours). To speed up compressing, we recommend running in parallel or on a cluster. For example, to run six jobs in parallel on the same machine with `xargs`:
```
cat compress_cmds.sh | xargs -P 6 -I {} bash -c {}
```

After the compression is complete, there should be 252 embedding files each in  `runs/embs/wiki_2017` and `runs/embs/wiki_2018`.

### 2. Train downstream models

To train downstream models on all embeddings for word2vec and matrix completion (MC) for the SST-2 sentiment analysis task, run:
```
bash run_models.sh
```
This will generate a list of commands to train downstream models (216 commands per embedding algorithm + downstream task) in `model_cmds.sh`. We recommend launching these jobs in parallel with `xargs` or in a cluster, if possible.

After the models have finished training, we can measure the downstream prediction disagreements between pairs of models trained on Wikipedia'17 and Wikipedia'18 embeddings (using the outputted `*preds` produced from model training in `runs/models`). We also want to compute several different embedding distance measures between pairs of embeddings, to see which of these measures is most predictive of the downstream prediction disagreement. To compute these values:
```
bash run_collect_results.sh
```
The final output of the above script will be a CSV file with the embedding distance measure results and downstream instabilities between pairs of compressed embeddings (and their corresponding downstream models) trained on Wiki'17 and Wiki'18 for both embedding algorithms.

### 3. Analyze the results

To reproduce the main results using for the SST-2 algorithm, run:
```
bash run_analysis.sh sst
```

This will print the linear-log slopes for the trends described in Section 3. Note, as these trends were computed over five downstream tasks for the results reported in the paper, the numbers will vary slightly; however you should see a negative slope indicating that as the embedding memory increases, the downstream instability decreases. This script will also output CSVs to `runs/analysis` with Spearman correlation results between the embedding distance measures and downstream prediction disagreement, as well as selection criterion error and robustness results for the two settings described in the paper (Tables 1, 2, and 3).

You can also generate graphs with graphing code provided in this [notebook](notebooks/stability-memory-tradeoff-figures-sst2.ipynb).

Finally, we provide the full CSVs in `results` for three additional sentiment analysis tasks (MR, Subj, MPQA) and an NER task (CoNLL-2003) for the two embedding algorithms. For the complete table and graphs, run:
```
bash run_analysis.sh all
```
This should output the linear-log trends reported as well as CSVs to `results` with the complete results presented in Tables 1, 2, and 3. Note that the order of the columns will be different. To reproduce Figures 1 and 2, use this [notebook](notebooks/stability-memory-tradeoff-figures.ipynb).
