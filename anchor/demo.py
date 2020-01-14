from algorithms import GloVe, MatrixCompletion, PowerIteration, Word2Vec

from anchor import Anchor, Ensemble, Initializer

# create the embedding management system (ems)
ems = Anchor(emb_dir=".", initializer=Initializer.WARMSTART, ensemble=Ensemble.ALL)

# choose your algorithm and set parameters if desired
# algo = Word2Vec(
#     exec_dir="/lfs/raiders10/0/mleszczy/anchor/word2vec/word2vec",
#     tag="default",
#     epoch=1,
# )

algo = PowerIteration(
    exec_dir="/lfs/raiders10/0/mleszczy/anchor/Hazy/build/bin", tag="default", epoch=1
)

# algo = MatrixCompletion(
#     exec_dir="/lfs/raiders10/0/mleszczy/anchor/Hazy/build/bin", tag="default", epoch=1
# )

# algo = GloVe(
#     exec_dir="/lfs/raiders10/0/mleszczy/anchor/GloVe/build",
#     tag="default_glove",
#     epoch=1,
# )

# original data
datapath = "/dfs/scratch1/mleszczy/exp/Hazy_git/data/text8/text8_incr/text8_12M"

# train first embedding
embedding = ems.gen_embedding(data=datapath, coo="coo_text8_12M.txt", algo=algo)

# new data
datapath = "/dfs/scratch1/mleszczy/exp/Hazy_git/data/text8/text8_incr/text8_24M"
# train new embedding with same algorithm + parameters
embedding = ems.gen_embedding(data=datapath, coo="coo_text8_24M.txt", algo=algo)
