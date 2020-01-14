EMBDIR=runs/embs

python scripts/embedding_generation/compression_experiment_wiki.py --resultdir1 $EMBDIR/wiki_2017 --resultdir2 $EMBDIR/wiki_2018 --algo mc > compress_cmds.sh

python scripts/embedding_generation/compression_experiment_wiki.py --resultdir1 $EMBDIR/wiki_2017 --resultdir2 $EMBDIR/wiki_2018 --algo w2v_cbow >> compress_cmds.sh