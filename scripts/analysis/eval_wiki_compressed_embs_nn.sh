set -x

ALGO=$1 # word embedding algorithm
NN=$2 # number of neighbors
NQUERY=$3 # number words to find nearest neighbors for
ANALYSISDIR=$4
RESULTDIR=$5

display_usage() {
    echo "Usage: bash eval_wiki_compressed_embs.sh algo nn nquery"
}

if [ $# -le 2 ];
then
    display_usage
    exit 1
fi

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 25 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 50 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn  --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 100 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 200 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 400 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist knn --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 800 --nseeds 3 --compressed --truncate 10000 --nneighbors ${NN} --nquery ${NQUERY}
