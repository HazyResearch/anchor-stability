set -x

TASK=$1
ALGO=$2
LR=$3
ANALYSISDIR=$4
RESULTDIR=$5

display_usage() {
    echo "Usage: bash eval_wiki_compressed.sh dataset algo lr"
}

if [ $# -le 2 ];
then
    display_usage
    exit 1
fi

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 25 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 50 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 100 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 200 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 400 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist pred --out $ANALYSISDIR --model la --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --lr ${LR} --wiki True --dim 800 --nseeds 3 --task ${TASK} --compressed --tag no_emb_norm