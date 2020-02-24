ALGO=$1
EXP=$2 # exponent alpha for eigenspace instability
ANALYSISDIR=$3
RESULTDIR=$4

display_usage() {
    echo "Usage: bash eval_wiki_compressed_embs_anchor.sh algo exp"
}

# if less than one arguments supplied, display usage
if [ $# -le 1 ];
then
    display_usage
    exit 1
fi

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP} --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 25 --nseeds 3 --compressed --truncate 10000

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP} --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 50 --nseeds 3 --compressed --truncate 10000

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP}  --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 100 --nseeds 3 --compressed --truncate 10000

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP} --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 200 --nseeds 3 --compressed --truncate 10000

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP} --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 400 --nseeds 3 --compressed --truncate 10000

python scripts/analysis/dim_stability_analysis.py --algo ${ALGO} --dist eis --exp ${EXP} --out $ANALYSISDIR --resultdir1 $RESULTDIR/wiki_2017 --resultdir2 $RESULTDIR/wiki_2018 --wiki True --dim 800 --nseeds 3 --compressed --truncate 10000
