HOMEDIR=$PWD
PYTHONVENV=$(which python)
RESULTDIR=runs/models
EMBDIR=runs/embs

DATASET=sst
# see Appendix for values selected from grid search for each algorithm
MC_LR=0.001
CBOW_LR=0.0001

# MC
ls $EMBDIR/wiki_2017/*mc*br* | python scripts/model_training/gen_model_cmds.py --homedir $HOMEDIR --python $PYTHONVENV --resultdir $RESULTDIR/wiki_2017 --dataset $DATASET --task sentiment --lr $MC_LR > model_cmds.sh
ls $EMBDIR/wiki_2018/*mc*br* | python scripts/model_training/gen_model_cmds.py --homedir $HOMEDIR --python $PYTHONVENV --resultdir $RESULTDIR/wiki_2018 --dataset $DATASET --task sentiment --lr $MC_LR >> model_cmds.sh

# W2V CBOW
ls $EMBDIR/wiki_2017/*cbow*br* | python scripts/model_training/gen_model_cmds.py --homedir $HOMEDIR --python $PYTHONVENV --resultdir $RESULTDIR/wiki_2017 --dataset $DATASET --task sentiment --lr $CBOW_LR >> model_cmds.sh
ls $EMBDIR/wiki_2018/*cbow*br* | python scripts/model_training/gen_model_cmds.py --homedir $HOMEDIR --python $PYTHONVENV --resultdir $RESULTDIR/wiki_2018 --dataset $DATASET --task sentiment --lr $CBOW_LR >> model_cmds.sh