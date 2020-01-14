#!/bin/bash 
 
display_usage() {
    echo "Usage: sh run.sh TASK EMBFILE_PATH MODEL TAG"
} 

# if less than two arguments supplied, display usage
if [ $# -le 3 ];
then
    display_usage
    exit 1
fi 

TASK=$1
EMBFILE=$2
MODEL=$3
TAG=$4

case "$TASK" in 
    mr|subj|cr|mpqa|trec|sst)
        echo "Running task ${TASK}"
        ;;
    *)
        echo "Wrong task ${TASK}"
        exit 1
        ;;
esac

case "$MODEL" in 
    lstm|cnn|la)
        echo "Testing model ${MODEL}"
        ;;
    *)
        echo "Wrong model ${MODEL}"
        exit 1
        ;;
esac

emb_dir=$(dirname "${2}")
emb_file=$(basename "${2}")
echo ${emb_file}

SEED=1234
CV=0

MODELDIR=/dfs/scratch1/senwu/embedding/baselines/sentence_classification
RESFOLDER=sa_log
mkdir -p ${RESFOLDER}

if [ ! -f ${MODELDIR}/vocab/${TAG}_${emb_file}_${TASK}.pkl ]; then
  echo "===== Preparing vocab.. ====="
  python ${MODELDIR}/prepare_vocab.py \
             --dataset ${TASK} \
             --path ${MODELDIR}/sent-conv-torch/data \
             --embedding ${EMBFILE} \
             --output ${TAG}_${emb_file}_${TASK}
fi

if [ "$TASK" != "sst" ];
then
    for DROPOUT in 0.1 0.3 0.5 0.7;
    do
        for CV in 0 1 2 3 4 5 6 7 8 9;
        do
            echo "$ python ${MODELDIR}/train_classifier.py --dataset ${TASK} --path ${MODELDIR}/sent-conv-torch/data/ --embedding ${MODELDIR}/vocab/${TAG}_${emb_file}_${TASK}.pkl --cv ${CV} --${MODEL} --dropout ${DROPOUT} --seed ${SEED} > ${RESFOLDER}/${TASK}_${TAG}_${emb_file}_cv_${CV}_model_${MODEL}_dropout_${DROPOUT}_seed_${SEED}.log"
            python ${MODELDIR}/train_classifier.py --dataset ${TASK} --path ${MODELDIR}/sent-conv-torch/data/ --embedding ${MODELDIR}/vocab/${TAG}_${emb_file}_${TASK}.pkl --cv ${CV} --${MODEL} --dropout ${DROPOUT} --seed ${SEED} > ${RESFOLDER}/${TASK}_${TAG}_${emb_file}_cv_${CV}_model_${MODEL}_dropout_${DROPOUT}_seed_${SEED}.log
        done
    done
else
    for DROPOUT in 0.1 0.3 0.5 0.7;
    do
        for SEED in 1234 1235 1236 1237 1238;
        do
            echo "$ python ${MODELDIR}/train_classifier.py --dataset ${TASK} --path ${MODELDIR}/sent-conv-torch/data/ --embedding ${MODELDIR}/vocab/${TAG}_${emb_file}_${TASK}.pkl --cv ${CV} --${MODEL} --dropout ${DROPOUT} --seed ${SEED} > ${RESFOLDER}/${TASK}_${TAG}_${emb_file}_cv_${CV}_model_${MODEL}_dropout_${DROPOUT}_seed_${SEED}.log"
            python ${MODELDIR}/train_classifier.py --dataset ${TASK} --path ${MODELDIR}/sent-conv-torch/data/ --embedding ${MODELDIR}/vocab/${TAG}_${emb_file}_${TASK}.pkl --cv ${CV} --${MODEL} --dropout ${DROPOUT} --seed ${SEED} > ${RESFOLDER}/${TASK}_${TAG}_${emb_file}_cv_${CV}_model_${MODEL}_dropout_${DROPOUT}_seed_${SEED}.log
        done
    done
fi


