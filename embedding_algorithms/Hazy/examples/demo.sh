#!/bin/bash
set -x

# Downloads sample data and builds an embedding using PI.

DATADIR=data/text8
DATAFILE=text8
BUILDDIR=build/bin
RESULTDIR=result

mkdir -p $DATADIR
mkdir -p $RESULTDIR

if [ ! -e $DATADIR/$DATAFILE ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip -P $DATADIR
  else
    curl http://mattmahoney.net/dc/text8.zip -o $DATADIR/text8.zip
  fi
  unzip $DATADIR/text8.zip -d $DATADIR
  rm $DATADIR/text8.zip
fi

# We use modified GloVe code to build a co-occurrence matrix as input

MIN_COUNT=5
WINDOW_SIZE=15
VERBOSE=2
MEMORY=500.0
VOCAB_FILE=${DATAFILE}_vocab_minCount_${MIN_COUNT}_ws_${WINDOW_SIZE}.txt
COOCCURRENCE_FILE=${DATAFILE}_cooccurrence_minCount_${MIN_COUNT}_ws_${WINDOW_SIZE}.bin

# Generate vocab file

if [ ! -e $DATADIR/$VOCAB_FILE ]; then
	$BUILDDIR/vocab_count -min-count $MIN_COUNT -verbose $VERBOSE -data-file $DATADIR/$DATAFILE -vocab-file $DATADIR/$VOCAB_FILE
fi

# Generate cooccurrence file

if [ ! -e $DATADIR/$COOCCURRENCE_FILE ]; then
	$BUILDDIR/cooccur -memory $MEMORY -vocab-file $DATADIR/$VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -data-file $DATADIR/$DATAFILE -output $DATADIR/$COOCCURRENCE_FILE
fi

# Build an embedding using power iteration

THREADS=4
SOLVER=pi
MAX_ITER=10
DIM=300
TOL=1e-4
OUTPUT_FILE=$RESULTDIR/text8
CKPT=0
LOG_FREQ=2

$BUILDDIR/embedding -x $SOLVER -f $DATADIR/$COOCCURRENCE_FILE -v $DATADIR/$VOCAB_FILE -i $MAX_ITER -d $DIM -t $THREADS -o $OUTPUT_FILE  -z $CKPT -p $LOG_FREQ
