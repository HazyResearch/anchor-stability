../build/glove \
  -vector-size 300 \
  -threads 12 \
  -iter 5 \
  -eta 0.05 \
  -alpha 0.75 \
  -x-max 100.0 \
  -input-file data/cooccurrence.bin \
  -vocab-file data/vocab.txt \
  -save-file tmp_test_freeze \
  -checkpoint-every 1 \
  -seed 1234 \
  -use-unk-vec 0 \
  -init-word data/pretrained.w.txt \
  -init-context data/pretrained.c.txt \
  -freeze-vocab data/vocab.txt \
  -freeze-iter 3

if cmp -s "tmp_test_freeze.001.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 1\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.001.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 1\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.002.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 2\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.002.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 2\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.003.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 3\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.003.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 3\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.004.c.txt" "data/pretrained.c.txt";
then 
  printf "Freeze context embedding error at epoch 4\n"
  exit 1  
else
  printf "Pass\n"
fi

if cmp -s "tmp_test_freeze.004.w.txt" "data/pretrained.w.txt";
then 
  printf "Freeze word embedding error at epoch 4\n"
  exit 1
else
  printf "Pass\n"
fi



if cmp -s "tmp_test_freeze.005.c.txt" "data/pretrained.c.txt";
then 
  printf "Freeze context embedding error at epoch 5\n"
  exit 1
else
  printf "Pass\n"
fi

if cmp -s "tmp_test_freeze.005.w.txt" "data/pretrained.w.txt";
then 
  printf "Freeze word embedding error at epoch 5\n"
  exit 1
else
  printf "Pass\n"
fi

rm -rf tmp_test_freeze*
