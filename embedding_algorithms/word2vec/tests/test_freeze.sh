../word2vec \
  -train data/corpus.txt \
  -output tmp_test_freeze \
  -size 300 \
  -window 4 \
  -sample 0.0 \
  -negative 5 \
  -threads 1 \
  -iter 5 \
  -min-count 100 \
  -read-vocab data/vocab.txt \
  -cbow 0 \
  -seed 1234 \
  -checkpoint_interval 1 \
  -init-word data/pretrained.w.txt \
  -init-context data/pretrained.c.txt \
  -freeze-vocab data/vocab.txt \
  -freeze-iter 3

if cmp -s "tmp_test_freeze.1.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 1\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.1.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 1\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.2.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 2\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.2.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 2\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.3.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Freeze context embedding error at epoch 3\n"
  exit 1
fi

if cmp -s "tmp_test_freeze.3.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Freeze word embedding error at epoch 3\n"
  exit 1
fi


if cmp -s "tmp_test_freeze.4.c.txt" "data/pretrained.c.txt";
then 
  printf "Freeze context embedding error at epoch 4\n"
  exit 1  
else
  printf "Pass\n"
fi

if cmp -s "tmp_test_freeze.4.w.txt" "data/pretrained.w.txt";
then 
  printf "Freeze word embedding error at epoch 4\n"
  exit 1
else
  printf "Pass\n"
fi



if cmp -s "tmp_test_freeze.5.c.txt" "data/pretrained.c.txt";
then 
  printf "Freeze context embedding error at epoch 5\n"
  exit 1
else
  printf "Pass\n"
fi

if cmp -s "tmp_test_freeze.5.w.txt" "data/pretrained.w.txt";
then 
  printf "Freeze word embedding error at epoch 5\n"
  exit 1
else
  printf "Pass\n"
fi

rm -rf tmp_test_freeze*

