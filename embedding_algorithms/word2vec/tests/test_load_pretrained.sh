../word2vec \
  -train data/corpus.txt \
  -output tmp_test_pretrained \
  -size 300 \
  -window 4 \
  -sample 0.0 \
  -negative 5 \
  -threads 1 \
  -iter 0 \
  -min-count 100 \
  -read-vocab data/vocab.txt \
  -cbow 0 \
  -seed 1234 \
  -checkpoint_interval 1 \
  -init-word data/pretrained.w.txt \
  -init-context data/pretrained.c.txt

if cmp -s "tmp_test_pretrained.0.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Context embedding load error\n"
  exit 1
fi

if cmp -s "tmp_test_pretrained.0.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Word embedding load error\n"
  exit 1
fi

rm -rf tmp_test_pretrained.0.w.txt tmp_test_pretrained.0.c.txt

