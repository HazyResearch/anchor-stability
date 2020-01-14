../build/glove \
  -vector-size 300 \
  -threads 12 \
  -iter 0 \
  -eta 0.05 \
  -alpha 0.75 \
  -x-max 100.0 \
  -input-file data/cooccurrence.bin \
  -vocab-file data/vocab.txt \
  -save-file tmp_test_pretrained \
  -checkpoint-every 1 \
  -seed 1234 \
  -use-unk-vec 0 \
  -init-word data/pretrained.w.txt \
  -init-context data/pretrained.c.txt

if cmp -s "tmp_test_pretrained.c.txt" "data/pretrained.c.txt";
then 
  printf "Pass\n"
else
  printf "Context embedding load error\n"
  exit 1
fi

if cmp -s "tmp_test_pretrained.w.txt" "data/pretrained.w.txt";
then 
  printf "Pass\n"
else
  printf "Word embedding load error\n"
  exit 1
fi

rm -rf tmp_test_pretrained.txt tmp_test_pretrained.w.txt tmp_test_pretrained.c.txt

