Note: This code is for Python 2.

Code used for sentence classification tasks.

## How to run
  - Download the datasets from [harvardnlp/sent-conv-torch/data](https://github.com/harvardnlp/sent-conv-torch/tree/master/data)
  
  - Download pre-trained word embeddings such as [word2vec](https://code.google.com/p/word2vec/); make it into text format
    
  - Run **train_classifier.py** and get the results.
  ```
    python train_classifier.py --help           # see all running options
  
    python train_classifier.py --dataset mr     # which dataset (mr, subj, cr, sst, trec, mpqa) 
          --path data_directory                 # path to the data directory
          --embedding google_word2vec.txt       # path to pre-trained embeddings
          --cv 0                                # 10-fold cross-validation, use split 0 as the test set

    An example command for the: 
    python train_classifier.py --dataset trec --path ../sent-conv-torch/data/ --embedding ../../glove.6B.100d.txt --cv 2 --cnn  --out ./test2 2>&1 | tee test2/run.log 
  ```
  
  - Check **run_sc.sh** for all more details.
  
  <br>
  
  ### Credits
  
  Part of the code (such as text preprocessing) is taken from https://github.com/harvardnlp/sent-conv-torch
  
  CNN model is the implementation of [(Kim, 2014)](http://arxiv.org/abs/1408.5882), following
   - torch / lua version: https://github.com/yoonkim/CNN_sentence
   - pytorch version: https://github.com/Shawn1993/cnn-text-classification-pytorch
  
  
  
