# Acknowledgements

Much of the code in intrinsic evaluation is original from [hyperwords](https://bitbucket.org/omerlevy/hyperwords). We thank all authors to provide these tasks available online.

# Requirements

The intrinsic evaluation requires Python3 with the following packages installed:
- numpy
- scipy
- docopt

# Evaluation
Intrinsic evaluation contains two types of tasks: word similarity and analogies.

## Word Similarity
- **ws_eval.py**
- Compares how a representation ranks pairs of related words by similarity versus human ranking.
- 6 readily-available datasets
    - ws353_similarity.txt
    - ws353_relatedness.txt
    - radinsky_mturk.txt
    - bruni_men.txt
    - luong_rare.txt
    - simlex999.txt

## Analogies
- **analogy_eval.py**
- Solves analogy questions, such as: "man is to woman asking is to...?" (answer: queen).
- 2 readily-available datasets
    - google.txt
    - msr.txt
- Shows results of two analogy recovery methods: 3CosAdd and 3CosMul. For more information, see:
**"Linguistic Regularities in Sparse and Explicit Word Representations". Omer Levy and Yoav Goldberg. CoNLL 2014.**

## Datasets
In order to download those datasets, run `bash download_data.sh`.

## Examples
To evaluate embeddings on word similarity tasks, just run:
```
python ws_eval.py GLOVE [PATH_TO_YOUR_EMBEDDING_FILE] testsets/ws/[TEST_NAME]
```

To evaluate embeddings on analogy tasks, just run:
```
python analogy_eval.py GLOVE [PATH_TO_YOUR_EMBEDDING_FILE] testsets/analogy/[TEST_NAME]
```

The evaluation assumes embedding files in the textual format such as word2vec, GLOVE, and FastText. You can just use `GLOVE` tag as the argument.
