#include "glove/cooccur.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>

int main(int argc, char **argv) {
    int i;
    std::string corpus_file;
    std::string cooccur_file;
    std::string vocab_file = "vocab.txt";
    std::string file_head = "overflow";
    int verbose = 2;
    int symmetric = 1;
    int window_size = 15;
    float memory = 3;
    long long max_product = 0;
    long long overflow_length = 0;

    if (argc == 1) {
        printf("Tool to calculate word-word cooccurrence statistics\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu). Modified by hazytensor authors.\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, 2 (default), or 3\n");
        printf("\t-symmetric <int>\n");
        printf("\t\tIf <int> = 0, only use left context; if <int> = 1 (default), use left and right\n");
        printf("\t-window-size <int>\n");
        printf("\t\tNumber of context words to the left (and to the right, if symmetric = 1); default 15\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-memory <float>\n");
        printf("\t\tSoft limit for memory consumption, in GB -- based on simple heuristic, so not extremely accurate; default 4.0\n");
        printf("\t-max-product <int>\n");
        printf("\t\tLimit the size of dense cooccurrence array by specifying the max product <int> of the frequency counts of the two cooccurring words.\n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-length <int>\n");
        printf("\t\tLimit to length <int> the sparse overflow array, which buffers cooccurrence data that does not fit in the dense array, before writing to disk. \n\t\tThis value overrides that which is automatically produced by '-memory'. Typically only needs adjustment for use with very large corpora.\n");
        printf("\t-overflow-file <file>\n");
        printf("\t\tFilename, excluding extension, for temporary files; default overflow\n");
        printf("\t-data-file <file>\n");
        printf("\t\tFile to extract cooccurrences from.\n");
        printf("\t-output <file>\n");
        printf("\t\tFile to write cooccurrences to.");

        printf("\nExample usage:\n");
        printf("./cooccur -verbose 2 -symmetric 0 -window-size 10 -vocab-file vocab.txt -memory 8.0 -overflow-file tempoverflow -data-file corpus.txt -output cooccurrences.bin\n\n");
        return 0;
    }

    if ((i = cooccur::find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-symmetric", argc, argv)) > 0) symmetric = atoi(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-window-size", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-vocab-file", argc, argv)) > 0) vocab_file = std::string(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-overflow-file", argc, argv)) > 0) file_head = std::string(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-data-file", argc, argv)) > 0) corpus_file = std::string(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-output", argc, argv)) > 0) cooccur_file = std::string(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-memory", argc, argv)) > 0) memory = atof(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-max-product", argc, argv)) > 0) max_product = atoll(argv[i + 1]);
    if ((i = cooccur::find_arg((char *)"-overflow-length", argc, argv)) > 0) overflow_length = atoll(argv[i + 1]);

    return cooccur::cooccur(corpus_file, cooccur_file, verbose, symmetric,
            window_size, vocab_file, memory, max_product, overflow_length, file_head);
}