#include "glove/vocab_count.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
    int i;
    std::string corpus_filename;
    std::string vocab_filename;
    int verbose;
    long long max_vocab;
    long long min_count;

    if (argc == 1) {
        printf("Simple tool to extract unigram counts\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu). Modified by hazytensor authors.\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-max-vocab <int>\n");
        printf("\t\tUpper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.\n");
        printf("\t-min-count <int>\n");
        printf("\t\tLower limit such that words which occur fewer than <int> times are discarded.\n");
        printf("\t-data-file <file>\n");
        printf("\t\tFile to extract vocab from.\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile to write vocab to.");
        printf("\nExample usage:\n");
        printf("./vocab_count -verbose 2 -max-vocab 100000 -min-count 10 -data-file corpus.txt -vocab-file vocab.txt\n");
        return 0;
    }

    if ((i = vocab_count::find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = vocab_count::find_arg((char *)"-max-vocab", argc, argv)) > 0) max_vocab = atoll(argv[i + 1]);
    if ((i = vocab_count::find_arg((char *)"-min-count", argc, argv)) > 0) min_count = atoll(argv[i + 1]);
    if ((i = vocab_count::find_arg((char *)"-data-file", argc, argv)) > 0) corpus_filename = std::string(argv[i + 1]);
    if ((i = vocab_count::find_arg((char *)"-vocab-file", argc, argv)) > 0) vocab_filename = std::string(argv[i + 1]);

    return vocab_count::vocab_count(corpus_filename, vocab_filename, verbose, max_vocab, min_count);
}