#ifndef VOCABCOUNT_H
#define VOCABCOUNT_H

#include <string>

namespace vocab_count {
int find_arg(char *str, int argc, char **argv);

int vocab_count(const std::string corpus_filename,
                const std::string vocab_filename, const int verbose,
                const long long max_vocab, const long long min_count);
}

#endif