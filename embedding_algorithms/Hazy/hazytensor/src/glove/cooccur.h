#ifndef COOCCUR_H
#define COOCCUR_H

#include <string>

namespace cooccur {

int find_arg(char *str, int argc, char **argv);

int cooccur(const std::string corpus_file, const std::string cooccur_file,
            const int verbose_, const int symmetric_, const int window_size_,
            const std::string vocab_file_, const float memory_,
            const long long max_product_, const long long overflow_length_,
            const std::string overflow_file_);
}

#endif