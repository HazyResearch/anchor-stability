#ifndef SHUFFLE_H
#define SHUFFLE_H

#include <string>

namespace shuffle {
int shuffle(const std::string cooccur_in, const std::string cooccur_out,
            const int verbose_, const float memory_,
            const long long array_size_,
            const std::string temp_file_);
}


#endif 