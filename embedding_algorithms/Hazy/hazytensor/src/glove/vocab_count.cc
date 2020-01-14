//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include "vocab_count.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>

namespace vocab_count {

#define MAX_STRING_LENGTH 1000
#define TSIZE 1048576
#define SEED 1159241
#define HASHFN bitwisehash

typedef struct vocabulary {
  char *word;
  long long count;
} VOCAB;

typedef struct hashrec {
  char *word;
  long long count;
  struct hashrec *next;
} HASHREC;

int verbose = 2;          // 0, 1, or 2
long long min_count = 1;  // min occurrences for inclusion in vocab
long long max_vocab = 0;  // max_vocab = 0 for no limit

/* Efficient string comparison */
int scmp(char *s1, char *s2) {
  while (*s1 != '\0' && *s1 == *s2) {
    s1++;
    s2++;
  }
  return (*s1 - *s2);
}

/* Vocab frequency comparison; break ties alphabetically */
int CompareVocabTie(const void *a, const void *b) {
  long long c;
  if ((c = ((VOCAB *)b)->count - ((VOCAB *)a)->count) != 0)
    return (c > 0 ? 1 : -1);
  else
    return (scmp(((VOCAB *)a)->word, ((VOCAB *)b)->word));
}

/* Vocab frequency comparison; no tie-breaker */
int CompareVocab(const void *a, const void *b) {
  long long c;
  if ((c = ((VOCAB *)b)->count - ((VOCAB *)a)->count) != 0)
    return (c > 0 ? 1 : -1);
  else
    return 0;
}

/* Move-to-front hashing and hash function from Hugh Williams,
 * http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
  char c;
  unsigned int h;
  h = seed;
  for (; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
  return ((unsigned int)((h & 0x7fffffff) % tsize));
}

/* Create hash table, initialise pointers to NULL */
HASHREC **inithashtable() {
  int i;
  HASHREC **ht;
  ht = (HASHREC **)malloc(sizeof(HASHREC *) * TSIZE);
  for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *)NULL;
  return (ht);
}

/* Search hash table for given string, insert if not found */
void hashinsert(HASHREC **ht, char *w) {
  HASHREC *htmp, *hprv;
  unsigned int hval = HASHFN(w, TSIZE, SEED);

  for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0;
       hprv = htmp, htmp = htmp->next)
    ;
  if (htmp == NULL) {
    htmp = (HASHREC *)malloc(sizeof(HASHREC));
    htmp->word = (char *)malloc(strlen(w) + 1);
    strcpy(htmp->word, w);
    htmp->count = 1;
    htmp->next = NULL;
    if (hprv == NULL)
      ht[hval] = htmp;
    else
      hprv->next = htmp;
  } else {
    /* new records are not moved to front */
    htmp->count++;
    if (hprv != NULL) {
      /* move to front on access */
      hprv->next = htmp->next;
      htmp->next = ht[hval];
      ht[hval] = htmp;
    }
  }
  return;
}

int get_counts(const std::string corpus_filename,
               const std::string vocab_filename) {
  long long i = 0, j = 0, vocab_size = 12500;
  char format[20];
  char str[MAX_STRING_LENGTH + 1];
  HASHREC **vocab_hash = inithashtable();
  HASHREC *htmp;
  VOCAB *vocab;
  FILE *fid = fopen (corpus_filename.c_str() , "r");

  std::ofstream write_file;
  write_file.open (vocab_filename);

  if (verbose > 1) fprintf(stderr, "Processed %lld tokens.", i);
  sprintf(format, "%%%ds", MAX_STRING_LENGTH);
  while (fscanf(fid, format, str) != EOF) {  // Insert all tokens into hashtable
    if (strcmp(str, "<unk>") == 0) {
      fprintf(stderr,
              "\nError, <unk> vector found in corpus.\nPlease remove <unk>s "
              "from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' "
              "> text8.new)");
      return 1;
    }
    hashinsert(vocab_hash, str);
    if (((++i) % 100000) == 0)
      if (verbose > 1) fprintf(stderr, "\033[11G%lld tokens.", i);
  }
  if (verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", i);
  vocab = (VOCAB*)malloc(sizeof(VOCAB) * vocab_size);
  for (i = 0; i < TSIZE; i++) {  // Migrate vocab to array
    htmp = vocab_hash[i];
    while (htmp != NULL) {
      vocab[j].word = htmp->word;
      vocab[j].count = htmp->count;
      j++;
      if (j >= vocab_size) {
        vocab_size += 2500;
        vocab = (VOCAB *)realloc(vocab, sizeof(VOCAB) * vocab_size);
      }
      htmp = htmp->next;
    }
  }
  if (verbose > 1) fprintf(stderr, "Counted %lld unique words.\n", j);
  if (max_vocab > 0 && max_vocab < j)
    // If the vocabulary exceeds limit, first sort full vocab by frequency
    // without alphabetical tie-breaks.
    // This results in pseudo-random ordering for words with same frequency, so
    // that when truncated, the words span whole alphabet
    qsort(vocab, j, sizeof(VOCAB), CompareVocab);
  else
    max_vocab = j;
  qsort(vocab, max_vocab, sizeof(VOCAB),
        CompareVocabTie);  // After (possibly) truncating, sort (possibly
                           // again), breaking ties alphabetically

  for (i = 0; i < max_vocab; i++) {
    if (vocab[i].count < min_count) {  // If a minimum frequency cutoff exists,
                                       // truncate vocabulary
      if (verbose > 0)
        fprintf(stderr, "Truncating vocabulary at min count %lld.\n",
                min_count);
      break;
    }
    write_file << vocab[i].word << " " << vocab[i].count << "\n";
  }

  if (i == max_vocab && max_vocab < j)
    if (verbose > 0)
      fprintf(stderr, "Truncating vocabulary at size %lld.\n", max_vocab);
  fclose(fid);
  write_file.close();
  fprintf(stderr, "Using vocabulary of size %lld.\n\n", i);
  return 0;
}

int find_arg(char *str, int argc, char **argv) {
  int i;
  for (i = 1; i < argc; i++) {
    if (!scmp(str, argv[i])) {
      if (i == argc - 1) {
        printf("No argument given for %s\n", str);
        exit(1);
      }
      return i;
    }
  }
  return -1;
}

int vocab_count(const std::string corpus_filename,
                const std::string vocab_filename, const int verbose_ = 2,
                const long long max_vocab_ = max_vocab,
                const long long min_count_ = min_count) {
  verbose = verbose_;
  max_vocab = max_vocab_;
  min_count = min_count_;
  return get_counts(corpus_filename, vocab_filename);
}

}  // end namespace vocab_count