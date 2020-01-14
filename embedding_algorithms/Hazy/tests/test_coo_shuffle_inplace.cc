#include "datastructures/COO.h"
#include <unordered_map>
#include <catch.hpp> 

inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;}

TEST_CASE("Test COO shuffle_inplace")
{
  // Load data. Note: HAZYTESTPATH is set in setup.py
  char* pPath;
  pPath = getenv("HAZYTESTPATH");
  std::string cooccurrence_file = std::string(pPath) + "/tests/data/sample_data_cooccur.bin";

  COO<double> coo = COO<double>::from_file(cooccurrence_file);
  std::unordered_map<size_t,double> map;
  std::unordered_map<size_t,double> old_indices;

  size_t cnt = 0;
  size_t new_order_cnt = 0;

  for (size_t i = 0; i < coo.nnz(); ++i) {
    map[key(coo.rowind()[i], coo.colind()[i])] = coo.val()[i];
    old_indices[key(coo.rowind()[i], coo.colind()[i])] = i;
    cnt += 1;
  }

  coo.shuffle_inplace(1234);

  for (size_t i = 0; i < coo.nnz(); ++i) {
    REQUIRE(map[key(coo.rowind()[i], coo.colind()[i])] == coo.val()[i]);
    // Tuple moved location
    if (old_indices[key(coo.rowind()[i], coo.colind()[i])] != i){
      new_order_cnt += 1;
    }
    cnt -= 1;
  }  

  // Check if shuffling occurred
  REQUIRE(new_order_cnt != 0);

  REQUIRE(cnt == 0);
} 
