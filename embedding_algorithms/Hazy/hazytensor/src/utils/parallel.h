#ifndef PARALLEL_H
#define PARALLEL_H

#include <atomic>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <functional>

#define PADDING 300

namespace par {
template <class T>
struct reducer {
  T* elem;
  std::function<T(T, T)> f;
  reducer(T init_in, std::function<T(T, T)> f_in);
  ~reducer() { delete elem; }
  void update(const size_t tid, T new_val);
  T get(const size_t tid);
  void clear();
  T evaluate(T init);
};

struct staticParFor {
  size_t tid;
  size_t work_start;
  size_t work_end;
  static std::function<void(size_t, size_t)> body;
  staticParFor(size_t tid_in, size_t work_start_in, size_t work_end_in) {
    tid = tid_in;
    work_start = work_start_in;
    work_end = work_end_in;
  }
  void* run();
};

size_t for_range(const size_t from, const size_t to,
                 std::function<void(const size_t, const size_t)> body);

struct parFor {
  size_t tid;
  static std::atomic<size_t> next_work;
  static size_t block_size;
  static size_t range_len;
  static size_t offset;
  static std::function<void(size_t, size_t)> body;
  parFor(size_t tid_in) { tid = tid_in; }
  void* run();
};

size_t for_range(const size_t from, const size_t to, const size_t block_size,
                 std::function<void(const size_t, const size_t)> body);
}
#endif