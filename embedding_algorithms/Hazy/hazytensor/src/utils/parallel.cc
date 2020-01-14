#include "utils/parallel.h"
#include "utils/thread_pool.h"

size_t ThreadPool::num_threads;

namespace par {
template <class T>
inline void reducer<T>::update(const size_t tid, T new_val) {
  elem[tid * PADDING] = f(elem[tid * PADDING], new_val);
}

template <class T>
inline T reducer<T>::get(const size_t tid) {
  return elem[tid * PADDING];
}

template <class T>
inline void reducer<T>::clear() {
  memset(elem, (uint8_t)0, sizeof(T) * ThreadPool::num_threads * PADDING);
}

template <class T>
inline T reducer<T>::evaluate(T init) {
  for (size_t i = 0; i < ThreadPool::num_threads; i++) {
    init = f(init, elem[i * PADDING]);
  }
  return init;
}

template <class T>
reducer<T>::reducer(T init_in, std::function<T(T, T)> f_in) {
  f = f_in;
  elem = new T[ThreadPool::num_threads * PADDING];
  for (size_t i = 0; i < ThreadPool::num_threads; i++) {
    elem[i * PADDING] = init_in;
  }
  memset(elem, (uint8_t)0, sizeof(T) * ThreadPool::num_threads * PADDING);
}

std::atomic<size_t> parFor::next_work;
size_t parFor::block_size;
size_t parFor::range_len;
size_t parFor::offset;
std::function<void(size_t, size_t)> parFor::body;

size_t for_range(const size_t from, const size_t to,
                 std::function<void(const size_t, const size_t)> body) {
  const size_t range_len = to - from;
  size_t chunk_size =
      (range_len + ThreadPool::num_threads) / ThreadPool::num_threads;
  const size_t THREADS_CHUNKS = (range_len % ThreadPool::num_threads);
  ThreadPool::init_threads();
  staticParFor::body = body;
  staticParFor **pf = new staticParFor *[ThreadPool::num_threads];
  for (size_t k = 0; k < THREADS_CHUNKS; k++) {
    const size_t work_start = k * chunk_size;
    const size_t work_end = work_start + chunk_size;
    pf[k] = new staticParFor(k, work_start, work_end);
    ThreadPool::submitWork(k, ThreadPool::general_body<staticParFor>,
                            (void *)(pf[k]));
  }
  const size_t offset = THREADS_CHUNKS * chunk_size;
  --chunk_size;
  for (size_t k = 0; k < (ThreadPool::num_threads - THREADS_CHUNKS); k++) {
    const size_t work_start = offset + k * chunk_size;
    const size_t work_end = work_start + chunk_size;
    const size_t tid = k + THREADS_CHUNKS;
    pf[tid] = new staticParFor(tid, work_start, work_end);
    ThreadPool::submitWork(tid, ThreadPool::general_body<staticParFor>,
                            (void *)(pf[tid]));
  }
  ThreadPool::join_threads();
  for (size_t k = 0; k < ThreadPool::num_threads; k++) {
    delete pf[k];
  }
  delete[] pf;
  return 1;
}

size_t for_range(const size_t from, const size_t to, const size_t block_size,
                 std::function<void(const size_t, const size_t)> body) {
  const size_t range_len = to - from;
  const size_t actual_block_size = (block_size > range_len) ? 1 : block_size;
  ThreadPool::init_threads();
  parFor::next_work = 0;
  parFor::block_size = actual_block_size;

  parFor::range_len = range_len;
  parFor::offset = from;
  parFor::body = body;
  parFor **pf = new parFor *[ThreadPool::num_threads];
  for (size_t k = 0; k < ThreadPool::num_threads; k++) {
    pf[k] = new parFor(k);
  }
  for (size_t k = 0; k < ThreadPool::num_threads; k++) {
    ThreadPool::submitWork(k, ThreadPool::general_body<parFor>,
                            (void *)(pf[k]));
  }
  ThreadPool::join_threads();
  for (size_t k = 0; k < ThreadPool::num_threads; k++) {
    delete pf[k];
  }
  delete[] pf;
  return 1;
}

void *parFor::run() {
  while (true) {
    const size_t work_start = parFor::next_work.fetch_add(
        parFor::block_size, std::memory_order_relaxed);
    if (work_start > parFor::range_len) break;

    const size_t work_end = std::min(
        parFor::offset + work_start + parFor::block_size, parFor::range_len);
    // std::cout << tid << " " << work_start << " " << work_end << " " <<
    // parFor::range_len << std::endl;

    for (size_t j = work_start; j < work_end; j++) {
      parFor::body(tid, parFor::offset + j);
    }
  }
  return NULL;
}

std::function<void(size_t, size_t)> staticParFor::body;
void *staticParFor::run() {
  for (size_t j = work_start; j < work_end; j++) {
    staticParFor::body(tid, j);
  }
  return NULL;
}
}

template struct par::reducer<double>;
template struct par::reducer<uint32_t>;
template struct par::reducer<int>;
template struct par::reducer<float>;
template struct par::reducer<long>;
template struct par::reducer<size_t>;