#include <stdio.h>
#include <cxxopts.hpp>
#include "datastructures/COO.h"
#include "solver/common.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include<iostream>
#include<fstream>


#ifndef __APPLE__
#include <omp.h>
#endif

namespace parser {
void exit_parser(const cxxopts::Options &options) {
  std::cerr << options.help() << std::endl;
  exit(1);
}

struct Args {
  std::string coo_filename;
  std::string sample_filename;
  int sample_size;
  bool symmetric;
  int nthreads;
  int seed;

  void log() {
    std::string myarg_string = "user options...";
    myarg_string += "\n\t(c,coo): " + coo_filename + "\n";
    myarg_string += "\t(o,out): " + sample_filename + "\n";
    myarg_string += "\t(s,size): " + std::to_string(sample_size) + "\n";
    myarg_string += "\t(m,sym): " + std::to_string(symmetric) + "\n";
    myarg_string += "\t(t,nthreads): " + std::to_string(nthreads) + "\n";
    myarg_string += "\t(s,seed): " + std::to_string(seed) + "\n";
    LOG(myarg_string);
  }
};

Args parse(int argc, char *argv[]) {
  cxxopts::Options options("sample", "Generates a sampled coo file (to disk) from an input cooccurrence file.");
  options.add_options()
    ("c,coo", "Input cooccurrence file name (string).", cxxopts::value<std::string>())
    ("o,out", "Output cooccurrence file name (string).", cxxopts::value<std::string>())
    ("n,size", "Percent to sample.", cxxopts::value<int>())
    ("m,sym", "Symmetric cooccurrence (default=true).", cxxopts::value<bool>())
    ("t,nthreads", "Number of threads to be used. (int) (default=4)", cxxopts::value<size_t>())
    ("s,seed", "Random seed (int) (default=1234)", cxxopts::value<size_t>());

  auto result = options.parse(argc, argv);

  Args args;
  if (result.count("coo") == 0) {
    std::cerr << "Missing cooccurrence file input." << std::endl;
    exit_parser(options);
  }
  args.coo_filename = result["coo"].as<std::string>();

  if (result.count("out") == 0) {
    std::cerr << "Missing ppmi file output name." << std::endl;
    exit_parser(options);
  }
  args.sample_filename = result["out"].as<std::string>();

  if (result.count("size") == 0) {
    std::cerr << "Missing size." << std::endl;
    exit_parser(options);
  }
  args.sample_size = result["size"].as<int>();

  if (result.count("sym") == 0) {
    args.symmetric = 1;
  } else {
    args.symmetric = result["sym"].as<bool>();
  }

  if (result.count("nthreads") == 0) {
    args.nthreads = 4;
  } else {
    args.nthreads = result["nthreads"].as<size_t>();
  }

  if (result.count("seed") == 0) {
    args.seed = 1234;
  } else {
    args.seed = result["seed"].as<size_t>();
  }

  return args;
}
}

int main(int argc, char *argv[]) {
  parser::Args args = parser::parse(argc, argv);
  //  Displays the selected options to the user.
  args.log();

  ThreadPool::initialize_thread_pool(args.nthreads);
#ifndef __APPLE__
  omp_set_num_threads(args.nthreads);  // set num threads
#endif

  const auto loading = timer::start_clock();
  COO<double> coo_cooccurrence =
      COO<double>::from_file(args.coo_filename);
  timer::stop_clock("LOADING", loading);

  const auto sampling = timer::start_clock();
  COO<double> sample_coo = coo_cooccurrence.sample(args.sample_size,
  	args.seed, args.symmetric);
  timer::stop_clock("SAMPLE TIME", sampling);

  const auto output = timer::start_clock();
  sample_coo.to_file(args.sample_filename);
  timer::stop_clock("SAVING TO DISK", output);
}
