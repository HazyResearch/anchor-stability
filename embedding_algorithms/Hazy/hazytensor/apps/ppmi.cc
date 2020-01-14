#include <stdio.h>
#include <cxxopts.hpp>
#include "datastructures/COO.h"
#include "solver/solver.h"
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
  std::string cooccurrence_filename;
  std::string ppmi_filename;
  int nthreads;

  void log() {
    std::string myarg_string = "user options...";
    myarg_string += "\n\t(c,coo): " + cooccurrence_filename + "\n";
    myarg_string += "\t(p,ppmi): " + ppmi_filename + "\n";
    myarg_string += "\t(t,nthreads): " + std::to_string(nthreads) + "\n";
    LOG(myarg_string);
  }
};

Args parse(int argc, char *argv[]) {
  cxxopts::Options options("ppmi", "Generates a ppmi file (to disk) from a input cooccurrence file.");
  options.add_options()
    ("c,coo", "Input cooccurrence file name (string).", cxxopts::value<std::string>())
    ("p,ppmi", "Output ppmi file name (string).", cxxopts::value<std::string>())
    ("t,nthreads", "Number of threads (int).", cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  Args args;
  if (result.count("coo") == 0) {
    std::cerr << "Missing cooccurrence file input." << std::endl;
    exit_parser(options);
  }
  args.cooccurrence_filename = result["coo"].as<std::string>();

  if (result.count("ppmi") == 0) {
    std::cerr << "Missing ppmi file output name." << std::endl;
    exit_parser(options);
  }
  args.ppmi_filename = result["ppmi"].as<std::string>();

  if (result.count("nthreads") == 0) {
    std::cerr << "Missing nthreads." << std::endl;
    exit_parser(options);
  }
  args.nthreads = result["nthreads"].as<int>();

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
      COO<double>::from_file(args.cooccurrence_filename);

  timer::stop_clock("LOADING", loading);

  const auto csr_build = timer::start_clock();
  CSR<double> csr_cooccurrence = CSR<double>::from_coo(
      coo_cooccurrence.n(), coo_cooccurrence.nnz(), coo_cooccurrence.rowind(),
      coo_cooccurrence.colind(), coo_cooccurrence.val(), /*sort_rows=*/false);
  timer::stop_clock("CSR BUILD", csr_build);

  const auto preprocessing = timer::start_clock();
  csr_cooccurrence.ppmi();
  COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
  timer::stop_clock("PPMI BUILD TIME", preprocessing);
  
  const auto output = timer::start_clock();
  new_coo.to_file(args.ppmi_filename);

  COO<double> ppmi = COO<double>::from_file(args.ppmi_filename);
  timer::stop_clock("SAVING TO DISK", csr_build);
}