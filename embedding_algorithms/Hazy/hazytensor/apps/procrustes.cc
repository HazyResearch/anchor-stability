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
  std::size_t n_threads;
  std::size_t n_words;
  std::size_t n_dimensions;
  std::size_t seed;
  std::string pretrained;
  std::string out_filename;
  std::string vocab_filename;

  void log() {
    std::string myarg_string = "user options...";
    myarg_string += "\t(t,nthreads): " + std::to_string(n_threads) + "\n";
    myarg_string += "\t(d,ndimensions): " + std::to_string(n_dimensions) + "\n";
    myarg_string += "\t(s,seed): " + std::to_string(seed) + "\n";
    myarg_string += "\t(r,pretrained): " + pretrained + "\n";
    myarg_string += "\t(o,out): " + out_filename + "\n";;
    LOG(myarg_string);
  }
};

Args parse(int argc, char *argv[]) {
  cxxopts::Options options("procrustes", "Performs orthogonal procrustes on a pretrained embedding");
  options.add_options()
    ("t,nthreads", "Number of threads to be used. (int, default=4)", cxxopts::value<size_t>())
    ("d,ndimensions", "Number of dimensions for embedding. (int, required)", cxxopts::value<size_t>())
    ("s,seed", "Random seed for B matrix (int, default=1234)", cxxopts::value<size_t>())
    ("r,pretrained", "Pre-trained model file name (required)", cxxopts::value<std::string>())
    ("o,out", "Output embedding file name (required).", cxxopts::value<std::string>());

  auto result = options.parse(argc, argv);

  Args args;
  if (result.count("nthreads") == 0) {
    args.n_threads = 4;
  } else {
    args.n_threads = result["nthreads"].as<size_t>();
  }

  if (result.count("ndimensions") == 0) {
    std::cerr << "Missing ndimensions." << std::endl;
    exit_parser(options);
  }
  args.n_dimensions = result["ndimensions"].as<size_t>();

  if (result.count("seed") == 0) {
    args.seed = 1234;
  } else {
    args.seed = result["seed"].as<size_t>();
  }

  if (result.count("pretrained") == 0) {
    std::cerr << "Missing pretrained embedding." << std::endl;
    exit_parser(options);
  }
  args.pretrained = result["pretrained"].as<std::string>();

  if (result.count("out") == 0) {
    std::cerr << "Missing output filename." << std::endl;
    exit_parser(options);
  }
  args.out_filename = result["out"].as<std::string>();

  return args;
}
}

int main(int argc, char *argv[]) {
  parser::Args args = parser::parse(argc, argv);
  //  Displays the selected options to the user.
  args.log();

  std::vector<std::string> vocab;
  const std::string output_file = args.out_filename;

#ifndef __APPLE__
  omp_set_num_threads(args.n_threads);  // set num threads
#endif

  const auto loading = timer::start_clock();
  DenseMatrix<double> pretrained_model = utils::load_pre_trained_model(
      args.n_dimensions, args.pretrained, vocab);
  timer::stop_clock("LOADING", loading);

  // Procrustes
  const auto orthogonal_procrustes = timer::start_clock();
  DenseMatrix<double> procrustes_model = utils::orthogonal_procrustes(pretrained_model, args.seed);
  timer::stop_clock("PROCRUSTES TIME", orthogonal_procrustes);

  const auto save_to_file = timer::start_clock();
  utils::save_to_file(procrustes_model, vocab,
                      output_file);
  timer::stop_clock("SAVE TO FILE", save_to_file);
}