#include <stdio.h>
#include <cxxopts.hpp>
#include "datastructures/COO.h"
#include "solver/solver.h"
#include "utils/timer.h"
#include "utils/utils.h"
#include <set>

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
  std::string vocab_filename;
  size_t n_dimensions;
  size_t n_threads;
  size_t n_epochs;
  size_t save_epochs;
  size_t log_epochs;
  bool gpu;
  bool debug;
  std::string out_filename;
  size_t init_seed;
  size_t sample_seed;
  double tol;
  size_t freq;
  size_t batch_size;
  double lr;
  double beta;
  double lambda;
  double lr_decay;
  std::string pre_trained;
  bool new_corpus;
  std::string solver;
  std::string freeze_vocab_filename;
  size_t freeze_n_epochs;

  void log() {
    std::string myarg_string = "user options...";
    myarg_string += "\n\t(f,file): " + cooccurrence_filename + "\n";
    myarg_string += "\t(v,vocab): " + vocab_filename + "\n";
    myarg_string += "\t(o,out): " + out_filename + "\n";
    myarg_string += "\t(x,solver): " + solver + "\n";
    myarg_string += "\t(t,nthreads): " + std::to_string(n_threads) + "\n";
    myarg_string += "\t(d,ndimensions): " + std::to_string(n_dimensions) + "\n";
    myarg_string += "\t(i,nepochs): " + std::to_string(n_epochs) + "\n";
    myarg_string += "\t(g,gpu): " + std::to_string(gpu) + "\n";
    myarg_string += "\t(u,debug): " + std::to_string(debug) + "\n";
    myarg_string += "\t(s,init_seed): " + std::to_string(init_seed) + "\n";
    myarg_string += "\t(w,sample_seed): " + std::to_string(sample_seed) + "\n";
    myarg_string += "\t(z,save): " + std::to_string(save_epochs) + "\n";
    myarg_string += "\t(p,log): " + std::to_string(log_epochs) + "\n";
    myarg_string += "\t(e,tol): " + std::to_string(tol) + "\n";
    myarg_string += "\t(l,lr): " + std::to_string(lr) + "\n";
    myarg_string += "\t(a,beta): " + std::to_string(beta) + "\n";
    myarg_string += "\t(m,lambda): " + std::to_string(lambda) + "\n";
    myarg_string += "\t(y,lr_decay): " + std::to_string(lr_decay) + "\n";
    myarg_string += "\t(q,freq): " + std::to_string(freq) + "\n";
    myarg_string += "\t(b,batch_size): " + std::to_string(batch_size) + "\n";
    myarg_string += "\t(r,pre_trained): " + pre_trained + "\n";
    myarg_string += "\t(n,new_corpus): " + std::to_string(new_corpus) + "\n";
    myarg_string += "\t(h,freeze_vocab): " + freeze_vocab_filename + "\n";
    myarg_string += "\t(c,freeze_n_epochs): " + std::to_string(freeze_n_epochs) + "\n";
    LOG(myarg_string);
  }
};

Args parse(int argc, char *argv[]) {
  cxxopts::Options options("embedding", "Faster embeddings for the masses.");
  options.add_options()
    ("f,file", "Cooccurrence input file name (required).", cxxopts::value<std::string>())
    ("v,vocab", "Vocab file name (required).", cxxopts::value<std::string>())
    ("o,out", "Output input file name (required).", cxxopts::value<std::string>())
    ("x,solver", "Solver (pi, dpi, gd, sgd, svrg, hybird_pi_sgd) (required)", cxxopts::value<std::string>())
    ("t,nthreads", "Number of threads to be used. (int) (default=4)", cxxopts::value<size_t>())
    ("d,ndimensions", "Number of dimensions for embedding. (int) (default=300)", cxxopts::value<size_t>())
    ("i,nepochs", "Number of epochs to run PCA for. (int) (default=50)", cxxopts::value<size_t>())
    ("g,gpu", "Use gpu? (bool) (default=false)", cxxopts::value<bool>())
    ("u,debug", "Debug by printing eigenvalues (only for PI) (default=false)", cxxopts::value<bool>())
    ("s,init_seed", "Random seed (int) (default=1234)", cxxopts::value<size_t>())
    ("w,sample_seed", "Sample seed (int) (default=1234)", cxxopts::value<size_t>())
    ("z,save", "Every specified number of epochs save embeddings to file (int) (default=0)", cxxopts::value<size_t>())
    ("p,log", "Every specified number of epochs print log information (int) (default= i / 10)", cxxopts::value<size_t>())
    ("e,tol", "Tolerance input (default=1e-4).", cxxopts::value<double>())
    ("l,lr", "Learning rate (default=1e-3).", cxxopts::value<double>())
    ("a,beta", "Momentum for sgd (default=0).", cxxopts::value<double>())
    ("m,lambda", "Regularization for sgd (default=0).", cxxopts::value<double>())
    ("y,lr_decay", "Learning rate decay for sgd (default=never).", cxxopts::value<double>())
    ("q,freq", "Calculate full gradient every freq epoch (only for SVRG) (default=never).", cxxopts::value<size_t>())
    ("b,batch_size", "Batch size for SGD/SVRG (default=128).", cxxopts::value<size_t>())
    ("r,pre_trained", "Pre-trained model file name (default="")", cxxopts::value<std::string>())
    ("n,new_corpus", "Indicates different vocab is used for pre-trained model. (bool) (default=false)", cxxopts::value<bool>())
    ("h,freeze_vocab", "Freeze vocab file name (default="")", cxxopts::value<std::string>())
    ("c,freeze_n_epochs", "Number of epochs to freeze the freezen vocab (int) (default=0)", cxxopts::value<size_t>());

  auto result = options.parse(argc, argv);

  Args args;
  if (result.count("file") == 0) {
    std::cerr << "Missing cooccurrence file input." << std::endl;
    exit_parser(options);
  }
  args.cooccurrence_filename = result["file"].as<std::string>();

  if (result.count("vocab") == 0) {
    std::cerr << "Missing vocab input." << std::endl;
    exit_parser(options);
  }
  args.vocab_filename = result["vocab"].as<std::string>();

  if (result.count("out") == 0) {
    std::cerr << "Missing output filename." << std::endl;
    exit_parser(options);
  }
  args.out_filename = result["out"].as<std::string>();

  if (result.count("nthreads") == 0) {
    args.n_threads = 4;
  } else {
    args.n_threads = result["nthreads"].as<size_t>();
  }

  if (result.count("ndimensions") == 0) {
    args.n_dimensions = 300;
  } else {
    args.n_dimensions = result["ndimensions"].as<size_t>();
  }

  if (result.count("nepochs") == 0) {
    args.n_epochs = 50;
  } else {
    args.n_epochs = result["nepochs"].as<size_t>();
  }

  if (result.count("gpu") == 0) {
    args.gpu = 0;
  } else {
    args.gpu = result["gpu"].as<bool>();
  }

  if (result.count("debug") == 0) {
    args.debug = 0;
  } else {
    args.debug = result["debug"].as<bool>();
  }

  if (result.count("init_seed") == 0) {
    args.init_seed = 1234;
  } else {
    args.init_seed = result["init_seed"].as<size_t>();
  }
  
  if (result.count("sample_seed") == 0) {
    args.sample_seed = 1234;
  } else {
    args.sample_seed = result["sample_seed"].as<size_t>();
  }

  if (result.count("save") == 0) {
    args.save_epochs = std::numeric_limits<size_t>::max();
  } else {
    args.save_epochs = result["save"].as<size_t>();
  }

  if (result.count("log") == 0) {
    args.log_epochs = args.n_epochs / 10;
  } else {
    args.log_epochs = result["log"].as<size_t>();
  }

  if (result.count("tol") == 0) {
    args.tol = 1e-4;
  } else {
    args.tol = result["tol"].as<double>();
  }

  if (result.count("lr") == 0) {
    args.lr = 1e-3;
  } else {
    args.lr = result["lr"].as<double>();
  }

  if (result.count("beta") == 0) {
    args.beta = 0.0;
  } else {
    args.beta = result["beta"].as<double>();
  }

  if (result.count("lambda") == 0) {
    args.lambda = 0.0;
  } else {
    args.lambda = result["lambda"].as<double>();
  }

  if (result.count("lr_decay") == 0) {
    args.lr_decay = args.n_epochs;
  } else {
    args.lr_decay = result["lr_decay"].as<double>();
  }

  if (result.count("freq") == 0) {
    args.freq = std::numeric_limits<size_t>::max();
  } else {
    args.freq = result["freq"].as<size_t>();
  }

  if (result.count("batch_size") == 0) {
    args.batch_size = 128;
  } else {
    args.batch_size = result["batch_size"].as<size_t>();
  }

  if (result.count("pre_trained") == 0) {
    args.pre_trained = "";
  } else {
    args.pre_trained = result["pre_trained"].as<std::string>();
  }

  if (result.count("new_corpus") == 0) {
    args.new_corpus = 0;
  } else {
    args.new_corpus = result["new_corpus"].as<bool>();
  }

  if (result.count("freeze_vocab") == 0) {
    args.freeze_vocab_filename = "";
  } else {
    args.freeze_vocab_filename = result["freeze_vocab"].as<std::string>();
  }

  if (result.count("freeze_n_epochs") == 0) {
    args.freeze_n_epochs = 0;
  } else {
    args.freeze_n_epochs = result["freeze_n_epochs"].as<size_t>();
  }

  if (result.count("solver") == 0) {
    std::cerr << "Missing solver input." << std::endl;
    exit_parser(options);
  }
  args.solver = result["solver"].as<std::string>();

  return args;
}
}

int main(int argc, char *argv[]) {
  parser::Args args = parser::parse(argc, argv);

  //  Displays the selected options to the user.
  args.log();

  size_t n_dimensions = args.n_dimensions;
  size_t n_epochs = args.n_epochs;
  size_t save_epochs = args.save_epochs;
  size_t log_epochs = args.log_epochs;
  bool gpu = args.gpu;
  bool debug = args.debug;
  double tol = args.tol;

#ifndef __APPLE__
  omp_set_num_threads(args.n_threads);  // set num threads
#endif
  const auto loading = timer::start_clock();

  std::string cooccurrence_file = args.cooccurrence_filename;
  const std::string vocab_file = args.vocab_filename;
  const std::string output_file = args.out_filename;

  COO<double> coo_cooccurrence = COO<double>::from_file(cooccurrence_file);
  const std::vector<std::string> vocab = utils::load_vocab(vocab_file);
  timer::stop_clock("LOADING", loading);

  std::set<size_t> freeze_idx_set;
  if (args.freeze_vocab_filename != "") {
    const std::vector<std::string> freeze_vocab = utils::load_vocab(args.freeze_vocab_filename);
    std::set<std::string> freeze_vocab_set;
    for (size_t i = 0; i < freeze_vocab.size(); ++i)
      freeze_vocab_set.insert(freeze_vocab.at(i));

    for (size_t i = 0; i < vocab.size(); ++i) {
      std::set<std::string>::iterator it = freeze_vocab_set.find(vocab.at(i));
      if(it != freeze_vocab_set.end())
        freeze_idx_set.insert(i);
    }
  }

  const auto csr_build = timer::start_clock();
  CSR<double> csr_cooccurrence = CSR<double>::from_coo(
      coo_cooccurrence.n(),
      coo_cooccurrence.nnz(),
      coo_cooccurrence.rowind(),
      coo_cooccurrence.colind(),
      coo_cooccurrence.val(), /*sort_rows=*/false);
  /*debug*/  // csr_cooccurrence.print();
  timer::stop_clock("CSR BUILD", csr_build);

  const auto preprocessing = timer::start_clock();
  csr_cooccurrence.ppmi();
  timer::stop_clock("PREPROCESSING TIME", preprocessing);
  // csr_cooccurrence.print();
  LOG("NNZ: " + std::to_string(csr_cooccurrence.nnz()));

  // update batch_size;
  size_t batch_size = std::min(args.batch_size, csr_cooccurrence.nnz());

  const auto solver = timer::start_clock();

  DenseMatrix<double> embedding;

  DenseMatrix<double> pre_trained_model = utils::load_pre_trained_model(
      coo_cooccurrence.n(), n_dimensions, args.pre_trained, args.new_corpus, vocab,
      args.init_seed);

  size_t number_of_epochs_ran = 0;
  if (args.solver == "pi") {
    LOG("Running power method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    embedding = solver::cpu_simultaneous_power_iteration(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        args.n_threads,
        std::move(pre_trained_model),
        number_of_epochs_ran,
        debug);
  }

  if (args.solver == "dpi") {
    LOG("Running deflation power method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    // new_coo.print();
    embedding = solver::cpu_deflation_power_iteration(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        args.n_threads,
        std::move(pre_trained_model),
        number_of_epochs_ran);
  }

  if (args.solver == "gd") {
    LOG("Running gd method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    new_coo.shuffle_inplace(args.sample_seed);
    embedding = solver::cpu_gd(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        args.lr,
        args.n_threads,
        std::move(pre_trained_model),
        number_of_epochs_ran);
  }

  if (args.solver == "sgd") {
    LOG("Running sgd method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    new_coo.shuffle_inplace(args.sample_seed);
    embedding = solver::cpu_sgd(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        batch_size,
        args.lr,
        args.beta,
        args.lambda,
        args.lr_decay,
        args.n_threads,
        freeze_idx_set,
        args.freeze_n_epochs,
        std::move(pre_trained_model),
        number_of_epochs_ran);
  }

  if (args.solver == "svrg") {
    LOG("Running svrg method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    new_coo.shuffle_inplace(args.sample_seed);
    embedding = solver::cpu_svrg(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        args.freq,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        batch_size,
        args.lr,
        args.n_threads,
        std::move(pre_trained_model),
        number_of_epochs_ran);
  }

  if (args.solver == "hybird_pi_sgd") {
    LOG("Running pi and sgd hybird method.");
    COO<double> new_coo = COO<double>::from_csr(csr_cooccurrence);
    new_coo.shuffle_inplace(args.sample_seed);
    embedding = solver::cpu_hybird_pi_sgd(
        csr_cooccurrence,
        new_coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        args.init_seed,
        batch_size,
        args.lr,
        args.beta,
        args.lambda,
        args.lr_decay,
        args.n_threads,
        freeze_idx_set,
        args.freeze_n_epochs,
        std::move(pre_trained_model));
  }

  ASSERT(embedding.is_valid(), "Embedding not valid.");

  timer::stop_clock("SOLVER TIME", solver);
}
