#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "bhtsne.h"

using namespace std;
namespace po = boost::program_options;
namespace fsys = boost::filesystem;

int main(int argc, char **argv) {
  // Declare the supported options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input-file", po::value<string>()->value_name("FILE")->default_value("P.dat"), "name of binary input file containing P matrix (see ComputeP)")
    ("out-dir", po::value<string>()->value_name("DIR")->default_value("out"), "where to create output files; directory will be created if it does not exist")
    ("out-dim", po::value<int>()->value_name("NUM")->default_value(2), "number of output dimensions")
    ("max-iter", po::value<int>()->value_name("NUM")->default_value(1000), "maximum number of iterations")
    ("rand-seed", po::value<int>()->value_name("NUM")->default_value(-1), "seed for random number generator; to use current time as seed set it to -1")
    ("theta", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "a value between 0 and 1 that controls the accuracy-efficiency tradeoff in SPTree for gradient computation; 0 means exact")
    ("learn-rate", po::value<double>()->value_name("NUM")->default_value(200, "200"), "learning rate for gradient steps")
    ("mom-init", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "initial momentum between 0 and 1")
    ("mom-final", po::value<double>()->value_name("NUM")->default_value(0.8, "0.8"), "final momentum between 0 and 1 (switch point controlled by --mom-switch-iter)")
    ("mom-switch-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of initial momentum")
    ("early-exag-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of early exaggeration")
    ("skip-random-init", po::bool_switch()->default_value(false), "skip random initialization")
    ("init-map", po::value<string>()->value_name("FILE"), "specify starting positions of a subset of data; rest are initialized based on the average of k nearest-neighbors; must be provided with other --init options")
    ("init-indices", po::value<string>()->value_name("FILE"), "list of indices (1-based) to go with the provided initial mappings; must be provided with other --init options")
    ("init-X", po::value<string>()->value_name("FILE"), "feature vectors for finding nearest neighbors; must be provided with other --init options")
    ("init-k", po::value<int>()->value_name("NUM")->default_value(5), "how many neighbors to use for knn-based initialization")
    ("batch-frac", po::value<double>()->value_name("NUM"), "what fraction of points to update for each iteration")
  ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
                options(desc).run(), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    cout << "Usage: RunBhtsne [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

  string infile = vm["input-file"].as<string>();
  string outdir = vm["out-dir"].as<string>();

  fsys::path dir(outdir);
  if (fsys::is_directory(dir)) {
    cout << "Error: Output directory already exists" << endl;
    return 1;
  }
  if (fsys::create_directory(dir)) {
    cout << "Output directory created: " << outdir << endl; 
  }

  TSNE *tsne = new TSNE();

  fsys::path paramfile = dir;
  paramfile /= "param.txt";
  ofstream ofs(paramfile.string().c_str());
  ofs << "input-file: " << infile << endl;
  ofs << "out-dir: " << fsys::canonical(dir).string() << endl;

  int no_dims = vm["out-dim"].as<int>(); ofs << "out-dim: " << no_dims << endl;
  double theta = vm["theta"].as<double>(); ofs << "theta: " << theta << endl;
  int rand_seed = vm["rand-seed"].as<int>(); ofs << "rand-seed: " << rand_seed << endl;
  bool skip_random_init = vm["skip-random-init"].as<bool>(); ofs << "skip-random-init: " << skip_random_init << endl;
  int max_iter = vm["max-iter"].as<int>(); ofs << "max-iter: " << max_iter << endl;
  int stop_lying_iter = vm["early-exag-iter"].as<int>(); ofs << "early-exag-iter: " << stop_lying_iter << endl;
  int mom_switch_iter = vm["mom-switch-iter"].as<int>(); ofs << "mom-switch-iter: " << mom_switch_iter << endl;
  double momentum = vm["mom-init"].as<double>(); ofs << "mom-init: " << momentum << endl;
  double final_momentum = vm["mom-final"].as<double>(); ofs << "mom-final: " << final_momentum << endl;
  double eta = vm["learn-rate"].as<double>(); ofs << "learn-rate: " << eta << endl;

  if (vm.count("init-map") || vm.count("init-indices") || vm.count("init-X")) {
    if (!(vm.count("init-map") && vm.count("init-indices") && vm.count("init-X"))) {
      cout << "Error: --init-map, --init-indices, and --init-X must be provided together" << endl;
      return 1;
    }

    string init_map = vm["init-map"].as<string>(); ofs << "init-map: " << init_map << endl;
    string init_indices = vm["init-indices"].as<string>(); ofs << "init-indices: " << init_indices << endl;
    string init_X = vm["init-X"].as<string>(); ofs << "init-X: " << init_X << endl;
    int init_k = vm["init-k"].as<int>(); ofs << "init-k: " << init_k << endl;

    tsne->INIT_MAP_FLAG = true;
    tsne->INIT_MAP = init_map;
    tsne->INIT_INDICES = init_indices;
    tsne->INIT_X = init_X;
    tsne->INIT_K = init_k;

  }

  if (vm.count("batch-frac")) {
    double batch_frac = vm["batch-frac"].as<double>(); ofs << "batch-frac: " << batch_frac << endl;
    tsne->BATCH_FLAG = true;
    tsne->BATCH_FRAC = batch_frac;
  }

  ofs.close();

  printf("Loading input similarities...\n");
  int N;
  unsigned int *row_P;
  unsigned int *col_P;
  double *val_P;
  if (!tsne->load_P(infile, N, &row_P, &col_P, &val_P)) {
    cout << "Error: failed to load P from " << infile << endl;
    return 1;
  }

  double *Y = (double *)malloc(N * no_dims * sizeof(double));
  if (Y == NULL) {
    cout << "Error: Memory allocation for the output failed" << endl;
    return 1;
  }

  if (!tsne->run(N, row_P, col_P, val_P, Y, no_dims, theta, rand_seed,
           skip_random_init, max_iter, stop_lying_iter, mom_switch_iter,
           momentum, final_momentum, eta, dir)) {
    return 1;
  }

  free(row_P);
  free(col_P);
  free(val_P);
  free(Y);
  delete(tsne);

  cout << "Done" << endl;

  return 0;
}
