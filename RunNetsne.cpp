#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "netsne.h"
#include <armadillo>

using namespace std;
namespace po = boost::program_options;
namespace fsys = boost::filesystem;

bool load_data(string infile_X, mat &X, int &num_instances, int &num_features) {

  FILE *fp = fopen(infile_X.c_str(), "rb");
	if (fp == NULL) {
		cout << "Error: could not open data file " << infile_X << endl;
		return false;
	}

  uint64_t ret;
	ret = fread(&num_instances, sizeof(int), 1, fp);
	ret = fread(&num_features, sizeof(int), 1, fp);

  X.set_size(num_features, num_instances);

  uint64_t nelem = (uint64_t)num_instances * num_features;

  size_t batch_size = 1e8;
  double *ptr = X.memptr();
  ret = 0;
  for (uint64_t remaining = nelem; remaining > 0; remaining -= batch_size) {
    if (remaining < batch_size) {
      batch_size = remaining;
    }
    ret += fread(ptr, sizeof(double), batch_size, fp);
    ptr += batch_size;
  }
  
  if (ret != nelem) {
    cout << "Error: reading input returned incorrect number of elements (" << ret
         << ", expected " << nelem << ")" << endl;
    return false;
  }
  
	fclose(fp);

	return true;
}

int main(int argc, char **argv) {
  // Declare the supported options
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input-P", po::value<string>()->value_name("FILE")->default_value("P.dat"), "name of binary input file containing P matrix (see ComputeP)")
    ("input-X", po::value<string>()->value_name("FILE")->default_value("data.dat"), "name of binary input file containing data feature matrix (see prepare_input.m)")
    ("input-Y", po::value<string>()->value_name("FILE"), "if this option is provided, net-SNE will train to match the provided embedding instead of using the P matrix")
    ("out-dir", po::value<string>()->value_name("DIR")->default_value("out"), "where to create output files; directory will be created if it does not exist")
    ("out-dim", po::value<int>()->value_name("NUM")->default_value(2), "number of output dimensions")
    ("max-iter", po::value<int>()->value_name("NUM")->default_value(1000), "maximum number of iterations")
    ("rand-seed", po::value<int>()->value_name("NUM")->default_value(-1), "seed for random number generator; to use current time as seed set it to -1")
    ("theta", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "a value between 0 and 1 that controls the accuracy-efficiency tradeoff in SPTree for gradient computation; 0 means exact")
    ("learn-rate", po::value<double>()->value_name("NUM")->default_value(0.02, "0.02"), "learning rate for gradient steps")
    ("mom-init", po::value<double>()->value_name("NUM")->default_value(0.5, "0.5"), "initial momentum between 0 and 1")
    ("mom-final", po::value<double>()->value_name("NUM")->default_value(0.8, "0.8"), "final momentum between 0 and 1 (switch point controlled by --mom-switch-iter)")
    ("mom-switch-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of initial momentum")
    ("early-exag-iter", po::value<int>()->value_name("NUM")->default_value(250), "duration (number of iterations) of early exaggeration")
    ("num-local-sample", po::value<int>()->value_name("NUM")->default_value(20), "number of local samples for each data point in the mini-batch")
    ("batch-frac", po::value<double>()->value_name("NUM")->default_value(0.05, "0.05"), "fraction of data to sample for mini-batch")
    ("min-sample-Z", po::value<double>()->value_name("NUM")->default_value(0.1, "0.1"), "minimum fraction of data to use for approximating the normalization factor Z in the gradient")
    ("no-batch-norm", po::bool_switch()->default_value(false), "turn off batch normalization")
    ("init-model-prefix", po::value<string>()->value_name("STR"), "prefix of model files for initialization")
    ("monte-carlo-pos", po::bool_switch()->default_value(false), "use monte-carlo integration for positive gradient term")
    ("match-pos-neg", po::bool_switch()->default_value(false), "compute negative forces for points sampled for positive force")
    ("step-method", po::value<string>()->value_name("STR")->default_value("adam"), "gradient step schedule; 'adam', 'mom' (momentum), 'mom_gain' (momentum with gains), 'fixed'")
    ("num-input-feat", po::value<int>()->value_name("NUM"), "if set, use only the first NUM features for the embedding function")
    ("init-map", po::bool_switch()->default_value(false), "output initial mapping for the entire data")
    ("num-layers", po::value<int>()->value_name("NUM")->default_value(2), "number of layers in the neural network")
    ("num-units", po::value<int>()->value_name("NUM")->default_value(50), "number of units for each layer in the neural network")
    ("act-fn", po::value<string>()->value_name("STR")->default_value("relu"), "activation function of the neural network; 'sigmoid' or 'relu'")
    ("test-model", po::bool_switch()->default_value(false), "if set, use the model provided with --init-model-prefix and visualize the entire data set then terminate without training")
    ("l2-reg", po::value<double>()->value_name("NUM")->default_value(0, "0"), "L2 regularization parameter")
    ("perm-iter", po::value<int>()->value_name("NUM")->default_value(INT_MAX, "INT_MAX"), "After every NUM iterations, permute the ordering of data points for fast mini-batching")
    ("cache-iter", po::value<int>()->value_name("NUM")->default_value(INT_MAX, "INT_MAX"), "After every NUM iterations, write intermediary embeddings and parameters to disk. Final embedding is always reported.")
    ("no-sgd", po::bool_switch()->default_value(false), "if set, do not use SGD acceleration; equivalent to t-SNE with an additional backpropagation step to train a neural network. Effective for small datasets")

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

  string infile_P = vm["input-P"].as<string>();
  string infile_X = vm["input-X"].as<string>();
  string outdir = vm["out-dir"].as<string>();

  fsys::path dir(outdir);
  if (fsys::is_directory(dir)) {
    cout << "Error: Output directory (" << outdir << ") already exists" << endl;
    return 1;
  }
  if (fsys::create_directory(dir)) {
    cout << "Output directory created: " << outdir << endl; 
  }

  fsys::path paramfile = dir;
  paramfile /= "param.txt";
  ofstream ofs(paramfile.string().c_str());

  bool use_known_Y = vm.count("input-Y");
  string infile_Y;

  if (use_known_Y) {
    infile_Y = vm["input-Y"].as<string>();
    ofs << "input-Y: " << infile_Y << endl;
    cout << "Learning to match the provided embedding: " << infile_Y << endl;
  } else {
    ofs << "input-P: " << infile_P << endl;
  }
  ofs << "input-X: " << infile_X << endl;
  ofs << "out-dir: " << fsys::canonical(dir).string() << endl;

  NETSNE* netsne = new NETSNE();

  netsne->BATCH_FRAC = vm["batch-frac"].as<double>(); ofs << "batch-frac: " << netsne->BATCH_FRAC << endl;
  netsne->N_SAMPLE_LOCAL = vm["num-local-sample"].as<int>(); ofs << "num-local-sample: " << netsne->N_SAMPLE_LOCAL << endl;
  netsne->MIN_SAMPLE_Z = vm["min-sample-Z"].as<double>(); ofs << "min-sample-Z: " << netsne->MIN_SAMPLE_Z << endl;
  netsne->STOP_LYING = vm["early-exag-iter"].as<int>(); ofs << "early-exag-iter: " << netsne->STOP_LYING << endl;
  netsne->BATCH_NORM = !vm["no-batch-norm"].as<bool>(); ofs << "no-batch-norm: " << !netsne->BATCH_NORM << endl;
  netsne->MONTE_CARLO_POS = vm["monte-carlo-pos"].as<bool>(); ofs << "monte-carlo-pos: " << netsne->MONTE_CARLO_POS << endl;
  netsne->MATCH_POS_NEG = vm["match-pos-neg"].as<bool>(); ofs << "match-pos-neg: " << netsne->MATCH_POS_NEG << endl;
  netsne->STEP_METHOD = vm["step-method"].as<string>(); ofs << "step-method: " << netsne->STEP_METHOD << endl;
  netsne->LEARN_RATE = vm["learn-rate"].as<double>(); ofs << "learn-rate: " <<  netsne->LEARN_RATE << endl;
  netsne->L2_REG = vm["l2-reg"].as<double>(); ofs << "l2-reg: " <<  netsne->L2_REG << endl;
  netsne->SGD_FLAG = !vm["no-sgd"].as<bool>(); ofs << "sgd: " << netsne->SGD_FLAG << endl;

  netsne->MODEL_PREFIX_FLAG = vm.count("init-model-prefix");
  if (netsne->MODEL_PREFIX_FLAG) {
    netsne->MODEL_PREFIX = vm["init-model-prefix"].as<string>();
  }
  ofs << "init-model-prefix: " << netsne->MODEL_PREFIX << endl;

  netsne->TEST_RUN = vm["test-model"].as<bool>();
  if (netsne->TEST_RUN) {
    netsne->COMPUTE_INIT = true;

    if (!netsne->MODEL_PREFIX_FLAG) {
      cout << "Error: if --test-model is set, then --init-model-prefix must be provided; see --help" << endl;
      return 1;
    }
  } else {
    netsne->COMPUTE_INIT = vm["init-map"].as<bool>();
  }
  ofs << "init-map: " << netsne->COMPUTE_INIT << endl;

  if (!netsne->MODEL_PREFIX_FLAG) {
    netsne->NUM_LAYERS = vm["num-layers"].as<int>(); ofs << "num-layers: " << netsne->NUM_LAYERS << endl;
    netsne->NUM_UNITS = vm["num-units"].as<int>(); ofs << "num-units: " << netsne->NUM_UNITS << endl;
    netsne->ACT_FN = vm["act-fn"].as<string>(); ofs << "act-fn: " << netsne->ACT_FN << endl;
  }

  if (netsne->STEP_METHOD == "mom" || netsne->STEP_METHOD == "mom_gain") {
    netsne->MOM_SWITCH_ITER = vm["mom-switch-iter"].as<int>(); ofs << "mom-switch-iter: " << netsne->MOM_SWITCH_ITER << endl;
    netsne->MOM_INIT = vm["mom-init"].as<double>(); ofs << "mom-init: " << netsne->MOM_INIT << endl;
    netsne->MOM_FINAL = vm["mom-final"].as<double>(); ofs << "mom-final: " << netsne->MOM_FINAL << endl;
  }

  int no_dims = vm["out-dim"].as<int>(); ofs << "out-dim: " << no_dims << endl;
  double theta = vm["theta"].as<double>(); ofs << "theta: " << theta << endl;
  int rand_seed = vm["rand-seed"].as<int>(); ofs << "rand-seed: " << rand_seed << endl;
  int max_iter = vm["max-iter"].as<int>(); ofs << "max-iter: " << max_iter << endl;

  ofs.close();

  if (vm.count("help")) {
    cout << "Usage: RunBhtsne [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

  if (netsne->STEP_METHOD != "adam" && netsne->STEP_METHOD != "mom" && netsne->STEP_METHOD != "mom_gain"
      && netsne->STEP_METHOD != "fixed") {
    cout << "Error: Unrecognized --step-method argument " << netsne->STEP_METHOD << "; see --help" << endl;
    return 1;
  } 

  if (netsne->ACT_FN != "sigmoid" && netsne->ACT_FN != "relu") {
    cout << "Error: Unrecognized --act-fn argument " << netsne->ACT_FN << "; see --help" << endl;
    return 1;
  } 

  int num_input_feat;
  if (vm.count("num-input-feat")) {
    num_input_feat = vm["num-input-feat"].as<int>();
  } else {
    num_input_feat = INT_MAX;
  }

  cout << "Loading input features ... ";
  mat X;
  int num_instances;
  int num_features;
	if (!load_data(infile_X, X, num_instances, num_features)) {
    return 1;
  }
  cout << endl;

  if (X.n_rows > num_input_feat) {
    cout << "Truncating to top " << num_input_feat << " features" << endl;
    X = X.head_rows(num_input_feat);
  }

  cout << "Data feature matrix is " << X.n_rows << " by " << X.n_cols << endl;

  int N;
  unsigned int *row_P = NULL;
  unsigned int *col_P = NULL;
  double *val_P = NULL;
  mat target_Y;

  if (use_known_Y) {

    cout << "Loading target Y ... ";
    target_Y.load(infile_Y, arma_ascii);
    target_Y = target_Y.t();
    cout << "done" << endl;

    N = target_Y.n_cols;

    if (N != X.n_cols) {
      cout << "Error: Y matrix dimensions (" << N << ") do not match with X matrix (" << X.n_cols << ")" << endl;
      return 1;
    }

  } else {

    cout << "Loading input similarities ... ";
    if (!netsne->load_P(infile_P, N, &row_P, &col_P, &val_P)) {
      cout << "Error: failed to load P from " << infile_P << endl;
      return 1;
    }
    cout << "done" << endl;

    if (N != X.n_cols) {
      cout << "Error: P matrix dimensions (" << N << ") do not match with X matrix (" << X.n_cols << ")" << endl;
      return 1;
    }

  }

  mat Y(no_dims, N);

  if (!netsne->run(N, row_P, col_P, val_P, target_Y, X, Y, no_dims, theta, rand_seed,
           max_iter, dir)) {
    return 1;
  }

  free(row_P);
  free(col_P);
  free(val_P);

  delete(netsne);

  cout << "Done" << endl;
}
