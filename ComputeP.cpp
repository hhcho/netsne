#include <iostream>
#include <boost/program_options.hpp>
#include "vptree.h"

using namespace std;
namespace po = boost::program_options;

/* Functions taken from Laurens van der Maaten's original implementation of t-SNE */
/* Source: https://github.com/lvdmaaten/bhtsne                                    */
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K);
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N);
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
static void zeroMean(double* X, int N, int D);
/**********************************************************************************/

static bool load_data(string infile, double **data, int &num_instances, int &num_features) {

  FILE *fp = fopen(infile.c_str(), "rb");
	if (fp == NULL) {
		cout << "Error: could not open data file " << infile << endl;
		return false;
	}

  uint64_t ret;
	ret = fread(&num_instances, sizeof(int), 1, fp);
	ret = fread(&num_features, sizeof(int), 1, fp);

	*data = (double *)malloc(num_instances * num_features * sizeof(double));
  if (*data == NULL) {
    cout << "Error: memory allocation of " << num_instances << " by " 
         << num_features << " matrix failed" << endl;
    return false;
  }

  uint64_t nelem = (uint64_t)num_instances * num_features;

  size_t batch_size = 1e8;
  double *ptr = *data;
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

static void truncate_data(double *X, int num_instances, int num_features, int target_dims) {
  size_t i_old = 0;
  size_t i_new = 0;
  for (int r = 0; r < num_instances; r++) {
    for (int c = 0; c < num_features; c++) {
      if (c < target_dims) {
        X[i_new++] = X[i_old];
      }
      i_old++;
    }
  }
}

static bool run(double *X, int num_instances, int num_features, double perplexity, string outfile) {

  // Apply lower bound on perplexity from original t-SNE implementation
  if (num_instances - 1 < 3 * perplexity) {
    cout << "Error: target perplexity (" << perplexity << ") is too large "
         << "for the number of data points (" << num_instances << ")" << endl;
    return false;
  }

  printf("Processing %d data points, %d features with target perplexity %f\n",
         num_instances, num_features, perplexity);

  // Normalize input data (to prevent numerical problems)
  zeroMean(X, num_instances, num_features);
  cout << "Normalizing the features" << endl;
  double max_X = 0;
  for (size_t i = 0; i < num_instances * num_features; i++) {
    if (fabs(X[i]) > max_X) {
      max_X = fabs(X[i]);
    }
  }

  for (size_t i = 0; i < num_instances * num_features; i++) {
    X[i] /= max_X;
  }

  // Compute input similarities for exact t-SNE
  double* P; unsigned int* row_P; unsigned int* col_P; double* val_P;

  // Compute asymmetric pairwise input similarities
  cout << "Computing conditional distributions" << endl;
  computeGaussianPerplexity(X, num_instances, num_features,
    &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity));

  // Symmetrize input similarities
  cout << "Symmetrizing matrix" << endl;
  symmetrizeMatrix(&row_P, &col_P, &val_P, num_instances);
  double sum_P = .0;
  for (int i = 0; i < row_P[num_instances]; i++) {
    sum_P += val_P[i];
  }
  for (int i = 0; i < row_P[num_instances]; i++) {
    val_P[i] /= sum_P;
  }

  cout << "Saving to " << outfile << endl;
	FILE *fp = fopen(outfile.c_str(), "wb");
	if (fp == NULL) {
		cout << "Error: could not open output file " << outfile << endl;
    return false;
	}

	fwrite(&num_instances, sizeof(int), 1, fp);
  fwrite(row_P, sizeof(unsigned int), num_instances + 1, fp);
	fwrite(col_P, sizeof(unsigned int), row_P[num_instances], fp);
  fwrite(val_P, sizeof(double), row_P[num_instances], fp);

  free(row_P);
  free(col_P);
  free(val_P);

  return true;
}

int main(int argc, char **argv) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("input-file", po::value<string>()->value_name("FILE")->default_value("data.dat"), "name of binary input file (see prepare_input.m)")
    ("output-file", po::value<string>()->value_name("FILE")->default_value("P.dat"), "name of output file to be created")
    ("perp", po::value<double>()->value_name("NUM")->default_value(30, "30"), "set target perplexity for conditional distributions")
    ("num-dims", po::value<int>()->value_name("NUM"), "if provided, only the first NUM features in the input will be used")
  ;

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).
                options(desc).run(), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    cout << "Usage: ComputeP [options]" << endl;
    cout << desc << "\n";
    return 1;
  }

	double perplexity = vm["perp"].as<double>();
  string infile = vm["input-file"].as<string>();
  string outfile = vm["output-file"].as<string>();

  double *data;
  int num_instances;
  int num_features;

	if (!load_data(infile, &data, num_instances, num_features)) {
    return 1;
  }

  cout << infile << " successfully loaded" << endl;

  if (vm.count("num-dims")) {
    int num_dims = vm["num-dims"].as<int>();
    cout << "Using only the first " << num_dims << " dimensions" << endl;
    truncate_data(data, num_instances, num_features, num_dims);
    num_features = num_dims;
  }
  
  if (!run(data, num_instances, num_features, perplexity, outfile)) {
    return 1;
  }

  cout << "Done" << endl;

  return 0;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
// Source: https://github.com/lvdmaaten/bhtsne
static void computeGaussianPerplexity(double* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double perplexity, int K) {

    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;
    double* cur_P = (double*) malloc((N - 1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    printf("Building tree...\n");
    vector<DataPoint> indices;
    vector<double> distances;
    for(int n = 0; n < N; n++) {

        if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);

        // Find nearest neighbors
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while(!found && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            double H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    delete tree;
}

// Symmetrizes a sparse matrix
// Source: https://github.com/lvdmaaten/bhtsne
static void symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    unsigned int* row_P = *_row_P;
    unsigned int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) present = true;
            }
            if(present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for(int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
    unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if(col_P[m] == n) {
                    present = true;
                    if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if(!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if(!present || (present && n <= col_P[i])) {
                offset[n]++;
                if(col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}

// Compute squared Euclidean distance matrix
// Source: https://github.com/lvdmaaten/bhtsne
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
    const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

// Makes data zero-mean
// Source: https://github.com/lvdmaaten/bhtsne
static void zeroMean(double* X, int N, int D) {

	// Compute data mean
	double* mean = (double*) calloc(D, sizeof(double));
    if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}
    free(mean); mean = NULL;
}
