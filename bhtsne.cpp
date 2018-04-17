/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include "vptree.h"
#include "sptree.h"
#include "bhtsne.h"
#include <iostream>
#include <map>
#include <vector>

#include <string>
#include <armadillo>

using namespace std;
using namespace arma;

// Fisher-Yates shuffle
static void randperm(ivec &selected, int N, int K) {
  ivec tmp(N);
  for (int i = 0; i < N; i++) {
    tmp(i) = i;
  }

  for (int i = 0; i <= min(N - 2, K); i++) {
    ivec r = randi(1, distr_param(i + 1, N - 1));

    // Swap
    int val = tmp(i);
    tmp(i) = tmp(r(0));
    tmp(r(0)) = val;
  }

  selected.set_size(K);
  for (int i = 0; i < K; i++) {
    selected(i) = tmp(i % N);
  }
}

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


// Perform t-SNE
bool TSNE::run(int N, unsigned int *row_P, unsigned int *col_P, double *val_P, 
               double* Y, int no_dims, double theta, int rand_seed,
               bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter,
               double momentum, double final_momentum, double eta, boost::filesystem::path outdir) {

  // Set random seed
  if (!skip_random_init) {
    if (rand_seed >= 0) {
      printf("Using random seed: %d\n", rand_seed);
      srand((unsigned int)rand_seed);
    } else {
      printf("Using current time as random seed...\n");
      srand(time(NULL));
    }
  }

  printf("Using no_dims = %d and theta = %f\n", no_dims, theta);

  float total_time = .0;
  clock_t start, end;
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();
  boost::filesystem::path newfile;

  // Allocate some memory
  double *dY    = (double *)malloc(N * no_dims * sizeof(double));
  double *uY    = (double *)malloc(N * no_dims * sizeof(double));
  double *gains = (double *)malloc(N * no_dims * sizeof(double));
  if (dY == NULL || uY == NULL || gains == NULL) {
    printf("Memory allocation failed!\n");
    return false;
  }
  for (int i = 0; i < N * no_dims; i++) {
    uY[i] = 0;
    gains[i] = 1;
  }

  if (!skip_random_init) {
    for (int i = 0; i < N * no_dims; i++) {
      Y[i] = randn() * 0.0001;
    }
  }

	// Perform main training loop
  printf("Learning embedding...\n");
  start = clock();
  t_start = chrono::high_resolution_clock::now();

  newfile = outdir;
  newfile /= "log.txt";
  ofstream logfs(newfile.c_str());
  logfs << "iter\tobjective\telapsed-time" << endl;

  double C0 = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta); // doing approximate computation here!
  iters.push_back(0);
  objectives.push_back(C0);
  elapsed_times.push_back(0);
  printf("Iteration 0: error is %f\n", C0);

  logfs << iters[0] << "\t" << objectives[0] << "\t" << elapsed_times[0] << endl;
  logfs.flush();

  int batch_N;

  if (BATCH_FLAG) {
    batch_N = (int)(N * BATCH_FRAC);
    cout << "batch N: " << batch_N << endl;
  }

	for(int iter = 0; iter < max_iter; iter++) {

    // Compute (approximate) gradient
    computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, iter < stop_lying_iter);

    if (BATCH_FLAG) {

      ivec selected;
      randperm(selected, N, batch_N);

      // Perform gradient update (with momentum)
      for (int i = 0; i < batch_N; i++) {
        int ind = selected(i) * no_dims;
        for (int d = 0; d < no_dims; d++) {
          uY[ind + d] = momentum * uY[ind + d] - eta * dY[ind + d];
           Y[ind + d] += uY[ind + d];
        }
      }
      
    } else {

      // Update gains
      for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
      for (int i = 0; i < N * no_dims; i++) if (gains[i] < .01) gains[i] = .01;

      // Perform gradient update (with momentum and gains)
      for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
		  for (int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];

    }

    // Make solution zero-mean
		zeroMean(Y, N, no_dims);

    if (iter == mom_switch_iter) {
      momentum = final_momentum;
    }

    // Print out progress
    if (iter > 0 && ((iter + 1) % 50 == 0 || iter == max_iter - 1)) {
      end = clock();
      t_end = chrono::high_resolution_clock::now();

      double C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);
      float elapsed = chrono::duration<float, std::milli>(t_end - t_start).count() / 1000.0;

      total_time += elapsed;
      printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n",
             iter + 1, C, (float)(end - start) / CLOCKS_PER_SEC);

      iters.push_back(iter + 1);
      objectives.push_back(C);
      elapsed_times.push_back((double)total_time);

      logfs << iters.back() << "\t" << objectives.back() << "\t" << elapsed_times.back() << endl;
      logfs.flush();

      if ((iter + 1) % CACHE_ITER == 0 || iter == max_iter - 1) {
        string filename;
        if (iter == max_iter - 1) {
          filename = "Y_final.txt";
        } else {
          filename = string("Y_") + to_string(iter + 1) + string(".txt");
        }
        newfile = outdir;
        newfile /= filename;

        //FILE *fp = fopen(newfile.string().c_str(), "wb");
        //fwrite(Y, sizeof(double), no_dims * N, fp);
        //fclose(fp);

        mat Y_tmp(Y, no_dims, N, true); // convert Y to armadillo matrix
        Y_tmp = Y_tmp.t();
        Y_tmp.save(newfile.string(), arma_ascii);
      }

		  start = clock();
      t_start = chrono::high_resolution_clock::now();
    }
  }

  logfs.close();

  free(dY);
  free(uY);
  free(gains);

  printf("Fitting performed in %4.2f seconds.\n", total_time);

  return true;
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, bool early_exaggeration)
{

    // Construct space-partitioning tree on current map
    SPTree* tree = new SPTree(D, Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = (double*) calloc(N * D, sizeof(double));
    double* neg_f = (double*) calloc(N * D, sizeof(double));
    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

    // Compute final t-SNE gradient
    for(int i = 0; i < N * D; i++) {
        if (early_exaggeration) {
          dC[i] = 12.0 * pos_f[i] - (neg_f[i] / sum_Q);
        } else {
          dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
        }
    }
    free(pos_f);
    free(neg_f);
    delete tree;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

    // Get estimate of normalization term
    SPTree* tree = new SPTree(D, Y, N);
    double* buff = (double*) calloc(D, sizeof(double));
    double sum_Q = .0;
    for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for(int n = 0; n < N; n++) {
        ind1 = n * D;
        for(int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * D;
            for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
            for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
            for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}

// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

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


// Generates a Gaussian random number
double TSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}

bool TSNE::load_P(string infile, int &N, unsigned int** row_P, unsigned int** col_P, double** val_P) {
  FILE *h;
	if((h = fopen(infile.c_str(), "rb")) == NULL) {
    return false;
	}

  size_t ret = 0;
  ret += fread(&N, sizeof(int), 1, h);
  *row_P = (unsigned int*)malloc((N+1) * sizeof(unsigned int));
  if (*row_P == NULL) {
    printf("Memory allocation error\n");
    exit(1);
  }
  ret += fread(*row_P, sizeof(unsigned int), N+1, h);
  *col_P = (unsigned int*)malloc((*row_P)[N] * sizeof(unsigned int));
  *val_P = (double*)malloc((*row_P)[N] * sizeof(double));
  if (*col_P == NULL || *val_P == NULL) {
    printf("Memory allocation error\n");
    exit(1);
  }
  ret += fread(*col_P, sizeof(unsigned int), (*row_P)[N], h);
  ret += fread(*val_P, sizeof(double), (*row_P)[N], h);
  fclose(h);

  printf("P successfully loaded\n");
  return true;
}
