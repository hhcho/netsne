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
#include "netsne.h"

auto t_prev = chrono::high_resolution_clock::now();
void tic() {
  t_prev = chrono::high_resolution_clock::now();
}
float toc() {
  auto t_cur = chrono::high_resolution_clock::now();
  return chrono::duration<float, std::milli>(t_cur - t_prev).count();
}

// arr[left] >= val > arr[right]
int binsearch(double *arr, double val, int left, int right) {
  if (right <= left + 1) {
    return left;
  }

  int mid = (left + right) / 2;
  if (arr[mid] >= val) {
    return binsearch(arr, val, mid, right);
  } else {
    return binsearch(arr, val, left, mid);
  }
}
//
// Fisher-Yates shuffle
void randperm(ivec &selected, int N, int K) {
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

// Fisher-Yates shuffle for large matrix
void randperm(ivec &perm_indices, mat &base) {
  int n = base.n_cols;
  perm_indices.set_size(n);
  for (int i = 0; i < n; i++) {
    perm_indices(i) = i;
  }

  ivec r(1);
  for (int i = 0; i <= n - 2; i++) {
    r = randi(1, distr_param(i + 1, n - 1));

    // Swap
    base.swap_cols(i, r(0));
    int val = perm_indices(i);
    perm_indices(i) = perm_indices(r(0));
    perm_indices(r(0)) = val;
  }
}

void sort_P(int N, unsigned int *row_P, unsigned int *col_P, double *val_P) {
  for (int i = 0; i < N; i++) {
    int L = row_P[i+1] - row_P[i];
    vec tmp_col(L);
    vec tmp_val(L);
    for (int j = 0; j < L; j++) {
      tmp_col(j) = col_P[row_P[i]+j];
      tmp_val(j) = val_P[row_P[i]+j];
    }
    uvec I = sort_index(tmp_val, "descend");
    for (int j = 0; j < L; j++) {
      col_P[row_P[i]+j] = tmp_col(I(j));
      val_P[row_P[i]+j] = tmp_val(I(j));
    }
  }
}

void permute_P(ivec &perm_indices, unsigned int *row_P, unsigned int *col_P, double *val_P) {
  int n = perm_indices.n_elem;
  unsigned int *rP = (unsigned int *)malloc((n+1) * sizeof(unsigned int));
  unsigned int *cP = (unsigned int *)malloc(row_P[n] * sizeof(unsigned int));
  double *vP = (double *)malloc(row_P[n] * sizeof(double));
  if (rP == NULL || cP == NULL || vP == NULL) {
    printf("Memory allocaiton error in permute_P\n");
    exit(1);
  }

  map<int,int> full2perm;
  for (int i = 0; i < n; i++) {
    full2perm[perm_indices(i)] = i;
  }

  rP[0] = 0;
  for (int i = 0; i < n; i++) {
    int ind = perm_indices(i);
    unsigned int nelem = row_P[ind+1] - row_P[ind];
    rP[i+1] = rP[i] + nelem;
    for (unsigned int j = 0; j < nelem; j++) {
      cP[rP[i]+j] = full2perm[(int)col_P[row_P[ind]+j]];
      vP[rP[i]+j] = val_P[row_P[ind]+j];
    }
  }

  memcpy(row_P, rP, (n+1) * sizeof(unsigned int));
  memcpy(col_P, cP, row_P[n] * sizeof(unsigned int));
  memcpy(val_P, vP, row_P[n] * sizeof(double));

  free(rP);
  free(cP);
  free(vP);
}

void reorder_columns(mat &out, mat &in, ivec &ord) {
  out.set_size(in.n_rows, in.n_cols);
  for (int i = 0; i < out.n_cols; i++) {
    out.col(ord[i]) = in.col(i);
  }
}

bool NETSNE::run(int N, unsigned int *row_P, unsigned int *col_P, double *val_P, mat &target_Y, 
               mat &X, mat &Y, int no_dims, double theta, int rand_seed,
               int max_iter, boost::filesystem::path outdir) {

  bool use_target_Y = val_P == NULL;
  if (use_target_Y) {
    N_SAMPLE_LOCAL = 0;
    MIN_SAMPLE_Z = 0;
  }

  int stop_lying_iter = STOP_LYING;
  int mom_switch_iter = MOM_SWITCH_ITER;
  double momentum = MOM_INIT;
  double final_momentum = MOM_FINAL;
  double eta = LEARN_RATE;
  int nunits = NUM_UNITS;
  int nlayers = NUM_LAYERS;
  int batch_norm = BATCH_NORM;
  bool use_relu = ACT_FN == "relu";

  int D = X.n_rows;

  printf("Feature matrix: %llu x %llu\n", X.n_rows, X.n_cols);
  printf("eta: %f, max_iter: %d, stop_lying: %d\n", LEARN_RATE, max_iter, stop_lying_iter);
  printf("batch_frac: %f, n_sample_local: %d\n", BATCH_FRAC, N_SAMPLE_LOCAL);

  if(rand_seed >= 0) {
    printf("Using random seed: %d\n", rand_seed);
    srand((unsigned int) rand_seed);
  } else {
    printf("Using current time as random seed...\n");
    srand(time(NULL));
  }

  printf("Using no_dims = %d  and theta = %f\n", no_dims, theta);

  // Set learning parameters
  float total_time = .0;
  clock_t start, end;
  auto t_start = chrono::high_resolution_clock::now();
  auto t_end = chrono::high_resolution_clock::now();
  boost::filesystem::path newfile;

  /* NETSNE */
  double adam_b1 = 0.9;
  double adam_b2 = 0.999;
  double adam_eps = 1e-8;
  double BN_eps = 1e-4;
  /**********/

  if (MODEL_PREFIX_FLAG) {
    printf("Initializing with an existing model: %s\n", MODEL_PREFIX.c_str());

    ifstream ifs(MODEL_PREFIX + "_param.txt");
    if (!ifs.is_open()) {
      cout << "Error: failed to open model parameter file " << MODEL_PREFIX + "_param.txt" << endl;
      return false;
    }
    ifs >> nlayers >> nunits >> use_relu >> batch_norm;
    ifs.close();

  } else {
    cout << "Creating a new model" << endl;
  }

  printf("nlayers: %d, nunits: %d, use_relu: %d, batch_norm: %d\n",
         nlayers, nunits, use_relu, batch_norm);

  // Allocate some memory
  vector<mat> W(nlayers + 1); // Network weights
  vector<mat> dW(nlayers + 1); // Gradients
  
  // Batch normalization parameters/gradients
  vector<vec> beta_BN(nlayers); 
  vector<vec> gamma_BN(nlayers);
  vector<vec> dbeta_BN(nlayers);
  vector<vec> dgamma_BN(nlayers);
  vector<vec> mean_BN(nlayers); // for BN transformation in the testing phase
  vector<vec> std_BN(nlayers); // for BN transformation in the testing phase
  
  vector<mat> dW_m(nlayers + 1); // Adam first moments
  vector<mat> dW_v(nlayers + 1); // Adam second moments
  vector<vec> dbeta_BN_m(nlayers);
  vector<vec> dbeta_BN_v(nlayers);
  vector<vec> dgamma_BN_m(nlayers);
  vector<vec> dgamma_BN_v(nlayers);
  
  vector<mat> W_update(nlayers + 1); // Update history
  vector<mat> W_gains(nlayers + 1); // Gains
  vector<vec> beta_BN_update(nlayers);
  vector<vec> beta_BN_gains(nlayers);
  vector<vec> gamma_BN_update(nlayers);
  vector<vec> gamma_BN_gains(nlayers);
  
  vector<mat> A(nlayers); // activations. for BN, final y values
  vector<mat> A_xhat(nlayers); // only for BN. normalized x_hat values
  vector<vec> A_mu(nlayers);
  vector<vec> A_sig2(nlayers);

  vec X_mean;
  vec X_std;

  if (MODEL_PREFIX_FLAG) {

    string filename = MODEL_PREFIX + "_X_mean.txt";
    bool success = X_mean.load(filename, arma_ascii);
    if (!success || X_mean.n_elem != D) {
      printf("Error loading %s: file not found or dimension mismatch\n",
             filename.c_str());
      cout << "Got " << X_mean.n_elem << ", expected " << D << endl;
      return false;
    }

    filename = MODEL_PREFIX + "_X_std.txt";
    success = X_std.load(filename, arma_ascii);
    if (!success || X_std.n_elem != D) {
      printf("Error loading %s: file not found or dimension mismatch\n",
             filename.c_str());
      cout << "Got " << X_std.n_elem << ", expected " << D << endl;
      return false;
    }
    
    for (int i = 0; i < nlayers + 1; i++) {
      int indim = (i == 0) ? D : nunits;
      int outdim = (i == nlayers) ? no_dims : nunits;
      if (i < nlayers) indim++; // add intercept
    
      filename = MODEL_PREFIX + "_L" + to_string(i) + "_W.txt";
      success = W[i].load(filename, arma_ascii);
      if (!success || W[i].n_rows != outdim || W[i].n_cols != indim) {
        printf("Error loading %s: file not found or dimension mismatch\n",
               filename.c_str());
        cout << "Got " << W[i].n_rows << "x" << W[i].n_cols << ", expected "
             << outdim << "x" << indim << endl;
        return false;
      }

      if (batch_norm && i < nlayers) {
        filename = MODEL_PREFIX + "_L" + to_string(i) + "_BN_beta.txt";
        success = beta_BN[i].load(filename, arma_ascii);
        if (!success || beta_BN[i].n_elem != outdim) {
          printf("Error loading %s: file not found or dimension mismatch\n",
                 filename.c_str());
          cout << "Got " << beta_BN[i].n_elem << ", expected " << outdim << endl;
          return false;
        }

        filename = MODEL_PREFIX + "_L" + to_string(i) + "_BN_gamma.txt";
        success = gamma_BN[i].load(filename, arma_ascii);
        if (!success || gamma_BN[i].n_elem != outdim) {
          printf("Error loading %s: file not found or dimension mismatch\n",
                 filename.c_str());
          cout << "Got " << gamma_BN[i].n_elem << ", expected " << outdim << endl;
          return false;
        }

        if (TEST_RUN) {
          filename = MODEL_PREFIX + "_L" + to_string(i) + "_BN_mean.txt";
          success = mean_BN[i].load(filename, arma_ascii);
          if (!success || mean_BN[i].n_elem != outdim) {
            printf("Error loading %s: file not found or dimension mismatch\n",
                   filename.c_str());
            cout << "Got " << mean_BN[i].n_elem << ", expected " << outdim << endl;
            return false;
          }

          filename = MODEL_PREFIX + "_L" + to_string(i) + "_BN_std.txt";
          success = std_BN[i].load(filename, arma_ascii);
          if (!success || std_BN[i].n_elem != outdim) {
            printf("Error loading %s: file not found or dimension mismatch\n",
                   filename.c_str());
            cout << "Got " << std_BN[i].n_elem << ", expected " << outdim << endl;
            return false;
          }
        }
      }
    }

  } else {

    X_mean = sum(X, 1) / N;
    X_std = sqrt(sum(square(X.each_col() - X_mean), 1) / N + BN_eps);

    for (int i = 0; i < nlayers + 1; i++) {
      int indim = (i == 0) ? D : nunits;
      int outdim = (i == nlayers) ? no_dims : nunits;
  
      if (i < nlayers) indim++; // add intercept
    
      W[i].randn(outdim, indim);
      //W[i].randu(outdim, indim) - .5;
      W[i] *= sqrt(2.0/(double)indim);// * sqrt(12.0);
  
      if (batch_norm && i < nlayers) {
        // For each output feature of the previous layer, create BN params
        beta_BN[i].zeros(outdim);
        gamma_BN[i].ones(outdim);
      }
    }

  }

  // Normalize input features
  X.each_col() -= X_mean;
  X.each_col() /= X_std;
  
  // Initialize auxiliary variables for SGD
  for (int i = 0; i < nlayers + 1; i++) {
    int indim = (i == 0) ? D : nunits;
    int outdim = (i == nlayers) ? no_dims : nunits;
    if (i < nlayers) indim++; // add intercept

    dW[i].zeros(outdim, indim);

    if (i < nlayers) {
      dbeta_BN[i].zeros(outdim);
      dgamma_BN[i].zeros(outdim);
    }
  
    if (STEP_METHOD == "adam") {
      dW_m[i].zeros(outdim, indim);
      dW_v[i].zeros(outdim, indim);
      if (i < nlayers) {
        dbeta_BN_m[i].zeros(outdim);
        dbeta_BN_v[i].zeros(outdim);
        dgamma_BN_m[i].zeros(outdim);
        dgamma_BN_v[i].zeros(outdim);
      }
    } else if (STEP_METHOD == "mom") {
      W_update[i].zeros(outdim, indim);
      W_gains[i].ones(outdim, indim);
      if (i < nlayers) {
        beta_BN_update[i].zeros(outdim);
        beta_BN_gains[i].zeros(outdim);
        gamma_BN_update[i].zeros(outdim);
        gamma_BN_gains[i].zeros(outdim);
      }
    }
  }
  

  if (!use_target_Y) {
    printf("Sorting P ... ");
    sort_P(N, row_P, col_P, val_P);
    //save_P(N, row_P, col_P, val_P, "P_sorted.dat");
    printf("done\n");
  }

	// Perform main training loop
  printf("Learning embedding...\n");
  t_start = chrono::high_resolution_clock::now();

  ivec vec_1toN(N);
  for (int i = 0; i < N; i++) {
    vec_1toN(i) = i;
  }

  vec P_rowsum(N);
  if (!use_target_Y) {
    for (int i = 0; i < N; i++) {
      // Note P is sorted
      // Add small values first for numerical stability
      P_rowsum(i) = 0;
      for (unsigned int j = 0; j < row_P[i+1]-row_P[i]; j++) {
        P_rowsum(i) += val_P[row_P[i+1]-1-j];
    
        if (MONTE_CARLO_POS) {
          val_P[row_P[i+1]-1-j] = P_rowsum(i); // replace with running sum
                                // needed for weighted sampling
        }
      }
      
      if (MONTE_CARLO_POS) {
        for (unsigned int j = row_P[i]; j < row_P[i+1]; j++) {
          if (j == row_P[i]) {
            val_P[j] = 1;
          } else {
            val_P[j] /= P_rowsum(i); // normalize
          }
        }
      }
    }
    printf("P_rowsum computed\n");
  }

  int subN = ceil(N * BATCH_FRAC);
  int nbatch = ((N-1) / subN) + 1;
  int min_sample = ceil(N * MIN_SAMPLE_Z);
  bool map_all_flag = min_sample == N;

  printf("subN: %d\n", subN);
  printf("nbatch: %d\n", nbatch);
  printf("min_sample: %d\n", min_sample);
  
  Col<unsigned int> row_P_mini(subN + 1);
  Col<unsigned int> col_P_mini(subN * N_SAMPLE_LOCAL);
  Col<double> val_P_mini(subN * N_SAMPLE_LOCAL);
  map<unsigned int,int> full2sub;
  uvec sub2full;
  ivec rand_indices(N_SAMPLE_LOCAL);
  
  for (int i = 0; i <= subN; i++) {
    row_P_mini(i) = i * N_SAMPLE_LOCAL;
  }
  
  if (!map_all_flag) {
    sub2full.set_size(subN * (1 + N_SAMPLE_LOCAL) + min_sample);
  }
  
  newfile = outdir;
  newfile /= "log.txt";
  ofstream logfs(newfile.c_str());
  logfs << "iter\tobjective\telapsed-time" << endl;

  double C0 = DBL_MAX;

  // Initial map
  // Forward propagation - Full pass
  if (COMPUTE_INIT || TEST_RUN) {
    for (int i = 0; i < nlayers; i++) {

      // Affine transformation
      A[i] = W[i].head_cols(W[i].n_cols - 1) * ((i == 0) ? X : A[i-1]);
      A[i].each_col() += W[i].tail_cols(1);

      // Nonlinear activation
      if (use_relu) {
        A[i] = clamp(A[i], 0, datum::inf);
      } else {
        A[i] = 1 / (1 + exp(-A[i]));
      }

      // Batch normalization
      if (batch_norm) { // NOTE this is only applied after each hidden layer
        if (!TEST_RUN) {
          mean_BN[i] = sum(A[i], 1) / A[i].n_cols;
          std_BN[i] = sqrt(BN_eps + sum(square(A[i].each_col() - mean_BN[i]), 1) / A[i].n_cols);
        }

        A[i].each_col() -= mean_BN[i];
        A[i].each_col() /= std_BN[i];
        A[i].each_col() %= gamma_BN[i];
        A[i].each_col() += beta_BN[i];
      }
    }

    Y = W[nlayers] * A[nlayers - 1];

    printf("Init Y computed\n");
    if (use_target_Y) {
      C0 = sqrt(sum(sum(square(target_Y - Y))));
      printf("Initial Norm: %f\n", C0);
    } else {
      C0 = evaluateError(row_P, col_P, val_P, Y.memptr(), N, no_dims, theta, P_rowsum);
      printf("Initial KL: %f\n", C0);
    }
    
    newfile = outdir;
    newfile /= "Y_0.txt";
    mat Y_tmp = Y.t();
    Y_tmp.save(newfile.string(), arma_ascii);

    iters.push_back(0);
    objectives.push_back(C0);
    elapsed_times.push_back(0);

    printf("Iteration 0: error is %f\n", C0);

    logfs << iters[0] << "\t" << objectives[0] << "\t" << elapsed_times[0] << endl;
    logfs.flush();

  }

  if (TEST_RUN) { 
    return true; // We are done if this is a test run
  }

  ivec X_perm_indices, perm_indices;

	for (int iter = 0; iter < max_iter; iter++) {

    tic();

    if (iter == 0 || (PERM_ITER > 0 && iter % PERM_ITER == 0)) {
      /* Permute X and P for fast mini-batching */
      printf("Permuting data points ... "); tic();
      randperm(perm_indices, X); // in-place shuffle
      if (use_target_Y) {
        mat tmp(no_dims, N);
        for (int i = 0; i < perm_indices.n_elem; i++) {
          tmp.col(i) = target_Y.col(perm_indices(i));
        }
        target_Y = tmp;
      } else {
        permute_P(perm_indices, row_P, col_P, val_P);
      }
      if (iter > 0) {
        for (int i = 0; i < perm_indices.n_elem; i++) {
          perm_indices(i) = X_perm_indices(perm_indices(i));
        }
      }
      X_perm_indices = perm_indices;
      printf("done (%.2f secs)\n", toc() / 1000);
    }

    int batch_id = iter % nbatch;
    int batch_start = batch_id * subN;
    int batch_end = min(N, batch_start + subN);
    int batch_N = batch_end - batch_start;

    int extra_N = max(0, min_sample - batch_N);
    
    full2sub.clear();
    if (!map_all_flag) {
      for (int i = 0; i < batch_N; i++) {
        full2sub[batch_start + i] = i;
        sub2full(i) = batch_start + i;
      }

      if (extra_N > 0) {
        randperm(rand_indices, N - batch_N, extra_N);
        for (int i = 0; i < extra_N; i++) {
          unsigned int selected = (batch_end + rand_indices(i)) % N;
          full2sub[selected] = batch_N + i;
          sub2full(batch_N + i) = selected;
        }
      }
    }
    
    vec pos_correct(batch_N);

    int N_mini = batch_N + extra_N;
    if (!use_target_Y) {
      for (int i = 0; i < batch_N; i++) {
        unsigned int full_ind;
        if (map_all_flag) {
          full_ind = batch_start + i;
        } else {
          full_ind = (unsigned int)sub2full(i);
        }
        int nnei = row_P[full_ind+1] - row_P[full_ind];

        if (MONTE_CARLO_POS) {
          vec u = randu(N_SAMPLE_LOCAL);
          // binary search to find index for each random number
          for (int s = 0; s < N_SAMPLE_LOCAL; s++) {
            rand_indices(s) = binsearch(&val_P[row_P[full_ind]], u(s), 0, nnei);
          }
        } else {
          randperm(rand_indices, nnei, N_SAMPLE_LOCAL);
        }

        for (int s = 0; s < N_SAMPLE_LOCAL; s++) {
          unsigned int ind2 = rand_indices(s) + row_P[full_ind];
          unsigned int full_ind2 = col_P[ind2];
      
          if (!map_all_flag && full2sub.find(full_ind2) == full2sub.end()) {
            sub2full(N_mini) = full_ind2;
            full2sub[full_ind2] = N_mini;
            N_mini++;
          }

          if (map_all_flag) {
            col_P_mini(i*N_SAMPLE_LOCAL + s) = full_ind2;
          } else {
            col_P_mini(i*N_SAMPLE_LOCAL + s) = full2sub[full_ind2];
          }

          double val;
          if (MONTE_CARLO_POS) {
            val = 1;
          } else {
            val = val_P[ind2];
          }

          val_P_mini(i*N_SAMPLE_LOCAL + s) = val;
        }

        // Correction factor for positive force
        if (MONTE_CARLO_POS) {
          pos_correct(i) = P_rowsum(full_ind) / N_SAMPLE_LOCAL;
        } else {
          pos_correct(i) = nnei / N_SAMPLE_LOCAL;
        }
      }
    }

    // Forward propagation
    for (int i = 0; i < nlayers; i++) {

      // Affine transformation
      if (i == 0) {
        if (map_all_flag) {
          A[i] = W[i].head_cols(W[i].n_cols - 1) * X;
        } else {
          A[i] = W[i].head_cols(W[i].n_cols - 1) * X.cols(sub2full.head(N_mini));
        }
      } else {
        A[i] = W[i].head_cols(W[i].n_cols - 1) * A[i-1];
      }
      A[i].each_col() += W[i].tail_cols(1);

      // Nonlinear activation
      if (use_relu) {
        A[i] = clamp(A[i], 0, datum::inf);
      } else {
        A[i] = 1 / (1 + exp(-A[i]));
      }

      // Batch normalization
      if (batch_norm) { // NOTE this is only applied after each hidden layer
        A_mu[i] = sum(A[i], 1) / A[i].n_cols;
        A_sig2[i] = sum(square(A[i].each_col() - A_mu[i]), 1) / A[i].n_cols;
        A[i].each_col() -= A_mu[i];
        A[i].each_col() /= sqrt(A_sig2[i] + BN_eps);
        A_xhat[i] = A[i]; // cache xhat values
        A[i].each_col() %= gamma_BN[i];
        A[i].each_col() += beta_BN[i];
      }
    }

    // Output layer
    mat Y_mini = W[nlayers] * A[nlayers - 1];

    // Compute (approximate) gradient
    mat dY(no_dims, batch_N);
    if (use_target_Y) {
      dY = 2 * (Y_mini - target_Y.cols(sub2full.head(N_mini)));
    } else {
      computeGradient(batch_N, extra_N, pos_correct, row_P_mini.memptr(),
          col_P_mini.memptr(), val_P_mini.memptr(),
          Y_mini.memptr(), N_mini, no_dims, dY.memptr(), theta,
          iter < stop_lying_iter, map_all_flag ? batch_start : 0);
    }

    // Take the gradients/other related data for only the primary points in the mini-batch
    // Note that the remaining points are used for approximating the gradients of the primary points
    for (int i = 0; i < nlayers; i++) {
      if (map_all_flag) {
        A[i] = A[i].cols(batch_start, batch_end-1);
        if (batch_norm) {
          A_xhat[i] = A_xhat[i].cols(batch_start, batch_end-1);
        }
      } else {
        A[i] = A[i].head_cols(batch_N);
        if (batch_norm) {
          A_xhat[i] = A_xhat[i].head_cols(batch_N);
        }
      }
    }

    // Backpropagation
    dW[nlayers] = dY * A[nlayers - 1].t() * (N / (double)batch_N);
    
    mat cur_dA = W[nlayers].t() * dY;

    for (int i = nlayers - 1; i >= 0; i--) {
      if (batch_norm) {
        vec sigeps = sqrt(A_sig2[i] + BN_eps);
        mat dxhat = cur_dA.each_col() % gamma_BN[i];
        mat tmp = dxhat % A_xhat[i];
        vec dsig2 = -0.5 * sum(tmp.each_col() / (A_sig2[i] + BN_eps), 1);
        vec dmu = -sum(dxhat.each_col() / sigeps, 1) - 2 * dsig2 % sum(A_xhat[i].each_col() % sigeps, 1) / batch_N;
        dgamma_BN[i] = sum(cur_dA % A_xhat[i], 1);
        dbeta_BN[i] = sum(cur_dA, 1);
        cur_dA = dxhat.each_col() / sigeps + 2 * (dsig2 % sigeps) % A_xhat[i].each_col() / batch_N;
        cur_dA.each_col() += dmu / batch_N;
        A[i] = A_xhat[i].each_col() % sigeps;
        A[i].each_col() += A_mu[i];
      }
    
      mat dO;
      if (use_relu) {
        dO = cur_dA % (A[i] > 0);
      } else {
        dO = cur_dA % A[i] % (1 - A[i]);
      }
    
      if (i > 0) {
        dW[i].head_cols(dW[i].n_cols - 1) = dO * A[i-1].t();
      } else {
        dW[i].head_cols(dW[i].n_cols - 1) = dO * X.cols(batch_start, batch_end-1).t();
      }
      dW[i].tail_cols(1) = sum(dO, 1);
    
      if (i > 0) {
        cur_dA = trans(W[i].head_cols(W[i].n_cols - 1)) * dO;
      }
    }

    // L2 regularization on W
    for (int i = 0; i <= nlayers; i++) {
      dW[i] += 2 * L2_REG * W[i];
    }
    
    // Gradient update
    if (STEP_METHOD == "adam") {
      for (int i = 0; i < nlayers + 1; i++) {
        dW_m[i] = adam_b1 * dW_m[i] + (1 - adam_b1) * dW[i];
        dW_v[i] = adam_b2 * dW_v[i] + (1 - adam_b2) * (dW[i] % dW[i]);
        W[i] -= eta * (dW_m[i] / (1.0 - adam_b1)) / (sqrt(dW_v[i] / (1.0 - adam_b2)) + adam_eps);
    
        if (batch_norm && i < nlayers) {
          dgamma_BN_m[i] = adam_b1 * dgamma_BN_m[i] + (1 - adam_b1) * dgamma_BN[i];
          dgamma_BN_v[i] = adam_b2 * dgamma_BN_v[i] + (1 - adam_b2) * (dgamma_BN[i] % dgamma_BN[i]);
          gamma_BN[i] -= eta * (dgamma_BN_m[i] / (1.0 - adam_b1)) / (sqrt(dgamma_BN_v[i] / (1.0 - adam_b2)) + adam_eps);
    
          dbeta_BN_m[i] = adam_b1 * dbeta_BN_m[i] + (1 - adam_b1) * dbeta_BN[i];
          dbeta_BN_v[i] = adam_b2 * dbeta_BN_v[i] + (1 - adam_b2) * (dbeta_BN[i] % dbeta_BN[i]);
          beta_BN[i] -= eta * (dbeta_BN_m[i] / (1.0 - adam_b1)) / (sqrt(dbeta_BN_v[i] / (1.0 - adam_b2)) + adam_eps);
        }
      }
    } else if (STEP_METHOD == "mom_gain") { // Momentum w/ gains
      for (int i = 0; i < nlayers + 1; i++) {
        // Update gains
        umat selector = (dW[i] % W_update[i]) > 0;
        W_gains[i] += 0.2 - selector % (W_gains[i] * 0.2 + 0.2);
        W_gains[i] = clamp(W_gains[i], 0.01, datum::inf);

        if (batch_norm && i < nlayers) {
          uvec sel = (dgamma_BN[i] % gamma_BN_update[i]) > 0;
          gamma_BN_gains[i] += 0.2 - sel % (gamma_BN_gains[i] * 0.2 + 0.2);
          gamma_BN_gains[i] = clamp(gamma_BN_gains[i], 0.01, datum::inf);

          sel = (dbeta_BN[i] % beta_BN_update[i]) > 0;
          beta_BN_gains[i] += 0.2 - sel % (beta_BN_gains[i] * 0.2 + 0.2);
          beta_BN_gains[i] = clamp(beta_BN_gains[i], 0.01, datum::inf);
        }
    
        double mom = (iter < mom_switch_iter) ? momentum : final_momentum;

        // Perform gradient update (with momentum and gains)
        W_update[i] *= mom;
        W_update[i] -= eta * W_gains[i] % dW[i];
        W[i] += W_update[i];
    
        if (batch_norm && i < nlayers) {
          gamma_BN_update[i] *= mom;
          gamma_BN_update[i] -= eta * gamma_BN_gains[i] % dgamma_BN[i];
          gamma_BN[i] += gamma_BN_update[i];

          beta_BN_update[i] *= mom;
          beta_BN_update[i] -= eta * beta_BN_gains[i] % dbeta_BN[i];
          beta_BN[i] += beta_BN_update[i];
        }
      }
    } else if (STEP_METHOD == "mom") { // Just momentum
      for (int i = 0; i < nlayers + 1; i++) {
        double mom = (iter < mom_switch_iter) ? momentum : final_momentum;
        W_update[i] *= mom;
        W_update[i] -= eta * dW[i];
        W[i] += W_update[i];

        if (batch_norm && i < nlayers) {
          gamma_BN_update[i] *= mom;
          gamma_BN_update[i] -= eta * dgamma_BN[i];
          gamma_BN[i] += gamma_BN_update[i];

          beta_BN_update[i] *= mom;
          beta_BN_update[i] -= eta * dbeta_BN[i];
          beta_BN[i] += beta_BN_update[i];
        }
      }
    } else if (STEP_METHOD == "fixed") { 
      for (int i = 0; i < nlayers + 1; i++) {
        W[i] -= eta * dW[i];
      }
    }

    // Print out progress
    if (((iter + 1) % 50 == 0 || iter == max_iter - 1)) {
      t_end = chrono::high_resolution_clock::now();

      printf("Norm of dW:");
      for (int i = 0; i < nlayers; i++) {
        printf(" L%d (%f)", i, norm(dW[i], "fro"));
      }
      printf("\n");

      // Forward propagation - Full pass
      for (int i = 0; i < nlayers; i++) {
        A[i] = W[i].head_cols(W[i].n_cols - 1) * ((i == 0) ? X : A[i-1]);
        A[i].each_col() += W[i].tail_cols(1);
        if (use_relu) {
          A[i] = clamp(A[i], 0, datum::inf);
        } else {
          A[i] = 1 / (1 + exp(-A[i]));
        }

        if (batch_norm) {
          mean_BN[i] = sum(A[i], 1) / A[i].n_cols;
          std_BN[i] = sqrt(BN_eps + sum(square(A[i].each_col() - mean_BN[i]), 1) / A[i].n_cols);

          A[i].each_col() -= mean_BN[i];
          A[i].each_col() /= std_BN[i];
          A[i].each_col() %= gamma_BN[i];
          A[i].each_col() += beta_BN[i];
        }
      }

      Y = W[nlayers] * A[nlayers - 1];

      double C;
      if (use_target_Y) {
        C = sqrt(sum(sum(square(target_Y - Y))));
      } else {
        C = evaluateError(row_P, col_P, val_P, Y.memptr(), N, no_dims, theta, P_rowsum);
      }

      float elapsed = chrono::duration<float, std::milli>(t_end - t_start).count() / 1000.0;
      printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter + 1, C, elapsed);
      total_time += elapsed;
      
      if ((iter + 1) % 100 == 0 || iter == max_iter - 1) {
        string it_str = to_string(iter + 1);
        if (iter == max_iter - 1) {
          it_str = "final";
        }

        // Revert the ordering of data points to the initial ordering
        mat Y_tmp;
        reorder_columns(Y_tmp, Y, X_perm_indices);

        newfile = outdir;
        newfile /= "Y_" + it_str + ".txt";
        Y_tmp = Y_tmp.t();
        Y_tmp.save(newfile.string(), arma_ascii);

        newfile = outdir;
        newfile /= "model_" + it_str + "_param.txt";
        ofstream ofs(newfile.string());
        ofs << nlayers << " " << nunits << " " << use_relu << " " << batch_norm;
        ofs.close();

        newfile = outdir;
        newfile /= "model_" + it_str + "_X_mean.txt";
        X_mean.save(newfile.string(), arma_ascii);

        newfile = outdir;
        newfile /= "model_" + it_str + "_X_std.txt";
        X_std.save(newfile.string(), arma_ascii);

        for (int i = 0; i <= nlayers; i++) {
          newfile = outdir;
          newfile /= "model_" + it_str + "_L" + to_string(i) + "_W.txt";
          W[i].save(newfile.string(), arma_ascii);

          if (i < nlayers) {
            newfile = outdir;
            newfile /= "model_" + it_str + "_L" + to_string(i) + "_BN_beta.txt";
            beta_BN[i].save(newfile.string(), arma_ascii);

            newfile = outdir;
            newfile /= "model_" + it_str + "_L" + to_string(i) + "_BN_gamma.txt";
            gamma_BN[i].save(newfile.string(), arma_ascii);

            newfile = outdir;
            newfile /= "model_" + it_str + "_L" + to_string(i) + "_BN_mean.txt";
            mean_BN[i].save(newfile.string(), arma_ascii);

            newfile = outdir;
            newfile /= "model_" + it_str + "_L" + to_string(i) + "_BN_std.txt";
            std_BN[i].save(newfile.string(), arma_ascii);
          }
        }
      }

      iters.push_back(iter + 1);
      objectives.push_back(C);
      elapsed_times.push_back((double)total_time);

      logfs << iters.back() << "\t" << objectives.back() << "\t" << elapsed_times.back() << endl;
      logfs.flush();

      t_start = chrono::high_resolution_clock::now();
    }
  }

  logfs.close();

  printf("Fitting performed in %4.2f seconds.\n", total_time);

  return true;
}

// Computes edge forces
void computeEdgeForces(unsigned int* row_P, unsigned int* col_P,
    double* val_P, int N, double* pos_f,
    double* neg_f, double* qsum,
    int dimension, double *data, int ind_offset)
{
    // Loop over all edges in the graph
    unsigned int ind1 = 0;
    unsigned int ind2 = 0;
    unsigned int offset = ind_offset * dimension;
    double D;
    double buff[3];
    for(unsigned int n = 0; n < N; n++) {
        for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
            // Compute pairwise distance and Q-value
            D = 1.0;
            ind2 = col_P[i] * dimension;
            for(unsigned int d = 0; d < dimension; d++) {
              buff[d] = data[offset + ind1 + d] - data[ind2 + d];
            }
            for(unsigned int d = 0; d < dimension; d++) {
              D += buff[d] * buff[d];
            }
            //D = val_P[i] / D;
            // Sum positive force
            //for(unsigned int d = 0; d < dimension; d++) pos_f[ind1 + d] += D * buff[d];
            
            D = 1 / D;
            for(unsigned int d = 0; d < dimension; d++) {
              pos_f[ind1 + d] += val_P[i] * D * buff[d];
              if (neg_f != NULL) {
                neg_f[ind1 + d] += D * D * buff[d];
              }
            }
            if (qsum != NULL) {
              qsum[n] += D;
            }
        }
        //for(unsigned int d = 0; d < dimension; d++) printf("%f ", pos_f[ind1 + d]);
        //printf("\n");
        ind1 += dimension;
    }
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void NETSNE::computeGradient(int subN, int extraN, vec &pos_correct, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, bool early_exaggeration, int ind_offset)
{
  //printf("subN(%d), extraN(%d), theta(%f), early(%d), offset(%d)\n", subN, extraN, theta, early_exaggeration, ind_offset);
    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    //double sum_Q_ans = .0;
    double* pos_f = (double*) calloc(subN * D, sizeof(double));
    double* neg_f = (double*) calloc(subN * D, sizeof(double));
    double* qsum;
    if (MATCH_POS_NEG) {
      qsum = (double*) calloc(subN, sizeof(double));
    } else {
      qsum = NULL;
    }

    if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, subN, pos_f, 
        MATCH_POS_NEG ? neg_f : NULL, qsum, D, Y, ind_offset);

    //printf("EdgeForces: %f, ", toc()); tic();

    SPTree* tree = new SPTree(D, Y, subN + extraN);
    for (int n = 0; n < subN; n++) {

      int ndup = 0;
      if (MATCH_POS_NEG) {
        // if an edge found in the tree, nullify its effect
        for (int j = inp_row_P[n]; j < inp_row_P[n+1]; j++) {
          if (inp_col_P[j] < subN + extraN) { 
            tree->modifyWeight(inp_col_P[j], -1);
            ndup++;
          }
        }
      }

      double cur_qsum = 0.0;
      double neg_f_tmp[3] = {0};
      double neg_correct;
      if (MATCH_POS_NEG) {
        neg_correct = (N - 1 - ndup) / (double)(subN + extraN - 1 - ndup);
      } else {
        neg_correct = 1.0;
      }

      //tree->computeNonEdgeForces(ind_offset + n, theta, neg_f + n * D, &sum_Q);
      tree->computeNonEdgeForces(ind_offset + n, theta, neg_f_tmp, &cur_qsum);
      for (int d = 0; d < D; d++) {
        neg_f[n * D + d] += neg_f_tmp[d] * neg_correct;
      }

      sum_Q += cur_qsum * neg_correct;
      if (MATCH_POS_NEG) {
        sum_Q += qsum[n];
      }

      if (MATCH_POS_NEG) {
        // restore the tree
        for (int j = inp_row_P[n]; j < inp_row_P[n+1]; j++) {
          if (inp_col_P[j] < subN + extraN) { 
            tree->modifyWeight(inp_col_P[j], +1);
          }
        }
      }

    }

    // Compute final t-SNE gradient
    for(int i = 0; i < subN; i++) {
      for (int j = 0; j < D; j++) {
        //dC[i] = pos_f[i] - (neg_f_ans[i] / sum_Q_ans);
        if (early_exaggeration) {
          dC[i*D+j] = 12.0 * pos_f[i*D+j] * pos_correct(i) - (neg_f[i*D+j] / sum_Q) * (subN / (double) N);
        } else {
          dC[i*D+j] = pos_f[i*D+j] * pos_correct(i) - (neg_f[i*D+j] / sum_Q) * (subN / (double) N);
        }
      }
    }
    free(pos_f);
    free(neg_f);
    if (qsum != NULL) free(qsum);
    delete tree;
}

// Evaluate t-SNE cost function (approximately)
double NETSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, vec& P_rowsum)
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

            double val = val_P[i];
            if (MONTE_CARLO_POS) {
              // New val_P is cumulative and normalized
              if (i < row_P[n+1] - 1) {
                val -= val_P[i+1];
              }
              val *= P_rowsum(n);
            }

            C += val * log((val + FLT_MIN) / (Q + FLT_MIN));
        }
    }

    // Clean up memory
    free(buff);
    delete tree;
    return C;
}

bool NETSNE::load_P(string infile, int &N, unsigned int** row_P, unsigned int** col_P, double** val_P) {
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

void NETSNE::save_P(int N, unsigned int* row_P, unsigned int* col_P, double* val_P, string filename) {
  FILE *h;
	if((h = fopen(filename.c_str(), "w+b")) == NULL) {
		printf("Error: could not open data file in save_P\n");
		return;
	}

  fwrite(&N, sizeof(int), 1, h);
  fwrite(row_P, sizeof(unsigned int), N+1, h);
  fwrite(col_P, sizeof(unsigned int), row_P[N], h);
  fwrite(val_P, sizeof(double), row_P[N], h);
  fclose(h);
  printf("P saved\n");
}
