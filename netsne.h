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


#ifndef NETSNE_H
#define NETSNE_H

#include <vector>
#include <string>
#include <armadillo>
#include <boost/filesystem.hpp>

using namespace arma;

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class NETSNE
{
public:
  bool run(int N, unsigned int *row_P, unsigned int *col_P, double *val_P, mat &target_Y,
           mat &X, mat &Y, int no_dims, double theta, int rand_seed,
           int max_iter, boost::filesystem::path outdir);
  bool load_data(mat &X, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter,
                 int* nlayers, int* nunits, int* use_relu);
  void save_data(double* data, int* landmarks, double* costs, int n, int d);
  void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, double** val_P, int N); // should be static!

  int N_SAMPLE_LOCAL = 20;
  double BATCH_FRAC = 0.05;
  double MIN_SAMPLE_Z = 0.1;
  double LEARN_RATE = 0.02;
  double L2_REG = 0.01;
  int STOP_LYING = 250;
  std::string MODEL_PREFIX = "";
  bool TEST_RUN = false;
  bool NO_TARGET = false;
  bool MODEL_PREFIX_FLAG = false;
  bool MONTE_CARLO_POS = false;
  std::string STEP_METHOD = "relu";
  int MOM_SWITCH_ITER = 250;
  double MOM_INIT = 0.5;
  double MOM_FINAL = 0.8;
  bool COMPUTE_INIT = false;
  bool MATCH_POS_NEG = false;
  bool BATCH_NORM = false;
  int NUM_LAYERS = 2;
  int NUM_UNITS = 50;
  int PERM_ITER = INT_MAX;
  int CACHE_ITER = INT_MAX;
  std::string ACT_FN = "relu";
  bool SGD_FLAG = true;

  bool load_P(std::string infile, int &N, unsigned int** row_P, unsigned int** col_P, double** val_P);
  void save_P(int N, unsigned int* row_P, unsigned int* col_P, double* val_P, std::string filename = "P.dat");
 
private:

  std::vector<int> iters;
  std::vector<double> objectives;
  std::vector<double> elapsed_times;

  void computeGradient(int subN, int extraN, vec &pos_correct, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta, bool early_exaggeration, int ind_offset);
  double evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta, vec& P_rowsum);
  void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);
  double randn();
};


#endif
