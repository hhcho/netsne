% This script demonstrates cross-dataset generalization 
% of net-SNE. We will first train net-SNE on PBMC68k 
% dataset and map FAC-sorted B cells onto the same visualization. 
%
% PBMC68k dataset: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
% Direct link to file: http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
%
% B-cell dataset: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/b_cells
% Direct link to file: http://cf.10xgenomics.com/samples/cell-exp/1.1.0/b_cells/b_cells_filtered_gene_bc_matrices.tar.gz
%
% *** NOTE ***
% Before running this script, download both datasets, unpack them,
% rename each "matrix.mtx" file as "pbmc68k.mtx" and "bcells.mtx",
% respectively, and put them in the example_data/ directory.

% Check if datasets are downloaded
if ~exist('example_data/pbmc68k.mtx', 'file') || ...
   ~exist('example_data/bcells.mtx', 'file')
  fprintf('Before running this script, please download the necessary\n');
  fprintf('datasets from the 10x Genomics website, as explained in\n');
  fprintf('the header section of example_run_crossdata.m.\n')
  return
end

% Preprocess pbmc68k
fprintf('Loading pbmc68k data ...\n');
X = load_mtx('example_data/pbmc68k.mtx', true);
zero_filt = sum(X,2) == 0; % remove zero rows
X = X(~zero_filt,:);

fprintf('Preprocessing ...\n');
prepare_input(X', 'example_data/pbmc68k_X.dat', 50, 3, 'example_data/pbmc68k_pca.mat')

% Compute input similarities
fprintf('Computing input similarities ...\n');
system('bin/ComputeP --input-file example_data/pbmc68k_X.dat --output-file example_data/pbmc68k_P.dat');

% Run t-SNE to obtain target map for net-SNE
fprintf('Initial map ...\n');
system('bin/RunBhtsne --input-P example_data/pbmc68k_P.dat --out-dir example_data/bhtsne_pbmc68k');

% Run net-SNE
fprintf('Train net-SNE ...\n');
system('bin/RunNetsne --input-Y example_data/bhtsne_pbmc68k/Y_final.txt --input-X example_data/pbmc68k_X.dat --out-dir example_data/netsne_pbmc68k');

% Preprocess bcells
fprintf('Loading bcells ...\n');
Xnew = load_mtx('example_data/bcells.mtx', true);
Xnew = Xnew(~zero_filt,:);
fprintf('Preprocessing using the parameters from pbmc68k ...\n');
prepare_input(full(Xnew'), 'example_data/bcells_X.dat', 50, 0, 'example_data/pbmc68k_pca.mat');

% Map bcells with the trained net-SNE model
fprintf('Mapping bcells onto visualization using the trained net-SNE ...\n');
system('bin/RunNetsne --input-X example_data/bcells_X.dat --init-model-prefix example_data/netsne_pbmc68k/model_final --test-model --no-target --out-dir example_data/netsne_bcells');

% Plot results
fprintf('Plotting results ...\n');
Y = dlmread('example_data/netsne_pbmc68k/Y_final.txt', '', 2, 0);
Ynew = dlmread('example_data/netsne_bcells/Y_final.txt', '', 2, 0);
alpha = .1;
figure;
subplot(1,2,1);
scatter(Y(:,1), Y(:,2), 3, [.5 .5 .5], 'filled', 'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
title('pbmc68k')
subplot(1,2,2);
scatter(Y(:,1), Y(:,2), 3, [.8 .8 .8], 'filled', 'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
hold on
scatter(Ynew(:,1), Ynew(:,2), 3, [1 0 0], 'filled', 'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
hold off
title('new bcells')
