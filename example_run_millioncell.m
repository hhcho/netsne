% This script demonstrates large-scale visualization of
% net-SNE on 1.3 million mouse brain cells from 10x Genomics.
% We first train net-SNE on a random subset of 100K cells and
% generalize to the remaining cells.
%
% Dataset URL: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
%
% We provide a dimension-reduced dataset where we projected each cell onto top 50
% principal components. This matrix is available as a MATLAB formatted data at:
%
% http://netsne.csail.mit.edu/data/brain1m_50pc.mat
%
% *** NOTE ***
% Before running this script, download our preprocessed file
% and put it in the example_data/ directory.

% Check if dataset is downloaded
if ~exist('example_data/brain1m_50pc.mat', 'file')
  fprintf('Before running this script, please download the necessary\n');
  fprintf('data file at this link: \n');
  fprintf('    http://netsne.csail.mit.edu/data/brain1m_50pc.mat\n');
  fprintf('and put it in the example_data/ directory.');
  return
end

% Preprocess pbmc68k
fprintf('Loading brain1m data ...\n');
load('example_data/brain1m_50pc.mat'); % Loads matrix X

fprintf('Subsampling ...\n');
n = size(X, 1);
rp = randperm(n, 1e5);
Xsub = X(rp,:);

fprintf('Saving to disk ...\n');
prepare_input(Xsub, 'example_data/brain100k_X.dat', inf); % PCA is skipped

% Compute input similarities
fprintf('Computing input similarities ...\n');
system('bin/ComputeP --input-file example_data/brain100k_X.dat --output-file example_data/brain100k_P.dat');

% Run t-SNE to obtain target map for net-SNE
fprintf('Initial map ...\n');
system('bin/RunBhtsne --input-P example_data/brain100k_P.dat --out-dir example_data/bhtsne_brain100k');

% Run net-SNE
fprintf('Train net-SNE ...\n');
system('bin/RunNetsne --input-Y example_data/bhtsne_brain100k/Y_final.txt --input-X example_data/brain100k_X.dat --learn-rate 0.05 --out-dir example_data/netsne_brain100k');

% Preprocess bcells
fprintf('Saving full data to disk ...\n');
prepare_input(X, 'example_data/brain1m_X.dat', inf); % PCA is skipped

% Map bcells with the trained net-SNE model
fprintf('Mapping all cells ...\n');
system('bin/RunNetsne --input-X example_data/brain1m_X.dat --init-model-prefix example_data/netsne_brain100k/model_final --test-model --no-target --out-dir example_data/netsne_brain1m');

% Plot results
fprintf('Plotting results ...\n');

% Find a good range to display
Y = dlmread('example_data/bhtsne_brain100k/Y_final.txt', '', 2, 0);
ax = [min(Y(:,1)), max(Y(:,1)), min(Y(:,2)), max(Y(:,2))]; 
xlen = ax(2) - ax(1); ylen = ax(4) - ax(3);
ax = ax + [-xlen, xlen, -ylen, ylen]/10; 

Y = dlmread('example_data/netsne_brain1m/Y_final.txt', '', 2, 0);
nbins = 400;
H = histcounts2(Y(:,1), Y(:,2), linspace(ax(1),ax(2),nbins), linspace(ax(3),ax(4),nbins));
figure;
imagesc(H);
c = linspace(1,0,100); colormap([c(:), c(:), ones(length(c),1)]);
colorbar;
title('brain1m');
