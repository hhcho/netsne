% Preprocess data
fprintf('Loading data ...\n');
X = dlmread('../data/pollen/matrix.txt');
fprintf('Preprocessing ...\n');
prepare_input(X', 'example_data/pollen_X.dat', 50, 1, 'example_data/pollen_pca.mat')

% Compute (approximate) input similarity matrix
fprintf('Computing input similarities ...\n');
system('bin/ComputeP --input-file example_data/pollen_X.dat --output-file example_data/pollen_P.dat');

% Learn embedding
fprintf('Running netsne ...\n');
system('bin/RunNetsne --input-P example_data/pollen_P.dat --input-X example_data/pollen_X.dat --out-dir example_data/netsne --no-sgd');

fprintf('Running bhtsne ...\n');
system('bin/RunBhtsne --input-P example_data/pollen_P.dat --out-dir example_data/bhtsne');

% Plot results
fprintf('Plotting ...\n');
YN = dlmread('example_data/netsne/Y_final.txt', '', 2, 0);
YB = dlmread('example_data/bhtsne/Y_final.txt', '', 2, 0);
labels = dlmread('../data/pollen/labels.txt');
figure;
subplot(1,2,1);
scatter(YN(:,1), YN(:,2), 10, labels, 'filled')
title('netsne')
subplot(1,2,2);
scatter(YB(:,1), YB(:,2), 10, labels, 'filled')
title('bhtsne')
