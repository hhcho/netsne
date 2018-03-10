function prepare_input(X, output_file, initial_dims, svd_type, transform_file)
% Takes a data matrix, optionally performs PCA, then creates an input data file
% for net-SNE or t-SNE
%   X: data matrix (# of instances by # of features)
%   output_file: name of the binary data file to be created [data.dat]
%   initial_dims: input dimensionality for net-SNE/t-SNE
%                 if this is smaller than # of columns of X, PCA is performed
%                 [1000]
%   svd_type: 0 for using existing transformation in transform_file,
%             1 for performing partial SVD (use if initial_dims is very
%               small compared to the # of columns of X), and
%             2 for performing full SVD then taking the top components
%             [2] 
%   transform_file: name of the MAT file to be loaded/created for
%                   the centering parameters and principal components
%                   used to transform the data
%                   [pca_transform.mat] 

  if ~exist('initial_dims', 'var') || isempty(initial_dims)
    initial_dims = 1000;
  end

  if ~exist('output_file', 'var') || isempty(output_file)
    output_file = 'data.dat';
  end

  if ~exist('transform_file', 'var') || isempty(transform_file)
    transform_file = 'pca_transform.mat';
  end

  if ~exist('svd_type', 'var') || isempty(svd_type)
    svd_type = 2;
  end

  % Perform the initial dimensionality reduction using PCA
  X = double(X);
  if svd_type == 0
    ws = load(transform_file);
    X_center = ws.X_center;
    V = ws.X_princomp;
    clear ws;
  end

  if initial_dims < size(X, 2)
    fprintf('Centering data ... ');
    if svd_type ~= 0
      X_center = mean(X, 1);
    end
    X = bsxfun(@minus, X, X_center); % Center each input dimension
    fprintf('done\n');

    if svd_type ~= 0
      fprintf('Performing SVD ... ');
      if svd_type == 2
        [~,~,V] = svd(X);
        V = V(:,1:initial_dims);
      else
        [~,~,V] = svds(X, initial_dims);
      end
      fprintf('done\n');
    end

    fprintf('Projecting data ... ');
    X = X * V;
    fprintf('done\n');

    if svd_type ~= 0
      X_princomp = V;
      save(transform_file, 'X_center', 'X_princomp', '-v7.3');
    end
  end
    
  fprintf('Writing to disk (%s) ... ', output_file);
  [num_instances, num_features] = size(X);
  fp = fopen(output_file, 'wb');
  fwrite(fp, num_instances, 'integer*4');
  fwrite(fp, num_features, 'integer*4');
  fwrite(fp, X', 'double'); % Write row by row
  fclose(fp);
  fprintf('done\n');

end
