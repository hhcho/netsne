function M = load_mtx(filename, log_transform)
  if ~exist('log_transform', 'var')
    log_transform = false;
  end
  X = dlmread(filename, ' ', 2, 0);

  if log_transform
    X(2:end,3) = log(1 + X(2:end,3));
  end

  M = sparse(X(2:end,1), X(2:end,2), X(2:end,3), X(1,1), X(1,2), X(1,3));
end
