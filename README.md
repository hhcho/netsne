# net-SNE

This repository contains software for net-SNE, a neural netork-based visualization tool developed for single cell RNA sequencing datasets.

##### Reference
"Generalizable and Scalable Visualization of Single-Cell Data Using Neural Networks"\
Hyunghoon Cho, Bonnie Berger, and Jian Peng\
Cell Systems, 2018

##### Dependencies
- [Armadillo](http://arma.sourceforge.net/) for matrix/vector operations (developed with version 8.2).
    - (*Optional*) To speed up matrix multiplication via multi-threading, we recommend installing [OpenBLAS](http://www.openblas.net/) before installing Armadillo (see Armadillo's README for more information). Many platforms come with pre-built OpenBLAS library. Armadillo's CMake installer will try to automatically detect whether OpenBLAS is available and link to it if it exists. You may need to explicitly provide the path if it is installed in a non-standard location.

- [Boost C++ Library](https://www.boost.org/) for command-line parser and file system operations (developed with version 1.66.0). We require only the FileSystem, System, and ProgramOptions libraries. These can be selectively installed by calling `bootstrap.sh` with the argument `--with-libraries=program_options,filesystem,system` before building Boost.

##### Installation
1. Update platform-specific settings in Makefile. Parameters to update/confirm are:
    - `CXX`: C++ compiler
    - `BOOSTROOT`, `BOOSTLIB`: Boost library paths (base directory for the package and library directory containing linkable .a/.so files, respectively)
    - `ARMAINC`,`ARMALIB`: Armadillo library paths (directory containing header files and  directory with linkable .a/.so files, respectively); only needed if Armadillo is installed in a non-standard location, in which case `ARMALIB` should also be included in the system's search path for shared libraries (e.g., `LD_LIBRARY_PATH` on Linux) before running net-SNE.

2. Inside the net-SNE directory, run `make`. This will create the executables `RunNetsne`, `RunBhtsne`, and `ComputeP` in a subdirectory named `bin`.

##### Example Runs
We provide three example MATLAB scripts for different use cases of net-SNE: `example_run_basic.m`, `example_run_crossdataset.m`, and `example_run_millioncell.m`. See the comments in each script for more information. The following is a step-by-step description of `example_run_basic.m`, which serves a good starting point: 

1. *Prepare dataset.* In `example_data` directory we provide an example single-cell RNA-seq dataset from [Pollen et al., *Nature Biotechnology*, 2014](https://www.nature.com/articles/nbt.2967). The data file `pollen.txt` contains a matrix in space-delimited text format, where each row is a gene (or a unique molecule) and each column is a sample (i.e., a cell). Each element represents a measure of expression (e.g., normalized read counts). Known subtypes of the cells are provided in `pollen_labels.txt`.

2. *Preprocess.* We provide a MATLAB script `prepare_input.m` for (optionally) performing dimensionality reduction via principal component analysis and saving the data in binary format for subsequent steps. In MATLAB, run:\
`X = dlmread('example_data/pollen.txt');`\
`prepare_input(X', 'example_data/pollen_X.dat', 50, 1, 'example_data/pollen_pca.mat')`\
This will save the processed data with respect to top 50 principal components to `example_data/pollen_X.dat`. See the comments in `prepare_input.m` for more information about the arguments.

3. *Compute input similarities.* Like t-SNE, net-SNE learns the embedding by placing cells closer to each other in the visualization if they are more similar in the input data. Using `bin/ComputeP`, which invokes a subroutine from the [original t-SNE implementation](https://github.com/lvdmaaten/bhtsne), we can construct a k-Nearest Neighbor (k-NN) graph approximation of the input similarity matrix (denoted *P* in t-SNE). In the command-line terminal, run:\
`bin/ComputeP --input-file example_data/pollen_X.dat --output-file example_data/pollen_P.dat`\
This saves the input similarity matrix to `example_data/pollen_P.dat`. To see the list of supported arguments, run `bin/ComputeP --help`.

4. *Run net-SNE (or t-SNE)*. In the command-line terminal, run:\
`bin/RunNetsne --input-P example_data/pollen_P.dat --input-X example_data/pollen_X.dat --out-dir example_data/netsne --no-sgd`\
This outputs the final model and the embedding in the `example_data/netsne` directory. The embedding file `Y_final.txt` is in the Armadillo text file format, which contains two lines of header followed by a matrix. Note the flag `--no-sgd` makes net-SNE use batch gradient descent as the example dataset is relatively small. To see the list of supported parameters, run `bin/RunNetsne --help`. To run t-SNE for comparison, run the following:\
`bin/RunBhtsne --input-P example_data/pollen_P.dat --out-dir example_data/bhtsne`

5. *Plot visualization*. The following MATLAB code visualizes the final embeddings and color the points according to the known cluster labels in `pollen_labels.txt`:\
`Y = dlmread('example_data/netsne/Y_final.txt', '', 2, 0);`\
`labels = dlmread('example_data/pollen_labels.txt');`\
`scatter(Y(:,1), Y(:,2), 10, labels, 'filled')`

##### Notes
Our software is based on the implementation of Barnes-Hut t-SNE developed by Laurens van der Maaten at Delft University of Technology, available at: https://github.com/lvdmaaten/bhtsne.

We are working on releasing wrappers in other programming languages (e.g. Python). If you would like an update when these become available, feel free to drop us a note at the email address below.

[2018/8/7] R wrapper is available at https://github.com/schwikowskilab/rNetSNE. Developed by Pierre Bost and Florian Specque.

##### Contact for Questions
Hoon Cho, hhcho@mit.edu
