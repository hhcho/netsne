# netsne

This repository contains software for net-SNE, a neural netork-based visualization tool developed for single cell RNA sequencing datasets.

##### Reference
"Neural Data Visualization for Scalable and Generalizable Single Cell Analysis"\
Hyunghoon Cho, Bonnie Berger, and Jian Peng\
Cell Systems (accepted), 2018

##### Dependencies
- [OpenBLAS](http://www.openblas.net/) for multi-threaded matrix multiplication (*optional*, but recommended). Many platforms come with OpenBLAS pre-installed. On macOS, the internal Accelerate framework can be used instead of OpenBLAS (see below).
- [Armadillo](http://arma.sourceforge.net/) for matrix/vector operations (developed with version 8.2). Armadillo installer will try to automatically detect whether OpenBLAS/Accelerate is available. You may need to explicitly provide the path if a non-standard location is used. Check the output of the `configure` script to see if they are recognized before installing Armadillo.
- [Boost C++ Library](https://www.boost.org/) for command-line parser and interaction with the file system (developed with version 1.66.0). We require only the FileSystem, System, and ProgramOptions libraries. These can be selectively installed by calling `bootstrap.sh` with the argument `--with-libraries=program_options,filesystem,system` before building Boost.

##### Installation
1. Update the platform-specific settings in Makefile. Parameters to update/confirm:
    - `CXX`: C++ compiler
    - `BOOSTROOT`, `BOOSTLIB`: Boost library paths (base directory for the package and library directory containing linkable .a/.so files, respectively) 
    - `ARMAINC`: Armadillo "include" path (directory containing header files)
    - `OBLASINC`,`OBLASLIB`: (*Optional*) OpenBLAS library paths (directory containing header files and linkable .a/.so files, respectively); only needed if you intend to use OpenBLAS with Armadillo to accelerate matrix multiplication

2. Inside the netsne directory, run `make`. This will create the executables `RunNetsne`, `RunBhtsne`, and `ComputeP` in a subdirectory named `bin`.

##### Example Run
The following is a step-by-step explanation of our example MATLAB script `example_run.m`:
1. *Prepare dataset.* In `example_data` directory we provide an example single-cell RNA-seq dataset from [Pollen et al., *Nature Biotechnology*, 2014](https://www.nature.com/articles/nbt.2967). The data file `pollen.txt` contains a matrix in space-delimited text format, where each row is a gene (or a unique molecule) and each column is a sample (i.e., a cell). Each element represents a measure of expression (e.g., normalized read counts). Known subtypes of the cells are provided in `pollen_labels.txt`.

2. *Preprocess.* We provide a MATLAB script `prepare_input.m` for (optionally) performing dimensionality reduction via principal component analysis and saving the data in binary format for subsequent steps. In MATLAB, first load the data matrix:
`X = dlmread('example_data/pollen.txt');`
For this dataset, we perform log-transformation as follows, which is common for analyzing count data from RNA-seq:
`X = log(1 + X);`
Now run:
`prepare_input(X', 'example_data/pollen_X.dat', 50, 1, 'example_data/pollen_pca.mat')`
This will save the processed data with respect to top 50 principal components to `example_data/pollen_X.dat`. See the comments in `prepare_input.m` for more information about the arguments.

3. *Compute input similarities.* Like t-SNE, net-SNE learns the embedding by placing cells closer to each other in the visualization if they are more similar in the input data. Using `bin/ComputeP`, which invokes a subroutine from the [original t-SNE implementation](https://github.com/lvdmaaten/bhtsne), we can construct a k-Nearest Neighbor (k-NN) graph approximation to the input similarity matrix (denoted *P* in t-SNE). In the command-line terminal, run:
`bin/ComputeP --input-file example_data/pollen_X.dat --output-file example_data/pollen_P.dat`
This saves the input similarity matrix to `example_data/pollen_P.dat`. To see the list of supported arguments, run `bin/ComputeP --help`. 
 
4. *Run net-SNE (or t-SNE)*. In the command-line terminal, run:
`bin/RunNetsne --input-P example_data/pollen_P.dat --input-X example_data/pollen_X.dat --out-dir example_data/netsne --no-sgd`
This outputs the final model and the embedding in the `example_data/netsne` directory. The embedding file `Y_final.txt` is in the Armadillo text file format, which contains two lines of header followed by a matrix. Note the flag `--no-sgd` makes net-SNE use batch gradient descent as the example dataset is relatively small. To see the list of supported parameters, run `bin/RunNetsne --help`. To run t-SNE for comparison, run the following:
`bin/RunBhtsne --input-P example_data/pollen_P.dat --out-dir example_data/bhtsne`

5. *Plot visualization*. The following MATLAB code visualizes the final embeddings and color the points according to the known cluster labels in `pollen_labels.txt`:
`Y = dlmread('example_data/netsne/Y_final.txt', '', 2, 0);`
`labels = dlmread('example_data/pollen_labels.txt');`
`scatter(Y(:,1), Y(:,2), 10, labels, 'filled')`

##### Notes
We are working on releasing wrappers in other programming languages such as R and Python. If you would like an update when these become available, feel free to drop us a note at the email address below.

##### Contact for Questions
Hoon Cho, hhcho@mit.edu
