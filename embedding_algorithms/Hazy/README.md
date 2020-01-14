# HazyTensor

[![Build Status](https://travis-ci.com/HazyResearch/Hazy.svg?branch=master)](https://travis-ci.com/HazyResearch/Hazy)

An embedding engine for fast, stable embeddings.

# Requirements

- Python 3
- CMake 3.6 or higher
- A modern compiler with C++14 support
- BLAS
- LAPACKE

An example command to install the `BLAS` and `LAPACKE` libraries on Ubuntu is below:

```
$ sudo apt-get install libblas-dev liblapack-dev liblapacke-dev libopenblas-dev libatlas-base-dev
```

# Installation

To build and install `hazytensor`, clone or download this repository:

```bash
$ git clone --recursive https://github.com/HazyResearch/Hazy.git && cd Hazy
```

For best performance, we recommend using the C++ version of the code. For convenience, we have limited support for a Python version as well (via pybind).

## C++ version

To install only the C++ version of the code:

```bash
$ mkdir build && cd build && cmake ..
$ make && cd ..
$ ./examples/demo.sh
```

To see run options for the embedding engine, run:
```bash
$ ./build/bin/embedding
```
In particular, the embedding engine supports multiple solvers, including simultaneous power iteration, power deflation, and SGD.

## Python version

### Conda

First make sure that [conda](https://conda.io/docs/user-guide/install/index.html) and [conda-build](https://github.com/conda/conda-build) are installed.

Then, from within the repository, run:

```bash
$ conda build conda.recipe
$ conda install --use-local hazy
```

Please note that this package is only compatible with Python3.X. If your conda install defaults to building Python2.X packages, you will need to specify the Python 3 version you want in the build command.

```
$ conda build conda.recipe --python 3.X
```

### Pip
Then, from within the repository, run:

```bash
$ pip3 install .
```

### Notes on Installing without Root Privileges

You must set the `HAZY_C_PREFIX` environment variable to a local folder. This folder will hold the backend's shared libary objects, so this folder must be added to your `LD_LIBRARY_PATH` (linux) or `DYLD_LIBRARY_PATH` (mac).  By default we set this location to `/usr/lib/` (linux) or `/usr/local/lib` (mac) which usually require root privileges to write to.

An example of this that works on a Ubuntu machine:

```
$ source local_build_env.sh # Sets HAZY_C_PREFIX and LD_LIBRARY_PATH
$ pip3 install . --user
```

# Tests

To execute all unit tests (currently supported by Python version only), you will need to download the external test data:

```bash
$ cd eval/intrinsic/
$ bash download_data.sh
$ cd ../..
```

Then, you’ll be able to run our tests:

```bash
$ python3 ./setup.py test
```

# Acknowledgements

Much of the code in this repository was adapted from the [`pbind11cpp` example](https://github.com/benjaminjack/python_cpp_example), [`pybind11` tutorial](http://pybind11.readthedocs.io/en/stable/basics.html) and the [`pybind11` example CMake repository](https://github.com/pybind/cmake_example).
