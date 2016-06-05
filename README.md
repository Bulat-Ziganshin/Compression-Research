This repository contains my experiments with compression-related algorithms.

### [BSL: the block-sorting lab](app_bsl)
This's the main executable processing input data with various algos and recording speed/outsize of every experiment.
[results.txt](app_bsl/results.txt) presents the current results on my GPU.
[profile.txt](app_bsl/profile.txt) is a profiling report of CUDA kernels implemented by BSL.

### [LZP](algo_lzp)
Various algorithms performing LZP preprocessing on CPU.

### [ST](algo_st)
BSC implementations of Sort Transform on CPU & GPU.

### [BWT](algo_bwt)
OpenBWT 2.0 implementation of BWT on CPU.

### [MTF](algo_mtf)
Various algorithms computing MTF transform on CPU & GPU.
