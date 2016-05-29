This repository contains my experiments with compression-related algorithms.

### [BSL: the block-sorting lab](bsl.cu)
That's the main executable processing input data with various algos and recording speed/outsize of every experiment.
[results.txt](results.txt) presents the current results on my GPU.
[profile.txt](profile.txt) is a profiling report of CUDA kernels implemented by BSL.

### [LZP](lzp)
Various algorithms performing LZP preprocessing on CPU.

### [BWT](bwt)
OpenBWT 2.0 implementation of BWT transformation on CPU.

### [MTF](mtf)
Various algorithms computing MTF transformation on CPUs and GPUs.
