[LZP]:   algo_lzp
[ST]:    algo_st
[BWT]:   algo_bwt
[MTF]:   algo_mtf
[EC]:    algo_ec

### [BSLab: the block-sorting lab](app_bslab)
This's the main executable that combines all the stages of typical block-sorting compressor together
and applies them to user-supplied input data. It allows to examine the impact of alternative algorithms
on each stage on the overall compression speed/ratio.

### [radix_sort: benchmarking the CUB radix sort](app_radix_sort)
This application benchmarks the CUB radix sort implementation with all possible combinations
of key/value lengths and allow user to select size of the array to sort.


### [LZP]
Various algorithms performing LZP preprocessing on CPU.

### [ST]
BSC implementations of Sort Transform on CPU & GPU.

### [BWT]
DivSufSort and OpenBWT 2.0 implementations of BWT on CPU.

### [MTF]
Various algorithms computing MTF transform on CPU & GPU.

### [EC]
Variants of EC (entropy coding) stage (not yet implemented).
