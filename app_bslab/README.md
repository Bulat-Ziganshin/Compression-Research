[LZP]:   ../algo_lzp
[ST]:    ../algo_st
[BWT]:   ../algo_bwt
[MTF]:   ../algo_mtf
[EC]:    ../algo_ec
[results-cpu.txt]:   (results-cpu.txt)
[results-cuda.txt]:  (results-cuda.txt)
[profile.txt]:       (profile.txt)


BSLab stands for the block-sorting laboratory.
It sequentially applies to input data all algorithms employed in real compressors such as BSC:
- [LZP] removes repeated strings
- [BWT] or [ST] sorts the data
- RLE squeezes repeating chars
- [MTF] converts local entropy (similar chars in similar contexts) into global entropy
- [EC] finally encodes the data (not implemented yet)

On every stage (except for RLE) we have a choice of algorithms, including those implemented in BSC as the baseline.
Output of the last algorithm completed on every stage (except for OpenMP LZP) goes as the input to the next stage.
Individual algorithms can be selected with options like -mtf1,3-4, or you may disable some algos with option like -mtf-1,3-4.
You can also completely disable some stages by optiions like -nolzp, and control LZP stage parameters with options -h and -l.
Blocksize is controlled by -b option, you may need to reduce it if program exits with memory allocation error.

We have tested the following compilers:
- Intel C++ Compiler 16.0
- Clang 3.8
- GCC 5.3
- Microsoft Visual C++ 2015 Update 1
- CUDA 8.0 RC with MSVC2015 Update 1

All compilers were tested in 32 and 64 bit versions, with SSE2 and with AVX2. 
You can find Windows executables in the release archive.


### Understanding the results

Once all data were processed, the program displays for each algorithm:
- compression ratio (on this stage and compared to original data)
- speed (on this stage and in terms of original data)
- worktime

For example, command `bslab-clang-x64-avx2.exe -lzp1 -bwt6 -mtf1,2 enwik8` produced the following output:
```
[ 1] lzp_cpu_bsc: 100,000,000 => 95,006,102 (95.01%)  307 MiB/s,  311.107 ms
[ 6] divsufsort:  10.3 / 9.76 MiB/s,  9286.634 ms
rle: 95,006,102 => 36,703,518 (38.63% / 36.70%)   >255: 15,171,  rank+len: 46,604,415,  1/2 encoding: 52,271,085
[ 1] mtf_cpu_bsc     :   181 / 66.5 MiB/s,  526.415 ms
[ 2] mtf_cpu_shelwien:   380 /  140 MiB/s,  250.876 ms
```
This means that data were compressed from 100,000,000 to 95,006,102 bytes at LZP stage, 
and then to 36,703,797 bytes at RLE stage, while BWT and MTF stages are 1:1 mappings.
36,703,797 bytes is 38.63% of RLE stage input (95,025,330 bytes) and 
36.70% of original data (100,000,000 bytes). Also, we see that 
- 15,164 run-length counts (out of total 36,703,797) produced by RLE stage were higher than 255
- 46,604,415 values represent ranks and lengths>1
- 52,271,085 values represent ranks and lengths in 1/2 encoding

In the speed department, we see that mtf_cpu_bsc algo was finished in 526.415 ms, that's 
66.5 MiB/s compared to its input (36,703,797 bytes) and
181 MiB/s compared to original data (100,000,000 bytes).
The alternative MTF algorithm, namely mtf_cpu_shelwien, was finished in just 250.876 ms.


### Full results

- [results-cpu.txt] are my CPU results
- [results-cuda.txt] are my CUDA GPU results
- [profile.txt] are my CUDA GPU profiling report (only MTF kernels are included)


### x64: enwik9 results on Haswell i7-4770
```
C:\app_bslab>bslab-icl-x64.exe enwik9
[ 1] lzp_cpu_bsc              : 1,000,000,000 => 855,183,966 (85.52%)  274 MiB/s,  3475.590 ms
[ 2] lzp_cpu_bsc_mod          : 1,000,000,000 => 855,168,159 (85.52%)  367 MiB/s,  2595.310 ms
[ 3] lzp_cpu_rollhash         : 1,000,000,000 => 855,369,315 (85.54%)  488 MiB/s,  1952.647 ms
[ 4] lzp_cpu_rollhash (OpenMP):  1501 MiB/s,  635.485 ms
[ 1] st3-cpu            :   206 /  177 MiB/s,  4620.462 ms
[ 2] st4-cpu            :   169 /  145 MiB/s,  5640.893 ms
[ 3] st5-cpu            :  66.5 / 56.9 MiB/s,  14338.477 ms
[ 4] st6-cpu            :  46.5 / 39.8 MiB/s,  20519.435 ms
[ 5] OpenBWT            :  6.61 / 5.65 MiB/s,  144384.098 ms
[ 6] divsufsort         :  11.1 / 9.51 MiB/s,  85773.096 ms
[ 7] divsufsort (OpenMP):  14.8 / 12.6 MiB/s,  64529.215 ms
rle: 855,369,315 => 317,196,734 (37.08% / 31.72%)   >255: 142,036
[ 1] mtf_cpu_bsc               :   186 / 58.8 MiB/s,  5140.753 ms
[ 2] mtf_cpu_shelwien          :   333 /  106 MiB/s,  2860.983 ms
[ 3] mtf_cpu_shelwien2         :   470 /  149 MiB/s,  2028.455 ms
[ 4] mtf_cpu_shelwien2 (OpenMP):  1349 /  428 MiB/s,  706.969 ms
```

### x86: enwik9 results on Haswell i7-4770
```
C:\app_bslab>bslab-icl.exe enwik9
[ 1] lzp_cpu_bsc              : 1,000,000,000 => 855,183,966 (85.52%)  247 MiB/s,  3860.045 ms
[ 2] lzp_cpu_bsc_mod          : 1,000,000,000 => 855,168,159 (85.52%)  341 MiB/s,  2797.712 ms
[ 3] lzp_cpu_rollhash         : 1,000,000,000 => 855,369,315 (85.54%)  363 MiB/s,  2629.060 ms
[ 4] lzp_cpu_rollhash (OpenMP):  1086 MiB/s,  878.130 ms
[ 1] st3-cpu            :   179 /  153 MiB/s,  5328.586 ms
[ 2] st4-cpu            :   166 /  142 MiB/s,  5750.610 ms
[ 3] st5-cpu            :  66.5 / 56.9 MiB/s,  14347.955 ms
[ 4] st6-cpu            :  45.9 / 39.2 MiB/s,  20786.048 ms
[ 5] OpenBWT            :  6.33 / 5.42 MiB/s,  150542.874 ms
[ 6] divsufsort         :  10.1 / 8.64 MiB/s,  94372.929 ms
[ 7] divsufsort (OpenMP):  13.8 / 11.8 MiB/s,  68977.150 ms
rle: 855,369,315 => 317,196,734 (37.08% / 31.72%)   >255: 142,036
[ 1] mtf_cpu_bsc               :   188 / 59.7 MiB/s,  5069.979 ms
[ 2] mtf_cpu_shelwien          :   321 /  102 MiB/s,  2968.044 ms
[ 3] mtf_cpu_shelwien2         :   456 /  145 MiB/s,  2091.740 ms
[ 4] mtf_cpu_shelwien2 (OpenMP):  1744 /  553 MiB/s,  546.876 ms
```

### CUDA: enwik9 results on GeForce 560 Ti
```
C:\app_bslab>bslab-cuda-x64.exe enwik9 -b34
GeForce GTX 560 Ti, CC 2.1.  VRAM 1.0 GB, 2004 MHz * 256-bit = 128 GB/s.  8 SM * 48 alu * 1800 MHz * 2 = 1.38 TFLOPS
[ 1] lzp_cpu_bsc              : 1,000,000,000 => 855,252,235 (85.53%)  275 MiB/s,  3464.797 ms
[ 2] lzp_cpu_bsc_mod          : 1,000,000,000 => 855,231,977 (85.52%)  366 MiB/s,  2603.438 ms
[ 3] lzp_cpu_rollhash         : 1,000,000,000 => 855,435,379 (85.54%)  433 MiB/s,  2204.165 ms
[ 4] lzp_cpu_rollhash (OpenMP):  1295 MiB/s,  736.536 ms
[ 1] st5-cuda           :   198 /  169 MiB/s,  4828.331 ms
[ 2] st6-cuda           :   181 /  154 MiB/s,  5281.056 ms
[ 3] st7-cuda           :   166 /  142 MiB/s,  5730.117 ms
[ 4] st8-cuda           :   127 /  109 MiB/s,  7503.673 ms
[ 5] st3-cpu            :   210 /  179 MiB/s,  4548.743 ms
[ 6] st4-cpu            :   160 /  137 MiB/s,  5958.020 ms
[ 7] st5-cpu            :  67.0 / 57.3 MiB/s,  14233.920 ms
[ 8] st6-cpu            :  46.9 / 40.1 MiB/s,  20352.996 ms
[ 9] OpenBWT            :  7.64 / 6.53 MiB/s,  124882.666 ms
[10] divsufsort         :  13.0 / 11.1 MiB/s,  73225.877 ms
[11] divsufsort (OpenMP):  17.3 / 14.8 MiB/s,  55135.494 ms
rle: 855,435,379 => 329,950,014 (38.57% / 33.00%)   >255: 130,769
[ 1] mtf_cpu_bsc               :   198 / 65.3 MiB/s,  4821.351 ms
[ 2] mtf_cpu_shelwien          :   212 / 69.9 MiB/s,  4502.694 ms
[ 3] mtf_cpu_shelwien2         :   252 / 83.1 MiB/s,  3788.163 ms
[ 4] mtf_cpu_shelwien2 (OpenMP):   805 /  266 MiB/s,  1184.861 ms
[ 5] mtf_cuda_scalar           :  1485 /  490 MiB/s,  642.313 ms
[ 6] mtf_cuda_2symbols         :  1357 /  448 MiB/s,  702.967 ms
[ 7] mtf_cuda_2buffers         :  1621 /  535 MiB/s,  588.376 ms
[ 8] mtf_cuda_2buffers<32>     :  2231 /  736 MiB/s,  427.482 ms
[ 9] mtf_cuda_3buffers<32>     :  2173 /  717 MiB/s,  438.775 ms
[10] mtf_cuda_4buffers<32>     :  2007 /  662 MiB/s,  475.079 ms
[11] mtf_cuda_4by8             :  1668 /  550 MiB/s,  571.844 ms
[12] mtf_cuda_8by4             :  1330 /  439 MiB/s,  717.125 ms
[13] mtf_cuda_thread           :   640 /  211 MiB/s,  1489.478 ms
[14] mtf_cuda_thread_by4       :   902 /  298 MiB/s,  1056.907 ms
[15] mtf_cuda_thread<8>        :  1172 /  387 MiB/s,  813.592 ms
[16] mtf_cuda_thread<16>       :  1061 /  350 MiB/s,  899.120 ms
[17] mtf_cuda_thread<32>       :   910 /  300 MiB/s,  1048.134 ms
[18] mtf_cuda_thread<64>       :   799 /  264 MiB/s,  1193.687 ms
[19] mtf_cuda_thread_by4<8>    :  1116 /  368 MiB/s,  854.805 ms
[20] mtf_cuda_thread_by4<16>   :  1096 /  362 MiB/s,  869.754 ms
[21] mtf_cuda_thread_by4<32>   :  1017 /  336 MiB/s,  937.602 ms
[22] mtf_cuda_thread_by4<64>   :   950 /  314 MiB/s,  1003.599 ms
```
