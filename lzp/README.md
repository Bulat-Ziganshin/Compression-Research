
This directory contains my experiments on fast LZP coder. Current results on enwik9 with i7-4770:
```
lzp_cpu_bsc              : 1,000,000,000 => 855,183,966 (85.52%)  314 MiB/s,  3033.297 ms
lzp_cpu_bsc_mod          : 1,000,000,000 => 855,168,159 (85.52%)  384 MiB/s,  2483.152 ms
lzp_cpu_rollhash         : 1,000,000,000 => 855,369,315 (85.54%)  476 MiB/s,  2005.023 ms
lzp_cpu_rollhash (OpenMP):                                       1520 MiB/s,   627.267 ms
```

### [lzp-cpu-bsc.cpp](lzp-cpu-bsc.cpp)

Original BSC implementation.

### [lzp-cpu-bsc-mod.cpp](lzp-cpu-bsc-mod.cpp)

Original BSC implementation slightly optimized with low-level x86 tricks.

### [lzp-cpu-rollhash.cpp](lzp-cpu-rollhash.cpp)

Each hash-table entry stores (in addition to a pointer) checksum of minLen bytes it points to.
This allows to spend most of time inside inner loop - in >99% cases, when we are going out
of the inner loop, we have real match. The checksum saved is multiplicative rolling hash
of these bytes.

Plus the same x86 tricks.
