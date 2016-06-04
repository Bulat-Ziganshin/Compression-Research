
This directory contains my experiments on fast LZP coder. Current results on enwik8 with i7-4770 (single thread):
```
lzp_cpu_bsc     : 100,000,000 => 95,006,102 (95.01%)  341 MiB/s,  280.002 ms
lzp_cpu_bsc_mod : 100,000,000 => 95,012,381 (95.01%)  361 MiB/s,  263.985 ms
lzp_cpu_rollhash: 100,000,000 => 95,025,330 (95.03%)  379 MiB/s,  251.672 ms
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
