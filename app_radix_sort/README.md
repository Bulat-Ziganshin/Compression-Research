Benchmark CUB radix sort with various parameters.

### Current results with CUDA 7.5 and CUB 1.5.2
```
GeForce GTX 560 Ti, CC 2.1.  1.0 GB VRAM (2004 MHz * 256-bit = 128 GB/s).  8 SM * 48 alu * 1800 MHz = 691 GIOPS
Sorting 32M elements:
1 of 4: Throughput = 3529.1539 MElements/s, Time = 0.00951 s
2 of 4: Throughput = 1762.2191 MElements/s, Time = 0.01904 s
3 of 4: Throughput = 1248.6999 MElements/s, Time = 0.02687 s
4 of 4: Throughput =  922.8657 MElements/s, Time = 0.03636 s

1 of 8: Throughput = 1983.0368 MElements/s, Time = 0.01692 s
2 of 8: Throughput =  991.4804 MElements/s, Time = 0.03384 s
3 of 8: Throughput =  728.3311 MElements/s, Time = 0.04607 s
4 of 8: Throughput =  516.3670 MElements/s, Time = 0.06498 s
5 of 8: Throughput =  437.8928 MElements/s, Time = 0.07663 s
6 of 8: Throughput =  355.0213 MElements/s, Time = 0.09451 s
7 of 8: Throughput =  298.3419 MElements/s, Time = 0.11247 s
8 of 8: Throughput =  266.6209 MElements/s, Time = 0.12585 s
Elapsed time = 18.042 seconds
```
