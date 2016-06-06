Benchmark CUB radix sort with various parameters.
```
Usage: radix_sort [N] [full]
  where N is the number [of millions] of elements to test
        "full" enables benchmarking of 8/16-bit elements which on my GPU shows the same speed as 32-bit ones
```

### Current x86 results with CUDA 7.5 and CUB 1.5.2

(x64 version is a few percents slower due to need to manage larger pointers)

```
GeForce GTX 560 Ti, CC 2.1.  VRAM 1.0 GB, 2004 MHz * 256-bit = 128 GB/s.  8 SM * 48 alu * 1800 MHz * 2 = 1.38 TFLOPS
Sorting 16M elements:
1/4  : Throughput = 3630.096 MElements/s, Time = 4.622 ms
2/4  : Throughput = 1807.721 MElements/s, Time = 9.281 ms
3/4  : Throughput = 1325.778 MElements/s, Time = 12.655 ms
4/4  : Throughput =  941.682 MElements/s, Time = 17.816 ms

1/8  : Throughput = 2033.248 MElements/s, Time = 8.251 ms
2/8  : Throughput = 1013.995 MElements/s, Time = 16.546 ms
3/8  : Throughput =  729.117 MElements/s, Time = 23.010 ms
4/8  : Throughput =  525.525 MElements/s, Time = 31.925 ms
5/8  : Throughput =  442.132 MElements/s, Time = 37.946 ms
6/8  : Throughput =  361.177 MElements/s, Time = 46.452 ms
7/8  : Throughput =  305.574 MElements/s, Time = 54.904 ms
8/8  : Throughput =  271.861 MElements/s, Time = 61.712 ms

1/4+4: Throughput = 2345.812 MElements/s, Time = 7.152 ms
2/4+4: Throughput = 1173.353 MElements/s, Time = 14.299 ms
3/4+4: Throughput =  874.986 MElements/s, Time = 19.174 ms
4/4+4: Throughput =  609.576 MElements/s, Time = 27.523 ms

1/4+8: Throughput = 1737.907 MElements/s, Time = 9.654 ms
2/4+8: Throughput =  869.434 MElements/s, Time = 19.297 ms
3/4+8: Throughput =  577.189 MElements/s, Time = 29.067 ms
4/4+8: Throughput =  428.730 MElements/s, Time = 39.132 ms

1/8+4: Throughput = 1483.201 MElements/s, Time = 11.311 ms
2/8+4: Throughput =  743.381 MElements/s, Time = 22.569 ms
3/8+4: Throughput =  517.357 MElements/s, Time = 32.429 ms
4/8+4: Throughput =  378.284 MElements/s, Time = 44.351 ms
5/8+4: Throughput =  312.690 MElements/s, Time = 53.654 ms
6/8+4: Throughput =  258.132 MElements/s, Time = 64.995 ms
7/8+4: Throughput =  220.360 MElements/s, Time = 76.135 ms
8/8+4: Throughput =  194.696 MElements/s, Time = 86.171 ms

1/8+8: Throughput = 1261.976 MElements/s, Time = 13.294 ms
2/8+8: Throughput =  630.888 MElements/s, Time = 26.593 ms
3/8+8: Throughput =  421.866 MElements/s, Time = 39.769 ms
4/8+8: Throughput =  315.385 MElements/s, Time = 53.196 ms
5/8+8: Throughput =  256.326 MElements/s, Time = 65.453 ms
6/8+8: Throughput =  213.615 MElements/s, Time = 78.539 ms
7/8+8: Throughput =  183.989 MElements/s, Time = 91.186 ms
8/8+8: Throughput =  161.552 MElements/s, Time = 103.851 ms
```
