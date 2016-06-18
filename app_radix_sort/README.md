Benchmark CUB radix sort with various parameters.
```
Usage: radix_sort [N] [full]
  where N is the number [of millions] of elements to test
        "full" enables benchmarking of 8/16-bit elements which on my GPU shows the same speed as 32-bit ones
```

The program displays perfromance of radix sort for rather large arrays (16M elements by default).
First column has the format "N/K" for sorting K-byte keys by N bytes.
It has format "N/K+V" for sorting with extra V-byte values attached to the keys.


### x64 results with CUDA 8.0RC and CUB 1.5.2

(x64 version is a few percents slower than x86 due to need to manage larger pointers). 
For full results see [results.txt](results.txt).

```
GeForce GTX 560 Ti, CC 2.1.  VRAM 1.0 GB, 2004 MHz * 256-bit = 128 GB/s.  8 SM * 48 alu * 1800 MHz * 2 = 1.38 TFLOPS
Sorting 16M elements:
1/4  : Throughput = 3532.966 MElements/s, Time = 4.749 ms
2/4  : Throughput = 1765.983 MElements/s, Time = 9.500 ms
3/4  : Throughput = 1298.415 MElements/s, Time = 12.921 ms
4/4  : Throughput =  921.279 MElements/s, Time = 18.211 ms

1/8  : Throughput = 1976.709 MElements/s, Time = 8.487 ms
2/8  : Throughput =  988.398 MElements/s, Time = 16.974 ms
3/8  : Throughput =  715.334 MElements/s, Time = 23.454 ms
4/8  : Throughput =  515.346 MElements/s, Time = 32.555 ms
5/8  : Throughput =  434.126 MElements/s, Time = 38.646 ms
6/8  : Throughput =  354.241 MElements/s, Time = 47.361 ms
7/8  : Throughput =  299.286 MElements/s, Time = 56.057 ms
8/8  : Throughput =  266.688 MElements/s, Time = 62.909 ms

1/4+4: Throughput = 2346.395 MElements/s, Time = 7.150 ms
2/4+4: Throughput = 1170.342 MElements/s, Time = 14.335 ms
3/4+4: Throughput =  868.724 MElements/s, Time = 19.312 ms
4/4+4: Throughput =  606.442 MElements/s, Time = 27.665 ms

1/4+8: Throughput = 1731.703 MElements/s, Time = 9.688 ms
2/4+8: Throughput =  868.007 MElements/s, Time = 19.328 ms
3/4+8: Throughput =  574.224 MElements/s, Time = 29.217 ms
4/4+8: Throughput =  425.807 MElements/s, Time = 39.401 ms

1/8+4: Throughput = 1447.258 MElements/s, Time = 11.592 ms
2/8+4: Throughput =  725.238 MElements/s, Time = 23.133 ms
3/8+4: Throughput =  506.463 MElements/s, Time = 33.126 ms
4/8+4: Throughput =  370.368 MElements/s, Time = 45.299 ms
5/8+4: Throughput =  306.365 MElements/s, Time = 54.762 ms
6/8+4: Throughput =  252.914 MElements/s, Time = 66.336 ms
7/8+4: Throughput =  215.716 MElements/s, Time = 77.774 ms
8/8+4: Throughput =  190.831 MElements/s, Time = 87.917 ms

1/8+8: Throughput = 1255.114 MElements/s, Time = 13.367 ms
2/8+8: Throughput =  627.455 MElements/s, Time = 26.739 ms
3/8+8: Throughput =  418.790 MElements/s, Time = 40.061 ms
4/8+8: Throughput =  312.513 MElements/s, Time = 53.685 ms
5/8+8: Throughput =  254.506 MElements/s, Time = 65.921 ms
6/8+8: Throughput =  212.192 MElements/s, Time = 79.066 ms
7/8+8: Throughput =  183.198 MElements/s, Time = 91.579 ms
8/8+8: Throughput =  160.342 MElements/s, Time = 104.634 ms
```
