Benchmark CUB radix sort with various parameters.
```
Usage: radix_sort [N] [full]
  where N is the number [of millions] of elements to test
        "full" enables all benchmarks
```

### Current results with CUDA 7.5 and CUB 1.5.2
```
GeForce GTX 560 Ti, CC 2.1.  VRAM 1.0 GB, 2004 MHz * 256-bit = 128 GB/s.  8 SM * 48 alu * 1800 MHz * 2 = 1.38 TFLOPS
Sorting 16M elements:
1/4+0: Throughput = 3487.920 MElements/s, Time = 4.810 ms
2/4+0: Throughput = 1745.233 MElements/s, Time = 9.613 ms
3/4+0: Throughput = 1200.088 MElements/s, Time = 13.980 ms
4/4+0: Throughput =  901.352 MElements/s, Time = 18.613 ms

1/8+0: Throughput = 1974.093 MElements/s, Time = 8.499 ms
2/8+0: Throughput =  987.452 MElements/s, Time = 16.990 ms
3/8+0: Throughput =  713.954 MElements/s, Time = 23.499 ms
4/8+0: Throughput =  512.924 MElements/s, Time = 32.709 ms
5/8+0: Throughput =  432.296 MElements/s, Time = 38.810 ms
6/8+0: Throughput =  353.732 MElements/s, Time = 47.429 ms
7/8+0: Throughput =  298.917 MElements/s, Time = 56.127 ms
8/8+0: Throughput =  266.235 MElements/s, Time = 63.017 ms

1/4+4: Throughput = 2354.350 MElements/s, Time = 7.126 ms
2/4+4: Throughput = 1174.615 MElements/s, Time = 14.283 ms
3/4+4: Throughput =  864.405 MElements/s, Time = 19.409 ms
4/4+4: Throughput =  603.807 MElements/s, Time = 27.786 ms

1/4+8: Throughput = 1678.648 MElements/s, Time = 9.994 ms
2/4+8: Throughput =  832.958 MElements/s, Time = 20.142 ms
3/4+8: Throughput =  571.701 MElements/s, Time = 29.346 ms
4/4+8: Throughput =  419.946 MElements/s, Time = 39.951 ms

1/8+4: Throughput = 1446.440 MElements/s, Time = 11.599 ms
2/8+4: Throughput =  723.684 MElements/s, Time = 23.183 ms
3/8+4: Throughput =  501.736 MElements/s, Time = 33.438 ms
4/8+4: Throughput =  368.421 MElements/s, Time = 45.538 ms
5/8+4: Throughput =  305.185 MElements/s, Time = 54.974 ms
6/8+4: Throughput =  251.235 MElements/s, Time = 66.779 ms
7/8+4: Throughput =  214.403 MElements/s, Time = 78.251 ms
8/8+4: Throughput =  189.606 MElements/s, Time = 88.485 ms

1/8+8: Throughput = 1250.835 MElements/s, Time = 13.413 ms
2/8+8: Throughput =  623.788 MElements/s, Time = 26.896 ms
3/8+8: Throughput =  416.163 MElements/s, Time = 40.314 ms
4/8+8: Throughput =  310.980 MElements/s, Time = 53.950 ms
5/8+8: Throughput =  253.448 MElements/s, Time = 66.196 ms
6/8+8: Throughput =  210.652 MElements/s, Time = 79.644 ms
7/8+8: Throughput =  181.395 MElements/s, Time = 92.490 ms
8/8+8: Throughput =  159.199 MElements/s, Time = 105.385 ms

Elapsed time = 20.352 seconds
```
