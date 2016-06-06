// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

#include <stdio.h>
#include <vector>
#include <functional>
#include <stdint.h>

#include <helper_functions.h>          // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>               // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>

#include <cub/cub.cuh>

#include "../util/cuda_common.h"       // my own cuda-specific helper functions
#include "../util/wall_clock_timer.h"  // StartTimer() and GetTimer()
#include "../util/cpu_common.h"        // my own helper functions
#include "../util/libbsc.h"            // BSC common definitions
#include "../util/lz4_common.h"        // Utility functions from LZ4


// Parameters
const int defaultNumElements = 32*1024*1024;
double MIN_BENCH_TIME = 0.5;  // mimimum seconds to run each bechmark


template <typename T>
__global__ void fill_with_random (T *d_array, uint32_t size)
{
    const uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= size)  return;

    uint32_t rnd = idx*1234567891u;
    rnd = 29943829*rnd + 1013904223;    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    rnd = 29943829*rnd + 1013904223;
    uint64_t rnd1 = rnd;
    rnd = 29943829*rnd + 1013904223;
    rnd = 29943829*rnd + 1013904223;
    d_array[idx] = T(rnd1<<32) + rnd;
}


//template <int sort_bytes, typename Key, typename Value>

template <typename Key>
double generic_sort (int SORT_BYTES, size_t n, void *d_array0, cudaEvent_t &start, cudaEvent_t &stop)
{
    // Bit subrange [begin_bit, end_bit) of differentiating key bits
    int begin_bit = 0,  end_bit = SORT_BYTES*8;

    auto d_array = (Key*) d_array0;

    // Create a DoubleBuffer to wrap the pair of device pointers
    cub::DoubleBuffer<Key> d_keys (d_array, d_array + n);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    checkCudaErrors( cub::DeviceRadixSort::SortKeys (d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));

    // Allocate temporary storage
    checkCudaErrors( cudaMalloc (&d_temp_storage, temp_storage_bytes));

    int numIterations = 0;
    double totalTime = 0;

    for ( ; totalTime < MIN_BENCH_TIME; numIterations++)
    {
        // Fill source buffer with random numbers
        fill_with_random<Key> <<< n/1024+1, 1024 >>> (d_array, n);
        checkCudaErrors( cudaDeviceSynchronize());

        checkCudaErrors( cudaEventRecord (start, nullptr));

        // Run sorting operation
        checkCudaErrors( cub::DeviceRadixSort::SortKeys (d_temp_storage, temp_storage_bytes, d_keys, n, begin_bit, end_bit));

        // Record time
        checkCudaErrors( cudaEventRecord (stop, nullptr));
        checkCudaErrors( cudaDeviceSynchronize());
        float start_stop;
        checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
        totalTime += start_stop/1000; // converts milliseconds to seconds
    }

    // Release temporary storage
    checkCudaErrors( cudaFree (d_temp_storage));

    return totalTime/numIterations;
}


int main (int argc, char **argv)
{
    int numElements = defaultNumElements;

    if (argc > 1) {
        char* endptr;
        numElements = strtol (argv[1], &endptr, 10) << 20;
        if (argc!=2 || *endptr) {
            printf ("radix_sort: benchmark CUB Radix Sort with various parameters.  Part of https://github.com/Bulat-Ziganshin/Compression-Research\n");
            printf ("  Usage: radix_sort [N]  where N is the number of millions of elements to test\n");
            return 1;
        }
    }

    DisplayCudaDevice();

    void* d_array;
    checkCudaErrors( cudaMalloc(&d_array, 2*numElements*sizeof(uint64_t)));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    auto print = [&] (int bytes, int keysize, double totalTime) {
        printf("%d of %d: Throughput = %.4lf MElements/s, Time = %.5lf s\n",
               bytes, keysize, 1e-6 * numElements / totalTime, totalTime);
    };

    printf("Sorting %dM elements:\n", numElements>>20);
    for(int i=1;i<=4;i++)  print (i, 4, generic_sort <uint32_t> (i, numElements, d_array, start, stop));  printf("\n");
    for(int i=1;i<=8;i++)  print (i, 8, generic_sort <uint64_t> (i, numElements, d_array, start, stop));
    return 0;
}
