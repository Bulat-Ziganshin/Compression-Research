#include <stdio.h>
#include <vector>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>


const int ALPHABET_SIZE = 256;
const int WARP_SIZE = 32;

const int NUM_WARPS = 8;
const int BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
typedef unsigned char byte;


__global__ void mtf (const byte* inbuf, byte* outbuf, int n, int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
//printf("%d ", warp_id);

    inbuf  += idx*chunk;
    outbuf += idx*chunk;

//    __shared__  byte in[128], out[128];
    __shared__  unsigned mtf0 [ALPHABET_SIZE*NUM_WARPS];
    auto mtf = mtf0 + ALPHABET_SIZE*warp_id;
    for (int i=0; i<ALPHABET_SIZE; i+=WARP_SIZE)
    {
        mtf[i+tid] = i+tid;
    }


    for (int i=0; i<chunk; i++)
    {
        #pragma unroll 1
        for( ; ; i++)
        {
            if (i>=chunk) return;
            auto cur = inbuf[i];
            auto old = mtf[tid];
            unsigned n = __ballot (cur==old);
            if (n==0)  break;

            int k = 0;
            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[k+tid+1] = old;
            outbuf[i] = k+minbit;
            mtf[0] = cur;
        }

        {
            auto cur = inbuf[i];
            auto old = mtf[tid];
            int k;  unsigned n;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                n = __ballot (cur==old);
                if (n)  break;
                auto next = mtf[k+WARP_SIZE+tid];
                mtf[k+tid+1] = old;
                old = next;
            }
            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[k+tid+1] = old;
            outbuf[i] = k+minbit;
            mtf[0] = cur;
        }
    }
}


int main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: mtf infile [outfile]\n");
        return 0;
    }

    unsigned char* d_inbuf;
    unsigned char* d_outbuf;
    checkCudaErrors( cudaMalloc((void**)(&d_inbuf),  BUFSIZE));
    checkCudaErrors( cudaMalloc((void**)(&d_outbuf), BUFSIZE));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    unsigned char* inbuf  = new unsigned char[BUFSIZE];
    unsigned char* outbuf = new unsigned char[BUFSIZE];
    double insize = 0,  outsize = 0,  duration = 0;

    FILE* infile  = fopen (argv[1], "rb");
    FILE* outfile = fopen (argv[2]? argv[2] : "nul", "wb");

    for (int inbytes; !!(inbytes = fread(inbuf,1,BUFSIZE,infile)); )
    {
        checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
        checkCudaErrors( cudaDeviceSynchronize());
        checkCudaErrors( cudaEventRecord (start, nullptr));

        mtf <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1, 32*NUM_WARPS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);
        checkCudaErrors( cudaEventRecord (stop, nullptr));
        checkCudaErrors( cudaDeviceSynchronize());

        float start_stop;
        checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
        duration += start_stop;

        checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
        checkCudaErrors( cudaDeviceSynchronize());

        auto ptr = outbuf;
        auto outbytes = outbuf+inbytes - ptr;
        fwrite (ptr, 1, outbytes, outfile);
        insize  += inbytes;
        outsize += outbytes;
    }

    printf("rle: %.0lf => %.0lf\n", insize, outsize);
    printf("%.6lf ms\n", duration);
    printf("%.6lf MiB/s\n", ((1000.0f/duration) * insize) / (1 << 20));
    fclose(infile);
    fclose(outfile);
    cudaProfilerStop();
    return 0;
}
