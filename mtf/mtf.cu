#include <stdio.h>
#include <vector>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>


const int ALPHABET_SIZE = 256;
const int WARP_SIZE = 32;
typedef unsigned char byte;

// Parameters
const int NUM_WARPS = 4;
const int BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
typedef unsigned MTF_WORD;
#define SYNC_WARP __syncthreads  /* alternatively, __syncthreads or, better, __threadfence_warp */


__global__ void mtf (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
//printf("%d ", warp_id);

    inbuf  += idx*chunk*2;
    outbuf += idx*chunk*2;

    auto inbuf1  = inbuf,  inbuf2  = inbuf+chunk; 
    auto outbuf1 = outbuf, outbuf2 = outbuf+chunk; 

//    __shared__  byte in[128], out[128];
    __shared__  MTF_WORD mtf0 [ALPHABET_SIZE*NUM_WARPS*2];
    auto mtf1 = mtf0 + ALPHABET_SIZE*warp_id*2;
    auto mtf2 = mtf1 + ALPHABET_SIZE;
    for (int i=0; i<ALPHABET_SIZE; i+=WARP_SIZE)
    {
        mtf1[i+tid] = i+tid;
        mtf2[i+tid] = i+tid;
    }
    SYNC_WARP();


    for (int i=0; ; i++)
    {    
        unsigned n1, n2;
        auto next1 = inbuf1[i];
        auto next2 = inbuf2[i];
        #pragma unroll 4
        for ( ; i<CHUNK-1; i++)
        {
            auto cur1 = next1;
            auto old1 = mtf1[tid];
            next1 = inbuf1[i+1];

            auto cur2 = next2;
            auto old2 = mtf2[tid];
            next2 = inbuf2[i+1];

            n1 = __ballot (cur1==old1);
            n2 = __ballot (cur2==old2);
            if (n1==0 || n2==0)  goto deeper;
    
            auto minbit = __ffs(n1) - 1;
            if (tid < minbit)  mtf1[tid+1] = old1;
            if (tid==0)        outbuf1[i] = minbit;
            mtf1[0] = cur1;          
    
            minbit = __ffs(n2) - 1;
            if (tid < minbit)  mtf2[tid+1] = old2;
            if (tid==0)        outbuf2[i] = minbit;
            mtf2[0] = cur2;          
            SYNC_WARP();
        }
        return;

    deeper:
        {
            auto cur = next1;
            auto old = mtf1[tid];

            int k;  unsigned n;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                n = __ballot (cur==old);
                if (n) break;
                auto next = mtf1[k+WARP_SIZE+tid];
                mtf1[k+tid+1] = old;
                old = next;
                SYNC_WARP();
            }
            
            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf1[k+tid+1] = old;
            if (tid==0)        outbuf1[i] = k+minbit;
            mtf1[0] = cur;
        }

        {
            auto cur = next2;
            auto old = mtf2[tid];

            int k;  unsigned n;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                n = __ballot (cur==old);
                if (n) break;
                auto next = mtf2[k+WARP_SIZE+tid];
                mtf2[k+tid+1] = old;
                old = next;
                SYNC_WARP();
            }
            
            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf2[k+tid+1] = old;
            if (tid==0)        outbuf2[i] = k+minbit;
            mtf2[0] = cur;
            SYNC_WARP();
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

        mtf <<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, 32*NUM_WARPS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);
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

    // printf("rle: %.0lf => %.0lf\n", insize, outsize);
    printf("%.6lf ms\n", duration);
    printf("%.6lf MiB/s\n", ((1000.0f/duration) * insize) / (1 << 20));
    fclose(infile);
    fclose(outfile);
    cudaProfilerStop();
    return 0;
}
