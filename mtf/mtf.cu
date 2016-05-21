#include <stdio.h>
#include <vector>
#include <functional>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>


const int ALPHABET_SIZE = 256;
const int WARP_SIZE = 32;
typedef unsigned char byte;

// Parameters
const int NUM_WARPS = 6;
const int BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
#define SYNC_WARP __threadfence_block  /* alternatively, __syncthreads or, better, __threadfence_warp */



template <int CHUNK>
__global__ void mtf_thread (const byte* inbuf,  byte* outbuf,  int inbytes,  int chunk)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = idx % WARP_SIZE;

    inbuf  += idx*CHUNK;
    outbuf += idx*CHUNK;

    volatile __shared__  byte mtf0 [ALPHABET_SIZE*WARP_SIZE];
    auto mtf = mtf0 + 4*tid;
    for (int k=0; k<ALPHABET_SIZE; k++)
    {
        auto index = (k&252)*32+(k&3);
        mtf[index] = k;
    }


    int i = 0,  k = 0;
    auto cur  = *inbuf++;
    auto next = *inbuf++;
    auto old  = mtf[0];

    for(;;)
    {
        if (cur != old) {
            k++;
            auto index = (k&252)*32+(k&3);
            auto next = mtf[index];
            mtf[index] = old;
            old = next;
        } else {
            mtf[0] = cur;
            *outbuf++ = k;
            if (++i >= CHUNK)  return;
            old = cur;
            cur = next;
            next = *inbuf++;
            k = 0;
        }
    }
}



template <int CHUNK>
__global__ void mtf_thread_by4 (const byte* inbuf,  byte* outbuf,  int inbytes,  int chunk)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = idx % WARP_SIZE;

    inbuf  += idx*CHUNK;
    outbuf += idx*CHUNK;

    volatile __shared__  unsigned mtf0 [ALPHABET_SIZE*WARP_SIZE/4];
    auto mtf = mtf0 + tid;
    for (int k=0; k<ALPHABET_SIZE; k++)
    {
        auto index = (k&252)*32+(k&3);
        ((byte*)mtf)[index] = k;
    }


    int i = 0,  k = 0;
    auto cur  = *inbuf++;
    auto next = *inbuf++;
    auto mtf_k = mtf;
    auto old  = *mtf_k;
    unsigned sym = cur;

    for(;;)
    {
        auto old1 = old&255, old2 = (old>>8)&255, old3 = (old>>16)&255, old4 = (old>>24);
        int x = cur==old1? 0 :
                cur==old2? 1 :
                cur==old3? 2 :
                cur==old4? 3 : -1;
        if (x < 0) {
            *mtf_k = (old<<8) + sym;
            sym = old4;
            k+=4;  mtf_k += 32;
if (k>=256) {printf("!"); return;}
            old = *mtf_k;
        } else {
            *outbuf++ = k+x;
            if (++i >= CHUNK)  return;

            if (x >= 3)   ((byte*)mtf_k)[3] = old3;
            if (x >= 2)   ((byte*)mtf_k)[2] = old2;
            if (x >= 1)   ((byte*)mtf_k)[1] = old1;
                          ((byte*)mtf_k)[0] = sym;

            mtf_k = (unsigned*)mtf;
            old  = *mtf_k;

            cur = next;
            next = *inbuf++;
            k = 0;
        }
    }
}



template <int NUM_WARPS,  int CHUNK,  typename MTF_WORD = unsigned>
__global__ void mtf (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    inbuf  += idx*chunk;
    outbuf += idx*chunk;

    __shared__  MTF_WORD mtf0 [ALPHABET_SIZE*NUM_WARPS];
    auto mtf = mtf0 + ALPHABET_SIZE*warp_id;
    for (int i=0; i<ALPHABET_SIZE; i+=WARP_SIZE)
    {
        mtf[i+tid] = i+tid;
    }
    __syncthreads();


    for (int i=0; ; i++)
    {
        auto next = inbuf[i];
        #pragma unroll 4
        for ( ; i<CHUNK; i++)
        {
            auto cur = next;
            auto old = mtf[tid];
            next = inbuf[i+1];

            unsigned n = __ballot (cur==old);
            if (n==0)  goto deeper;

            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[tid+1] = old;
            if (tid==0)        outbuf[i] = minbit;
            mtf[0] = cur;
            __syncthreads();
        }
        return;

    deeper:
        {
            auto cur = next;
            auto old = mtf[tid];

            int k;  unsigned n;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                n = __ballot (cur==old);
                if (n) break;
                auto next = mtf[k+WARP_SIZE+tid];
                mtf[k+tid+1] = old;
                old = next;
                __syncthreads();
            }

            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[k+tid+1] = old;
            if (tid==0)        outbuf[i] = k+minbit;
            mtf[0] = cur;
            __syncthreads();
        }
    }
}



template <int NUM_WARPS,  int CHUNK,  typename MTF_WORD = unsigned>
__global__ void mtf_2symbols (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    inbuf  += idx*chunk;
    outbuf += idx*chunk;

    __shared__  MTF_WORD mtf0 [ALPHABET_SIZE*NUM_WARPS];
    auto mtf = mtf0 + ALPHABET_SIZE*warp_id;
    for (int i=0; i<ALPHABET_SIZE; i+=WARP_SIZE)
    {
        mtf[i+tid] = i+tid;
    }
    SYNC_WARP();


    for (int i=0; ; i+=2)
    {
        auto next1 = inbuf[i];
        auto next2 = inbuf[i+1];
        #pragma unroll 8
        for ( ; i<CHUNK; i+=2)
        {
            auto cur1 = next1;
            auto cur2 = next2;
            auto old = mtf[tid];
            next1 = inbuf[i+2];
            next2 = inbuf[i+3];

            unsigned n1 = __ballot (cur1==old);
            unsigned n2 = __ballot (cur2==old);
            if (n1==0 || n2==0)  goto deeper;

            auto minbit1 = __ffs(n1) - 1;
            if (tid < minbit1)  mtf[tid+1] = old;
            if (tid==0)         outbuf[i] = minbit1;
            mtf[0] = cur1;
            SYNC_WARP();

            auto minbit2 = __ffs(n2) - 1;
            if (minbit2 < minbit1)  minbit2++;     // the second symbol was shifted one more position down by the first one
            if (cur1==cur2)         minbit2 = 0;   // not required after RLE
            if (tid < minbit2)  mtf[tid+1] = mtf[tid];
            if (tid==0)         outbuf[i+1] = minbit2;
            mtf[0] = cur2;
            SYNC_WARP();
        }
        return;

    deeper:
        #pragma unroll
        for (int add=0; add<2; add++)
        {
            auto cur = inbuf[i+add];
            auto old = mtf[tid];

            int k;  unsigned n;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                n = __ballot (cur==old);
                if (n) break;
                auto next = mtf[k+WARP_SIZE+tid];
                mtf[k+tid+1] = old;
                old = next;
                SYNC_WARP();
            }

            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[k+tid+1] = old;
            if (tid==0)        outbuf[i+add] = k+minbit;
            mtf[0] = cur;
            SYNC_WARP();
        }
    }
}



template <int NUM_WARPS,  int CHUNK,  typename MTF_WORD = unsigned>
__global__ void mtf_2buffers (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
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
        for ( ; i<CHUNK; i++)
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
    checkCudaErrors( cudaMalloc((void**)(&d_inbuf),  BUFSIZE+CHUNK));
    checkCudaErrors( cudaMalloc((void**)(&d_outbuf), BUFSIZE+CHUNK));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    unsigned char* inbuf  = new unsigned char[BUFSIZE];
    unsigned char* outbuf = new unsigned char[BUFSIZE];
    double insize = 0,  outsize = 0,  duration[5] = {0};

    FILE* infile  = fopen (argv[1], "rb");
    FILE* outfile = fopen (argv[2]? argv[2] : "nul", "wb");

    for (int inbytes; !!(inbytes = fread(inbuf,1,BUFSIZE,infile)); )
    {
        checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
        checkCudaErrors( cudaDeviceSynchronize());

        auto time_run = [&] (std::function<void(void)> f) {
            checkCudaErrors( cudaEventRecord (start, nullptr));
            f();
            checkCudaErrors( cudaEventRecord (stop, nullptr));
            checkCudaErrors( cudaDeviceSynchronize());

            float start_stop;
            checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
            return start_stop;
        };

        duration[0]  +=  time_run ([&] {mtf           <NUM_WARPS,CHUNK> <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        duration[1]  +=  time_run ([&] {mtf_2symbols  <NUM_WARPS,CHUNK> <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        duration[2]  +=  time_run ([&] {mtf_2buffers  <NUM_WARPS,CHUNK> <<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        duration[3]  +=  time_run ([&] {mtf_thread    <CHUNK>           <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        duration[4]  +=  time_run ([&] {mtf_thread_by4<CHUNK>           <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

        checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
        checkCudaErrors( cudaDeviceSynchronize());

        auto ptr = outbuf;
        auto outbytes = outbuf+inbytes - ptr;
        fwrite (ptr, 1, outbytes, outfile);
        insize  += inbytes;
        outsize += outbytes;
    }

    // printf("rle: %.0lf => %.0lf\n", insize, outsize);
    char *mtf_name[] = {"scalar mtf", "2-symbol mtf", "2-buffer mtf", "thread mtf", "thread-by4 mtf"};
    for (int i=0; i<5; i++)
        if (duration[i])
            printf("%-12s:  %.6lf ms, %.6lf MiB/s\n", mtf_name[i], duration[i], ((1000.0f/duration[i]) * insize) / (1 << 20));
    fclose(infile);
    fclose(outfile);
    cudaProfilerStop();
    return 0;
}
