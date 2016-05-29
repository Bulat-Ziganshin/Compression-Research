// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  typename MTF_WORD = unsigned>
__global__ void mtf_scalar (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (idx*CHUNK >= inbytes)  return;
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

            unsigned n = __ballot (cur==old);
            if (n==0)  goto deeper;
            next = inbuf[i+1];

            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[tid+1] = old;
            *outbuf++ = minbit;
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
            for (k=WARP_SIZE; k<ALPHABET_SIZE; k+=WARP_SIZE)
            {
                auto next = mtf[k+tid];
                mtf[k+tid+1-WARP_SIZE] = old;
                old = next;

                n = __ballot (cur==old);
                if (n) break;
                __syncthreads();
            }

            auto minbit = __ffs(n) - 1;
            if (tid < minbit)  mtf[k+tid+1] = old;
            *outbuf++ = k+minbit;
            mtf[0] = cur;
            __syncthreads();
        }
    }
}
