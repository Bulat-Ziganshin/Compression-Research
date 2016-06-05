// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  typename MTF_WORD = unsigned>
__global__ void mtf_cuda_2symbols (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
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
    SYNC_WARP();


    for (int i=0; ; i+=2)
    {
        auto next1 = inbuf[i];
        auto next2 = inbuf[i+1];
        #pragma unroll 4
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
