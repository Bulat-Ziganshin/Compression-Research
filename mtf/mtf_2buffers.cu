// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  typename MTF_WORD = unsigned>
__global__ void mtf_2buffers (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (idx*CHUNK*2 >= inbytes)  return;
    inbuf  += idx*chunk*2;
    outbuf += idx*chunk*2;

    auto inbuf1  = inbuf,  inbuf2  = inbuf+chunk;
    auto outbuf1 = outbuf, outbuf2 = outbuf+chunk;

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

            auto cur2 = next2;
            auto old2 = mtf2[tid];

            n1 = __ballot (cur1==old1);
            n2 = __ballot (cur2==old2);
            if (n1==0 || n2==0)  goto deeper;

            next1 = inbuf1[i+1];
            next2 = inbuf2[i+1];

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

            int k;  unsigned n = n1;
            #pragma unroll
            for (k=0; k<ALPHABET_SIZE-WARP_SIZE; k+=WARP_SIZE)
            {
                if (n) break;

                auto next = mtf1[k+tid+WARP_SIZE];
                mtf1[k+tid+1] = old;
                old = next;

                n = __ballot (cur==old);
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
