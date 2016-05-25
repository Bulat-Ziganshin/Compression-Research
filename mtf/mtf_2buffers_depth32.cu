// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  typename MTF_WORD = unsigned,  int MTF_SYMBOLS = WARP_SIZE>
__global__ void mtf_2buffers_depth32 (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (idx*CHUNK*2 >= inbytes)  return;
    inbuf  += idx*chunk*2;
    outbuf += idx*chunk*2;

    auto inbuf1  = inbuf,  inbuf2  = inbuf+chunk;
    auto outbuf1 = outbuf, outbuf2 = outbuf+chunk;

    __shared__  MTF_WORD mtf0 [(MTF_SYMBOLS+1)*NUM_WARPS*2];
    auto mtf1 = mtf0 + MTF_SYMBOLS*warp_id*2;
    auto mtf2 = mtf1 + MTF_SYMBOLS;
    for (int i=0; i<MTF_SYMBOLS; i+=WARP_SIZE)
    {
        mtf1[i+tid] = i+tid;
        mtf2[i+tid] = i+tid;
    }
    SYNC_WARP();


    auto next1 = inbuf1[0];
    auto next2 = inbuf2[0];
    #pragma unroll 4
    for (int i=0; i<CHUNK; i++)
    {
        auto cur1 = next1;
        auto old1 = mtf1[tid];

        auto cur2 = next2;
        auto old2 = mtf2[tid];

        unsigned n1 = __ballot (cur1==old1);
        unsigned n2 = __ballot (cur2==old2);

        next1 = inbuf1[i+1];
        next2 = inbuf2[i+1];

        auto minbit = __ffs(n1) - 1;
        if (minbit < 0)    minbit = MTF_SYMBOLS;
        if (tid < minbit)  mtf1[tid+1] = old1;
        if (tid==0)        outbuf1[i] = minbit;
        mtf1[0] = cur1;

        minbit = __ffs(n2) - 1;
        if (minbit < 0)    minbit = MTF_SYMBOLS;
        if (tid < minbit)  mtf2[tid+1] = old2;
        if (tid==0)        outbuf2[i] = minbit;
        mtf2[0] = cur2;
        SYNC_WARP();
    }
}
