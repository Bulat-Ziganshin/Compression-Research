// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  typename MTF_WORD = unsigned,  int MTF_SYMBOLS = WARP_SIZE>
__global__ void mtf_2buffers_depth32 (const byte* __restrict__ _inbuf,  byte* __restrict__ _outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (idx*CHUNK*2 >= inbytes)  return;

    const byte* inbuf[2];
    inbuf[0]  = _inbuf + idx*chunk*2;
    inbuf[1]  = inbuf[0] + chunk;

    byte* outbuf[2];
    outbuf[0]  = _outbuf + idx*chunk*2;
    outbuf[1]  = outbuf[0] + chunk;

    __shared__  MTF_WORD mtf0 [(MTF_SYMBOLS+1)*NUM_WARPS*2];
    MTF_WORD* mtf[2];
    mtf[0] = mtf0 + MTF_SYMBOLS*warp_id*2;
    mtf[1] = mtf[0] + MTF_SYMBOLS;
    for (int i=0; i<MTF_SYMBOLS; i+=WARP_SIZE)
    {
        mtf[0][i+tid] = i+tid;
        mtf[1][i+tid] = i+tid;
    }
    SYNC_WARP();


    byte next[] = {*inbuf[0]++, *inbuf[1]++};
    #pragma unroll 16
    for (int i=0; i<CHUNK; i++)
    {
        #pragma unroll
        for (int k=0; k<2; k++)
        {
            auto cur = next[k];
            auto old = mtf[k][tid];
            unsigned n = __ballot (cur==old);
            next[k] = *inbuf[k]++;
            mtf[k][0] = cur;

            auto minbit = __ffs(n) - 1;
            if (minbit < 0)    minbit = MTF_SYMBOLS;
            if (tid < minbit)  mtf[k][tid+1] = old;
            *outbuf[k]++ = minbit;
        }
        SYNC_WARP();
    }
}
