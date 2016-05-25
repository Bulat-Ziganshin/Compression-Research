// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_WARPS,  int NUM_BUFFERS = 2,  typename MTF_WORD = unsigned,  int MTF_SYMBOLS = WARP_SIZE>
__global__ void mtf_2buffers_depth32 (const byte* __restrict__ _inbuf,  byte* __restrict__ _outbuf,  int inbytes,  int chunk)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int tid = (blockIdx.x * blockDim.x + threadIdx.x) % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (idx*CHUNK*NUM_BUFFERS >= inbytes)  return;

    const byte* __restrict__ inbuf[NUM_BUFFERS];
    byte* __restrict__ outbuf[NUM_BUFFERS];
    byte next[NUM_BUFFERS];

    __shared__  MTF_WORD mtf0 [(MTF_SYMBOLS+1)*NUM_WARPS*NUM_BUFFERS];
    MTF_WORD* mtf[NUM_BUFFERS];

    #pragma unroll
    for (int k=0; k<NUM_BUFFERS; k++)
    {
        inbuf[k]  =  _inbuf + CHUNK * (idx*NUM_BUFFERS + k);
        outbuf[k] = _outbuf + CHUNK * (idx*NUM_BUFFERS + k);
        next[k]   = *inbuf[k]++;

        mtf[k] = mtf0 + (MTF_SYMBOLS+1) * (warp_id*NUM_BUFFERS + k);
        #pragma unroll
        for (int i=0; i<MTF_SYMBOLS; i+=WARP_SIZE)
        {
            mtf[k][i+tid] = i+tid;
        }
    }
    SYNC_WARP();


    #pragma unroll 16
    for (int i=0; i<CHUNK; i++)
    {
        #pragma unroll
        for (int k=0; k<NUM_BUFFERS; k++)
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
