// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_THREADS = WARP_SIZE,  int MTF_SYMBOLS = ALPHABET_SIZE>
__global__ void mtf_thread_by4 (const byte* inbuf,  byte* outbuf,  int inbytes,  int chunk)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (idx*CHUNK >= inbytes)  return;
    inbuf  += idx*CHUNK;
    outbuf += idx*CHUNK;
    auto cur  = *inbuf++;
    auto next = *inbuf++;

    volatile __shared__  byte mtf0 [MTF_SYMBOLS*NUM_THREADS];
    auto mtf = mtf0 + 4*tid;
    for (int k=0; k<MTF_SYMBOLS; k+=4)
    {
        *(unsigned*)(mtf+k*NUM_THREADS)  =  k + ((k+1)<<8) + ((k+2)<<16) + ((k+3)<<24);
    }


    int i = 0,  k = 0;
    auto mtf_k = mtf;
    auto old = cur;

    for(;;)
    {
        #pragma unroll
        for (int x=0; x<4; x++)
        {
            auto next = mtf_k[x];
            mtf_k[x] = old;
            old = next;
            if (cur==old)  goto found;
            k++;
        }
        mtf_k += 4*NUM_THREADS;
        if (MTF_SYMBOLS == ALPHABET_SIZE  ||  k < MTF_SYMBOLS)
            continue;

found:
        *outbuf++ = k;
        if (++i >= CHUNK)  return;

        old = cur = next;
        next = *inbuf++;

        mtf_k = mtf;
        k = 0;
    }
}
