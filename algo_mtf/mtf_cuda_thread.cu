// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_THREADS = WARP_SIZE,  int MTF_SYMBOLS = ALPHABET_SIZE>
__global__ void mtf_cuda_thread (const byte* inbuf,  byte* outbuf,  int inbytes,  int chunk)
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
    auto old  = mtf[0];

    for(;;)
    {
        if (cur != old  &&  !(MTF_SYMBOLS < ALPHABET_SIZE  &&  k >= MTF_SYMBOLS-1)) {
            k++;
            auto index = (k&252)*NUM_THREADS+(k&3);
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
