// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_THREADS,  int NUM_BUFFERS,  typename MTF_WORD = byte,  int MTF_SYMBOLS = ALPHABET_SIZE>
__global__ void mtf_cuda_4by8 (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    // NUM_BUFFERS   - how many buffers processed by each thread block
    // NUM_POSITIONS - how many positions in each buffer are checked simultaneously (i.e. by the single warp)
    const int NUM_POSITIONS  =  NUM_THREADS / NUM_BUFFERS;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_POSITIONS;
    const int buf = threadIdx.x / NUM_POSITIONS;
    const int pos = threadIdx.x % NUM_POSITIONS;
    const int first_bit = (buf * NUM_POSITIONS) % WARP_SIZE;

    if (idx*CHUNK >= inbytes)  return;
    inbuf  += idx*CHUNK;
    outbuf += idx*CHUNK;
    auto cur  = *inbuf++;
    auto next = *inbuf++;

    volatile __shared__  MTF_WORD mtf0 [MTF_SYMBOLS*NUM_BUFFERS];
    auto mtf = mtf0 + buf*MTF_SYMBOLS;
    auto mtf_pos = mtf+pos;
    for (int k=0; k<MTF_SYMBOLS; k+=NUM_POSITIONS)
    {
        mtf_pos[k] = k+pos;
    }
    //__syncthreads();


    int i = 0,  k = 0;
    auto old  = *mtf_pos;

    for(;;)
    {
        unsigned n = __ballot (cur==old);                       // combined flags for NUM_POSITIONS in NUM_BUFFERS
        if (NUM_POSITIONS < WARP_SIZE)
            n  =  (n >> first_bit) % (1<<NUM_POSITIONS);        // only NUM_POSITIONS flags for the current buffer
        if (n==0) {                                             // if there is no match among these positions in the current buffer
            auto next = mtf_pos[k+NUM_POSITIONS];
            mtf_pos[k+1] = old;
            old = next;
            k += NUM_POSITIONS;
            //__syncthreads();
        } else {
            auto minbit = __ffs(n) - 1;
            if (pos < minbit)  mtf_pos[k+1] = old;
            *outbuf++ = k+minbit;
            mtf[0] = cur;
            //__syncthreads();
            old = *mtf_pos;
            if (++i >= CHUNK)  return;

            cur = next;
            next = *inbuf++;
            k = 0;
        }
    }
}
