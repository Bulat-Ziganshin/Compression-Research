// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

template <int CHUNK,  int NUM_THREADS,  int NUM_BUFFERS,  typename MTF_WORD = byte,  int MTF_SYMBOLS = ALPHABET_SIZE>
__global__ void mtf_4by8 (const byte* __restrict__ inbuf,  byte* __restrict__ outbuf,  int inbytes,  int chunk)
{
    // NUM_BUFFERS   - how many buffers processed by each thread block
    // NUM_POSITIONS - how many positions in each buffer are checked simultaneously (i.e. by the single warp)
    const int NUM_POSITIONS  =  NUM_THREADS / NUM_BUFFERS;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_POSITIONS;
    const int buf = threadIdx.x / NUM_POSITIONS;
    const int pos = threadIdx.x % NUM_POSITIONS;
    const int first_bit = (buf * NUM_POSITIONS) % WARP_SIZE;
    const int BYTES_PER_WORD = 4;
    const int offset = threadIdx.x / (NUM_THREADS/BYTES_PER_WORD);  // two highest bits of tid defines use of byte 0..3 of the word in mtf0[]

    if (idx*CHUNK >= inbytes)  return;
    inbuf  += idx*CHUNK;
    outbuf += idx*CHUNK;
    auto cur  = *inbuf++;
    auto next = *inbuf++;

    volatile __shared__  MTF_WORD mtf0 [MTF_SYMBOLS*NUM_BUFFERS];
    auto mtf = mtf0 + offset + (buf % (NUM_BUFFERS/BYTES_PER_WORD)) * NUM_POSITIONS * BYTES_PER_WORD;
    auto mtf_pos = mtf + pos*BYTES_PER_WORD;
    const int SKIP = NUM_POSITIONS * NUM_BUFFERS;
    for (int k=0; k<MTF_SYMBOLS; k+=NUM_POSITIONS)
    {
        mtf_pos[k*SKIP/NUM_POSITIONS] = k+pos;
    }
    //__syncthreads();
    auto mtf_pos_next = mtf_pos + BYTES_PER_WORD;
    if (pos == NUM_POSITIONS-1)
        mtf_pos_next += SKIP - NUM_POSITIONS*BYTES_PER_WORD;


    int i = 0,  k = 0;
    auto old = *mtf_pos;

    for(;;)
    {
// to do: for(int _=0;_<4;_++) around if-then
        unsigned n = __ballot (cur==old);                       // combined flags for NUM_POSITIONS in NUM_BUFFERS
        if (NUM_POSITIONS < WARP_SIZE)
            n  =  (n >> first_bit) % (1<<NUM_POSITIONS);        // only NUM_POSITIONS flags for the current buffer
        if (n==0) {                                             // if there is no match among these positions in the current buffer
            auto next = mtf_pos[k+SKIP];
            mtf_pos_next[k] = old;
            old = next;
            k += SKIP;
            //__syncthreads();
        } else {
            auto minbit = __ffs(n) - 1;
            if (pos < minbit)  mtf_pos_next[k] = old;
            *outbuf++ = minbit + k/(SKIP/NUM_POSITIONS);
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
