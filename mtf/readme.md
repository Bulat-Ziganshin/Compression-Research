
Further GPU optimizations:
* global loads/stores (inbuf/outbuf)
* use 4-8 lanes to find ranks of 4-8 symbols simultaneously, then combine them and shift mtf[] elements by 1-4 positions
* the same, but combine search with shift
* the same with r2c[] and c2r[] machinery
* first pass processing only ranks up to 8-32

Further CPU optimizations:
* use SSE/AVX to check 16-32 positions simultaneously (PCMPEQB+PMOVMSKB)
* check multiple symbols/buffers simultaneously in order to increase ILP

