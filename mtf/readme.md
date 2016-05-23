### CPU implementations

The only one currently included is qlfc-cpu.cpp, borrowed from BSC 3.1

Further CPU optimizations:
* use SSE/AVX to check 16-32 positions simultaneously (PCMPEQB+PMOVMSKB)
* check multiple symbols/buffers simultaneously in order to increase ILP
* use PCMPSTR instructions to compare multiple symbols against multiple positions


### GPU implementations

Current GPU MTF implementations:
* `mtf_scalar` - process single buffer per warp, compare 32 mtf positions in single operation
* `mtf_2symbols` - the same, but check 2 input symbols interleaved, increasing ILP
* `mtf_2buffers` - the same, but process 2 buffers interleaved, increasing ILP (not yet provides correct results!!)
* `mtf_thread` - process 32 buffers per warp, on every algorithm step going 1 mtf position deeper and/or one input symbol further
* `mtf_thread_by4` - the same, but process 4 mtf positions on every step
* `mtf_thread<N>` and `mtf_thread_by4<N>` - mtf search depth limited to N, should be used together with second-pass algorithm

Further GPU optimizations:
* global loads/stores (inbuf/outbuf)
* process 4/8 positions from 8/4 buffers in the single warp
* use 4-8 lanes to find ranks of 4-8 symbols simultaneously, then combine them and shift mtf[] elements by 1-4 positions
* the same, but combine search with shift
* the same with r2c[] and c2r[] machinery
* first pass processing only ranks up to 8-32


### How to implement MTF on GPU?

Overall, we can explore 3 levels of parallelism, employing the single warp to:
* process multiple buffers
* process multiple input symbols from the single buffer
* compare multiple mtf queue positions to the same symbol

Although, each version of parallelism has its own drawbacks:
* Since each MTF queue occupy 256 bytes of shared memory, the entire 32-thread warp occupies as much as 8 KB.
This means that we cannot run more than 6..12 warps per SM, i.e. 1.5 .. 3 warps per one sheduler.
This requires very careful programming that should provide a lot of ILP.
In particular, input data should be prefetched, and probably multiple symbols/positions are checked interleaved.
* ...
