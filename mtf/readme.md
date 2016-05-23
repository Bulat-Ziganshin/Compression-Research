[mtf_thread]:     https://github.com/Bulat-Ziganshin/Compression-Research/blob/6709d3ceb368e59824d35ed58b0e3211b26281a4/mtf/mtf.cu#L31
[mtf_thread_by4]: https://github.com/Bulat-Ziganshin/Compression-Research/blob/6709d3ceb368e59824d35ed58b0e3211b26281a4/mtf/mtf.cu#L76
[mtf_scalar]:     https://github.com/Bulat-Ziganshin/Compression-Research/blob/6709d3ceb368e59824d35ed58b0e3211b26281a4/mtf/mtf.cu#L129
[mtf_2symbols]:   https://github.com/Bulat-Ziganshin/Compression-Research/blob/6709d3ceb368e59824d35ed58b0e3211b26281a4/mtf/mtf.cu#L200
[mtf_2buffers]:   https://github.com/Bulat-Ziganshin/Compression-Research/blob/6709d3ceb368e59824d35ed58b0e3211b26281a4/mtf/mtf.cu#L283


### CPU implementations

The only one currently included is qlfc-cpu.cpp, borrowed from BSC 3.1

Further CPU optimizations:
* use SSE/AVX to check 16-32 positions simultaneously (PCMPEQB+PMOVMSKB)
* check multiple symbols/buffers simultaneously in order to increase ILP
* use PCMPSTR instructions to compare multiple symbols against multiple positions


### GPU implementations

Current GPU MTF implementations:
* [mtf_scalar] - processes single buffer per warp, comparing 32 mtf positions in single operation
* [mtf_2symbols] - the same, but checks 2 input symbols interleaved, increasing ILP
* [mtf_2buffers] - the same, but processes 2 buffers interleaved, increasing ILP (don't yet provides correct results!!)
* [mtf_thread] - process 32 buffers per warp, on every algorithm step going 1 mtf position deeper and/or one input symbol further
* [mtf_thread_by4] - the same, but process 4 mtf positions on every step
* `mtf_thread<N>` and `mtf_thread_by4<N>` - mtf search depth limited to N, should be used together with second-pass algorithm

Further GPU optimizations:
* global loads/stores (inbuf/outbuf)
* process 4/8 positions from 8/4 buffers in the single warp
* use 4-8 lanes to find ranks of 4-8 symbols simultaneously, then combine them and shift mtf[] elements by 1-4 positions
* the same, but combine search with shift
* the same with r2c[] and c2r[] machinery (see OpenBWT implementation)
* first pass process only ranks up to 8-32, last pass - only a few remaining chars with rank>32


### How to implement MTF on GPU?

Overall, we can explore 3 versions of parallelism, employing the single warp to:
* process multiple buffers
* process multiple input symbols from the single buffer
* compare multiple mtf queue positions to the same symbol

Unfortunately, each version of parallelism has its own drawbacks:
* Since each MTF queue occupy 256 bytes of shared memory, the entire 32-thread warp occupies as much as 8 KB.
This means that we cannot run more than 6..12 warps per SM, i.e. 1.5 .. 3 warps per one sheduler.
This requires very careful programming that should provide a lot of ILP.
In particular, input data should be prefetched, and probably multiple symbol/position checks should be interleaved.
Otherwise, we would stall a lot at memory delays and execution dependencies.
* Processimg multiple input symbols for the same buffer simultaneously makes it harder to simultaneously shift data
in the MTF queue. I.e. when we are looking for 4 (different) symbols, we should shift MTF queue by 4 positions until we got
the first match, then by 3 positions until we get second match and so on. Alternatively, we can use 2 stages - the first stage
only discovers symbol ranks and the second stage performs actual data shift.
* Checking multiple MTF positions by the single warp is easy to implement, but results in significant inefficiency,
especially on low-entropy data. Peter Fenwick discovered that average rank (on Calgary corpus) is ~6,
meaning that ~80% of comparisons are wasted, in addition to equivalent operations performed by multiple lanes.

The same 3 versions of parallelism can be exploited at ILP level:
* multiple buffers are processed by [mtf_2buffers]
* multiple input symbols are processed by [mtf_2symbols]
* multiple mtf positions are processed by [mtf_thread_by4]

Moreover, we can combine multiple parallelisms at warp level (f.e. check 4 positions in 8 buffers by the single warp instruction)
and/or simultaneously at ILP level. This creates a large space of possible combinations, which we can explore
in order to find most efficient solutions for existing GPUs.


### Multi-pass MTF algorithm

Like many other algos, MTF implementation on GPU is seriously limited by the shared memory size.
This can make reasonable a multi-pass approach, f.e. first pass may find only ranks up to 7,
second pass - ranks of 8..31, and last pass - all remaining ranks.

This means that the first passes will have much lower shared memory usage, allowing them to run more warps per SM
and reach 100% occupancy even with some memory-aggressive algo like [mtf_thread_by4],
while the last pass process only a few remaining symbols, and can check 32 positions at each step without losing much efficiency
with some simple algo like [mtf_scalar]. Isn't it beautiful?!

The key point, of course, is how they can be combined? The low-rank (first pass) algorithm should save at each position
where it was "overflowed", the symbol that was pushed out of its short MTF queue:
```C
symbol = inbuf[i]
for (int rank=0; rank<8; rank++)
    if (mtf[rank] == symbol)  {outbuf[i] = rank;  mtf[0] = symbol;  goto next_symbol;}
    ...
// Oh, we've found symbol with rank>=8
*tempbuf++ = {i, symbol, mtf[7]}
```

While the high-rank (last pass) algorithm should process only symbols overflowed at previous pass, looking for the current
symbol f.e. in mtf[32..255] positions, saving the position (rank) where this symbol was found, but inserting to the head
of its queue the symbol that was pushed out of low-rank mtf queue at the previous pass:
```C
{i, symbol, overflowed_symbol} = *tempbuf++
for (int rank=32; rank<256; rank++)
    if (mtf[rank] == symbol)  {outbuf[i] = rank;  mtf[32] = overflowed_symbol;  goto next_symbol;}
    ...
```

And any intermediate algorithm should combine both features to process symbols only of its own rank range.
