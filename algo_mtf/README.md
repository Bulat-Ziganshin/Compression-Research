[mtf_cpu_bsc]:         mtf_cpu_bsc.cpp
[mtf_cpu_shelwien]:    mtf_cpu_shelwien.cpp
[mtf_cpu_shelwien2]:   mtf_cpu_shelwien2.cpp
[mtf_cuda_thread]:     mtf_cuda_thread.cu
[mtf_cuda_thread_by4]: mtf_cuda_thread_by4.cu
[mtf_cuda_scalar]:     mtf_cuda_scalar.cu
[mtf_cuda_2symbols]:   mtf_cuda_2symbols.cu
[mtf_cuda_2buffers]:   mtf_cuda_2buffers.cu
[mtf_cuda_4by8]:       mtf_cuda_4by8.cu


### CPU implementations

Current CPU MTF implementations:
* [mtf_cpu_bsc] - borrowed from BSC 3.1
* [mtf_cpu_shelwien] - vectorizable constant-speed algo developed by Eugene Shelwien
* [mtf_cpu_shelwien2] - the same plus checking 2 input symbols simultaneously

Further optimizations:
- use SSE/AVX to check 16-32 positions simultaneously (PCMPEQB+PMOVMSKB+TZCNT+Jxx)
- check multiple symbols/buffers simultaneously in order to increase ILP
- use PCMPSTR instructions to compare multiple symbols against multiple positions
- SRC (sorted MTF) require storing `64*256` intermediate bytes, with full 64-byte lines stored to the memory, like in the radix sort

Combined algo:
- check for first 32 ranks using MTF queue in AVX2 register or two SSE2 registers, going into shelwien cycle only for rare ranks>32
- in order to provide sufficient ILP to deal with latency of PCMPEQB+PMOVMSKB+TZCNT+Jxx, interleave processing of 2 symbols from each of 2 blocks


### CUDA implementations

Current CUDA MTF implementations:
* [mtf_cuda_scalar] - processes single buffer per warp, comparing 32 mtf positions in single operation
* [mtf_cuda_2symbols] - the same, but checks 2 input symbols interleaved, increasing ILP
* [mtf_cuda_2buffers] - the same, but processes 2 buffers interleaved, increasing ILP
* [mtf_cuda_4by8] - process 4/8 positions from 8/4 buffers in the single warp
* [mtf_cuda_thread] - process 32 buffers per warp, on every algorithm step going 1 mtf position deeper and/or one input symbol further
* [mtf_cuda_thread_by4] - the same, but process 4 mtf positions on every step
* `mtf_cuda_Kbuffers<N>`, `mtf_cuda_thread<N>` and `mtf_cuda_thread_by4<N>` - mtf search depth limited to N, for use in multi-pass algorithm

Further optimizations:
- colesced global loads/stores (inbuf/outbuf)
- use 4-8 lanes to find ranks of 4-8 symbols simultaneously, then combine them and shift mtf[] elements by 1-4 positions
- the same, but combine search with shift
- the same with r2c[] and c2r[] machinery (see OpenBWT implementation)
- multi-pass: first pass process only ranks up to 8-32, last pass - only a few remaining chars with rank>32
- increase registers used/ILP by adding call limits
- maxwell simd instructions
- global loads through tex1D / `cub::TexRefInputIterator`
- 2-pass: combine improved `mtf_cuda_thread_by4<32>` with mtf_cuda_shelwien

Improving [mtf_cuda_thread_by4]:
- try by8-16-32
- try `NUM_THREADS = 4*WARP_SIZE` and check profiler info
- resolve shmem conflicts
- load 32 bytes/thread into shmem, replace them with mtf ranks and then save ranks to memory
- xor.u32, cmp.u8/u16 without byte_extract, `cub::SHL_ADD/PRMT+st.u32` for `mtf[]` update


### How to implement MTF on GPU?

Overall, we can explore 3 ways to parallelism, employing the single warp to:
* process multiple buffers
* process multiple input symbols from the single buffer
* compare multiple mtf queue positions to the same symbol

Unfortunately, each version of parallelism has its own drawbacks:
* Since each MTF queue occupies 256 bytes of shared memory, the entire 32-thread warp occupies as much as 8 KB.
This means that we cannot run more than 6..12 warps per SM, i.e. 1.5 .. 3 warps per one sheduler.
This requires very careful programming that should provide a lot of ILP.
In particular, input data should be prefetched, and probably multiple symbol/position checks should be interleaved.
Otherwise, we would stall a lot at memory delays and execution dependencies.
* Processing multiple input symbols for the same buffer simultaneously makes it harder to simultaneously shift data
in the MTF queue. I.e. when we are looking for 4 (different) symbols, we should shift MTF queue by 4 positions until we got
the first match, then by 3 positions until we got second match and so on. Alternatively, we can use 2 stages - the first stage
only discovers symbol ranks and the second stage shifts the data.
* Checking multiple MTF positions by the single warp is easy to implement, but results in significant inefficiency,
especially on low-entropy data. Peter Fenwick discovered that average rank of BWT output (on Calgary corpus) is ~6,
meaning that ~80% of comparisons are wasted, in addition to duplicating operations performed by multiple lanes.

The same 3 versions of parallelism can be exploited at ILP level:
* multiple buffers are processed by [mtf_cuda_2buffers]
* multiple input symbols are processed by [mtf_cuda_2symbols]
* multiple mtf positions are processed by [mtf_cuda_thread_by4]

Moreover, we can combine multiple parallelisms at warp level (f.e. check 4 positions in 8 buffers by the single warp instruction)
and/or simultaneously at ILP level. This creates a large space of possible combinations, which we can explore
in order to find most efficient solutions for existing GPUs.


### Multi-pass MTF algorithm

Like many other algos, MTF implementation on GPU is seriously limited by the shared memory size.
This can make reasonable a multi-pass approach, f.e. first pass may find only ranks up to 7,
second pass - ranks of 8..31, and last pass - all remaining ranks.

This means that the first passes will have much lower shared memory usage, allowing them to run more warps per SM
and reach 100% occupancy even with some memory-aggressive algo like [mtf_cuda_thread_by4],
while the last pass process only a few remaining symbols, and can check 32 positions at each step without losing much efficiency
with some simple algo like [mtf_cuda_scalar]. Isn't it beautiful?!

The key point, of course, is how they can be combined? The low-rank (first pass) algorithm should save at each position
where it was "overflowed", the symbol that was pushed out of its short MTF queue:
```C
symbol = inbuf[i]
for (int rank=0; rank<8; rank++)
    if (mtf[rank] == symbol)  {outbuf[i] = rank;  mtf[0] = symbol;  goto next_symbol;}
    ...
// Oh, we've found symbol with rank>=8. It will shift in at mtf[0], while mtf[7] is going to leave the queue.
*tempbuf++ = {i, symbol, mtf[7]}
mtf[0] = symbol
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

And any intermediate pass should combine both features to process symbols only of its own rank range.


### Parallel MTF algorithm

Parallel MTF algorithm was described in the
[Parallel Lossless Data Compression on the GPU](http://idav.ucdavis.edu/publications/print_pub?pub_id=1087)
and implemented in the [CUDPP library](https://github.com/cudpp/cudpp/blob/279eb8654b5a1e6b02573c568beafbb2b1344cc7/src/cudpp/app/compress_app.cu#L120).

The algorithm consists of three steps:

1. Split data into blocks and compute the local (partial) outbound MTF queue for every block - it's just the list of all symbols
appearing in the block, in the order of their **last** appearance.

2. Perform a (parallel) scan in order to combine partial MTF queues of the blocks into full outbound MTF queue for every block.
The full outbound MTF queue of the block is equal to its local outbound MTF queue plus any remaining symbols
in the order of their appearance in its inbound MTF queue (i.e. full outbound MTF queue of the previous block).
The MTF queue preceding all blocks is the trivial [0, 1 .. 255] list.

3. And finally, perform MTF on each block using full outbound MTF queue of the previous block as the initial MTF queue contents.

In order to simplify implementation, the second step may be performed sequentially on CPU.
