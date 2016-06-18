This directory contains BSC implementations of Sort Transform on CPU & GPU.

Now CUDA ST4 sorting of 30 MB block requires (on GF560Ti) about 150 ms, of those only 65 ms spent in CUB Radix sort,
and copying data in/out spends 12 ms.

Further optimizations:
- perform cudaMalloc/cudaFree only once - this should double the performance
- keep data in pinned memory - this shoud double the cudaMemcpy speed and give another 5-10% performance boost

Use some combination of the following ideas to shave off remaining times over 65 ms
- overload pre/post-sorting procedures and RLE compression with memcpy
- process only 4-byte elements at last sorting stages, and simultaneously copy-in next block to process -
4-byte sorting should also be faster than sorting of 4+4 (key+value) bytes (43 ms total instead of 65 ms)
- use zero-copy memory instead of copy in/out

So, after all optimizations, ST4 should become more than 3x faster!
