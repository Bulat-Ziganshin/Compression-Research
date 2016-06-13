This repository will contain variants of EC (entropy coding) stage of block-sorting compressor:
- bred-like: split data into 16 KB blocks and encode each block with static FSE coder
- bzip2-like: split data into 32/64 byte blocks, build 4-8 FSE tables per 1 MB shared by multiple data blocks
- bzip-like: encode rank/run-length with structured model (8-16 elements in top model, 8-16 elements in sub-models plus direct encoding of remaining bits) using dynamic SIMD RC/RANS coder

This stage input data are ranks+lengths after lzp+bwt/st+rle+mtf pipe, with size for text files in rnage of 10-40% of original data. 
In particular, for enwik8/9, it's 30-40% of input data, depending on BWT/ST order.
The number of bytes to process with entropy coding may be one of:
- 2x if each rank and length is encoded separately (this requires two 256-entry FSE tables)
- 1.3x if only lengths>1 are encoded (this requires two FSE tables - 512-entry and 256-entry)
- 1.4x-1.5x if lengths are 1-2 encoded (this requires single 257-entry table and compression ratio should be a bit lower)

FSE encodes at 400 MB/s, and Huff0 at 700 MB/s (here, all speeds are for single core of Haswell i7-4770).
Each optimization pass in bzip2 approach runs probably at 1 GB/s.

So, time required to encode enwik8/9 may be:
- bred-like: from 0.6-0.8 seconds with Huff0 and 1.3x scheme, 0.8-1 seconds with FSE and 1.3x scheme, to 2.5 seconds with FSE and 2x scheme
- bzip2-like: two optimization passes should add 0.6-0.8 seconds to the bred approach times, if we optimize only rank encoding, and 1.2-1.6 seconds if we also count lengths
- bzip-like: i think that the best approach is to encode 32 rank + 16 length values in top-level model (including 26+12 direct codes),
this should reduce to a minimum amount of extra codes (and low-probability jumps to their encoding), so encoding may be done in 2-3 seconds with RC (3-4 sec with RANS);
probably, input data should be preprocessed into (top-code,sub-model,sub-code) tuples in 1 KB chunks, with any extra bits saved into separate stream
