This repository will contain variants of EC (entropy coding) stage of block-sorting compressor:
- bred: split data into 16 KB blocks and encode each block with static FSE coder
- bzip2: split data into 32/64 byte blocks, build 4-8 FSE tables per 1 MB shared by multiple data blocks
- bzip: encode rank/run-length with structured model (8-16 elements in top model, 8-16 elements in sub-models plus direct encoding of remaining bits) using dynamic SIMD RANS coder
