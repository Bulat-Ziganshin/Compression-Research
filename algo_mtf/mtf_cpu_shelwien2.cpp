
/* Vectorizable MTF algo by Eugene Shelwien, processing 2 bytes simultaneously
   It requires c1!=c2, that is true for post-RLE data.
*/

void mtf_cpu_shelwien2 (const byte* input, byte* output, int n)
{
    typedef signed char ranktype;
    ranktype Rank[ALPHABET_SIZE];

    for (int i = 0; i < ALPHABET_SIZE; ++i)
        Rank[i] = i-128;

#ifndef _MSC_BUILD  // MSC but not ICC
#pragma unroll(4)
#endif
    for (int i=0; i<n; i+=2)
    {
        auto c1 = *input++;
        auto c2 = *input++;
        auto d1 = Rank[c1];
        auto d2 = Rank[c2];
        d2 += d1>d2;   // c1 popup will shift c2 one position higher if d1>d2
#ifndef _MSC_BUILD  // MSC but not ICC
#pragma unroll(16)
#endif
        for (int j=0; j<ALPHABET_SIZE; j++)
        {
            auto r = Rank[j];   // clang-avx2 goes 10% faster with direct update of Rank[j]
            r -= (Rank[j]<d1)? ranktype(0xFF) : ranktype(0),
            r -= (Rank[j]<d2)? ranktype(0xFF) : ranktype(0);
            Rank[j] = r;
        }
        Rank[c1] = -127;
        Rank[c2] = -128;
        *output++ = d1+128;
        *output++ = d2+128;
    }
}
