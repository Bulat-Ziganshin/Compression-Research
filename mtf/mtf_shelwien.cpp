
/* Vectorizable MTF algo by Eugene Shelwien (original: gmtf_v2f.zip)
*/

void mtf_shelwien (const byte* input, byte* output, int n)
{
    typedef signed char ranktype;
    ranktype Rank[ALPHABET_SIZE];

    for (int i = 0; i < ALPHABET_SIZE; ++i)
        Rank[i] = i-128;

    for (int i=0; i<n; i++)
    {
        auto c = *input++;
        auto d = Rank[c];
#pragma unroll(16)
        for (int j=0; j<ALPHABET_SIZE; j++)
            Rank[j] -= (Rank[j]<d)?ranktype(0xFF):ranktype(0);
        Rank[c] = -128;
        *output++ = d+128;
    }
}
