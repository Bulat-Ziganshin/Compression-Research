
/* Vectorizable MTF algo by Eugene Shelwien (original: gmtf_v1.zip)
*/

void mtf_shelwien (const byte* input, byte* output, int n)
{
    short Rank[ALPHABET_SIZE];

    for (int i = 0; i < ALPHABET_SIZE; ++i)
        Rank[i] = i;

    for (int i=0; i<n; i++)
    {
        auto c = *input++;
        auto d = Rank[c];
        for (int j=0; j<ALPHABET_SIZE; j++)
            Rank[j] += ( Rank[j]<d );
        Rank[c] = 0;
        *output++ = d;
    }
}
