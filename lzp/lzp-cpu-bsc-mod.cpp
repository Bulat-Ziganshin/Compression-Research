// Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@gmail.com>
// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

/*  Slightly optimized BSC LZP implementation
*/

int lzp_cpu_bsc_mod (const unsigned char * input, const unsigned char * inputEnd, unsigned char * output, unsigned char * outputEnd, int hashSize, int minLen)
{
    if (inputEnd - input < 16)
    {
        return LIBBSC_NOT_COMPRESSIBLE;
    }

    if (int * lookup = (int *)bsc_zero_malloc((int)(1 << hashSize) * sizeof(int)))
    {
        const unsigned char *   inputStart  = input;
        const unsigned char *   outputStart = output;
        const unsigned char *   outputEOB   = outputEnd - 4;

        for (int i = 0; i < 4; ++i)
        {
            *output++ = *input++;
        }

        const unsigned char * heuristic      = input;
        const unsigned char * inputMinLenEnd = inputEnd - minLen - 8;
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned int index  =  (*(unsigned int *)(input-4) * 123456791)  >>  (32 - hashSize);
            int value = lookup[index];  lookup[index] = (int)(input - inputStart);
            if (value > 0)
            {
                const unsigned char * reference = inputStart + value;
                if ((*(unsigned int *)(input + minLen - 4) == *(unsigned int *)(reference + minLen - 4)) && (*(unsigned int *)(input) == *(unsigned int *)(reference)))
                {
                    if ((heuristic > input) && (*(unsigned int *)heuristic != *(unsigned int *)(reference + (heuristic - input))))
                    {
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    int len = 4;
                    for (; input + len < inputMinLenEnd; len += 4)
                    {
                        if (*(unsigned int *)(input + len) != *(unsigned int *)(reference + len)) break;
                    }
                    if (len < minLen)
                    {
                        if (heuristic < input + len) heuristic = input + len;
                        goto LIBBSC_LZP_MATCH_NOT_FOUND;
                    }

                    if (input[len] == reference[len]) len++;
                    if (input[len] == reference[len]) len++;
                    if (input[len] == reference[len]) len++;

                    input += len;

                    *output++ = LIBBSC_LZP_MATCH_FLAG;

                    len -= minLen; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; }

                    *output++ = (unsigned char)(len);
                }
                else
                {

LIBBSC_LZP_MATCH_NOT_FOUND:

                    unsigned char next = *output++ = *input++;
                    if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
                }
            }
            else
            {
                *output++ = *input++;
            }
        }

        while ((input < inputEnd) && (output < outputEOB))
        {
            unsigned int index  =  (*(unsigned int *)(input-4) * 123456791)  >>  (32 - hashSize);
            int value = lookup[index]; lookup[index] = (int)(input - inputStart);
            if (value > 0)
            {
                unsigned char next = *output++ = *input++;
                if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
            }
            else
            {
                *output++ = *input++;
            }
        }

        bsc_free(lookup);

        return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
    }

    return LIBBSC_NOT_ENOUGH_MEMORY;
}
