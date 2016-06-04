// Copyright (c) 2009-2011 Ilya Grebnov <ilya.grebnov@gmail.com>
// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research


/*  Highly optimized CPU LZP implementation employing rolling hash of next minLen bytes
*/

int lzp_cpu_rollhash (const unsigned char * input, const unsigned char * inputEnd, unsigned char * output, unsigned char * outputEnd, int hashSize, int minLen)
{
    if (inputEnd - input < 16)
        return LIBBSC_NOT_COMPRESSIBLE;

    typedef uint32_t HASH_T;
    typedef uint64_t LOOKUP_T;
    LOOKUP_T* lookup  =  (LOOKUP_T *) bsc_zero_malloc ((int)(1 << hashSize) * sizeof(LOOKUP_T));
    if (!lookup)
        return LIBBSC_NOT_ENOUGH_MEMORY;

    const unsigned char *inputStart  = input;
    const unsigned char *outputStart = output;
    const unsigned char *outputEOB   = outputEnd - 4;

    for (int i = 0; i < 4; ++i)
    {
        *output++ = *input++;
    }
    const HASH_T PRIME1 = 1234567891, PRIME2 = 1414213573;
    HASH_T rolling_hash = 0,  PRIME2_POW = 1;
    for (int i = 0; i < minLen; ++i)
    {
        rolling_hash  =  rolling_hash*PRIME2 + input[i];
        PRIME2_POW *= PRIME2;
    }

    const unsigned char * inputMinLenEnd = inputEnd - minLen - 8 - 8*sizeof(size_t); // 8*8 for the find-match-len loop unrolling
    while ((input < inputMinLenEnd) && (output < outputEOB))
    {
        LOOKUP_T value;
        while ((input < inputMinLenEnd) && (output < outputEOB))
        {
            unsigned int index  =  (*(unsigned int *)(input-4) * PRIME1)  >>  (32 - hashSize);
            value = lookup[index];  lookup[index] = (LOOKUP_T(input - inputStart) << 32) + rolling_hash;
            if (uint32_t(value) == rolling_hash)  goto FOUND;
            rolling_hash  =  rolling_hash*PRIME2 + (input[minLen] - input[0]*PRIME2_POW);
            unsigned char next = *output++ = *input++;
            if (unlikely (next == LIBBSC_LZP_MATCH_FLAG)  &&  value > 0)
                *output++ = 255;
        }
        break;

FOUND:
        const unsigned char * reference = inputStart + (value>>32);

        size_t* in  = (size_t*) input;
        size_t* ref = (size_t*) reference;
        size_t x;
        for (; (void*) in < inputMinLenEnd; in+=8, ref+=8)
        {
            x = in[0] ^ ref[0];  if (x) break;
            x = in[1] ^ ref[1];  if (x) {in++; break;}
            x = in[2] ^ ref[2];  if (x) {in+=2; break;}
            x = in[3] ^ ref[3];  if (x) {in+=3; break;}
            x = in[4] ^ ref[4];  if (x) {in+=4; break;}
            x = in[5] ^ ref[5];  if (x) {in+=5; break;}
            x = in[6] ^ ref[6];  if (x) {in+=6; break;}
            x = in[7] ^ ref[7];  if (x) {in+=7; break;}
            x = (size_t)-1;
        }
        int len = (unsigned char *)in - input + LZ4_NbCommonBytes(x);

        if (unlikely (len < minLen))
        {
            rolling_hash  =  rolling_hash*PRIME2 + (input[minLen] - input[0]*PRIME2_POW);
            unsigned char next = *output++ = *input++;
            if (next == LIBBSC_LZP_MATCH_FLAG) *output++ = 255;
            continue;
        }

        input += len;
        *output++ = LIBBSC_LZP_MATCH_FLAG;
        len -= minLen; while (len >= 254) { len -= 254; *output++ = 254; if (output >= outputEOB) break; }
        *output++ = (unsigned char)(len);

        rolling_hash = 0;
        for (int i = 0; i < minLen; ++i)
            rolling_hash  =  rolling_hash*PRIME2 + input[i];
    }


    while ((input < inputEnd) && (output < outputEOB))
    {
        unsigned int index  =  (*(unsigned int *)(input-4) * PRIME1)  >>  (32 - hashSize);
        LOOKUP_T value = lookup[index];  lookup[index] = LOOKUP_T(input - inputStart) << 32;
        unsigned char next = *output++ = *input++;
        if (next == LIBBSC_LZP_MATCH_FLAG  &&  value > 0)
            *output++ = 255;
    }

    bsc_free(lookup);

    return (output >= outputEOB) ? LIBBSC_NOT_COMPRESSIBLE : (int)(output - outputStart);
}
