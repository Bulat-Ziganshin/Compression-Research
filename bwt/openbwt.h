/*
 * openbwt.h for the OpenBWT project
 * Copyright (c) 2008-2010 Yuta Mori. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _OPENBWT_H
#define _OPENBWT_H 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define OBWT_VERSION "2.0.0"

#ifndef OBWT_API
# ifdef OBWT_BUILD_DLL
#  define OBWT_API
# else
#  define OBWT_API
# endif
#endif

/**
 * Returns the version of the openbwt library.
 * @return The version number string.
 */
OBWT_API
const char *
obwt_version(void);



/*---------------------------------------------------------------------------*/
/*- Burrows-Wheeler Transforms -*/

/*
  Burrows-Wheeler Transform
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output BWTed-string.
  @param A[0..n-1] The temporary array.
  @param n The length of the string.
  @return The primary index if no error occurred, -1 or -2 otherwise.
*/
OBWT_API
int
obwt_bwt(const unsigned char *T, unsigned char *U, int *A, int n);

/*
  Construct the suffix array
  @param T[0..n-1] The input string.
  @param SA[0..n-1] The output suffix array.
  @param n The length of the string.
  @return 0 if no error occurred, -1 or -2 otherwise.
*/
OBWT_API
int
obwt_sufsort(const unsigned char *T, int *A, int n);


/*---------------------------------------------------------------------------*/
/*- Inverse Burrows-Wheeler Transforms -*/

/*
  Inverse Burrows-Wheeler Transform
  @param T[0..n-1] The input BWTed-string.
  @param U[0..n-1] The output string. (can be T)
  @param n The length of the string.
  @param pidx The primary index.
  @return 0 if no error occurred, -1, -2 or -3 otherwise.
*/

/* 6n algorithms */
OBWT_API
int
obwt_unbwt_basisPSI(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_basisLF(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_bw94(const unsigned char *T, unsigned char *U, int n, int pidx);

/* 5n algorithms, O(n) time */
OBWT_API
int
obwt_unbwt_saPSI(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_saLF(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_mergedTPSI(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_mergedTL(const unsigned char *T, unsigned char *U, int n, int pidx);

/* 5n algorithms */
OBWT_API
int
obwt_unbwt_indexPSIv1(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_indexPSIv2(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_indexLFv1(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_indexLFv2(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_unlimTPSI(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_unlimTLF(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_biPSIv1(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_biPSIv2(const unsigned char *T, unsigned char *U, int n, int pidx);

/* Medium-Space algorithms */
OBWT_API
int
obwt_unbwt_LL(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_LRI(const unsigned char *T, unsigned char *U, int n, int pidx);

OBWT_API
int
obwt_unbwt_LRI8v1(const unsigned char *T, unsigned char *U, int n, int pidx);
OBWT_API
int
obwt_unbwt_LRI8v2(const unsigned char *T, unsigned char *U, int n, int pidx);


/*- Inverse Suffix Array Samples for multi-threading -*/

typedef struct _obwt_ISAs_t obwt_ISAs_t;
struct _obwt_ISAs_t {
  int *samples; /* inverse suffix array samples */
  int lgsamplerate;
  int numsamples;
};

OBWT_API
int
obwt_ISAs_alloc(obwt_ISAs_t *ISAs, int n, int lgsamplerate);

OBWT_API
int
obwt_ISAs_build_from_BWT(obwt_ISAs_t *ISAs, const unsigned char *T, int n, int pidx);

OBWT_API
int
obwt_ISAs_build_from_SA(obwt_ISAs_t *ISAs, const int *SA, int n);

OBWT_API
void
obwt_ISAs_dealloc(obwt_ISAs_t *ISAs);


/*
  OpenMP supported Inverse Burrows-Wheeler Transform
  @param ISAs The inverse suffix array samples
  @param T[0..n-1] The input BWTed-string.
  @param U[0..n-1] The output string. (can be T)
  @param n The length of the string.
  @param numthreads The number of threads.
  @return 0 if no error occurred, -1, -2 or -3 otherwise.
*/

/* 6n algorithms */
OBWT_API
int
obwt_unbwt_basisPSI_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                        unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_basisLF_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                       unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_bw94_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                    unsigned char *U, int n, int numthreads);

/* 5n algorithms, O(n) time */
OBWT_API
int
obwt_unbwt_saPSI_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                     unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_saLF_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                    unsigned char *U, int n, int numthreads);

OBWT_API
int
obwt_unbwt_mergedTPSI_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                          unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_mergedTL_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                        unsigned char *U, int n, int numthreads);

/* 5n algorithms */
OBWT_API
int
obwt_unbwt_indexPSIv1_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                          unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_indexPSIv2_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                          unsigned char *U, int n, int numthreads);

OBWT_API
int
obwt_unbwt_indexLFv1_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                         unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_indexLFv2_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                         unsigned char *U, int n, int numthreads);

OBWT_API
int
obwt_unbwt_unlimTPSI_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                         unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_unlimTLF_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                        unsigned char *U, int n, int numthreads);

OBWT_API
int
obwt_unbwt_biPSIv1_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                       unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_biPSIv2_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                       unsigned char *U, int n, int numthreads);

/* Medium-Space algorithms */
OBWT_API
int
obwt_unbwt_LL_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                  unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_LRI_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                   unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_LRI8v1_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                      unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_LRI8v2_omp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                      unsigned char *U, int n, int numthreads);

/* Full OpenMP supported unbwt */
OBWT_API
int
obwt_unbwt_biPSIv1_fullomp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                           unsigned char *U, int n, int numthreads);
OBWT_API
int
obwt_unbwt_biPSIv2_fullomp(const obwt_ISAs_t *ISAs, const unsigned char *T,
                           unsigned char *U, int n, int numthreads);


/*---------------------------------------------------------------------------*/
/*- Burrows-Wheeler Transform Scottifed -*/

/*
  Burrows-Wheeler Transform Scottifed
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output BWTS-string.
  @param A[0..n-1] The temporary array.
  @param n The length of the string.
  @return 0 if no error occurred, -1 or -2 otherwise.
  @memspace 6n+@ bytes
*/
OBWT_API
int
obwt_bwts(const unsigned char *T, unsigned char *U, int *A, int n);

/*
  Inverse Burrows-Wheeler Transform Scottifed
  @param T[0..n-1] The input BWTS-string.
  @param U[0..n-1] The output string. (can be T)
  @param n The length of the string.
  @return 0 if no error occurred, -1 or -2 otherwise.
  @memspace 5n bytes (excluding T and U)
*/
OBWT_API
int
obwt_unbwts(const unsigned char *T, unsigned char *U, int n);


/*---------------------------------------------------------------------------*/
/*- Second Stage Transforms -*/

/*
  Move To Front
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
*/
OBWT_API
void
obwt_mtf_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_mtf_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Move One From Front
*/
OBWT_API
void
obwt_m1ff_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_m1ff_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Move One From Front 2
*/
OBWT_API
void
obwt_m1ff2_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_m1ff2_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Sorted Inversion Coder: Encoder
  @param T[0..n-1] The input string.
  @param IFV[0..n-1] The output IF-vector.
  @param C[0..255] The output array of the frequency of each symbol.
  @param n The length of the string.
  @return The length of IFV.
  @example
    m = SIF_Encode(T, IFV, C, n);
    encode_number(n);
    encode_number(m);
    for(i = 0; i < 256; ++i) {
      encode_number(C[i]);
    }
    for(i = 0; i < m; ++i) {
      encode_number(IFV[i]);
    }
*/
OBWT_API
int
obwt_inversioncoder_encode(const unsigned char *T, int *IFV, int *C, int n);

/*
  Sorted Inversion Coder: Decoder
  @param IFV[0..m-1] The input IF-vector.
  @param C[0..255] The input array of the frequency of each symbol.
  @param T[0..n-1] The output string.
  @param n The length of the string.
  @param m The length of IFV.
  @example
    n = decode_number();
    m = decode_number();
    for(i = 0; i < 256; ++i) {
      C[i] = decode_number();
    }
    for(i = 0; i < m; ++i) {
      IFV[i] = decode_number();
    }
    SIF_Decode(IFV, C, T, n, m);
*/
OBWT_API
void
obwt_inversioncoder_decode(const int *IFV, const int *C, unsigned char *T, int n, int m);

/*
  Distance Coder: Encoder
  @param T[0..n-1] The input string.
  @param DCV[0..n-1] The output DC-vector.
  @param P[0..255] The output array of the first position of each symbol.
  @param n The length of the string.
  @return The length of DCV.
  @example
    m = DC_Encode(T, DCV, P, n);
    encode_number(n);
    encode_number(m);
    for(i = 0; i < 256; ++i) {
      encode_number(P[i] + 1);
    }
    for(i = 0; i < m; ++i) {
      encode_number(DCV[i]);
    }
*/
OBWT_API
int
obwt_distancecoder_encode(const unsigned char *T, int *DCV, int *P, int n);

/*
  Distance Coder: Decoder
  @param DCV[0..m-1] The input DC-vector.
  @param P[0..255] The input array of the first position of each symbol.
  @param T[0..n-1] The output string.
  @param n The length of the string.
  @param m The length of DCV.
  @example
    n = decode_number();
    m = decode_number();
    for(i = 0; i < 256; ++i) {
      P[i] = decode_number() - 1;
    }
    for(i = 0; i < m; ++i) {
      DCV[i] = decode_number();
    }
    DC_Decode(DCV, P, T, n, m);
*/
OBWT_API
void
obwt_distancecoder_decode(const int *DCV, const int *P, unsigned char *T, int n, int m);

/*
  Sorted Distance Coder
  @example (encoding)
    m = obwt_sdcoder_encode(T, SDCV, C, n);
    encode_number(n);
    encode_number(m);
    for(i = 0; i < 256; ++i) {
      encode_number(C[i]);
    }
    for(i = 0; i < m; ++i) {
      encode_number(SDCV[i]);
    }
*/
OBWT_API
int
obwt_sdcoder_encode(const unsigned char *T, int *DCV, int *C, int n);
OBWT_API
void
obwt_sdcoder_decode(const int *DCV, const int *C, unsigned char *T, int n);

/*
  Sorted Rank Coder
  @example (encoding)
    SRC_Encode(T, SRCV, C, n);
    encode_number(n);
    for(i = 0; i < 256; ++i) {
      encode_number(C[i]);
    }
    for(i = 0; i < n; ++i) {
      encode_symbol(SRCV[i]);
    }
*/
OBWT_API
void
obwt_srcoder_encode(const unsigned char *T, unsigned char *RCV, int *C, int n);
OBWT_API
void
obwt_srcoder_decode(const unsigned char *RCV, const int *C, unsigned char *T, int n);

/*
  Transpose
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
*/
OBWT_API
void
obwt_transpose_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_transpose_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Best x of 2x - 1
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
  @param x
    x = 1 .. MTF
    x = 2 .. TimeStamp
  @return 0 if no error occurred, -1 otherwise.
*/
OBWT_API
int
obwt_Bx_encode(const unsigned char *T, unsigned char *U, int n, int x);
OBWT_API
int
obwt_Bx_decode(const unsigned char *T, unsigned char *U, int n, int x);

/*
  TimeStamp(0)
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
*/
OBWT_API
void
obwt_timestamp0_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_timestamp0_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Sort-By-Rank(0.5)
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
*/
OBWT_API
void
obwt_sortbyrank_encode(const unsigned char *T, unsigned char *U, int n);
OBWT_API
void
obwt_sortbyrank_decode(const unsigned char *T, unsigned char *U, int n);

/*
  Frequency Count
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
  @param incr The increment value.
  @param threshold The threshold value for each frequency.
*/
OBWT_API
void
obwt_frequencycount_encode(const unsigned char *T, unsigned char *U, int n,
                           int incr, int threshold);
OBWT_API
void
obwt_frequencycount_decode(const unsigned char *T, unsigned char *U, int n,
                           int incr, int threshold);

/*
  Weighted Frequency Count
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
  @param numlevels The number of levels.
  @param Levels[0..numlevels-1] end position of each level
  @param W[0..numlevels] weight value of each level
  @return 0 if no error occurred, -1 otherwise.

  Deorowicz's w6 function
  int numlevels = 12;
  int Levels[12] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  int W[13] = {65536, 8192, 4096, 2048, 1024, 512, 256, 64, 32, 8, 4, 1, 0};

  Abel's Advanced WFC
  int numlevels = 12;
  int Levels[12] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  int W[13];
  int p0 = 2600;
  int p1 = 4185;
  int s = obwt_awfc_calculate_s(T, n);
  W[0] = 1 << 17, W[1] = 1 << 14;
  obwt_awfc_calculate_W(s, p0, p1, numlevels, W);
  W[12] = 0;
  output(S);
*/
OBWT_API
int
obwt_wfc_encode(const unsigned char *T, unsigned char *U, int n,
                int numlevels, const int *Levels, const int *W);
OBWT_API
int
obwt_wfc_decode(const unsigned char *T, unsigned char *U, int n,
                int numlevels, const int *Levels, const int *W);
OBWT_API
int
obwt_awfc_calculate_s(const unsigned char *T, int n);
OBWT_API
void
obwt_awfc_calculate_W(int s, int p0, int p1, int numlevels, int *W);

/*
  Incremental Frequency Count
  @param T[0..n-1] The input string.
  @param U[0..n-1] The output string.
  @param n The length of the string.
  @param dm The maximum for difference between current and last avg.
  @param threshold The threshold for rescaling
  @param windowsize The size of the sliding window
*/
OBWT_API
void
obwt_ifc_encode(const unsigned char *T, unsigned char *U, int n,
                int dm, int threshold, int windowsize);
OBWT_API
void
obwt_ifc_decode(const unsigned char *T, unsigned char *U, int n,
                int dm, int threshold, int windowsize);


#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* _OPENBWT_H */
