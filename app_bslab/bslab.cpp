// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

#include <stdio.h>
#include <vector>
#include <functional>
#include <stdint.h>

#ifdef LIBBSC_CUDA_SUPPORT
#include <helper_functions.h>          // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>               // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>
#include "../util/cuda_common.h"       // my own cuda-specific helper functions
#endif // LIBBSC_CUDA_SUPPORT

#ifdef _OPENMP
#include <omp.h>
#define LIBBSC_OPENMP
#endif // _OPENMP

#include "../util/wall_clock_timer.h"  // StartTimer() and GetTimer()
#include "../util/cpu_common.h"        // my own helper functions
#include "../util/libbsc.h"            // BSC common definitions
#include "../util/lz4_common.h"        // Utility functions from LZ4

const int WARP_SIZE = 32;
typedef unsigned char byte;

// Parameters
const int DEFAULT_BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
#define SYNC_WARP __threadfence_block  /* alternatively, __syncthreads or, better, __threadfence_warp */

#include "../algo_lzp/lzp-cpu-bsc.cpp"
#include "../algo_lzp/lzp-cpu-bsc-mod.cpp"
#include "../algo_lzp/lzp-cpu-rollhash.cpp"

#include "../algo_bwt/sais.c"              // OpenBWT implementation
#include "../algo_bwt/divsufsort.c"        // divsufsort
#define LIBBSC_SORT_TRANSFORM_SUPPORT
#include "../algo_st/st.cpp"               // BSC CPU Sort Transform implementation
#include "../algo_st/st.cu"                // BSC GPU Sort Transform implementation

#include "../algo_mtf/mtf_cpu_bsc.cpp"
#include "../algo_mtf/mtf_cpu_shelwien.cpp"
#include "../algo_mtf/mtf_cpu_shelwien2.cpp"
#ifdef LIBBSC_CUDA_SUPPORT
#include "../algo_mtf/mtf_cuda_scalar.cu"
#include "../algo_mtf/mtf_cuda_2symbols.cu"
#include "../algo_mtf/mtf_cuda_2buffers.cu"
#include "../algo_mtf/mtf_cuda_2buffers_depth32.cu"
#include "../algo_mtf/mtf_cuda_4by8.cu"
#include "../algo_mtf/mtf_cuda_thread.cu"
#include "../algo_mtf/mtf_cuda_thread_by4.cu"
#endif // LIBBSC_CUDA_SUPPORT

// In-place RLE transformation (run lengths are dropped!)
// Sum up amount of runs>255 chars into long_runs
// Sum up amount of chars + non-trivial runs into ranks_plus_lens
// Sum up amount of chars plus size of 1/2 length encoding into _1_2_codes
int rle (byte* buf, int size, uint64_t &long_runs, uint64_t &ranks_plus_lens, uint64_t &_1_2_codes)
{
    int c = -1,  run = 0;
    auto out = buf;
    for (size_t i = 0; i < size; i++)
    {
        if (buf[i]==c) {
            run++;
        } else {
            if (run>255)  long_runs++;
            if (run>1)    ranks_plus_lens++;
            while (run>1) {
                _1_2_codes++;  // alternatively, _1_2_codes += logb(run);
                run /= 2;
            }
            run = 1;
            c = *out++ = buf[i];
        }
    }
    auto len = out-buf;
    ranks_plus_lens += len;
    _1_2_codes += len;
    return len;
}

int main (int argc, char **argv)
{
    const int MANY = 100;
    bool display_gpu = true;
    bool apply_lzp = true;
    bool apply_bwt = true;
    bool apply_rle = true;
    bool apply_mtf = true;
    enum STAGE {LZP, BWT, RLE, MTF, STAGES};
    bool enabled[STAGES][MANY];
    int lzpHashSize = 15,  lzpMinLen = 32;
    size_t bufsize = DEFAULT_BUFSIZE;
    uint64_t long_runs = 0,  ranks_plus_lens = 0,  _1_2_codes = 0;
    char *comment;
    int error = 0;
    for (int stage=0; stage<STAGES; stage++) 
        for (int i=0; i<MANY; i++) 
            enabled[stage][i] = true;
    
    auto src_argv = argv,  dst_argv = argv;
    while (*++src_argv) {
      ParseBool    (*src_argv, "-gpu", "-nogpu", &display_gpu) ||
      ParseBool    (*src_argv, "-lzp", "-nolzp", &apply_lzp)            ||
      ParseBool    (*src_argv, "-bwt", "-nobwt", &apply_bwt)            ||
      ParseBool    (*src_argv, "-rle", "-norle", &apply_rle)            ||
      ParseBool    (*src_argv, "-mtf", "-nomtf", &apply_mtf)            ||
      ParseIntList (*src_argv, "-lzp",           enabled[LZP], MANY)    ||
      ParseIntList (*src_argv, "-bwt",           enabled[BWT], MANY)    ||
      ParseIntList (*src_argv, "-mtf",           enabled[MTF], MANY)    ||
      ParseInt     (*src_argv, "-b",             &bufsize)              ||
      ParseInt     (*src_argv, "-h",             &lzpHashSize)          ||
      ParseInt     (*src_argv, "-l",             &lzpMinLen)            ||
      ParseStr     (*src_argv, "-rem",           &comment)              ||
      UnknownOption (*src_argv, &error) ||
      (*++dst_argv = *src_argv);
    }
    *++dst_argv = 0;  argc = dst_argv - argv;

    if (bufsize < 100*1000)
        bufsize <<= 20;  // if value is small enough, consider it as mebibytes

    if (!(argc==2 || argc==3) || error) {
        printf ("BSL: the block-sorting lab 1.0 (June 18 2016).  Part of https://github.com/Bulat-Ziganshin/Compression-Research\n"
                "Usage: bsl [options] infile [outfile]\n"
                "  -bN      buffer N (mebi)bytes (default %d MiB - reduce if program fails)\n"
                "  -nogpu   skip GPU name output\n"
                "  -nolzp   skip LZP transform\n"
                "  -nobwt   skip BWT/ST\n"
                "  -norle   skip RLE transform\n"
                "  -nomtf   skip MTF transform\n"
                "  -lzpLIST perform only LZP transforms specified by the LIST\n"
                "  -hN      set LZP hash size log (default 2^%d hash entries)\n"
                "  -lN      set LZP minLen (default %d)\n"
                "  -bwtLIST perform only sorting transforms specified by the LIST\n"
                "  -mtfLIST perform only MTF transforms specified by the LIST\n"
                "  -rem...  ignored by the program\n"
                "LIST has format \"[+/-]n,m-k...\" which means enable/disable/enable_only transforms number n and m..k\n"
                , DEFAULT_BUFSIZE>>20, lzpHashSize, lzpMinLen);
        return argc==1 && !error?  0 : 1;
    }

#ifdef LIBBSC_CUDA_SUPPORT
    unsigned char* d_inbuf;
    unsigned char* d_outbuf;
    checkCudaErrors( cudaMalloc((void**)(&d_inbuf),  bufsize+CHUNK*2+256));  // up to CHUNK*2 extra bytes may be processed, plus a few extra bytes may be read after bufend
    checkCudaErrors( cudaMalloc((void**)(&d_outbuf), bufsize+CHUNK*2+256));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));
#endif // LIBBSC_CUDA_SUPPORT

    unsigned char* inbuf  = new unsigned char[bufsize];
    unsigned char* outbuf = new unsigned char[bufsize];
    int*      bwt_tempbuf = apply_bwt? new int[bufsize] : 0;

    int num, stage, retval;  int64_t inbytes;  bool ret_outsize;
    uint64_t outsize = 0,  insize[STAGES] = {0},  size[STAGES][MANY] = {0};
    char *name[STAGES][MANY] = {0};  double duration[STAGES][MANY] = {0};

    auto cpu_time_run = [&] (char *this_name, std::function<int64_t(void)> stage_f) {
        name[stage][num] = this_name;        
        if (enabled[stage][num])
        {
            StartTimer();
            retval  =  stage_f();
            duration[stage][num] += GetTimer();

            if (retval < 0  &&  retval != LIBBSC_NOT_COMPRESSIBLE) {
                printf ("%s failed with errcode %d\n", this_name, retval);
                exit(4);
            }
            size[stage][num]  +=  (ret_outsize && retval != LIBBSC_NOT_COMPRESSIBLE?  retval : inbytes);
        }
        num++;
    };


    FILE* infile  = fopen (argv[1], "rb");
    FILE* outfile = fopen (argv[2]? argv[2] : "nul", "wb");
    if (!infile) {
        printf ("Can't open infile %s\n", argv[1]);
        return 2;
    }
    if (!outfile) {
        printf ("Can't open outfile %s\n", argv[2]);
        return 3;
    }
#ifdef LIBBSC_CUDA_SUPPORT
    if (display_gpu)
        DisplayCudaDevice();
#endif // LIBBSC_CUDA_SUPPORT


    // All preparations now are done. Now we are in the Analysis stage, processing input data with various algos and recording speed/outsize of every experiment
    while (!!(inbytes = fread(inbuf,1,bufsize,infile)))
    {
        byte *ptr = inbuf;  size_t outbytes = inbytes;  // output buffer


        stage = LZP,  insize[stage] += inbytes,  ret_outsize = true,  num = 1;
        if (apply_lzp) {
            lzp_cpu_bsc (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);   // "massage" data in order to provide equal conditions for the both following lzp routines

#ifdef _OPENMP
            num = 4;
            cpu_time_run ("lzp_cpu_rollhash (OpenMP)", [&] {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int64_t base=0; base<inbytes; base+=8 MB)
                {
                    auto size = mymin(inbytes-base,8 MB);
                    lzp_cpu_rollhash (inbuf+base, inbuf+base+size, outbuf+base, outbuf+base+size, lzpHashSize, lzpMinLen);
                }
                return inbytes;
            });
#endif // _OPENMP

            num = 1;
            cpu_time_run ("lzp_cpu_bsc     ", [&] {return lzp_cpu_bsc      (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            cpu_time_run ("lzp_cpu_bsc_mod ", [&] {return lzp_cpu_bsc_mod  (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            cpu_time_run ("lzp_cpu_rollhash", [&] {return lzp_cpu_rollhash (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});

            if (retval != LIBBSC_NOT_COMPRESSIBLE)
                memcpy (inbuf, outbuf, inbytes=retval);
        }


        stage = BWT,  insize[stage] += inbytes,  ret_outsize = false,  num = 1;
        if (apply_bwt) {
            int retval = bsc_st_init(0);
            if (retval != LIBBSC_NO_ERROR) {
                printf ("bsc_st_init failed with errcode %d\n", retval);
                exit(4);
            }

#ifdef LIBBSC_CUDA_SUPPORT
            if (enabled[stage][1] || enabled[stage][2] || enabled[stage][3] || enabled[stage][4]) {  // CUDA will be used, so we need to warm it up
                memcpy (outbuf, inbuf, inbytes);  bsc_st_encode_cuda (outbuf, inbytes, 8, 0);
            }

            char *cuda_st_name[] = {"st0-cuda", "st1-cuda", "st2-cuda", "st3-cuda", "st4-cuda", "st5-cuda", "st6-cuda", "st7-cuda", "st8-cuda"};
            for (int i=5; i<=8; i++)
                cpu_time_run (cuda_st_name[i], [&] {memcpy (outbuf, inbuf, inbytes);  return bsc_st_encode_cuda (outbuf, inbytes, i, 0);});
#endif // LIBBSC_CUDA_SUPPORT

            char *cpu_st_name[] = {"st0-cpu", "st1-cpu", "st2-cpu", "st3-cpu", "st4-cpu", "st5-cpu", "st6-cpu", "st7-cpu", "st8-cpu "};
            for (int i=3; i<=6; i++)
                cpu_time_run (cpu_st_name[i], [&] {memcpy (outbuf, inbuf, inbytes);  return bsc_st_encode (outbuf, inbytes, i, 0);});

            int             indexes[256];
            unsigned char   num_indexes;
            cpu_time_run ("OpenBWT",             [&] {return sais_bwt (inbuf, outbuf, bwt_tempbuf, inbytes);});
            cpu_time_run ("divsufsort",          [&] {return divbwt(inbuf, outbuf, bwt_tempbuf, inbytes, &num_indexes, indexes, 0);});
#ifdef _OPENMP
            cpu_time_run ("divsufsort (OpenMP)", [&] {return divbwt(inbuf, outbuf, bwt_tempbuf, inbytes, &num_indexes, indexes, 1);});
#endif // _OPENMP

            memcpy (inbuf, outbuf, inbytes);
        }


        stage = MTF,  num = 1;
        if (apply_mtf  &&  enabled[stage][num]) {
            name[stage][num] = "mtf_cpu_bsc";
            StartTimer();
                unsigned char MTFTable[ALPHABET_SIZE];
                ptr = mtf_cpu_bsc (inbuf, outbuf, inbytes, MTFTable);
                outbytes = outbuf+inbytes - ptr;
            duration[stage][num] += GetTimer();
        }


        stage = RLE,  insize[stage] += inbytes,  ret_outsize = true,  num = 1;
        if (apply_rle) {
            cpu_time_run ("rle", [&] {return rle (inbuf, inbytes, long_runs, ranks_plus_lens, _1_2_codes);});
            inbytes = retval;
        }


        stage = MTF,  insize[stage] += inbytes,  ret_outsize = false,  num = 2;
        if (apply_mtf)
        {
            cpu_time_run ("mtf_cpu_shelwien",  [&] {mtf_cpu_shelwien  (inbuf, outbuf, inbytes);  return inbytes;});
            cpu_time_run ("mtf_cpu_shelwien2", [&] {mtf_cpu_shelwien2 (inbuf, outbuf, inbytes);  return inbytes;});

#ifdef _OPENMP
            cpu_time_run ("mtf_cpu_shelwien2 (OpenMP)", [&] {
                #pragma omp parallel for schedule(dynamic, 1)
                for (int64_t base=0; base<inbytes; base+=1 MB)
                {
                    mtf_cpu_shelwien2 (inbuf+base, outbuf+base, mymin(inbytes-base,1 MB));
                }
                return inbytes;
            });
#endif // _OPENMP


#ifdef LIBBSC_CUDA_SUPPORT
            checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
            checkCudaErrors( cudaDeviceSynchronize());

            auto time_run = [&] (char *this_name, std::function<void(void)> f) {
                name[stage][num] = this_name;
                if (enabled[stage][num])
                {
                    checkCudaErrors( cudaEventRecord (start, nullptr));
                    f();
                    checkCudaErrors( cudaEventRecord (stop, nullptr));
                    checkCudaErrors( cudaDeviceSynchronize());

                    if (enabled[stage][num]) {
                        checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
                        checkCudaErrors( cudaDeviceSynchronize());
                        ptr = outbuf;
                        outbytes = inbytes;
                    }

                    float start_stop;
                    checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
                    duration[stage][num] += start_stop;
                }
                num++;
            };


{
            const int NUM_WARPS = 4;
            time_run ("mtf_cuda_scalar        ", [&] {mtf_cuda_scalar    <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_2symbols      ", [&] {mtf_cuda_2symbols  <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_2buffers      ", [&] {mtf_cuda_2buffers  <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_2buffers<32>  ", [&] {mtf_cuda_2buffers_depth32 <CHUNK,NUM_WARPS><<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_3buffers<32>  ", [&] {mtf_cuda_2buffers_depth32 <CHUNK,NUM_WARPS,3><<<(inbytes-1)/(CHUNK*NUM_WARPS*3)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_4buffers<32>  ", [&] {mtf_cuda_2buffers_depth32 <CHUNK,NUM_WARPS,4><<<(inbytes-1)/(CHUNK*NUM_WARPS*4)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
{
            const int NUM_THREADS = 8*WARP_SIZE,  NUM_BUFFERS = NUM_THREADS/8;
            time_run ("mtf_cuda_4by8          ", [&]{mtf_cuda_4by8<CHUNK,NUM_THREADS,NUM_BUFFERS><<<(inbytes-1)/(CHUNK*NUM_BUFFERS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
{
            const int NUM_THREADS = 4*WARP_SIZE,  NUM_BUFFERS = NUM_THREADS/4;
            time_run ("mtf_cuda_8by4          ", [&]{mtf_cuda_4by8<CHUNK,NUM_THREADS,NUM_BUFFERS><<<(inbytes-1)/(CHUNK*NUM_BUFFERS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
            time_run ("mtf_cuda_thread        ", [&] {mtf_cuda_thread    <CHUNK>                 <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread_by4    ", [&] {mtf_cuda_thread_by4<CHUNK>                 <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

            const int NUM_THREADS = 4*WARP_SIZE;
            time_run ("mtf_cuda_thread<8>     ", [&] {mtf_cuda_thread    <CHUNK,NUM_THREADS,8>   <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread<16>    ", [&] {mtf_cuda_thread    <CHUNK,NUM_THREADS,16>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread<32>    ", [&] {mtf_cuda_thread    <CHUNK,NUM_THREADS,32>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread<64>    ", [&] {mtf_cuda_thread    <CHUNK,NUM_THREADS,64>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

            time_run ("mtf_cuda_thread_by4<8> ", [&] {mtf_cuda_thread_by4<CHUNK,NUM_THREADS,8>   <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread_by4<16>", [&] {mtf_cuda_thread_by4<CHUNK,NUM_THREADS,16>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread_by4<32>", [&] {mtf_cuda_thread_by4<CHUNK,NUM_THREADS,32>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
            time_run ("mtf_cuda_thread_by4<64>", [&] {mtf_cuda_thread_by4<CHUNK,NUM_THREADS,64>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
#endif // LIBBSC_CUDA_SUPPORT
        }

        fwrite (ptr, 1, outbytes, outfile);
        outsize += outbytes;
    }


    // The Analysis stage now is finished, we are going to display the collected data in fancy way
    auto print_stage_stats = [&] (int num, int name_width, char *name, double _insize, double outsize, double duration, const char *extra) {
        if (num >= 0)
            printf ("[%2d] ", num);
        printf("%-*s: ", name_width, name);
        if (outsize  &&  outsize != _insize) {
            char extra[99], temp1[99], temp2[99];
            sprintf (extra, " / %.2lf%%", outsize*100.0/insize[0]);
            printf ("%s => %s (%.2lf%%%s)",  show3(_insize,temp1),  show3(outsize,temp2),  outsize*100.0/_insize,  (insize[0]!=_insize? extra:""));
        }
        if (duration) {
            auto print_speed = [&] (double insize, double duration, char *suffix) {
                auto speed = ((1000/duration) *  insize) / (1 << 20);
                int digits = speed<10?2:speed<100?1:0;
                printf ("%*.*lf%s", (num>=0?5:0), digits, speed, suffix);
            };
            if (insize[0] != _insize)    // if incoming data are already compacted, print both raw and effective speeds
                print_speed (insize[0], duration, " /");
            print_speed (_insize, duration, " MiB/s");
            printf (",  %.3lf ms", duration);
        }
        printf("%s\n", extra);
    };

    for (int stage=0; stage<STAGES; stage++) {
        int name_width = 0;
        for (int i=0; i<MANY; i++) {
            if (duration[stage][i] && name[stage][i]!=0) {
                name_width = mymax(name_width, strlen(name[stage][i]));    // first, compute the width for the name column
            }
        }
        for (int i=0; i<MANY; i++) {
            if (duration[stage][i]) {
                char extra[99], temp1[99], temp2[99], temp3[99];
                sprintf (extra, "   >255: %s,  rank+len: %s,  1/2 encoding: %s", show3(long_runs,temp1), show3(ranks_plus_lens,temp2), show3(_1_2_codes,temp3));
                print_stage_stats (stage==RLE?-1:i, name_width, name[stage][i], insize[stage], size[stage][i], stage==RLE?0:duration[stage][i], stage==RLE?extra:"");
            }
        }
        // printf("\n");
    }

    fclose(infile);
    fclose(outfile);
#ifdef LIBBSC_CUDA_SUPPORT
    cudaProfilerStop();
#endif // LIBBSC_CUDA_SUPPORT
    return 0;
}
