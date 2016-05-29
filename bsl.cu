// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

#include <stdio.h>
#include <vector>
#include <functional>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>

#include "lib/wall_clock_timer.h"  // StartTimer() and GetTimer()
#include "lib/cpu_common.h"        // my own helper functions
#include "lib/cuda_common.h"       // my own cuda-specific helper functions

#include "lib/libbsc.h"            // BSC common definitions
const int WARP_SIZE = 32;
typedef unsigned char byte;

// Parameters
const int DEFAULT_BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
#define SYNC_WARP __threadfence_block  /* alternatively, __syncthreads or, better, __threadfence_warp */

#include "lzp/lzp-cpu-bsc.cpp"
#include "lzp/lzp-cpu-bsc-mod.cpp"
#include "lzp/lzp-cpu-rollhash.cpp"

#include "bwt/sais.c"              // OpenBWT implementation

#include "mtf/qlfc-cpu.cpp"
#include "mtf/mtf_scalar.cu"
#include "mtf/mtf_2symbols.cu"
#include "mtf/mtf_2buffers.cu"
#include "mtf/mtf_2buffers_depth32.cu"
#include "mtf/mtf_4by8.cu"
#include "mtf/mtf_thread.cu"
#include "mtf/mtf_thread_by4.cu"

// In-place RLE transformation (run lengths are dropped!)
int rle (byte* buf, int size)
{
    int c = -1,  run = 0;
    auto out = buf;
    for (size_t i = 0; i < size; i++)
    {
        buf[i]==c?  run++  :  (run=1, c = *out++ = buf[i]);
    }
    return out-buf;
}

int main (int argc, char **argv)
{
    bool display_gpu = true;
    bool apply_lzp = true;
    bool apply_bwt = true;
    bool apply_rle = true;
    bool apply_mtf = true;
    int  mtf_num = -1,  lzp_num = -1,  lzpHashSize = 15,  lzpMinLen = 32;
    size_t bufsize = DEFAULT_BUFSIZE;
    char *comment;
    int error = 0;

    auto src_argv = argv,  dst_argv = argv;
    while (*++src_argv) {
      ParseBool (*src_argv, "-gpu", "-nogpu", &display_gpu) ||
      ParseBool (*src_argv, "-lzp", "-nolzp", &apply_lzp) ||
      ParseBool (*src_argv, "-bwt", "-nobwt", &apply_bwt) ||
      ParseBool (*src_argv, "-rle", "-norle", &apply_rle) ||
      ParseBool (*src_argv, "-mtf", "-nomtf", &apply_mtf) ||
      ParseInt  (*src_argv, "-lzp",           &lzp_num) ||
      ParseInt  (*src_argv, "-mtf",           &mtf_num) ||
      ParseInt  (*src_argv, "-b",             &bufsize) ||
      ParseInt  (*src_argv, "-h",             &lzpHashSize) ||
      ParseInt  (*src_argv, "-l",             &lzpMinLen) ||
      ParseStr  (*src_argv, "-rem",           &comment) ||
      UnknownOption (*src_argv, &error) ||
      (*++dst_argv = *src_argv);
    }
    *++dst_argv = 0;  argc = dst_argv - argv;

    if (bufsize < 100*1000)
        bufsize <<= 20;  // megabytes

    if (!(argc==2 || argc==3) || error) {
        printf ("BSL: the block-sorting lab.  Part of https://github.com/Bulat-Ziganshin/Compression-Research\n"
                "Usage: mtf [options] infile [outfile]\n"
                "  -nogpu   skip GPU name output\n"
                "  -nolzp   skip LZP transformation\n"
                "  -nobwt   skip BWT transformation\n"
                "  -norle   skip RLE transformation\n"
                "  -nomtf   skip MTF transformation\n"
                "  -lzpN    perform only LZP transformation number N\n"
                "  -hN      set LZP hash size (default 2^%d hash entries)\n"
                "  -lN      set LZP minLen (default %d)\n"
                "  -mtfN    perform only MTF transformation number N\n"
                "  -bN      buffer N (mebi)bytes (default %d MiB)\n"
                , lzpHashSize, lzpMinLen, DEFAULT_BUFSIZE>>20);
        return argc==1 && !error?  0 : 1;
    }

    unsigned char* d_inbuf;
    unsigned char* d_outbuf;
    checkCudaErrors( cudaMalloc((void**)(&d_inbuf),  bufsize+CHUNK*2+256));  // up to CHUNK*2 extra bytes may be processed, plus a few extra bytes may be read after bufend
    checkCudaErrors( cudaMalloc((void**)(&d_outbuf), bufsize+CHUNK*2+256));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    unsigned char* inbuf  = new unsigned char[bufsize];
    unsigned char* outbuf = new unsigned char[bufsize];
    int*      bwt_tempbuf = apply_bwt? new int[bufsize] : 0;

    double insize = 0,  after_lzp = 0,  outsize = 0,  bwt_duration = 0,  duration[100] = {0};  char *mtf_name[100] = {"cpu (1 thread)"};
    double lzp_size[100] = {0},  lzp_duration[100] = {0};  char *lzp_name[100];

    FILE* infile  = fopen (argv[1], "rb");
    FILE* outfile = fopen (argv[2]? argv[2] : "nul", "wb");
    if (!infile) {
        printf ("Can't open infile %s\n", argv[1]);
        return 2;
    }
    if (!outfile) {
        printf ("Can't open outfile %s\n", argv[3]);
        return 3;
    }
    if (display_gpu)
        DisplayCudaDevice();


    // All preparations now are done. Now we are in the Analysis stage, processing input data with various algos and recording speed/outsize of every experiment
    for (int inbytes; !!(inbytes = fread(inbuf,1,bufsize,infile)); )
    {
        insize += inbytes;
        byte *ptr = inbuf;  size_t outbytes = inbytes;  // output buffer

        if (apply_lzp) {
            int num = 1,  lzp_errcode;
            lzp_cpu_bsc (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);   // "massage" data in order to provide equal conditions for the both following lzp routines

            auto lzp_time_run = [&] (char *name, std::function<int(void)> lzp_f) {
                lzp_name[num] = name;
                if (num == lzp_num  ||  lzp_num < 0)
                {
                    StartTimer();
                    lzp_errcode  =  lzp_f();
                    lzp_duration[num] += GetTimer();

                    if (lzp_errcode < 0  &&  lzp_errcode != LIBBSC_NOT_COMPRESSIBLE) {
                        printf ("%s failed with errcode %d\n", name, lzp_errcode);
                        exit(4);
                    }
                    lzp_size[num]  +=  (lzp_errcode != LIBBSC_NOT_COMPRESSIBLE?  lzp_errcode : inbytes);
                }
                num++;
            };
            lzp_time_run ("lzp_cpu_bsc     ", [&] {return lzp_cpu_bsc      (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            lzp_time_run ("lzp_cpu_bsc_mod ", [&] {return lzp_cpu_bsc_mod  (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            lzp_time_run ("lzp_cpu_rollhash", [&] {return lzp_cpu_rollhash (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});

            if (lzp_errcode != LIBBSC_NOT_COMPRESSIBLE)
                memcpy (inbuf, outbuf, inbytes=lzp_errcode);
        }
        after_lzp += inbytes;

        if (apply_bwt) {
            StartTimer();
            auto bwt_errcode  =  sais_bwt (inbuf, outbuf, bwt_tempbuf, inbytes);
            bwt_duration += GetTimer();
            if (bwt_errcode < 0) {
                printf ("BWT failed with errcode %d\n", bwt_errcode);
                return 5;
            }
            memcpy (inbuf, outbuf, inbytes);
        }

        if (0 == mtf_num  ||  mtf_num < 0) {
            StartTimer();
                unsigned char MTFTable[ALPHABET_SIZE];
                ptr = qlfc (inbuf, outbuf, inbytes, MTFTable);
                outbytes = outbuf+inbytes - ptr;
            duration[0] += GetTimer();
        }
        int num = 1;

        if (apply_rle) {
            inbytes = rle(inbuf,inbytes);
        }

        checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
        checkCudaErrors( cudaDeviceSynchronize());

        auto time_run = [&] (char *name, std::function<void(void)> f) {
            mtf_name[num] = name;
            if (num == mtf_num  ||  mtf_num < 0)
            {
                checkCudaErrors( cudaEventRecord (start, nullptr));
                f();
                checkCudaErrors( cudaEventRecord (stop, nullptr));
                checkCudaErrors( cudaDeviceSynchronize());

                if (num == mtf_num) {
                    checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
                    checkCudaErrors( cudaDeviceSynchronize());
                    ptr = outbuf;
                    outbytes = inbytes;
                }

                float start_stop;
                checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
                duration[num] += start_stop;
            }
            num++;
        };

{
        const int NUM_WARPS = 4;
        time_run ("mtf_scalar        ", [&] {mtf_scalar    <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_2symbols      ", [&] {mtf_2symbols  <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS)+1,   NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_2buffers      ", [&] {mtf_2buffers  <CHUNK,NUM_WARPS>       <<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_2buffers<32>  ", [&] {mtf_2buffers_depth32 <CHUNK,NUM_WARPS><<<(inbytes-1)/(CHUNK*NUM_WARPS*2)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_3buffers<32>  ", [&] {mtf_2buffers_depth32 <CHUNK,NUM_WARPS,3><<<(inbytes-1)/(CHUNK*NUM_WARPS*3)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_4buffers<32>  ", [&] {mtf_2buffers_depth32 <CHUNK,NUM_WARPS,4><<<(inbytes-1)/(CHUNK*NUM_WARPS*4)+1, NUM_WARPS*WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
{
        const int NUM_THREADS = 8*WARP_SIZE,  NUM_BUFFERS = NUM_THREADS/8;
        time_run ("mtf_4by8          ", [&]{mtf_4by8<CHUNK,NUM_THREADS,NUM_BUFFERS><<<(inbytes-1)/(CHUNK*NUM_BUFFERS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
{
        const int NUM_THREADS = 4*WARP_SIZE,  NUM_BUFFERS = NUM_THREADS/4;
        time_run ("mtf_8by4          ", [&]{mtf_4by8<CHUNK,NUM_THREADS,NUM_BUFFERS><<<(inbytes-1)/(CHUNK*NUM_BUFFERS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
}
        time_run ("mtf_thread        ", [&] {mtf_thread    <CHUNK>                 <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread_by4    ", [&] {mtf_thread_by4<CHUNK>                 <<<(inbytes-1)/(CHUNK*WARP_SIZE)+1,             WARP_SIZE>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

        const int NUM_THREADS = 1*WARP_SIZE;
        time_run ("mtf_thread<8>     ", [&] {mtf_thread    <CHUNK,NUM_THREADS,8>   <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread<16>    ", [&] {mtf_thread    <CHUNK,NUM_THREADS,16>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread<32>    ", [&] {mtf_thread    <CHUNK,NUM_THREADS,32>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread<64>    ", [&] {mtf_thread    <CHUNK,NUM_THREADS,64>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

        time_run ("mtf_thread_by4<8> ", [&] {mtf_thread_by4<CHUNK,NUM_THREADS,8>   <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread_by4<16>", [&] {mtf_thread_by4<CHUNK,NUM_THREADS,16>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread_by4<32>", [&] {mtf_thread_by4<CHUNK,NUM_THREADS,32>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});
        time_run ("mtf_thread_by4<64>", [&] {mtf_thread_by4<CHUNK,NUM_THREADS,64>  <<<(inbytes-1)/(CHUNK*NUM_THREADS)+1,         NUM_THREADS>>> (d_inbuf, d_outbuf, inbytes, CHUNK);});

        fwrite (ptr, 1, outbytes, outfile);
        outsize += outbytes;
    }


    // The Analysis stage now is finished, we are going to display the collected data in fancy way
    auto print_stage_stats = [&] (int num, char *name, double insize, double outsize, double duration) {
        if (num >= 0)
            printf("[%2d] ", num);
        printf("%s: ", name);
        if (outsize >= 0  &&  outsize != insize)
            printf("%.0lf => %.0lf (%.2lf%%)", insize, outsize, outsize*100/insize);
        if (duration) {
            auto speed = ((1000/duration) *  insize) / (1 << 20);
            int digits = speed<10?3:speed<100?2:0;
            printf("%*.*lf MiB/s,  %.3lf ms", (num>=0?5:0), digits, speed, duration);
        }
        printf("\n");
    };

    for (int i=1; i<sizeof(lzp_duration)/sizeof(*lzp_duration); i++) {
        if (lzp_duration[i]) {
            print_stage_stats (i, lzp_name[i], insize, lzp_size[i], lzp_duration[i]);
        }
    }

    if (apply_bwt)  print_stage_stats (-1, "bwt", after_lzp, -1, bwt_duration);
    if (apply_rle)  print_stage_stats (-1, "rle", after_lzp, outsize, 0);

    for (int i=0; i<sizeof(duration)/sizeof(*duration); i++) {
        if (duration[i]) {
            char in_speed[100], out_speed[100];
            sprintf( in_speed,   "%5.0lf", ((1000/duration[i]) *  insize) / (1 << 20));
            sprintf(out_speed, " /%5.0lf", ((1000/duration[i]) * outsize) / (1 << 20));
            printf("[%2d] %-*s: %s%s MiB/s,  %.3lf ms\n", i, strlen(mtf_name[2]), mtf_name[i], in_speed, (outsize!=insize?out_speed:""), duration[i]);
        }
    }
    fclose(infile);
    fclose(outfile);
    cudaProfilerStop();
    return 0;
}
