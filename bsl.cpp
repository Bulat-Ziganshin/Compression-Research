// Copyright (c) 2016 Bulat Ziganshin <Bulat.Ziganshin@gmail.com>
// All rights reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

#include <stdio.h>
#include <vector>
#include <functional>
#include <stdint.h>

#ifdef LIBBSC_CUDA_SUPPORT
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>
#include "lib/cuda_common.h"       // my own cuda-specific helper functions
#endif // LIBBSC_CUDA_SUPPORT

#include "lib/wall_clock_timer.h"  // StartTimer() and GetTimer()
#include "lib/cpu_common.h"        // my own helper functions
#include "lib/libbsc.h"            // BSC common definitions
#include "lib/lz4_common.h"        // Utility functions from LZ4

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
#define LIBBSC_SORT_TRANSFORM_SUPPORT
#include "st/st.cpp"               // BSC CPU Sort Transform implementation
#include "st/st.cu"                // BSC GPU Sort Transform implementation

#include "mtf/qlfc-cpu.cpp"
#include "mtf/mtf_shelwien.cpp"
#ifdef LIBBSC_CUDA_SUPPORT
#include "mtf/mtf_scalar.cu"
#include "mtf/mtf_2symbols.cu"
#include "mtf/mtf_2buffers.cu"
#include "mtf/mtf_2buffers_depth32.cu"
#include "mtf/mtf_4by8.cu"
#include "mtf/mtf_thread.cu"
#include "mtf/mtf_thread_by4.cu"
#endif // LIBBSC_CUDA_SUPPORT

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
    enum STAGE {LZP, BWT, RLE, MTF, STAGES};
    int num[] = {-1,-1,-1,-1},  lzpHashSize = 15,  lzpMinLen = 32;
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
      ParseInt  (*src_argv, "-lzp",           &num[LZP]) ||
      ParseInt  (*src_argv, "-bwt",           &num[BWT]) ||
      ParseInt  (*src_argv, "-mtf",           &num[MTF]) ||
      ParseInt  (*src_argv, "-b",             &bufsize) ||
      ParseInt  (*src_argv, "-h",             &lzpHashSize) ||
      ParseInt  (*src_argv, "-l",             &lzpMinLen) ||
      ParseStr  (*src_argv, "-rem",           &comment) ||
      UnknownOption (*src_argv, &error) ||
      (*++dst_argv = *src_argv);
    }
    *++dst_argv = 0;  argc = dst_argv - argv;

    if (bufsize < 100*1000)
        bufsize <<= 20;  // if value is small enough, consider it as mebibytes

    if (!(argc==2 || argc==3) || error) {
        printf ("BSL: the block-sorting lab.  Part of https://github.com/Bulat-Ziganshin/Compression-Research\n"
                "Usage: bsl [options] infile [outfile]\n"
                "  -bN      buffer N (mebi)bytes (default %d MiB)\n"
                "  -nogpu   skip GPU name output\n"
                "  -nolzp   skip LZP transform\n"
                "  -nobwt   skip BWT/ST transform\n"
                "  -norle   skip RLE transform\n"
                "  -nomtf   skip MTF transform\n"
                "  -lzpN    perform only LZP transform number N\n"
                "  -hN      set LZP hash size (default 2^%d hash entries)\n"
                "  -lN      set LZP minLen (default %d)\n"
                "  -bwtN    perform only sorting transform number N\n"
                "  -mtfN    perform only MTF transform number N\n"
                "  -rem...  ignored by the program\n"
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

    int _num, stage, retval;  int64_t inbytes;  bool ret_outsize;
    uint64_t outsize = 0,  insize[STAGES] = {0},  size[STAGES][100] = {0};
    char *name[STAGES][100] = {0};  double duration[STAGES][100] = {0};

    auto cpu_time_run = [&] (char *_name, std::function<int64_t(void)> stage_f) {
        name[stage][_num] = _name;
        if (_num == num[stage]  ||  num[stage] < 0)
        {
            StartTimer();
            retval  =  stage_f();
            duration[stage][_num] += GetTimer();

            if (retval < 0  &&  retval != LIBBSC_NOT_COMPRESSIBLE) {
                printf ("%s failed with errcode %d\n", _name, retval);
                exit(4);
            }
            size[stage][_num]  +=  (ret_outsize && retval != LIBBSC_NOT_COMPRESSIBLE?  retval : inbytes);
        }
        _num++;
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


        stage = LZP,  insize[stage] += inbytes,  ret_outsize = true,  _num = 1;
        if (apply_lzp) {
            lzp_cpu_bsc (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);   // "massage" data in order to provide equal conditions for the both following lzp routines

            cpu_time_run ("lzp_cpu_bsc     ", [&] {return lzp_cpu_bsc      (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            cpu_time_run ("lzp_cpu_bsc_mod ", [&] {return lzp_cpu_bsc_mod  (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});
            cpu_time_run ("lzp_cpu_rollhash", [&] {return lzp_cpu_rollhash (inbuf, inbuf+inbytes, outbuf, outbuf+inbytes, lzpHashSize, lzpMinLen);});

            if (retval != LIBBSC_NOT_COMPRESSIBLE)
                memcpy (inbuf, outbuf, inbytes=retval);
        }


        stage = BWT,  insize[stage] += inbytes,  ret_outsize = false,  _num = 1;
        if (apply_bwt) {
            int retval = bsc_st_init(0);
            if (retval != LIBBSC_NO_ERROR) {
                printf ("bsc_st_init failed with errcode %d\n", retval);
                exit(4);
            }

#ifdef LIBBSC_CUDA_SUPPORT
            if (num[BWT] <= 4) {  // CUDA will be used, so we need to warm it up
                memcpy (outbuf, inbuf, inbytes);  bsc_st_encode_cuda (outbuf, inbytes, 5, 0);
            }

            char *cuda_st_name[] = {"st0-cuda", "st1-cuda", "st2-cuda", "st3-cuda", "st4-cuda", "st5-cuda", "st6-cuda", "st7-cuda", "st8-cuda"};
            for (int i=5; i<=8; i++)
                cpu_time_run (cuda_st_name[i], [&] {memcpy (outbuf, inbuf, inbytes);  return bsc_st_encode_cuda (outbuf, inbytes, i, 0);});
#endif // LIBBSC_CUDA_SUPPORT

            char *cpu_st_name[] = {"st0-cpu ", "st1-cpu ", "st2-cpu ", "st3-cpu ", "st4-cpu ", "st5-cpu ", "st6-cpu ", "st7-cpu ", "st8-cpu "};
            for (int i=3; i<=6; i++)
                cpu_time_run (cpu_st_name[i], [&] {memcpy (outbuf, inbuf, inbytes);  return bsc_st_encode (outbuf, inbytes, i, 0);});

            cpu_time_run ("OpenBWT ", [&] {return sais_bwt (inbuf, outbuf, bwt_tempbuf, inbytes);});
            memcpy (inbuf, outbuf, inbytes);
        }


        if (apply_mtf  &&  (1 == num[MTF]  ||  num[MTF] < 0)) {
            name[MTF][1] = "cpu (1 thread)    ";
            StartTimer();
                unsigned char MTFTable[ALPHABET_SIZE];
                ptr = qlfc (inbuf, outbuf, inbytes, MTFTable);
                outbytes = outbuf+inbytes - ptr;
            duration[MTF][1] += GetTimer();
        }


        stage = RLE,  insize[stage] += inbytes,  ret_outsize = true,  _num = 1;
        if (apply_rle) {
            cpu_time_run ("rle", [&] {return rle(inbuf,inbytes);});
            inbytes = retval;
        }


        stage = MTF,  insize[stage] += inbytes,  ret_outsize = false,  _num = 2;
        if (apply_mtf  &&  (1 != num[MTF]))
        {
            cpu_time_run ("mtf_shelwien      ", [&] {mtf_shelwien (inbuf, outbuf, inbytes);  return inbytes;});

#ifdef LIBBSC_CUDA_SUPPORT
            checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
            checkCudaErrors( cudaDeviceSynchronize());

            auto time_run = [&] (char *_name, std::function<void(void)> f) {
                name[MTF][_num] = _name;
                if (_num == num[MTF]  ||  num[MTF] < 0)
                {
                    checkCudaErrors( cudaEventRecord (start, nullptr));
                    f();
                    checkCudaErrors( cudaEventRecord (stop, nullptr));
                    checkCudaErrors( cudaDeviceSynchronize());

                    if (_num == num[MTF]) {
                        checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
                        checkCudaErrors( cudaDeviceSynchronize());
                        ptr = outbuf;
                        outbytes = inbytes;
                    }

                    float start_stop;
                    checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
                    duration[MTF][_num] += start_stop;
                }
                _num++;
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
#endif // LIBBSC_CUDA_SUPPORT
        }

        fwrite (ptr, 1, outbytes, outfile);
        outsize += outbytes;
    }


    // The Analysis stage now is finished, we are going to display the collected data in fancy way
    auto print_stage_stats = [&] (int _num, char *name, double _insize, double outsize, double duration) {
        char extra[99], temp1[99], temp2[99];
        if (_num >= 0)
            printf ("[%2d] ", _num);
        printf("%s: ", name);
        if (outsize  &&  outsize != _insize) {
            sprintf (extra, " / %.2lf%%", outsize*100.0/insize[0]);
            printf ("%s => %s (%.2lf%%%s)",  show3(_insize,temp1),  show3(outsize,temp2),  outsize*100.0/_insize,  (insize[0]!=_insize? extra:""));
        }
        if (duration) {
            auto print_speed = [&] (double insize, double duration, char *suffix) {
                auto speed = ((1000/duration) *  insize) / (1 << 20);
                int digits = speed<10?2:speed<100?1:0;
                printf ("%*.*lf%s", (_num>=0?5:0), digits, speed, suffix);
            };
            if (insize[0] != _insize)    // if incoming data are already compacted, print both raw and effective speeds
                print_speed (insize[0], duration, " /");
            print_speed (_insize, duration, " MiB/s");
            printf (",  %.3lf ms", duration);
        }
        printf("\n");
    };

    for (int stage=0; stage<STAGES; stage++) {
        for (int i=0; i<100; i++) {
            if (duration[stage][i]) {
                print_stage_stats (stage==RLE?-1:i, name[stage][i], insize[stage], size[stage][i], stage==RLE?0:duration[stage][i]);
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
