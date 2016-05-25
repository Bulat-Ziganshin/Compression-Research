// Copyright (C) 2016 Bulat Ziganshin
// All right reserved
// Part of https://github.com/Bulat-Ziganshin/Compression-Research

#include <stdio.h>
#include <vector>
#include <functional>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_profiler_api.h >
#include <cuda.h>

#include "wall_clock_timer.h"  // StartTimer() and GetTimer()
#include "cuda_common.h"       // my own helper functions
#include "sais.c"              // OpenBWT implementation

const int ALPHABET_SIZE = 256;
const int WARP_SIZE = 32;
typedef unsigned char byte;

// Parameters
const int BUFSIZE = 128*1024*1024;
const int CHUNK = 4*1024;
#define SYNC_WARP __threadfence_block  /* alternatively, __syncthreads or, better, __threadfence_warp */

#include "qlfc-cpu.cpp"
#include "mtf_scalar.cu"
#include "mtf_2symbols.cu"
#include "mtf_2buffers.cu"
#include "mtf_2buffers_depth32.cu"
#include "mtf_4by8.cu"
#include "mtf_thread.cu"
#include "mtf_thread_by4.cu"

// In-place RLE transformation (run lengths are dropped!)
int rle (byte* buf, int size)
{
    int c = -1, run=0;
    auto out = buf;
    for (size_t i = 0; i < size; i++)
    {
        buf[i]==c?  run++  :  (run=1, c = *out++ = buf[i]);
    }
    return out-buf;
}


int main (int argc, char **argv)
{
    bool apply_gpu_display = true;
    if (argv[1] && strcmp(argv[1],"-nogpu")==0) {
        apply_gpu_display = false;
        argv++, argc--;
    }

    bool apply_bwt = true;
    if (argv[1] && strcmp(argv[1],"-nobwt")==0) {
        apply_bwt = false;
        argv++, argc--;
    }

    bool apply_rle = true;
    if (argv[1] && strcmp(argv[1],"-norle")==0) {
        apply_rle = false;
        argv++, argc--;
    }

    if (!(argc==2 || argc==4)) {
        printf ("Usage: mtf [options] infile [N outfile]\n"
                "  N is the number of function those output will be saved\n"
                "  -nogpu   skip GPU name output\n"
                "  -nobwt   skip BWT transformation\n"
                "  -norle   skip RLE transformation\n"
                );
        return 0;
    }

    unsigned char* d_inbuf;
    unsigned char* d_outbuf;
    checkCudaErrors( cudaMalloc((void**)(&d_inbuf),  BUFSIZE+CHUNK*2));  // up to CHUNK*2 extra bytes may be processed
    checkCudaErrors( cudaMalloc((void**)(&d_outbuf), BUFSIZE+CHUNK*2));

    cudaEvent_t start, stop;
    checkCudaErrors( cudaEventCreate(&start));
    checkCudaErrors( cudaEventCreate(&stop));

    unsigned char* inbuf  = new unsigned char[BUFSIZE];
    unsigned char* outbuf = new unsigned char[BUFSIZE];
    int*      bwt_tempbuf = new int          [BUFSIZE];

    double insize = 0,  outsize = 0,  duration[100] = {0};  char *mtf_name[100] = {"cpu (1 thread)"};

    FILE* infile  = fopen (argv[1], "rb");
    FILE* outfile = fopen (argv[2]? argv[3] : "nul", "wb");
    if (!infile) {
        printf ("Can't open infile %s\n", argv[1]);
        return 1;
    }
    if (!outfile) {
        printf ("Can't open outfile %s\n", argv[3]);
        return 1;
    }
    int save_num  =  argc==4? atoi(argv[2]) : 0;
    if (apply_gpu_display)
        DisplayCudaDevice();


    for (int inbytes; !!(inbytes = fread(inbuf,1,BUFSIZE,infile)); )
    {
        if (apply_bwt) {
            auto bwt_errcode  =  sais_bwt (inbuf, outbuf, bwt_tempbuf, inbytes);
            if (bwt_errcode < 0) {
                printf ("BWT failed with errcode %d\n", bwt_errcode);
                return 2;
            }
            memcpy (inbuf, outbuf, inbytes);
        }

        StartTimer();
            unsigned char MTFTable[ALPHABET_SIZE];
            auto ptr = qlfc (inbuf, outbuf, inbytes, MTFTable);
            auto outbytes = outbuf+inbytes - ptr;
        duration[0] += GetTimer();
        int num = 1;

        insize += inbytes;
        if (apply_rle)
            inbytes = rle(inbuf,inbytes);

        checkCudaErrors( cudaMemcpy (d_inbuf, inbuf, inbytes, cudaMemcpyHostToDevice));
        checkCudaErrors( cudaDeviceSynchronize());

        auto time_run = [&] (char *name, std::function<void(void)> f) {
            mtf_name[num] = name;
            checkCudaErrors( cudaEventRecord (start, nullptr));
            f();
            checkCudaErrors( cudaEventRecord (stop, nullptr));
            checkCudaErrors( cudaDeviceSynchronize());

            if (num == save_num) {
                checkCudaErrors( cudaMemcpy (outbuf, d_outbuf, inbytes, cudaMemcpyDeviceToHost));
                checkCudaErrors( cudaDeviceSynchronize());
                ptr = outbuf;
                outbytes = inbytes;
            }

            float start_stop;
            checkCudaErrors( cudaEventElapsedTime (&start_stop, start, stop));
            duration[num] += start_stop;
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

    printf("rle: %.0lf => %.0lf (%.2lf%%)\n", insize, outsize, outsize*100.0/insize);
    for (int i=0; i<sizeof(duration)/sizeof(*duration); i++) {
        if (duration[i]) {
            char in_speed[100], out_speed[100];
            sprintf( in_speed,   "%5.0lf", ((1000/duration[i]) *  insize) / (1 << 20));
            sprintf(out_speed, " /%5.0lf", ((1000/duration[i]) * outsize) / (1 << 20));
            printf("[%2d] %-*s: %s%s MiB/s,  %.3lf ms\n", i, strlen(mtf_name[2]), mtf_name[i], in_speed, (apply_rle?out_speed:""), duration[i]);
        }
    }
    fclose(infile);
    fclose(outfile);
    cudaProfilerStop();
    return 0;
}
