@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
mkdir m:\cuda-temp 2>nul
set defines=
set opts=-m64 -O3 -Xcompiler -openmp --expt-extended-lambda -lineinfo -keep --keep-dir m:\cuda-temp
set includes=-IC:\Base\Compiler\GPGPU\cub-1.5.2 "-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include" "-IC:\Documents and Settings\All Users\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"
::  --resource-usage timer
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc" %defines% %opts% %includes% bslab.cu cuda.lib -o bslab-cuda-x64.exe %*
del *.exp *.lib 2>nul
::"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\ptxas" -arch sm_52 mtf.ptx
::"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvdisasm.exe" elf.o >mtf.sm_52.sass
