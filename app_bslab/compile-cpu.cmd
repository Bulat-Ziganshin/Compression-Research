@echo off
set boost=C:\Base\Compiler\boost_1_61_0
set sources=bslab.cpp
set options=-I%boost% %sources%
set options_gcc=%options% -O3 -fopenmp -std=c++14 -s -static -lstdc++ -pipe -lshell32 -lole32 -loleaut32 -luuid -Wno-write-strings -Wno-shift-count-overflow
set options_clang=%options% -O2 -Wno-write-strings -Wno-deprecated-declarations
:: -g -Wa,-adhlns=srep.lst  libacof64o.lib -fno-builtin  -fno-asynchronous-unwind-tables
set options_ms=-openmp -MP -Gy -GR- -nologo -Fo%TEMP%\ -Fp%TEMP%\ %options% user32.lib shell32.lib ole32.lib advapi32.lib %* -link -LARGEADDRESSAWARE
set options_ms_cl=-O2 -EHsc %options_ms%
:: -Fa
set options_ms_icl=-O3 -Qipo /Qdiag-disable:864 %options_ms%
:: /QxHOST /Qopt-prefetch:4 /Qunroll32 /Qinline-factor:10000 /Qipo
:: -QaxCORE-AVX2,AVX,SSE4.2,SSE2  /Qprofile-functions /Qprofile-loops  /Oa /Ow /Qalias-args[-]  -fno-exceptions  /Qopt-report-file:name
set options_ms_x86=-MACHINE:x86 -SUBSYSTEM:CONSOLE,5.01
set options_ms_x64=-MACHINE:x64 -SUBSYSTEM:CONSOLE,5.02

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
cl -Febslab-msvc-x64.exe %options_ms_cl% %options_ms_x64%
if errorlevel 1   goto :exit
::goto :exit
cl -Febslab-msvc-x64-avx2.exe -arch:AVX2 %options_ms_cl% %options_ms_x64%

::gcc x64
PATH C:\Base\Compiler\MinGW-5.3\mingw64\bin;%PATH%
start /BELOWNORMAL /B g++ %options_gcc% -m64        -obslab-gcc-x64.exe %*
start /BELOWNORMAL /B g++ %options_gcc% -m64 -mavx2 -obslab-gcc-x64-avx2.exe %*

::gcc x86
PATH C:\Base\Compiler\MinGW-5.3\mingw32\bin;%PATH%
start /BELOWNORMAL /B g++ %options_gcc% -m32 -msse2 -obslab-gcc.exe %*
start /BELOWNORMAL /B g++ %options_gcc% -m32 -mavx2 -obslab-gcc-avx2.exe %*

::clang x64
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM-3.8\compile-llvm-cl-x64.cmd %options_clang% -Febslab-clang-x64.exe
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM-3.8\compile-llvm-cl-x64.cmd %options_clang% -Febslab-clang-x64-avx2.exe -arch:AVX2

::clang x86
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM-3.8\compile-llvm-cl.cmd -O2 %options_clang% -Febslab-clang.exe      -arch:SSE2
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM-3.8\compile-llvm-cl.cmd -O2 %options_clang% -Febslab-clang-avx2.exe -arch:AVX2

::msvc x86
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
cl -Febslab-msvc.exe      -arch:SSE2 %options_ms_cl% %options_ms_x86%
cl -Febslab-msvc-avx2.exe -arch:AVX2 %options_ms_cl% %options_ms_x86%

::icl x64
call C:\Base\Compiler\ICL16\bin-intel64\icl_vars.bat
icl -Febslab-icl-x64.exe                   %options_ms_icl% %options_ms_x64%
icl -Febslab-icl-x64-avx2.exe -Qxcore-AVX2 %options_ms_icl% %options_ms_x64%
iccpatch.exe bslab-icl-x64.exe >nul
iccpatch.exe bslab-icl-x64-avx2.exe >nul

::icl x86
call C:\Base\Compiler\ICL16\bin-ia32\icl_vars.bat
icl -Febslab-icl.exe      -arch:SSE2   %options_ms_icl% %options_ms_x86%
icl -Febslab-icl-avx2.exe -Qxcore-AVX2 %options_ms_icl% %options_ms_x86%
iccpatch.exe bslab-icl.exe >nul
iccpatch.exe bslab-icl-avx2.exe >nul

:exit
del *.exe.bak *.obj *.res >nul 2>nul
