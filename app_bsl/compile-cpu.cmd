@echo off
set boost=C:\Base\Compiler\boost_1_61_0
set sources=bsl.cpp
set options=-I%boost% %sources%
set options_gcc=%options% -O3 -fopenmp -std=c++14 -s -static -lstdc++ -pipe -lshell32 -lole32 -loleaut32 -luuid -Wno-write-strings -Wno-shift-count-overflow
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
cl -Febsl-msvc-x64.exe %options_ms_cl% %options_ms_x64%
if errorlevel 1   goto :exit
::goto :exit
cl -Febsl-msvc-x64-avx2.exe -arch:AVX2 %options_ms_cl% %options_ms_x64%

::gcc x64
PATH C:\Base\Compiler\MinGW-5.3\mingw64\bin;%PATH%
start /BELOWNORMAL /B g++ %options_gcc% -m64        -obsl-gcc-x64.exe %*
start /BELOWNORMAL /B g++ %options_gcc% -m64 -mavx2 -obsl-gcc-x64-avx2.exe %*

::gcc x86
PATH C:\Base\Compiler\MinGW-5.3\mingw32\bin;%PATH%
start /BELOWNORMAL /B g++ %options_gcc% -m32        -obsl-gcc.exe %*
start /BELOWNORMAL /B g++ %options_gcc% -m32 -mavx2 -obsl-gcc-avx2.exe %*

::clang x64
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM\compile-llvm-cl.cmd -O2 -Wno-write-strings -Wno-deprecated-declarations bsl.cpp -Febsl-clang-x64.exe
start /BELOWNORMAL /B cmd /c C:\Base\Compiler\LLVM\compile-llvm-cl.cmd -O2 -Wno-write-strings -Wno-deprecated-declarations bsl.cpp -Febsl-clang-x64-avx2.exe -arch:AVX2

::msvc x86
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
cl -Febsl-msvc.exe                 %options_ms_cl% %options_ms_x86%
cl -Febsl-msvc-avx2.exe -arch:AVX2 %options_ms_cl% %options_ms_x86%

::icl x64
call C:\Base\Compiler\ICL16\bin-intel64\icl_vars.bat
icl -Febsl-icl-x64.exe                   %options_ms_icl% %options_ms_x64%
icl -Febsl-icl-x64-avx2.exe -Qxcore-AVX2 %options_ms_icl% %options_ms_x64%
iccpatch.exe bsl-icl-x64.exe >nul
iccpatch.exe bsl-icl-x64-avx2.exe >nul

::icl x86
call C:\Base\Compiler\ICL16\bin-ia32\icl_vars.bat
icl -Febsl-icl.exe      -arch:SSE2   %options_ms_icl% %options_ms_x86%
icl -Febsl-icl-avx2.exe -Qxcore-AVX2 %options_ms_icl% %options_ms_x86%
iccpatch.exe bsl-icl.exe >nul
iccpatch.exe bsl-icl-avx2.exe >nul

:exit
del *.exe.bak *.obj *.res >nul 2>nul
