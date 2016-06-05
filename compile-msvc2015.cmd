@echo off
set boost=C:\Base\Compiler\boost_1_58_0
set sources=bsl.cpp
set options=-I%boost% %sources%
set options_gcc4=%options% -O2 -s -static -std=c++1y -lstdc++ -pipe -lshell32 -lole32 -loleaut32 -luuid -Wno-write-strings
:: -g -Wa,-adhlns=srep.lst  libacof64o.lib -fno-builtin  -fno-asynchronous-unwind-tables
set options_ms=-openmp -MP -Gy -GR- -nologo -Fo%TEMP%\ -Fp%TEMP%\ %options% user32.lib shell32.lib ole32.lib advapi32.lib %* -link -LARGEADDRESSAWARE
set options_ms_cl=-O2 -EHsc %options_ms%
:: -Fa
set options_ms_icl=-w -O3 -Qipo %options_ms%
:: /QxHOST /Qopt-prefetch:4 /Qunroll32 /Qinline-factor:10000 /Qipo
:: -QaxCORE-AVX2,AVX,SSE4.2,SSE2  /Qprofile-functions /Qprofile-loops  /Oa /Ow /Qalias-args[-]  -fno-exceptions  /Qopt-report-file:name
set options_ms_x86=-MACHINE:x86 -SUBSYSTEM:CONSOLE,5.01
set options_ms_x64=-MACHINE:x64 -SUBSYSTEM:CONSOLE,5.02

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64
cl -Febsl-msvc-x64.exe %options_ms_cl% %options_ms_x64%
cl -Febsl-msvc-x64-avx2.exe -arch:AVX2 %options_ms_cl% %options_ms_x64%

:exit
del *.exe.bak *.obj *.res >nul 2>nul
