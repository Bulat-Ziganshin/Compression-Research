for %a in (boost e8 e9 100m 1g 1g.tor3) do for %x in (-x64-avx2.exe -x64.exe -avx2.exe .exe) do for %c in (icl clang gcc msvc) do bslab-%c%x z:\%a
for %a in (boost e8 e9 100m 1g 1g.tor3) do for %e in (bslab-cuda-x64.exe bslab-cuda.exe) do %e -nogpu z:\%a
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvprof.exe" --events all --metrics all  --log-file profile --replay-mode application bslab-cuda-x64.exe -bwt11 -lzp3 -mtf-1-4 z:\e8
