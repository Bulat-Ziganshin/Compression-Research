for %a in (boost e8 e9 100m 1g 1g.tor3) do for %x in (-x64-avx2.exe -x64.exe -avx2.exe .exe) do for %c in (icl clang gcc msvc) do bslab-%c%x z:\%a
for %a in (boost e8 e9 100m 1g 1g.tor3) do for %e in (bslab-cuda-x64.exe bslab-cuda.exe) do %e -nogpu z:\%a
