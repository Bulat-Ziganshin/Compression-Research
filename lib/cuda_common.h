#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

// Display CUDA device
inline void DisplayCudaDevice()
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    int sm_per_multiproc  =  deviceProp.major == 9999 && deviceProp.minor == 9999
                                ? 1
                                : _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    // printf("Device %d", dev);
    printf("%s, SM %d.%d, %.1lf GB VRAM", deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem/double(1<<30));
    printf(".  %d SM * %d alu * %.0lf MHz = %.0lf GIOPS\n", deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate*0.001,
                                                     1e-6 * deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate);
}
