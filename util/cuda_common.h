#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda_runtime.h>



// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}



// Display CUDA device
inline void DisplayCudaDevice()
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
    int memoryClock, memBusWidth;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
    getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);

    int sm_per_multiproc  =  deviceProp.major == 9999 && deviceProp.minor == 9999
                                ? 1
                                : _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    // printf("Device %d", dev);
    printf("%s, CC %d.%d.  VRAM %.1lf GB,", deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem/double(1<<30));
    printf(" %.0lf MHz * %d-bit = %.0lf GB/s.", memoryClock*0.001, memBusWidth, 1e-6*memoryClock*memBusWidth/4);
    printf("  %d SM * %d alu * %.0lf MHz * 2 = %.2lf TFLOPS\n", deviceProp.multiProcessorCount, sm_per_multiproc, deviceProp.clockRate*0.001,
                                                     1e-9 * deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate * 2);

}
