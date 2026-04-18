// IN c++
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simpleKernel() {
    printf("Hello World from GPU!\n");
}

int main() {
    simpleKernel << <1, 1 >> > ();  //1 block, 1 thread
    cudaDeviceSynchronize();
    // forces the CPU (host) to wait until all previously launched GPU (device) tasks—kernels and memory operations—have finished executing.
    return 0;
}
//end of file