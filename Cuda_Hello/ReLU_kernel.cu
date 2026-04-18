# include<stdio.h>
# include<cuda_runtime.h>
#include <math.h>

__global__ void relu(float* x, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        x[idx] = fmaxf(0.0f, x[idx]);   // ReLU
        // The function fmaxf in C/C++ is a math library function that returns 
        // the larger (maximum) of two floating-point values of type float
    }
}

int main() {
    int N = 10;
    float h_x[10] = { -1, 2, -3, 4, -5, 6, -7, 8, -9, 10 };

    float* d_x;     // initialized gpu array

    //Allocate memory on GPU
    cudaMalloc((void**)&d_x, N * sizeof(float));
    //Why void**? Because cudaMalloc needs to modify your pointer (d_x) 
    // to point to the newly allocated GPU memory.
    
    //Copy data from CPU -> GPU
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    //Launch Kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    relu <<<numBlocks, blockSize >>> (d_x, N);

    //Wait for GPU to finish
    cudaDeviceSynchronize();

    //Copy Result back to CPU
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    //Print result
    printf("ReLU Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_x[i]);
    }

    //Free GPU Memory
    cudaFree(d_x);

    return 0;
}