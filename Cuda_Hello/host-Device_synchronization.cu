# include<stdio.h>
# include<cuda_runtime.h>

__global__ void increment_gpu(float* d_A, float b, int N)
{
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (VERY important)
    if (idx < N)
    {
        d_A[idx] += b;      // takes one element → adds b → writes back
    }
}
int main() {
    // Define parametersf
    int N = 1024;
    int blockSize = 256;
    float b = 2.0f;
    // N → number of elements, blockSize → threads per block, b → value to add

    // Host Code
    unsigned int numBytes = N * sizeof(float);   // allocate total memory needed for array
    float* h_A = (float*)malloc(numBytes);		// host array
    // Creates array in RAM (CPU memory)

    // Allocate device memory 
    float* d_A = 0;     // d_A -> pointer in GPU memory
    cudaMalloc((void**)&d_A, numBytes);     //cudaMalloc → allocates memory on GPU

    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
    }

    //Copy data from host to device
    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);     // moves data from CPU to GPU

    //Execute the kernel
    int numBlocks = (N + blockSize - 1) / blockSize;    //Ensures enough threads for all elements
    increment_gpu << <N / numBlocks, blockSize >> > (d_A, b, N);       // <<< number_of_blocks, threads_per_block >>>
    // GPU creates Blocks × Threads = total threads, each thread runs increment_gpu, each thread processes 1 element of the array

    //Copy data from device back to host
    cudaMemcpy(h_A, d_A, numBytes, cudaMemcpyDeviceToHost);

    // Print first 10 results (for verification)
    printf("First 10 elements after increment:\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", h_A[i]);

    printf("\n");

    // Free memory
    cudaFree(d_A);
    free(h_A);

    return 0;
}

