# include <stdio.h>
#include <cuda_runtime.h>

//Parallel Reduction: Each thread computes the sum of a chunk of the input array, and writes one partial sum per block to the output array

__global__ void sum_kernel(int* g_input, int* g_output)        // pointers to where data is read from and written to
{
    extern __shared__ int s_data[];  // shared memory allocated during kernel launch

    // read input into shared memory
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;   //global thread index
    s_data[threadIdx.x] = g_input[idx];     //reading from g_input and writing to s_data 
    __syncthreads();        // the threads must finish writing into s_data[] before any thread starts reading from it.

    // compute sum for the threadblock
    for (int dist = blockDim.x / 2; dist > 0; dist /= 2)
    {
        if (threadIdx.x < dist)
            s_data[threadIdx.x] += s_data[threadIdx.x + dist];
        __syncthreads();
    }
    // write the block's sum to global memory
    if (threadIdx.x == 0)
        g_output[blockIdx.x] = s_data[0];
}

// CPU (main) → send data to GPU → run kernel → get results back → print
int main()
{
    int h_input[8] = { 1,2,3,4,5,6,7,8 };       // host input array created
    int h_output[2];        // host output array with 2 elememts (because 2 blocks) created

    // Create pointers for GPU memory
    int* d_input, * d_output;   // d_ -> device (GPU memory)

    // allocate memory on GPU
    cudaMalloc(&d_input, 8 * sizeof(int));      // creates memory on GPU     
    cudaMalloc(&d_output, 2 * sizeof(int));

    cudaMemcpy(d_input, h_input, 8 * sizeof(int), cudaMemcpyHostToDevice);      // Copy data from CPU → GPU

    // launch kernel
    // <<< blocks, threads_per_block, shared_memory >>>
    sum_kernel << <2, 4, 4 * sizeof(int) >> > (d_input, d_output);
    // total threads = 8
    // Each thread : Loads 1 element → stores in shared memory → participates in reduction
    // Each block : computes 1 sum

    cudaMemcpy(h_output, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);    // Copy result back GPU → CPU

    printf("Block sums: %d %d\n", h_output[0], h_output[1]);
    // Block 0 → 1+2+3+4 = 10
    // Block 1 → 5 + 6 + 7 + 8 = 26

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}