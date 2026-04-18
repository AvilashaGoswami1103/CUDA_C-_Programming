# include<stdio.h>
# include<cuda_runtime.h>

// computing matrix multiplication:
// C = A×B
// Each thread computes one element of C.

__global__ void matmul(float* A, float* B, float* C, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Each thread computes C[row][col]
	// threadIdx.x -> column, threadIdx.y -> row, blockIdx.x -> column blocks, blockIdx.y -> row blocks

	if (row < N && col < N) {	// boundary check
		float sum = 0.0f;

		for (int k = 0; k < N; k++) {
			sum += A[row * N + k] * B[k * N + col];
		}
		// Each element: C[row][col] = k=0∑N−1​ A[row][k]×B[k][col]


		C[row * N + col] = sum;
		//Writes final value back to global memory
	}
}

int main()
{
    int N = 4;  // matrix size (NxN)

    size_t size = N * N * sizeof(float);
    // total = 4 x N x N bytes = 64 bytes

    // Host memory
    float* h_A = (float*)malloc(size);  // input matrix A
    float* h_B = (float*)malloc(size);  // input matrix B
    float* h_C = (float*)malloc(size);  // output matrix

    // Initialize matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = i + 1;
        printf("%f ", h_A[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N * N; i++)
    {
        h_B[i] = 1;   // simple matrix for easy verification
        printf("%f ", h_B[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    // Device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockSize(16, 16);     // 16 × 16 = 256 threads per block (standard)
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x,
        (N + blockSize.y - 1) / blockSize.y);
    // so, for N = 4, numBlocks = (1,1)

    // Launch kernel
    matmul << <numBlocks, blockSize >> > (d_A, d_B, d_C, N);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("\nResult Matrix C:\n");
    for (int i = 0; i < N * N; i++)
    {
        printf("%f ", h_C[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    // Free memory
    // cpu memory
    free(h_A);
    free(h_B);
    free(h_C);

    //gpu memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}