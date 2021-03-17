#include <stdio.h>
#include <assert.h>

#define N 64

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrixMulGPU(int *a, int *b, int *c) {
    // define matrix indexes
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    dim3 stride(blockDim.x * gridDim.x, blockDim.y * gridDim.y);

    // perform stride avoiding out of bound access
    for (; row < N; row += stride.x) {
        for(; col < N; col += stride.y) {
            // perform matrix multiplication
            int val = 0;
            for (int k = 0; k < N; ++k) val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
    }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

void matrixMulCPU(int *a, int *b, int *c) {
    int val = 0;

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            val = 0;
            for (int k = 0; k < N; ++k) val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

int main() {
    int *a, *b, *c_cpu, *c_gpu;  // Allocate a solution matrix for both the CPU
                                 // and the GPU operations

    int size = N * N * sizeof(int);  // Number of bytes of an N x N matrix

    // Allocate memory
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c_cpu, size));
    checkCuda(cudaMallocManaged(&c_gpu, size));

    // Initialize memory; create 2D matrices
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            a[row * N + col] = row;
            b[row * N + col] = col + 2;
            c_cpu[row * N + col] = 0;
            c_gpu[row * N + col] = 0;
        }

    /*
     * Assign `threads_per_block` and `number_of_blocks` 2D values
     * that can be used in matrixMulGPU above.
     */

    // 256 threads per block
    dim3 threads_per_block(16, 16);
    // enough blocks in each dimension to perform the whole calculation
    dim3 number_of_blocks(
        (N + threads_per_block.x - 1) / threads_per_block.x, 
        (N + threads_per_block.y - 1) / threads_per_block.y
    );
    matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a, b, c_gpu);
    cudaDeviceSynchronize();

    checkCuda(cudaGetLastError());

    // Call the CPU version to check our work
    matrixMulCPU(a, b, c_cpu);

    // Compare the two answers to make sure they are equal
    bool error = false;
    for (int row = 0; row < N && !error; ++row)
        for (int col = 0; col < N && !error; ++col)
            if (c_cpu[row * N + col] != c_gpu[row * N + col]) {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error) printf("Success!\n");

    // Free all our allocated memory
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c_cpu));
    checkCuda(cudaFree(c_gpu));
}
