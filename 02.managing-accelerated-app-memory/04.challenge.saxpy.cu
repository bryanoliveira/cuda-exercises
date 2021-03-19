#include <stdio.h>
#include <assert.h>

#define N 2048 * 2048 // Number of elements in each vector

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Initialize memory
__global__ void initVectors(int * a, int * b, int * c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(; i < N; i += stride) {
        a[i] = 2;
        b[i] = 1;
        c[i] = 0;
    }
}

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 20us.
 *
 * EDIT: I made it run under 77 us :)
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (; i < N; i += stride)
        c[i] = 2 * a[i] + b[i];
}

int main()
{
    int *a, *b, *c;
    int size = N * sizeof (int); // The total number of bytes per vector
    
    int deviceId;
    cudaDeviceProp props;
    
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);

    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c, size));

    int threads_per_block = props.warpSize * 8;
    int number_of_blocks = props.multiProcessorCount * 32; // using stride is better than arbitrary blocks
    
    // prefetch to gpu
    checkCuda(cudaMemPrefetchAsync(a, size, deviceId));
    checkCuda(cudaMemPrefetchAsync(b, size, deviceId));
    checkCuda(cudaMemPrefetchAsync(c, size, deviceId));
    
    initVectors <<< number_of_blocks, threads_per_block >>>( a, b, c );
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    
    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
