/*
 * This exercise sadly requires extra code (timer.h, files.h)
 * which I didn't have access. This code won't compile without them :(
 * Anyway, this is my optimized solution for the n-body problem:
 */
 
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

inline cudaError_t checkCuda(cudaError_t result) {
    if(result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        // break the program
        assert(result == cudaSuccess);
    }
    return result;
}

/*
* Each body contains x, y, and z coordinate positions,
* as well as velocities in the x, y, and z directions.
*/

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
* Calculate the gravitational impact of all bodies in the system
* on all others.
*/

__global__ void bodyForce(Body *bodies, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        bodies[i].vx += dt*Fx; 
        bodies[i].vy += dt*Fy; 
        bodies[i].vz += dt*Fz;
    }
}

__global__ void integratePos(Body *bodies, float dt, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        bodies[i].x += bodies[i].vx*dt;
        bodies[i].y += bodies[i].vy*dt;
        bodies[i].z += bodies[i].vz*dt;
    }
}


int main(const int argc, const char** argv) {

    // The assessment will test against both 2<11 and 2<15.
    // Feel free to pass the command line argument 15 when you gernate ./nbody report files
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);

    // The assessment will pass hidden initialized values to check for correctness.
    // You should not make changes to these files, or else the assessment will not work.
    const char * initialized_values;
    const char * solution_values;

    if (nBodies == 2<<11) {
        initialized_values = "files/initialized_4096";
        solution_values = "files/solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "files/initialized_65536";
        solution_values = "files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *buf;

    // cuda initializations
    
    int deviceId;
    cudaDeviceProp props;

    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&props, deviceId);
    
    dim3 blocks(props.multiProcessorCount * 32); // 80*32
    dim3 threads_per_block(props.warpSize * 8); // 256
    
    checkCuda(cudaMallocManaged(&buf, bytes));
    
    // further assessment initializations
    
    Body *bodies = (Body*)buf;

    read_values_from_file(initialized_values, buf, bytes);

    double totalTime = 0.0;

    // prefetch to GPU
    checkCuda(cudaMemPrefetchAsync(buf, bytes, deviceId));
    
    /*
    * This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions to reflect.
    */

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // synchronize from the last integration execution (to allow for cpu co-processing)
        checkCuda(cudaDeviceSynchronize());
        
        /*
        * You will likely wish to refactor the work being done in `bodyForce`,
        * and potentially the work to integrate the positions.
        */

        bodyForce<<<blocks, threads_per_block>>>(bodies, dt, nBodies); // compute interbody forces
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        /*
        * This position integration cannot occur until this round of `bodyForce` has completed.
        * Also, the next round of `bodyForce` cannot begin until the integration is complete.
        */

        integratePos<<<blocks, threads_per_block>>>(bodies, dt, nBodies); // integrate position
        checkCuda(cudaGetLastError());

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    // prefetch back to CPU
    checkCuda(cudaMemPrefetchAsync(buf, bytes, cudaCpuDeviceId));
    
    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, buf, bytes);

    // You will likely enjoy watching this value grow as you accelerate the application,
    // but beware that a failure to correctly synchronize the device might result in
    // unrealistically high values.
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    cudaFree(buf);
}
 