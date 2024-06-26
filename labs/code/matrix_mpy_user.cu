#include <iostream>
#include "ee155_utils.hxx"
#include "matrix.hxx"
const int BS = 32;	// The blocks are BS x BS.
template<int size> __device__ void print_array (float arr [size][size]);


///////////////////////////////
// This is the CUDA kernel function for you to write.
//
__global__ void mat_mult (float *d_A, float *d_B, float *d_C, int N) {
    int rB=blockIdx..., 

    __shared__ float SA[BS][BS], SB[BS][BS];
    //printf("In thread with r=(%d,%d) c=(%d,%d)\n", rB,rI,cB,cI);

    ...

}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
void Matrix::mpy1 (const Matrix &A, const Matrix &B) {
    // Copy A from host memory to device memory.
    int numElem=N()*N(), sizeBytes = numElem*4;
    float *d_A = NULL;
    cudaError_t err = cudaMalloc((void **)&d_A, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix A");

    err = cudaMemcpy (d_A, A.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix A from host to device");

    // Allocate device memory for B.
    ...

    // Copy B from host memory to device memory.
    ...

    // Allocate device memory for C.
    ...

    // Launch the CUDA Kernel
    ...
    err = cudaGetLastError();
    ERR_CHK (err, "Failed to launch/complete the mat_mult() kernel");

    // Copy the result from device memory to host memory.
    ...

    // Free device memory.
    err = cudaFree(d_A);
    ERR_CHK (err, "Failed to free CUDA matrix A");
    ...
}

// Simple function for printing out an array for debug. Make sure to
// only call it from one thread, or else you'll get *lots* of output!
template<int size>
__device__ void print_array (float arr [size][size]) {
    for (int r=0; r<size; ++r) {
	printf ("[");
	for (int c=0; c<size; ++c)
	    printf ("%d ", int(arr[r][c]));
	printf ("]\n");
    }
}
