/*
 * CUDA convolutional neural net
 */

#include <iostream>
#include <iomanip>
#include "ee155_utils.hxx"
#include "matrix.hxx"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
template<int size> __device__ void print_array (float arr [size][size]);

const int BS=32;		// The blocks are BS x BS.
const int FILT_SIZE_MAX = 12;	// The filter size (needs not be a power of 2)

///////////////////////////////
// This is the CUDA kernel function for you to write.
//////////////////////////////
__global__ void CNN (...) {
}


///////////////////////////////
// This is the host function for you to write.
// It allocates memory and moves data between CPU<->GPU
//
void Matrix::CNN2 (const Matrix &inp, const Matrix &f, int dummy) {
    auto start1 = start_time();

    // Allocate input matrix in device memory. It's a nice 2^N size, so don't
    // bother with cudaMallocPitch().
    assert (1<<inp._log2NColsAlc == inp._nCols);
    int numElem=inp.data.size(), sizeBytes = numElem*4;
    float *d_inp = NULL;
    cudaError_t err = cudaMalloc((void **)&d_inp, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix 'inp'");
    //LOG ("\nInput matrix: rows="<<inp._nRows<<", cols="<<inp._nCols);

    // Copy inp from host memory to device memory.
    err = cudaMemcpy (d_inp, inp.data.data(), sizeBytes,cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix inp from host to device");

    // Allocate device memory for filter. Again, don't bother with
    // cudaMallocPitch(); the filter is small, and Matrix has already picked 
    // a power of 2 columns
    float *d_f = NULL;
    sizeBytes = static_cast<int> (f.data.size()) * 4;
    err = cudaMalloc((void **)&d_f, sizeBytes);
    ERR_CHK (err, "Failed to allocate device matrix for the filter f");

    // Copy f from host memory to device memory.
    err = cudaMemcpy (d_f, f.data.data(), sizeBytes, cudaMemcpyHostToDevice);
    ERR_CHK (err, "Failed to copy matrix f from host to device");

    // Allocate device memory for the output matrix. In fact, allocate the
    // entire thing (with padding).
    err = cudaMallocPitch(...);				for full credit
or  err = cudaMalloc((void **)&d_f, sizeBytes);		or, not
    ERR_CHK (err, "Failed to allocate device matrix 'out'");
    cudaDeviceSynchronize(); long int time1 = delta_usec (start1);
    auto start2 = start_time();

    // Launch the CUDA Kernel
    CNN <<<grid, block>>> (...);
    err = cudaGetLastError();
    ERR_CHK (err, "Failed to launch or finish CNN_kernel");

    cudaDeviceSynchronize(); long int time2 = delta_usec (start2);
    auto start3 = start_time();

    // Copy the result from device memory to host memory.
    err = cudaMemcpy2D (...); if you're doing the full-credit version
or  err = cudaMemcpy (...);	or, not
    ERR_CHK (err, "Failed to copy result from device to host");
    cudaDeviceSynchronize(); long int time3 = delta_usec (start3);

    err = cudaFree(d_inp);
    ERR_CHK (err, "Failed to free CUDA matrix inp");
    err = cudaFree(d_f);
    ERR_CHK (err, "Failed to free CUDA matrix f");
    err = cudaFree(d_out);
    ERR_CHK (err, "Failed to free CUDA matrix out");

    cout << setprecision(3) << fixed;
    LOG ("\tCUDA " <<inp.nRows()<<"x"<<inp.nRows()
	 << " CNN with "<<f.nRows()<<"x"<<f.nRows()<<" filter took "
	 <<(time1+time2+time3)/1000000.0<<" sec; "<<(time1/1000000.0)<<"s copy to, "
	 << (time2/1000000.0)<<"s for computation, "<< (time3/1000000.0)<<"s copy back ");
}
