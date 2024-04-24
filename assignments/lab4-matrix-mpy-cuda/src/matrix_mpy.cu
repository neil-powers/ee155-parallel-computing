/*
 * CUDA matrix multiply
 */

#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>
#include <chrono>
#include "ee155_utils.hxx"
#include "bits.hxx"
#include "matrix.hxx"
using namespace std;

const int BS = 32;	// The blocks are BS x BS.

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// Simple algorithm for multiplying two matrices.
void Matrix::mpy_dumb (const Matrix &A, const Matrix &B) {
    int N = this->N();
    for (int r=0; r<N; ++r)
	for (int k=0; k<N; ++k)
	    for (int c=0; c<N; ++c) {
		if (k==0)
		    (*this)(r,c) = 0.0F;
		(*this)(r,c) += (A(r,k) * B(k,c));
	    }
}

// This function executes the various pieces for a given matrix size.
static void run (int N) {
    Matrix a(N), b(N), c(N), d(N);
    a.init_random(5);
    b.init_random(10);
    //a.init_cyclic_order();
    //b.init_count_order();
    //b.init_identity();
    LOG (endl<<"Working on "<<N<<"x"<<N<<" matrices with BS="<<BS);
    //LOG ("A="<<a.str());
    //LOG ("B="<<b.str());

    // Compute the reference solution
    auto start = start_time();
    c.mpy_dumb (b, a);
    long int time = delta_usec (start);
    LOG ("Dumb mpy took "<<(time/1000000.0)<<"sec");
    //LOG ("Golden="<<c.str());

    long int total_time=0;
    for (int i=0; i<4; ++i) {
	auto start = start_time();
	d.mpy1 (b, a, BS);
	long int time = delta_usec (start);
	total_time += time;
	//LOG ("GPU="<<d.str());
	c.compare (d, "ref", "you");
    }
    LOG ("mpy1 averaged "<<(total_time/4000000.0)<<"sec");
}

// Main() lives on the CPU.
int main() {
    // Check the compute capability
    cudaDeviceProp prop; int device=0;
    cudaError_t err = cudaGetDeviceProperties (&prop, device);
    ERR_CHK (err, "Failed to get compute capability");
    int major = prop.major, minor=prop.minor;
    LOG ("Compute capability = "<<major<<"."<<minor);

    //run (4);	// Matrix size 4x4
    run (32);	// Matrix size 32x32
    run (64);	// Matrix size 64x64
    run (256);
    run (1024);	// Matrix size 1Kx1K
    run (2048);	// Matrix size 2Kx2K
    run (4096);	// Matrix size 4Kx4K
    return (0);
}
