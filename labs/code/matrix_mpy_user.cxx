#include <thread>
#include <iostream>

#include "matrix.hxx"
#include "ee155_utils.hxx"
using namespace std;

const int BS=64;

////////////////////////////////////////////////////////////////
// One thread, blocked. Loop order rB, kB, cB, r, k, c.
// This function is for you to write.
//
void Matrix::mpy1 (const Matrix &A, const Matrix &B) {
    int NBLK= this->N()/BS;     // An NBLKxNBLK grid of blocks
    assert (this->N() >= BS);
    ...
}

////////////////////////////////////////////////////////////////
// Multithreaded, blocked version.
//
// This function, th_func2(), does the per-thread work of multithreaded, blocked
// matrix multiplication.
static void th_func2 (...) {
    ...
}

////////////////////////////////////////////////////////////////
// This function does multithreaded, blocked matrix multiplication. It is for
// you to write. The parameters:
//	A, B: the input matrices
//	n_threads: how many threads to use.
// You must store the output in (*this), which already has its .data array
// allocated (but not necessarily cleared).
// Note that you can find out the size of the A, B and (*this) matrices by
// either looking at the _N member variable, or calling Matrix.N().
void Matrix::mpy2 (const Matrix &A, const Matrix &B, int n_threads) {
    ...
}
