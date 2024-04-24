/*
 * CUDA convolutional neural net
 */

#include <iostream>
#include <assert.h>
#include "ee155_utils.hxx"
#include "matrix.hxx"
using namespace std;

// Simple CNN algorithm. 1 core, no blocking. Used as a reference to check
// the GPU work.
// Inputs: array_in (the input), f (the filter)
// Outputs: set this' data (which should already be allocated to the correct
//	    size, slightly smaller than array_in.
// Assume: 1 input channel, 1 output channel.
void Matrix::CNN_dumb (const Matrix &array_in, const Matrix &f) {
    assert (this->N() == array_in.N() + 1 - f.N());
    int N = this->N();
    for (int ro=0; ro<N; ++ro)		// for each output row...
	for (int co=0; co<N; ++co) {	// and output column...
	    float sum=0;
	    for (int rf=0; rf<f.N(); ++rf)		// for each filter row,
		for (int cf=0; cf<f.N(); ++cf) {	// column...
		    int ri = ro + rf;
		    int ci = co + cf;
		    sum += array_in(ri,ci) * f(rf,cf);	// Sum it up.
		}
	    (*this)(ro,co) = sum;
	}
}

static void run (int mat_size, int filt_size) {
    LOG ("Running "<<mat_size<<"x"<<mat_size<<" with "
		<<filt_size<<"x"<<filt_size<<" filter");
    // Create the filter matrix.
    Matrix f(filt_size);
    f.init_identity();
    f.init_random (5);
    //LOG ("Filter is "<<filt_size<<"x"<<filt_size<<": "<<f.str());

    // Create input and output matrices.
    Matrix in(mat_size);
    in.init_cyclic_order();
    //in.init_identity();
    in.init_random(5);
    Matrix out(mat_size+1-filt_size);

    // Run and time the simple dumb CPU algorithm, just once.
    auto start = start_time();
    Matrix dumb(mat_size+1-filt_size);
    dumb.CNN_dumb (in, f);
    long int time = delta_usec (start);
    LOG ("\t"<<mat_size<<"x"<<mat_size<<" CNN_dumb() took "<<(time/1000000.0)<<"sec");
    //LOG ("Dumb is"<<endl<<dumb.str());

    // Now, the GPU version.
    time=0;
    for (int rep=0; rep<3; ++rep) {
	out.init_identity();//So we don't get a correct answer by accident!
	auto start = start_time();
	out.CNN2 (in, f, 0);
	long int dt = delta_usec (start);
	time += dt;
	dumb.compare (out, "ref", "you");
    }
    LOG ("    "<<mat_size<<"x"<<mat_size<<" CUDA CNN took "<<(time/3000000.0)
		<<" sec on average.");
}

int main () {
    run (32, 2);
	run (64, 2);
	run (4096, 4);
    run (4096, 8);
    run (4096, 12);
    run (8192, 4);
    run (8192, 8);
    run (8192, 12);
}
