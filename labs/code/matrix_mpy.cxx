// Matrix multiplication using C++ threads

//	c++ -std=c++11 -pthread -O2 matrix_mpy.cxx matrix.cxx matrix_mpy_user.cxx ee155_utils.cxx 

#include <iostream>
#include <sstream>
#include <mutex>

using namespace std;
#include "bits.hxx"
#include "ee155_utils.hxx"
#include "matrix.hxx"

// Simple algorithm for multiplying two matrices. One thread, unblocked.
void Matrix::mpy_dumb (const Matrix &A, const Matrix &B) {
    int N = this->N();
    for (int r=0; r<N; ++r)
	for (int c=0; c<N; ++c) {
	    float sum=0.0;
	    for (int k=0; k<N; ++k)
		sum += (A(r,k) * B(k,c));
	    this->data[index(r,c)] = sum;
	}
}

// Wrapper function around Matrix::mpy2(), which is the multithreaded/blocked
// algorithm. It just runs ::mpy2() several times and checks how long it took.
static void run_mpy2 (int n_threads, const Matrix &a, const Matrix &b,
				     const Matrix &c, Matrix &d) {
    vector<double> times;
    for (int i=0; i<5; ++i) {
	auto start = start_time();
	d.mpy2 (a, b, n_threads);
	double time_sec = delta_usec (start) / 1000000.0;
	times.push_back (time_sec);
	c.compare (d, "reference", "you");
	cout<<"mpy2 with "<<n_threads<<" threads="<<time_sec<<"sec"<<endl;
    }
    analyze_times (to_string(n_threads)+" threads", times, "s");
}

int main () {
    // Time mpy_dumb() for 1Kx1K.
    LOG ("Timing mpy_dumb() on 1Kx1K matrices");
    int LOG2_N=10, N=1<<LOG2_N;
    Matrix a(N), b(N), c(N), d(N);
    a.init_cyclic_order();
    b.init_identity();

    auto start = start_time();
    c.mpy_dumb (a, b);
    long int time = delta_usec (start);
    LOG ("1Kx1K mpy_dumb() took "<<(time/1000000.0)<<"sec");

    // Set up the matrices a and b (the two inputs), c (the golden reference)
    // and d (the output).
    LOG2_N=11; N=1<<LOG2_N;
    a = Matrix(N); b=Matrix(N); c=Matrix(N); d=Matrix(N);
    a.init_cyclic_order();
    b.init_count_order();
    //a.init_identity()
    //a.init_random();
    //b.init_random();

    // Time 2Kx2K mpy_dumb() and make the output into our golden reference c.
    start = start_time();
    c.mpy_dumb (a, b);
    time = delta_usec (start);
    LOG ("2Kx2K mpy_dumb() took "<<(time/1000000.0)<<"sec");

    // Run 2Kx2K mpy1 (single-threaded, blocked) 5 times and report the results.
    vector<double> times;
    for (int i=0; i<5; ++i) {
	auto start = start_time();
	d.mpy1 (a, b);
	double time_sec = delta_usec (start) / 1000000.0;
	times.push_back (time_sec);
	c.compare (d, "reference", "you");
	LOG ("2Kx2K mpy1 took "<<time_sec<<"sec");
    }
    analyze_times ("Mpy1", times, "s");

    // mpy2: using 1, 2, 4, 8 and 16 threads.
    run_mpy2 (1, a, b, c, d);	//Parameters are # threads, matrices
    run_mpy2 (2, a, b, c, d);
    run_mpy2 (4, a, b, c, d);
    run_mpy2 (8, a, b, c, d);
    run_mpy2 (16,a, b, c, d);
}
