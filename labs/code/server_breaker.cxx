// Lab 2: Server breaker
// This is the code for everything except the per-thread function (which is
// in server_breaker_thread.cxx).

//	c++ -std=c++11 -pthread -O2 server_breaker.cxx ee155_utils.cxx server_breaker_thread.cxx

// We will always allocate a vector of this many lines, and then choose which
// ones to actually use on any run.
static const int MAX_LINES = 10000000;

static const int N_ITERATIONS=5;	// Top-level iterations...

// Fixed by the CPU microarchitecture.
static const int BYTES_PER_LINE=64;

#include <iostream>
#include <thread>
#include <vector>
#include <random> 
#include <algorithm>
#include <mutex>
#include "ee155_utils.hxx"
using namespace std;

// This is all of the memory lines that we can use.
// We just allocate the biggest version we might ever need.
static unsigned char g_mem[MAX_LINES * BYTES_PER_LINE];

void compute_thread (int me, const vector<int> &indices, unsigned char g_mem[],
		     int n_stores, int n_loads, int n_line_accesses);
static void pick_unique_lines (vector<int> &indices, int n_lines);
static void run (int n_lines, int n_stores, int n_loads, int n_threads, int n_line_accesses);

mutex g_mut;	// For debug printing.

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    LOG ("There are "<<thread::hardware_concurrency() <<" threads");
    
    // Run through all of the parameter combinations that the homework asks for.
    // Parameters are n_lines, n_stores, n_loads, n_threads.
    for (int n_lines=100; n_lines<=100000; n_lines *= 10)
	for (int n_threads=1; n_threads<=16; n_threads *= 2)
	    run (n_lines, 0, 2, n_threads, 400000000);

    for (int n_lines=100; n_lines<=100000; n_lines *= 10)
	for (int n_threads=1; n_threads<=16; n_threads *= 2)
	    run (n_lines, 4, 1, n_threads, 400000000);
}

// This function takes care of picking lines, timing how long your code takes,
// and instantiating the threads.
static void run (int n_lines, int n_stores, int n_loads, int n_threads,
		 int n_total_line_accesses) {
    vector<int> indices;	// Indices into g_mem[] of our chosen lines
    pick_unique_lines (indices, n_lines);

    // Print a summary of the upcoming run's parameters.
    LOG ("\nPicking "<<n_lines<<" lines from a total of "<<MAX_LINES);
    int n_loops = n_total_line_accesses/(n_lines * (n_loads+n_stores));
    LOG (n_threads<<" threads, each doing "<<n_loops<<" loops of "
	 <<n_lines<<" lines ("<< n_stores << " stores+"<< n_loads<<" loads)");

    // The outer loop of iterations. Each iteration is really an entire
    // run; we do it multiple times to see any run-to-run variability.
    vector<double> times;
    for (int its=0; its<N_ITERATIONS; ++its) {
	// The main loop. We start the timer, call compute_thread() to do the
	// work, end the timer, and write out statistics.
	auto start = start_time();	// Time the loads and stores.

	// Instantiate all of the threads.
	vector<thread> threads;
	for (int i=0; i<n_threads; ++i) {
	    threads.push_back (thread (compute_thread,i,ref(indices), g_mem,
			       n_stores,n_loads,n_loops));
	    assign_to_core (threads[i].native_handle(), i);
	}

	for (auto &th:threads)	// Wait for all threads to finish.
	    th.join();

	// Collect & print timing statistics.
	double time = delta_usec (start)/1000.0;
	times.push_back (time);
	LOG ("Execution took " << time << "ms");
    }
    analyze_times (to_string(n_threads)+" threads", times, "ms");

    // Check that they stored the right data.
    if (n_stores > 0)
	for (int i=0; i<indices.size(); ++i) {
	    for (int t=0; t<n_threads; ++t) {
		int my_idx = indices[i]*BYTES_PER_LINE + t;
		unsigned int data = g_mem[my_idx];
		if (data != ((i+1) & 0xFF))
		    DIE ("Thread "<<t<<" expected mem["<<my_idx
			 <<"]="<<((i+1) & 0xFF)<<", received "<<data);
	    }
	}
}

// Randomly choose 'n_lines' lines. Return them implicitly via 'indices'.
// Specifically, we return a vector with the integers [0,MAX_LINES) randomly
// permuted. These will serve (after a small bit of munging) as indices into
// g_mem[].
static void pick_unique_lines (vector<int> &indices, int n_lines) {
    // Initialize it to integer values in [0,N_BUCKETS).
    default_random_engine gen;
    uniform_int_distribution<int> dist(0,MAX_LINES-1);

    //LOG ("Picking rand in [0,"<<MAX_LINES-1<<"]");

    indices.clear();
    // Random numbers may pick the same number multiple times. So we are a
    // bit tricky; we do two loops, and remove duplicates on the inner loop.
    // Yes, there are more efficient ways to do this.
    while (indices.size() < n_lines) {
	// First fill up indices with the correct number of lines.
	while (indices.size() < n_lines) {
	    int line = dist(gen);
	    indices.push_back (line);
        }

	// However, we may have put in some duplicates.
	// So, sort-uniquify-merge on our indices.
	sort (indices.begin(), indices.end());
	indices.erase (unique (indices.begin(), indices.end()), indices.end());
    }
    //cout << "Lines: {";
    //for (int ln : indices) cout <<ln<<" ";
    //cout << "}\n";
}
