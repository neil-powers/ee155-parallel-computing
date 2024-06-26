// Lab 1: Histogram generation 
// This is the "user" file that you fill in.

#include <vector>
#include <mutex>
#include <thread>
#include <pthread.h>
#include "ee155_utils.hxx"

// How many buckets the histogram has.
extern const int N_BUCKETS;
static std::mutex g_mut;	// For printing with LOGM

//************************************************************************
// This is the function that you (mostly) write yourself. It spawns the thread
// function th_func() as many times as requested and waits for them to finish.
//************************************************************************
std::vector<int> compute_multithread
	(const std::vector<int> &input_data, int n_threads) {
}


//************************************************************************
// You write this function yourself also. It gets instantiated multiple times
// in different threads.
//************************************************************************
static void th_func (int me, std::vector<int> &final_hist,
		     const std::vector<int> &input_data, int n_threads) {

}
