// Lab 1: Histogram generation 
// This is the "user" file that you fill in.

#include <vector>
#include <mutex>
#include <thread>
#include <pthread.h>
#include "ee155_utils.hxx"

using namespace std;

static void th_func(int me, std::vector<int> &final_hist,
                     const std::vector<int> &input_data, int n_threads);

// How many buckets the histogram has.
extern const int N_BUCKETS;
static std::mutex g_mut;        // For printing with LOGM

//************************************************************************
// This is the function that you (mostly) write yourself. It spawns the thread
// function th_func() as many times as requested and waits for them to finish.
//************************************************************************
std::vector<int> compute_multithread
        (const std::vector<int> &input_data, int n_threads) {
    vector<int> final_hist(N_BUCKETS, 0);
    vector<thread> threads; threads.reserve(n_threads);

    for (int i = 0; i < n_threads; i++) {
        threads.push_back(thread(th_func, i, ref(final_hist), cref(input_data), n_threads));
    }

    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }

    return final_hist;
}


//************************************************************************
// You write this function yourself also. It gets instantiated multiple times
// in different threads.
//************************************************************************
static void th_func(int me, std::vector<int> &final_hist,
                     const std::vector<int> &input_data, int n_threads) {
    static std::mutex mut; // For adding data to the histogram

    vector<int> hist(N_BUCKETS, 0);

    int region_size = input_data.size() / n_threads;
    int start = me * region_size;
    int end = (me == n_threads - 1) ? input_data.size() : (me + 1) * region_size;

    for (int i = start; i < end; i++) {
        hist[input_data[i]]++;
    }

    mut.lock();
    for (int i = 0; i < N_BUCKETS; i++) {
        final_hist[i] += hist[i];
    }
    mut.unlock();
}
