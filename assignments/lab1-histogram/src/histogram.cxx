// Lab 1: Histogram generation

// Compile as
//	c++ -std=c++11 -O2 -pthread histogram.cxx ee155_utils.cxx histogram_user.cxx

#include <iostream>
#include <random>
#include <thread>
#include <vector>
using namespace std;
#include "ee155_utils.hxx"

// How many buckets the histogram has.
extern const int N_BUCKETS = 5;

static void compute_1thread(vector<int> &hist, const vector<int> &input_data);
static void multithreaded(const vector<int> &input_data,
                          const vector<int> &ref_hist, int n_threads);
static void sanity_check_hist(const vector<int> &histogram, int n_input_vals);

vector<int> compute_multithread(const vector<int> &input_data, int n_threads);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    if (argc != 2)
        DIE("Usage: histogram <n_input_vals>");
    // int n_input_vals = std::stoi(argv[1]);
    int n_input_vals = strtol(argv[1], NULL, 10);
    vector<int> ref_hist, input_data(n_input_vals);
    LOG("Building histograms with " << n_input_vals << " entries");
    LOG("There are " << thread::hardware_concurrency() << " total cores");

    // Randomly generate input data.
    // Initialize it to integer values in [0,N_BUCKETS).
    default_random_engine gen;
    uniform_int_distribution<int> dist(0, N_BUCKETS - 1);
    for (int i = 0; i < n_input_vals; i++)
        input_data[i] = dist(gen);
    LOG("Created the input data array");

    // Create the golden solution, using one thread and timing it. Do this a
    // few times; to make up for cold caches & gauge test repeatability.
    vector<double> times;
    for (int loop = 0; loop < 4; ++loop) {
        LOG("\nCreating the golden histogram.");
        auto start = start_time();
        compute_1thread(ref_hist, input_data);
        double time = delta_usec(start) / 1000.0;
        LOG("Single-threaded creation took " << time << "msec");
        sanity_check_hist(ref_hist, n_input_vals);
        times.push_back(time);
    }
    analyze_times("Golden", times, "msec");

    multithreaded(input_data, ref_hist, 1); // Multithreaded (just 1 thread)
    multithreaded(input_data, ref_hist, 2); // 2 threads
    multithreaded(input_data, ref_hist, 4); // 4 threads
    multithreaded(input_data, ref_hist, 8); // 8 threads
    multithreaded(input_data, ref_hist, 16);
    multithreaded(input_data, ref_hist, 32);
    multithreaded(input_data, ref_hist, 64);
    multithreaded(input_data, ref_hist, 128);
}

// This function computes the reference solution without any threading.
static void compute_1thread(vector<int> &hist, const vector<int> &input_data) {
    // Initialize histogram
    hist = vector<int>(N_BUCKETS, 0);

    // Bin the elements in the input stream
    for (int input_val : input_data)
        ++hist[input_val];
}

//************************************************************************
// Now for the multithreaded solution. This function is just a wrapper around
// compute_multithread(), which everyone writes themself. This wrapper takes
// care of timing and sanity checking.
//************************************************************************
static void multithreaded(const vector<int> &input_data,
                          const vector<int> &ref_hist, int n_threads) {
    int n_input_vals = input_data.size();
    vector<double> times;
    for (int loop = 0; loop < 5; ++loop) {
        // LOG(endl<<"Creating histogram using "<<n_threads<<" threads.");
        auto start = start_time();
        vector<int> user_hist = compute_multithread(input_data, n_threads);
        double time = delta_usec(start) / 1000.0;
        times.push_back(time);
        // LOG (n_threads << " threads took " << time << "ms");
        sanity_check_hist(user_hist, n_input_vals);

        // Check the multithreaded results vs. the reference.
        float diff = 0.0;
        for (int i = 0; i < N_BUCKETS; i++)
            diff += abs(ref_hist[i] - user_hist[i]);
        if (diff > 0)
            DIE(diff << " errors between the reference and user results.");
    }
    analyze_times(to_string(n_threads) + " threads", times, "msec");
}

// This function sanity-checks that the histogram contains a total of
// 'n_input_vals' elements. It does so by summing the number in each bucket.
static void sanity_check_hist(const vector<int> &histogram, int n_input_vals) {
    int sum = 0;
    for (int bucket : histogram)
        sum += bucket;

    if (sum != n_input_vals)
        DIE("Histogram error: expected " << n_input_vals << " values, saw " << sum);
}
