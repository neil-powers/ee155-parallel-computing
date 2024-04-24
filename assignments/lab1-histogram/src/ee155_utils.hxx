#include <chrono>
#include <vector>
#include <string>

// Two functions for timing how fast your code runs. Use them as
// auto t1 = start_time ();
//	... your code...
// long int usec = delta_usec(t1);
std::chrono::time_point<std::chrono::high_resolution_clock> start_time ();
long int delta_usec
	(std::chrono::time_point<std::chrono::high_resolution_clock> start);

// Two simple macros for printing, if you want them.
#define LOG(args) cout << args << endl
#define DIE(args) { cout << args << endl; exit(0); }

// This macro is for printing in a multithreaded environment. It uses a
// lock to ensure that output from different threads doesn't interleave with
// each other. The idea is that
// 1. You declare a mutex yourself.
// 2. You pass it as the first argument to every time you call LOGM.
#define LOGM(mut, args) { mut.lock(); cout << args << endl; mut.unlock(); }

// Check the return status after a CUDA call.
#define ERR_CHK(status, args) if (status != cudaSuccess) DIE (args << " (error code "<<cudaGetErrorString(status)<<")")

#define DEBUG_CORES1(n_loops) \
    int n_loops_to_print= n_loops/10, print_counter=0; \
    vector<int> cores;

#define DEBUG_CORES2(print_counter, n_loops_to_print, cores) \
    if (++print_counter==n_loops_to_print) { \
	print_counter=0; \
	cores.push_back (sched_getcpu()); \
	if (cores.size()==10) { \
	    cout <<"Thread #"<<me<<" on core #"; \
	    for (int core:cores) cout << core << ","; \
	    cout<<endl; \
	} \
    }

// Assign a thread to a particular "core" (really, to a logical core, where
// the two threads in a single hyperthreaded core each count as a separate
// "core").
#include <thread>
void assign_to_core (std::thread::native_handle_type th_handle, int i);

// Given a vector of execution times (typically from running the same code
// multiple times & timing it each time), print statistics: average time and
// standard deviation. Ignore times[0]; it probably represents a cold cache.
void analyze_times (std::string message, std::vector<double> &times,
		    std::string units);
