#include <iostream>
#include <cmath>
#include "ee155_utils.hxx"
using namespace std;

std::chrono::time_point<std::chrono::high_resolution_clock> start_time () {
    return (std::chrono::high_resolution_clock::now());
}

long int delta_usec
	(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
    std::chrono::time_point<std::chrono::high_resolution_clock>
		end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000000> > elapsed_us=end-start;
    long int ticks_us = elapsed_us.count();
    return (ticks_us);
}

/* Here's an overview of how the C++ timing facility works.
 * A *duration* is a span of time. It is a number of ticks, where a tick can
 *	be, e.g., ms or us. The number of ticks is usually int or float; the
 *	tick length gets remembered as a std::ratio object that is implicit in
 *	the type name (but takes no space).
 *	* There are predefined duration types such as chrono::milliseconds. They
 *	  store the # of ticks as an integer.
 *	* You can convert duration types to each other;
 *		chrono::seconds sec(2); chrono::milliseconds millie(sec);
 *		Then millie.count() -> 2000.
 *	* You cannot convert from ms to seconds, since integer division loses
 *	  precision. To convert from ms to seconds, you would to convert to
 *	  duration <double, ratio<1,1> > instead.
 *	* We declare our own microsecond duration type above:
 *		std::chrono::duration<double, std::ratio<1, 1000000> >
 *	  If we had simply used chrono::microseconds, it would be based on int
 *	  rather than double, and we would not be able to measure anything less
 *	  than one microsecond. Thus, we rolled our own.
 * A *clock* has a starting point and a tick rate (i.e., number of ticks/second)
 * A *time_point* is essentially a duration that is relative to a particular
 *	clock's beginning of time. However, it is its own type; the particular
 *	clock it uses is encoded in the type.
 *
 */

// Assign a thread to a particular "core" (really, to a logical core, where
// the two threads in a single hyperthreaded core each count as a separate
// "core").
void assign_to_core (std::thread::native_handle_type th_handle, int i) {
#if !defined(_WIN32)
    // Ensure we don't try to assign to thread #10 if there are only 8 cores.
    int n_cores = thread::hardware_concurrency();
    i = i % n_cores;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int err = pthread_setaffinity_np(th_handle, sizeof(cpu_set_t), &cpuset);
    if (err != 0) {
	string str="???";
	if (err==EFAULT) str = "EFAULT";
	if (err==EINVAL) str = "EINVAL";
	if (err==ESRCH ) str = "ESRCH";
	DIE ("Error calling pthread_setaffinity_np: " << str)
    }
#endif
}

// Given a vector of execution times (typically from running the same code
// multiple times & timing it each time), print statistics: average time and
// standard deviation. Ignore times[0]; it probably represents a cold cache.
void analyze_times (string message, vector<double> &times, string units) {
    double sum=0, sum_sq=0;
    int n_times = times.size();
    for (int i=1; i<n_times; ++i) {
	sum += times[i];
	sum_sq += times[i] * times[i];
    }
    double mean = sum / (n_times-1),
	   dev  = sqrt(sum_sq/(n_times-1)-mean*mean);
    LOG (message << " summary without initial run: mean=" << mean << units
	 << ", std.dev.=" << dev << ", std.dev/mean = " << dev/mean);
}
