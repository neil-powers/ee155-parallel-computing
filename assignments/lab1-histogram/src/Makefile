# c++ -pthread -std=c++11 -O2 histogram.cxx ee155_utils.cxx

all: histogram

histogram: histogram.cxx ee155_utils.cxx
	c++ -pthread -std=c++11 -g histogram.cxx ee155_utils.cxx histogram_user.cxx -o histogram

histogram_opt:
	c++ -pthread -std=c++11 -O2 histogram.cxx ee155_utils.cxx histogram_user.cxx -o histogram

.PHONY: clean
clean:
	rm -f histogram *.o a.out

