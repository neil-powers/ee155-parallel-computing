# c++ -pthread -std=c++11 -O2 histogram.cxx ee155_utils.cxx

SOURCE = histogram.cxx ee155_utils.cxx histogram_user.cxx

all: histogram

histogram: $(SOURCE)
	c++ -pthread -std=c++11 -g $(SOURCE) -o histogram

histogram_opt: $(SOURCE)
	c++ -pthread -std=c++11 -O2 $(SOURCE) -o histogram

.PHONY: clean
clean:
	rm -f histogram *.o a.out

