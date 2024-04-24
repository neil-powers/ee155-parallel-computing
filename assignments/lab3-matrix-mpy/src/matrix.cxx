#include <vector>
#include <sstream>
#include <iostream>
#include <random> 
using namespace std;
#include "bits.hxx"
#include "ee155_utils.hxx"
#include "matrix.hxx"

Matrix::Matrix (int nRows, int nCols) {
    if (nCols == -1)	// If we're just called as Matrix(5), then
	nCols=nRows;	// build a square 5x5 matrix.

    // Allocate power of 2 # of cols, to make index() faster.
    // All element access goes through operator()(r,c), which uses index(r,c),
    // which is just r<<_log2NColsAlc + c. I.e., we pad the matrix to a power-
    // of-two number of columns. It's mostly invisible to the user, since
    // Matrix::nRows(), ::nCols() and ::N() give the unpadded sizes.
    for (_log2NColsAlc=0;1<<_log2NColsAlc<nCols; ++_log2NColsAlc)
	;

    this->_nRows = nRows;
    this->_nCols = nCols;
    unsigned int n_elements = (1<<_log2NColsAlc) * nRows;
    this->data = vector<float> (n_elements);
}

bool Matrix::operator== (const Matrix &other) const {
    return (this->data == other.data);
}

// Like ==. But: on mismatch, prints the first mismatching element and dies.
void Matrix::compare (const Matrix &M2, string my_name, string his_name) const {
    if ((_nRows != M2._nRows) || (_nCols != M2._nCols))
	DIE ("Comparing unequal-sized matrices");

    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    if ((*this)(r,c) != M2(r,c))
		DIE (my_name<<"["<<r<<","<<c<<"]="
			<< static_cast<long int>((*this)(r,c))
			<< ", "<<his_name<<"["<<r<<","<<c<<"]="
			<< static_cast<long int>(M2(r,c)))
}

void Matrix::init_identity() {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = ((r==c)?1.0F:0.0F);
}


// For the rest of these, make sure to keep all elements in the range [0,63].
// The "&0x3F" makes sure that, even for a 2Kx2K matrix, all dot products are
// less than 2^11 * 2^6 * 2*6 = 2^23, which will fit in the 24-bit mantissa of
// a float.
void Matrix::init_cyclic_order() {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float> ((r+c) & 0x3F);
}

void Matrix::init_count_order () {
    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float>	((r*this->_nCols + c) & 0x3F);
}

void Matrix::init_random (int max=64) {
    default_random_engine gen;
    uniform_int_distribution<int> dist(0,max);

    for (int r=0; r<_nRows; ++r)
	for (int c=0; c<_nCols; ++c)
	    this->data[index(r,c)] = static_cast<float> (dist(gen));
}

// Printing support.
string Matrix::row_str(int row) const {
    ostringstream os;
    os << "{";
    for (int c=0; c<_nCols; ++c)
	os << (c==0?"":", ") << (*this)(row,c);
    os << "}";
    return (os.str());
}
string Matrix::str() const {
    string s = "{";
    for (int r=0; r<_nRows; ++r)
	s += this->row_str(r);
    s += "}";
    return (s);
}
