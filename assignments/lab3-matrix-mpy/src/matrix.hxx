//******************************************************
//	MATRIX LIBRARY
//	Basic matrix operations used by matrix mpy and CNN on CPU & GPU.
//******************************************************

#include <vector>
#include <sstream>
#include <cassert>

///////////////////////////////
// Matrix is the main class that we use.
// It has methods to declare a matrix, allocate space, initialize it, do slow
// single-threaded matrix multiply, and printing support.
// It also has one fast matrix-multiply method that you'll write yourself.
///////////////////////////////
class Matrix {
    // The private members.
    std::vector<float> data;	// Data is stored in a 1D vector.
    int _nRows, _nCols;	// The usual
    int _log2NColsAlc;	// Always a power of 2, to make index() faster.
    int index (int r, int c) const { return ((r << _log2NColsAlc) | c); }

  public:
    Matrix (int rows, int cols= -1);	// Create a matrix, allocate its storage
    int nRows() const { return (this->_nRows); }
    int nCols() const { return (this->_nCols); }
    int N() const { assert(_nRows==_nCols); return (_nRows); }

    // Access an element (note that operator[] can only take 1 arg, not 2).
    float &operator() (int r,int c) {return(this->data[this->index(r,c)]);}
    float operator() (int r,int c) const {return(this->data[this->index(r,c)]);}

    bool operator== (const Matrix &other) const;	// Full equality check
    // Die on first mismatch
    void compare (const Matrix &M2, std::string my_name, std::string his_name) const;

    // Initialize a matrix; to I, to random #s in [0,1], or cyclic ints.
    void init_identity();
    void init_random (int max);
    void init_cyclic_order ();
    void init_count_order ();

    std::string row_str(int row) const;	// Print one matrix row to a string.
    std::string str() const;		// Ditto for the entire matrix.

    void CNN_dumb (const Matrix &array_in, const Matrix &f);// 1 thr, unblocked

    // 1 thread, but blocked.
    void CNN1 (const Matrix &in, const Matrix &f, int n_procs, int tile_size);

    // multithreaded & blocked.
    void CNN2 (const Matrix &array_in, const Matrix &f, int n_procs);

    void mpy_dumb (const Matrix &A, const Matrix &B);	// 1 thread, unblocked
    // 1 thread, but blocked.
    void mpy1 (const Matrix &A, const Matrix &B, int BS);
    // multithreaded & blocked.
    void mpy2 (const Matrix &A, const Matrix &B, int BS, int n_procs);
};
