#include <assert.h>
#include <cstdint>

// Bit setting and checking.
static inline uint64_t bit_get (uint64_t numb, int end, int start) {
    uint64_t mask = (((uint64_t)1)<<(1+end-start))-1;
    return ((numb>>start)&mask);
}
static inline uint64_t bit_get (uint64_t numb, int pos) {
    return ((numb>>pos)&1);
}
static inline void bit_set (uint64_t &numb, int end, int start, uint64_t val) {
    uint64_t mask=(((uint64_t) 1)<<(1+end-start))-1;
    assert (val <= mask);
    mask = mask << start;
    numb &= ~mask;
    numb |= (val<<start);
}
static inline void bit_set (uint64_t &numb, int pos, bool val) {
    bit_set (numb, pos, pos, val);
}
