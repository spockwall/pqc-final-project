#include <stdint.h>
#include "ntt.h"

// ---------- mod-arithmetic --------------------------
static inline uint32_t add_mod(uint32_t a, uint32_t b)
{
    uint32_t r = a + b;
    return (r >= Q) ? r - Q : r;
}
static inline uint32_t sub_mod(uint32_t a, uint32_t b)
{
    return (a >= b) ? a - b : (uint32_t)(Q + (uint64_t)a - b);
}

// 32×32-bit Montgomery multiply, branch-free
static inline uint32_t montmul(uint32_t a, uint32_t b)
{
    uint64_t t = (uint64_t)a * b;
    uint32_t m = (uint32_t)t * QINV;
    uint64_t u = (t + (uint64_t)m * Q) >> 32;
    return (u >= Q) ? u - Q : (uint32_t)u;
}

static inline uint32_t pow_mod(uint32_t g, uint32_t e)
{
    uint32_t r = 1;
    while (e)
    {
        if (e & 1)
            r = montmul(r, g);
        g = montmul(g, g);
        e >>= 1;
    }
    return r;
}

// check an uint32_t value is a power of 2
static inline int is_power_of_2(uint32_t x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

// compute log n
static inline unsigned log2n(uint32_t n)
{
    unsigned log = 0;
    while (n > 1)
    {
        n >>= 1;
        log++;
    }
    return log;
}