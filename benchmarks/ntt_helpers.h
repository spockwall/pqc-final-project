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
static inline uint32_t mul_mod(uint32_t a, uint32_t b)
{
    return (uint32_t)(((uint64_t)a * b) % Q);
}

// 32×32 → 32 Montgomery 乘：回傳 a·b·R^{-1} (mod Q)
static inline uint32_t mont_mul(uint32_t a, uint32_t b)
{
    uint64_t t = (uint64_t)a * b;    /* 0..(Q-1)^2 < 2^62            */
    uint32_t m = (uint32_t)t * QINV; /* (t mod R)*(-Q^{-1}) mod R    */
    uint64_t u = (t + (uint64_t)m * Q) >> 32;
    return u >= Q ? (uint32_t)(u - Q) : (uint32_t)u;
}

static inline uint32_t to_mont(uint32_t x)
{
    /* x·R mod Q = mont_mul(x, R^2)          (因為 mont_mul 會再乘 R^{-1}) */
    return mont_mul(x, R2INV);
}

// 從 Montgomery 域拿掉因子 R：x·R^{-1} mod Q
static inline uint32_t from_mont(uint32_t x)
{
    return mont_mul(x, 1); /* a·1·R^{-1} = a·R^{-1} */
}

static inline uint32_t mont_pow_mod(uint32_t g, uint32_t e)
{
    uint32_t r = to_mont(1);
    while (e)
    {
        if (e & 1)
            r = mont_mul(r, g);
        g = mont_mul(g, g);
        e >>= 1;
    }
    return r;
}

static inline uint32_t pow_mod(uint32_t g, uint32_t e)
{
    uint32_t r = 1;
    while (e)
    {
        if (e & 1)
            r = mul_mod(r, g);
        g = mul_mod(g, g);
        e >>= 1;
    }
    return r;
}

// Q is prime ⇒ x^(Q) ≡ x^‑1 (mod Q)
static inline uint32_t modinv(uint32_t x)
{
    return pow_mod(x, Q - 2);
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