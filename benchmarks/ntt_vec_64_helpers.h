// -----------------------------------------------------------------------------
//  Helpers – all 100 % NEON   (A72 / armv8-a)
// -----------------------------------------------------------------------------
#include <arm_neon.h>
#include "ntt.h"

/* 64-bit NTT prime -------------------------------------------------------- */
static const uint64_t Q_64 = 0xFFFFFFFF00000001ULL; /* 2⁶⁴ − 2³² + 1   */

/* Montgomery machinery for that Q (modulus R = 2⁶⁴) ---------------------- */
static const uint64_t QINV_64 = 0xFFFFFFFEFFFFFFFFULL;     /* −Q⁻¹  mod 2⁶⁴   */
static const uint64_t R2MODQ_64 = 0xFFFFFFFE00000001ULL;   /* R²    mod Q     */
static const uint64_t MONT_ONE_64 = 0x00000000FFFFFFFFULL; /* R     mod Q     */

static inline __int128_t vmull_u64(uint64x1_t a, uint64x1_t b)
{
    // get a from uint64x1_t to __int128_t
    uint64_t a_val = vget_lane_u64(a, 0);
    uint64_t b_val = vget_lane_u64(b, 0);

    return (__int128_t)a_val * b_val;
}

static inline uint64_t mont_mul(uint64_t a, uint64_t b)
{
    __int128_t t = (__int128_t)a * b;     /* 0 .. (Q-1)² < 2¹²⁶          */
    uint64_t m = (uint64_t)t * QINV;      /* (t mod R)·QINV  (mod 2⁶⁴)   */
    __int128_t u = t + (__int128_t)m * Q; /* divisible by R              */
    u >>= 64;                             /* divide by R                 */
    return (u >= Q) ? (uint64_t)u - Q : (uint64_t)u;
}

static uint64_t mont_reduce(__uint128_t t) /* helper */
{
    /* m = (t mod R) * QINV mod R  – needs only low 64 bits           */
    uint64_t m = (uint64_t)t * QINV;
    __uint128_t u = t + (__uint128_t)m * Q; /* t + m·Q is divisible by R */
    u >>= 64;                               /* divide by R               */
    return (u >= Q) ? (uint64_t)(u - Q) : (uint64_t)u;
}

static inline uint64_t to_mont(uint64_t x) { return mont_mul(x, R2MODQ_64); }
static inline uint64_t from_mont(uint64_t x) { return mont_mul(x, 1); }

static inline uint64_t add_mod(uint64_t a, uint64_t b)
{
    uint64_t s = a + b;
    return (s >= Q) ? s - Q : s;
}

static inline uint64_t sub_mod(uint64_t a, uint64_t b)
{
    return (a >= b) ? a - b : (uint64_t)(Q + (__uint128_t)a - b);
}
static inline uint64x2_t add_mod_vec64(uint64x2_t a, uint64x2_t b)
{
    uint64x2_t sum = vaddq_u64(a, b);
    uint64x2_t qvec = vdupq_n_u64(Q_64);
    uint64x2_t ge = vcgeq_u64(sum, qvec);
    return vsubq_u64(sum, vandq_u64(ge, qvec));
}

static inline uint64x2_t sub_mod_vec64(uint64x2_t a, uint64x2_t b)
{
    uint64x2_t diff = vsubq_u64(a, b);
    uint64x2_t qvec = vdupq_n_u64(Q_64);
    uint64x2_t lt = vcltq_u64(a, b);
    return vaddq_u64(diff, vandq_u64(lt, qvec));
}

static inline uint64_t mont_pow_mod(uint64_t g, uint64_t e)
{
    uint64_t r = to_mont(1);
    while (e)
    {
        if (e & 1)
            r = mont_mul(r, g);
        g = mont_mul(g, g);
        e >>= 1;
    }
    return r;
}
static inline uint64x2_t mont_mul_vec64(uint64x2_t a, uint64x2_t b)
{
    __uint128_t p0 = vmull_u64(vget_low_u64(a), vget_low_u64(b));
    __uint128_t p1 = vmull_u64(vget_high_u64(a), vget_high_u64(b));

    uint64_t t0 = (uint64_t)p0;
    uint64_t t1 = (uint64_t)p1;

    uint64x2_t m = {t0 * QINV_64, t1 * QINV_64};

    __uint128_t u0 = p0 + (__uint128_t)vget_lane_u64(vget_low_u64(m), 0) * Q_64;
    __uint128_t u1 = p1 + (__uint128_t)vget_lane_u64(vget_high_u64(m), 0) * Q_64;

    uint64x2_t u = {(uint64_t)(u0 >> 64), (uint64_t)(u1 >> 64)};

    uint64x2_t qv = vdupq_n_u64(Q_64);
    uint64x2_t u_q = vsubq_u64(u, qv);
    uint64x2_t mask = vcltq_u64(u, qv); /* 0xFFFF… if u < Q */

    return vbslq_u64(mask, u, u_q);
}