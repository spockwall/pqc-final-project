// -----------------------------------------------------------------------------
//  Helpers – all 100 % NEON   (A72 / armv8-a)
// -----------------------------------------------------------------------------
#include <arm_neon.h>
#include "ntt.h"

static inline uint32x4_t add_mod_q(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t result;
    uint32x4_t qvec = vdupq_n_u32(Q); // broadcast modulus into all lanes

    __asm__ volatile(
        // sum = a + b  (unsigned)
        "add    %[sum].4s, %[a].4s, %[b].4s         \n\t"

        // needs_sub = (sum >= Q) → sets bits to 0xFFFFFFFF if true
        "cmhs   %[mask].4s, %[sum].4s, %[q].4s      \n\t"

        // masked = needs_sub & Q (only subtract Q where sum ≥ Q)
        "and    %[mask].16b, %[mask].16b, %[q].16b  \n\t"

        // result = sum - masked
        "sub    %[sum].4s, %[sum].4s, %[mask].4s    \n\t"

        : [sum] "=&w"(result), // output: final result
          [mask] "=&w"(b)      // temp reuse of b for mask
        : [a] "w"(a),          // input a
          [b] "w"(b),          // input b
          [q] "w"(qvec)        // constant vector Q
        : "cc"                 // condition flags clobbered
    );
    return result;
}

static inline uint32x4_t sub_mod_q(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t result;
    uint32x4_t qvec = vdupq_n_u32(Q); // Q broadcast to all lanes

    __asm__ volatile(
        // diff = a - b (unsigned)
        "sub    %[diff].4s, %[a].4s, %[b].4s         \n\t"

        // needs_add = (b >= a) → 0xFFFFFFFF where true
        "cmhs   %[mask].4s, %[b].4s, %[a].4s         \n\t"

        // mask = (a < b) ? Q : 0
        "and    %[mask].16b, %[mask].16b, %[q].16b   \n\t"

        // result = diff + (a < b ? Q : 0)
        "add    %[diff].4s, %[diff].4s, %[mask].4s   \n\t"

        : [diff] "=&w"(result), // output: final result
          [mask] "=&w"(b)       // reuse b as temp mask
        : [a] "w"(a),           // input vector a
          [b] "w"(b),           // input vector b
          [q] "w"(qvec)         // modulus Q vector
        : "cc");
    return result;
}

/* 4-lane Montgomery multiplication:
 *     return  a·b·R^{-1}  (mod Q)
 *  Uses the standard “(t + m·Q) >> 32” trick.
 */
static inline uint32x4_t mont_mul_vec(uint32x4_t a, uint32x4_t b)
{
    /* Step-1: 64-bit product per lane  p = a·b                            */
    uint64x2_t p0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));   /* lanes 0-1 */
    uint64x2_t p1 = vmull_u32(vget_high_u32(a), vget_high_u32(b)); /* lanes 2-3 */

    /* Step-2: m = (p mod R) · QINV  (mod R) -- need only the low 32 bits   */
    uint32x4_t t_lo = vcombine_u32(vmovn_u64(p0), vmovn_u64(p1)); /* trunc p  */
    uint32x4_t m = vmulq_u32(t_lo, vdupq_n_u32(QINV));            /* wrap mul */

    /* Step-3: u = (p + m·Q) >> 32  (“shift-div” by R)                     */
    uint64x2_t mq0 = vmull_u32(vget_low_u32(m), vdup_n_u32(Q));
    uint64x2_t mq1 = vmull_u32(vget_high_u32(m), vdup_n_u32(Q));

    uint64x2_t u0 = vshrq_n_u64(vaddq_u64(p0, mq0), 32);
    uint64x2_t u1 = vshrq_n_u64(vaddq_u64(p1, mq1), 32);

    uint32x4_t u32 = vcombine_u32(vmovn_u64(u0), vmovn_u64(u1));

    /* Step-4: if (u >= Q) u -= Q   (per lane)                             */
    uint32x4_t qvec = vdupq_n_u32(Q);
    uint32x4_t u_minusQ = vsubq_u32(u32, qvec);
    uint32x4_t mask = vcltq_u32(u32, qvec); /* 0xFFFF… if u < Q       */

    /* select(u < Q ? u : u-Q)  */
    return vbslq_u32(mask, u32, u_minusQ);
}

// static inline uint32x4_t mont_mul_vec(uint32x4_t a, uint32x4_t b)
//{
//     uint32x4_t result;
//     uint32x4_t qvec = vdupq_n_u32(Q);
//     uint32x4_t qinv_vec = vdupq_n_u32(QINV);

//    __asm__ volatile(
//        // Step 1: 32×32 → 64-bit unsigned multiply (lanes 0-1 and 2-3)
//        "umull  v8.2d,  %w[a].2s,  %w[b].2s       \n\t" // p0 = a[0,1] * b[0,1]
//        "umull2 v9.2d,  %w[a].4s,  %w[b].4s       \n\t" // p1 = a[2,3] * b[2,3]

//        // Extract low 32 bits from each product lane into v10.4s
//        "xtn    v10.2s, v8.2d                     \n\t" // low half:  2 × 64-bit → 2 × 32-bit
//        "xtn2   v10.4s, v9.2d                     \n\t" // high half: 2 × 64-bit → 4 × 32-bit

//        // Step 2: m = t_lo * QINV
//        "mul    v11.4s, v10.4s, %w[qinv].4s       \n\t"

//        // Step 3: m * Q (64-bit), then add p, then >> 32
//        "umull  v12.2d, v11.2s, %w[qv].s[0]       \n\t"
//        "umull2 v13.2d, v11.4s, %w[qv].s[0]       \n\t"

//        "add    v12.2d, v12.2d, v8.2d             \n\t"
//        "add    v13.2d, v13.2d, v9.2d             \n\t"

//        "ushr   v12.2d, v12.2d, #32               \n\t"
//        "ushr   v13.2d, v13.2d, #32               \n\t"

//        "xtn    v14.2s, v12.2d                    \n\t"
//        "xtn2   v14.4s, v13.2d                    \n\t"

//        // Step 4: if (u >= Q) u -= Q
//        "cmhs   v15.4s, v14.4s, %w[qv].4s         \n\t"
//        "sub    v16.4s, v14.4s, %w[qv].4s         \n\t"
//        "bsl    v15.16b, v16.16b, v14.16b         \n\t"

//        "mov    %0.16b, v15.16b                   \n\t"
//        : "=w"(result)
//        : [a] "w"(a),
//          [b] "w"(b),
//          [qv] "w"(qvec),
//          [qinv] "w"(qinv_vec),
//          [mm] "w"(qinv_vec) // reused for m
//        : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "cc");

//    return result;
//}
