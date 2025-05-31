// -----------------------------------------------------------------------------
//  Helpers – all 100 % NEON   (A72 / armv8-a)
// -----------------------------------------------------------------------------
#include <arm_neon.h>
#include "ntt.h"

static inline uint32x4_t add_mod_q(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t sum = vaddq_u32(a, b);
    uint32x4_t qvec = vdupq_n_u32(Q);
    /* if (sum >= Q) sum -= Q */
    uint32x4_t needs_sub = vcgeq_u32(sum, qvec);
    return vsubq_u32(sum, vandq_u32(needs_sub, qvec));
}

static inline uint32x4_t sub_mod_q(uint32x4_t a, uint32x4_t b)
{
    uint32x4_t diff = vsubq_u32(a, b);
    uint32x4_t qvec = vdupq_n_u32(Q);
    /* if (a < b) diff += Q */
    uint32x4_t needs_add = vcltq_u32(a, b);
    return vaddq_u32(diff, vandq_u32(needs_add, qvec));
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