#include <arm_neon.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "lib.h"
#include "karatsuba.h"

// dst[16] – zeroed by caller or earlier code
static inline void sb8_mul_neon(uint32_t *dst,
                                const uint32_t *A,
                                const uint32_t *B)
{
    uint64_t acc[16] = {0};

    uint32x4_t a0 = vld1q_u32(A);     // limbs 0-3
    uint32x4_t a1 = vld1q_u32(A + 4); // limbs 4-7

    for (int j = 0; j < 8; j += 2)
    {
        uint32x2_t b0 = vdup_n_u32(B[j]);
        uint32x2_t b1 = vdup_n_u32(B[j + 1]);

        // A low × B[j .. j+1]
        uint64x2_t p0 = vmull_u32(vget_low_u32(a0), b0);  // A0,A1 * B[j]
        uint64x2_t p1 = vmull_u32(vget_high_u32(a0), b0); // A2,A3 * B[j]
        uint64x2_t p2 = vmull_u32(vget_low_u32(a0), b1);  // A0,A1 * B[j+1]
        uint64x2_t p3 = vmull_u32(vget_high_u32(a0), b1); // A2,A3 * B[j+1]

        // A high × B[j .. j+1]
        uint64x2_t p4 = vmull_u32(vget_low_u32(a1), b0);  // A4,A5 * B[j]
        uint64x2_t p5 = vmull_u32(vget_high_u32(a1), b0); // A6,A7 * B[j]
        uint64x2_t p6 = vmull_u32(vget_low_u32(a1), b1);  // A4,A5 * B[j+1]
        uint64x2_t p7 = vmull_u32(vget_high_u32(a1), b1); // A6,A7 * B[j+1]

        int o = j; // base output index

        // accumulate (read-modify-write)
        vst1q_u64(acc + (o + 0), vaddq_u64(p0, vld1q_u64(acc + (o + 0))));
        vst1q_u64(acc + (o + 2), vaddq_u64(p1, vld1q_u64(acc + (o + 2))));
        vst1q_u64(acc + (o + 1), vaddq_u64(p2, vld1q_u64(acc + (o + 1))));
        vst1q_u64(acc + (o + 3), vaddq_u64(p3, vld1q_u64(acc + (o + 3))));

        vst1q_u64(acc + (o + 4), vaddq_u64(p4, vld1q_u64(acc + (o + 4))));
        vst1q_u64(acc + (o + 6), vaddq_u64(p5, vld1q_u64(acc + (o + 6))));
        vst1q_u64(acc + (o + 5), vaddq_u64(p6, vld1q_u64(acc + (o + 5))));
        vst1q_u64(acc + (o + 7), vaddq_u64(p7, vld1q_u64(acc + (o + 7))));
    }

    // ------ single carry propagation ---------------------------
    uint64_t carry = 0;
    for (int k = 0; k < 16; ++k)
    {
        uint64_t t = acc[k] + carry;
        dst[k] = (uint32_t)t;
        carry = t >> 32;
    } // carry is 0 at k = 16
}

// n must be a multiple of 8 limbs (256 bits).  dst length = 2n.
void karatsuba32_vec(uint32_t *restrict dst, const uint32_t *restrict A, const uint32_t *restrict B, size_t n)
{
    // base case : 8 × 8 limbs = 256-bit × 256-bit
    if (n == 8)
    {
        sb8_mul_neon(dst, A, B);
        return;
    }

    // ---- split operands -----------------------------------------
    size_t m = n >> 1;  // n / 2
    size_t nm = n << 1; // 2*n
    size_t mm = m << 1; // 2*m

    const uint32_t *A0 = A;
    const uint32_t *A1 = A + m;
    const uint32_t *B0 = B;
    const uint32_t *B1 = B + m;

    // ---- Z₀ and Z₂ ----------------------------------------------
    // Z₀  → dst[0 .. 2m-1]
    karatsuba32_vec(dst, A0, B0, m);
    // Z₂  → dst[2m .. 4m-1]
    karatsuba32_vec(dst + mm, A1, B1, m);

    // ---- (A0+A1) and (B0+B1)  (vectorised) ----------------------
    // uint32_t SA[m], SB[m];
    uint32_t *SA = (uint32_t *)malloc(m * sizeof(uint32_t));
    uint32_t *SB = (uint32_t *)malloc(m * sizeof(uint32_t));

    for (size_t i = 0; i < m; i += 4)
    {
        uint32x4_t a0 = vld1q_u32(&A0[i]);
        uint32x4_t a1 = vld1q_u32(&A1[i]);
        uint32x4_t b0 = vld1q_u32(&B0[i]);
        uint32x4_t b1 = vld1q_u32(&B1[i]);
        vst1q_u32(&SA[i], vaddq_u32(a0, a1));
        vst1q_u32(&SB[i], vaddq_u32(b0, b1));
    }

    // ---- TMP = (A0+A1)(B0+B1) ----------------------------------
    // uint32_t TMP[mm];
    uint32_t *TMP = (uint32_t *)malloc(mm * sizeof(uint32_t));

    karatsuba32_vec(TMP, SA, SB, m);

    // ---- Z₁ = TMP − Z₀ − Z₂ ------------------------------------
    for (size_t i = 0; i < mm; ++i)
        TMP[i] = TMP[i] - dst[i] - dst[i + mm];

    // ---- add  Z₁ << m  into result (with carry) ----------------
    uint64_t carry = 0;
    for (size_t i = 0; i < mm; ++i)
    {
        uint64_t t = (uint64_t)dst[i + m] + TMP[i] + carry;
        dst[i + m] = (uint32_t)t;
        carry = t >> 32;
    }
    dst[3 * m] += (uint32_t)carry; // last carry

    // ---- one final full carry sweep to clean up -----------------
    carry = 0;
    for (size_t i = 0; i < nm; ++i)
    {
        uint64_t t = dst[i] + carry;
        dst[i] = (uint32_t)t;
        carry = t >> 32;
    }
    // if carry ≠ 0 here, dst has room (length = 2 n limbs)
    dst[nm] = (uint32_t)carry;

    free(SA);
    free(SB);
    free(TMP);
}
