#include <arm_neon.h>
#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include "hal.h"
#include "lib.h"

// ---------- tiny carry-aware school-book for n = 8 ----------------
static inline void sb8_mul(uint32_t *restrict dst,
                           const uint32_t *restrict A,
                           const uint32_t *restrict B)
{
    // zero the 16-limb result
    for (int k = 0; k < 16; ++k)
        dst[k] = 0;

    for (int i = 0; i < 8; ++i)
    {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j)
        {
            uint64_t t = (uint64_t)A[i] * B[j] + dst[i + j] + carry;
            dst[i + j] = (uint32_t)t;
            carry = t >> 32;
        }
        dst[i + 8] += (uint32_t)carry; // final carry out
    }
}
// n must be a multiple of 8 limbs (256 bits).  dst length = 2n.
static void karatsuba32_vec(uint32_t *restrict dst,
                            const uint32_t *restrict A,
                            const uint32_t *restrict B,
                            size_t n)
{
    // base case : 8 × 8 limbs = 256-bit × 256-bit
    if (n % 8 != 0)
    {
        fprintf(stderr, "Error: n must be a multiple of 8 limbs\n");
        exit(EXIT_FAILURE);
    }
    if (n == 8)
    {
        sb8_mul(dst, A, B);
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

static void bench_karatsuba()
{
    // gnereate random A and B
    uint32_t A[LIMBS_NUM] = {0};
    uint32_t B[LIMBS_NUM] = {0};
    uint32_t dst[LIMBS_NUM << 1] = {0};
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    srand((unsigned int)time(NULL));
    generate_random_bigint(A, LIMBS_NUM * BITS_PER_LIMB);
    generate_random_bigint(B, LIMBS_NUM * BITS_PER_LIMB);

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            karatsuba32_vec(dst, A, B, LIMBS_NUM);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            karatsuba32_vec(dst, A, B, LIMBS_NUM);
        }
        t1 = get_cyclecounter();

        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    printf("Karatsuba multiplication took %lu cycles\n", cycles[NTESTS / 2] / NITERATIONS);
}

int main()
{
    enable_cyclecounter();
    bench_karatsuba();
    disable_cyclecounter();
    return 0;
}