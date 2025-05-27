#include <arm_neon.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "hal.h"
#include "lib.h"
#include "ntt.h"
#include "ntt_helpers.h"

#define MAX_N 2048

//  Twiddle tables – initialised once
static uint32_t Wfwd[MAX_N], Winv[MAX_N];
static unsigned W_log_max = 0;
static int W_ready = 0;

// Fill Wfwd / Winv for 2^logn-point (or smaller) transforms.
static inline void ntt_init(unsigned logn)
{
    if (W_ready && logn <= W_log_max)
        return;

    uint32_t n = 1u << logn;
    uint32_t root = pow_mod(G, (Q - 1) / n);
    uint32_t root_i = pow_mod(root, Q - 2); /* inverse via Fermat */

    Wfwd[0] = Winv[0] = 1;
    for (uint32_t i = 1; i < n / 2; ++i)
    {
        Wfwd[i] = montmul(Wfwd[i - 1], root);
        Winv[i] = montmul(Winv[i - 1], root_i);
    }
    W_log_max = logn;
    W_ready = 1;
}

// Vectorised butterfly (handles 4 butterflies at once)
// inv = 0 ⇒ forward  (a,b) → (a+w·b, a−w·b)
// inv = 1 ⇒ inverse  (a,b) → (a+b, (a−b)·w)
static inline void butterfly4(uint32_t *a, uint32_t *b,
                              uint32_t w, int inv)
{
    uint32x4_t va = vld1q_u32(a);
    uint32x4_t vb = vld1q_u32(b);
    uint32x4_t vQ = vdupq_n_u32(Q);
    uint32x4_t vw = vdupq_n_u32(w);

    // results
    uint32x4_t x, y;

    if (!inv)
    {
        /* t = w * b */
        uint64x2_t lo = vmull_u32(vget_low_u32(vb), vget_low_u32(vw));
        uint64x2_t hi = vmull_u32(vget_high_u32(vb), vget_high_u32(vw));
        uint32x4_t vt = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));

        x = vaddq_u32(va, vt);
        y = vsubq_u32(va, vt);
    }
    else
    {
        uint32x4_t vs = vsubq_u32(va, vb);
        uint64x2_t lo = vmull_u32(vget_low_u32(vs), vget_low_u32(vw));
        uint64x2_t hi = vmull_u32(vget_high_u32(vs), vget_high_u32(vw));
        uint32x4_t vt = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));

        x = vaddq_u32(va, vb); // a + b
        y = vt;
    }

    // conditional reduction (lazy but safe)
    uint32x4_t x_ge = vcgeq_u32(x, vQ);
    uint32x4_t y_ge = vcgeq_u32(y, vQ);
    x = vsubq_u32(x, vandq_u32(x_ge, vQ));
    y = vsubq_u32(y, vandq_u32(y_ge, vQ));

    vst1q_u32(a, x);
    vst1q_u32(b, y);
}

// Unified recursive NTT and iNTT
// inv = 0 ⇒ forward NTT
// inv = 1 ⇒ inverse NTT
static void ntt(uint32_t *a, unsigned logn, unsigned step, int inv)
{
    if (logn == 0)
        return;

    unsigned half = 1u << (logn - 1);

    if (!inv)
    {
        ntt(a, logn - 1, step * 2, inv);
        ntt(a + step, logn - 1, step * 2, inv);

        for (unsigned j = 0; j < half; j += 4)
            butterfly4(a + j * step,
                       a + (j + half) * step,
                       Wfwd[j], 0);
    }
    else
    {
        for (unsigned j = 0; j < half; j += 4)
            butterfly4(a + j * step,
                       a + (j + half) * step,
                       Winv[j], 1);

        ntt(a, logn - 1, step * 2, inv);
        ntt(a + step, logn - 1, step * 2, inv);
    }
}

//  Big-int multiplication via NTT
static void ntt32_vec(uint32_t *c,
                      const uint32_t *a,
                      const uint32_t *b,
                      unsigned m)
{
    unsigned logn = log2n(m) + 1; // m is expected to be a power of 2
    unsigned n = 2 * m;           // n = 2^logn ≥ m

    static uint32_t A[MAX_N], B[MAX_N]; // n ≤ 64 here

    for (uint32_t i = 0; i < n; ++i)
    {
        A[i] = (i < m) ? (a[i] & RADIX30) : 0;
        B[i] = (i < m) ? (b[i] & RADIX30) : 0;
    }

    ntt(A, logn, 1, 0);
    ntt(B, logn, 1, 0);

    for (unsigned i = 0; i < n; ++i)
        A[i] = montmul(A[i], B[i]); // pointwise mult

    ntt(A, logn, 1, 1); // inverse NTT

    uint32_t n_inv = pow_mod(1, Q - 1 - (Q - 1) / n); // n⁻¹ in Montgomery form
    for (unsigned i = 0; i < n; ++i)
        A[i] = montmul(A[i], n_inv);

    // carry propagation in radix-2³⁰
    uint64_t carry = 0;
    for (unsigned i = 0; i < 2 * m; ++i)
    {
        uint64_t t = carry + (i < n ? A[i] : 0);
        c[i] = (uint32_t)(t & RADIX30);
        carry = t >> 30;
    }
}

void bench_ntt(uint32_t *A, uint32_t *B)
{
    // gnereate random A and B

    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[LIMBS_NUM << 1] = {0};

    if (!is_power_of_2(LIMBS_NUM))
    {
        fprintf(stderr, "LIMBS_NUM must be a power of 2.\n");
        exit(EXIT_FAILURE);
    }

    unsigned logn = log2n(LIMBS_NUM) + 1;
    ntt_init(logn);

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            ntt32_vec(dst, A, B, LIMBS_NUM);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            ntt32_vec(dst, A, B, LIMBS_NUM);
        }
        t1 = get_cyclecounter();

        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("NTT_vec", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result("NTT multiplication", A, B, dst, LIMBS_NUM, 0);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    for (i = 0; i < LIMBS_NUM << 1; i++)
        dst[i] = 0; // zero the result
}
