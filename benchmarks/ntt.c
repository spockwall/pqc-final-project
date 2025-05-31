#include <arm_neon.h>
#include <inttypes.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hal.h"
#include "ntt.h"
#include "lib.h"
#include "ntt_helpers.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------
static uint32_t wtbl[N];  // forward roots  w^k * R
static uint32_t iwtbl[N]; // inverse roots  w^{‑k} * R
static uint32_t n_inv;    // n^{‑1} * R mod Q  (for final scaling)
static uint32_t mont_one; // 1 in Montgomery form

static void ntt_init(void)
{
    uint32_t w = mont_pow_mod(to_mont(G), (Q - 1U) / N);
    uint32_t iw = mont_pow_mod(w, Q - 2U); // w^{‑1}
    mont_one = to_mont(1U);

    wtbl[0] = iwtbl[0] = mont_one;
    for (size_t i = 1; i < N; ++i)
    {
        wtbl[i] = mont_mul(wtbl[i - 1], w);
        iwtbl[i] = mont_mul(iwtbl[i - 1], iw);
    }
    n_inv = mont_pow_mod(to_mont(N), Q - 2U); // N^{‑1} * R mod Q
}

// bit‑reversal permutation (in‑place)
static inline void bit_reverse(uint32_t *x)
{
    unsigned j = 0;
    for (unsigned i = 1; i < N; ++i)
    {
        unsigned bit = N >> 1;
        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
        {
            uint32_t tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

static void ntt(uint32_t *x, int invert)
{
    // ---------------- 主循環：同樣 m 由 N/2 → 1 ----------------
    for (unsigned m = N >> 1, step = 1; m; m >>= 1, step <<= 1)
    {
        const unsigned seg = m << 1; // 一段 = 2m
        uint32_t *wt = invert ? wtbl : iwtbl;
        for (unsigned s = 0; s < N; s += seg) // 每段各做一次蝴蝶輪
        {
            uint32_t w = mont_one; // 本段的 twiddle 起點
            for (unsigned i = 0; i < m; ++i)
            {
                unsigned j = s + i;  // 上半段索引
                unsigned j2 = j + m; // 下半段索引 (= j+m)

                uint32_t u = x[j];
                uint32_t v = x[j2];
                x[j] = add_mod(u, v);
                x[j2] = mont_mul(sub_mod(u, v), wt[step * i]);
            }
        }
    }

    bit_reverse(x);

    // ---------------- 反 NTT 時再乘 n^{-1} ----------------
    if (!invert)
        return;

    for (unsigned i = 0; i < N; ++i)
        x[i] = mont_mul(x[i], n_inv);
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt(fa, 0);
    ntt(fb, 0);
    for (unsigned i = 0; i < N; ++i)
        dst[i] = mont_mul(fa[i], fb[i]);
    ntt(dst, 1);

    for (unsigned i = 0; i < N; ++i)
        dst[i] = from_mont(dst[i]);

    // carry propagation in radix‑2^12
    for (unsigned i = 0; i < N; ++i)
    {
        dst[i + 1] += dst[i] >> BITS_PER_LIMB; // BITS_per_limb = 12
        dst[i] &= RADIX_MASK;
    }
}

void bench_ntt(const uint32_t *A, const uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t A_copy[N] = {0};
    uint32_t B_copy[N] = {0};
    uint32_t dst[N + 1] = {0};
    uint32_t fa[N] = {0};
    uint32_t fb[N] = {0};

    for (unsigned k = 0; k < N / 2; ++k)
    {
        A_copy[k] = to_mont(A[k]);
        B_copy[k] = to_mont(B[k]);
    }

    ntt_init();

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            memcpy(fa, A_copy, N * sizeof(uint32_t));
            memcpy(fb, B_copy, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            multiply(dst, fa, fb);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            memcpy(fa, A_copy, N * sizeof(uint32_t));
            memcpy(fb, B_copy, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            multiply(dst, fa, fb);
        }
        t1 = get_cyclecounter();
        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("Montgomery NTT", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result_ntt("Montgomery NTT multiplication", A, B, dst, LIMBS_NUM);
    printf("------------------------------------------\n\n");
#endif
}