#include <arm_neon.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hal.h"
#include "ntt.h"
#include "lib.h"
#include "ntt_helpers.h"
#include "ntt_vec_helpers.h"
#include "ntt_vec.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------

// Group all NTT data in a single cache-aligned structure
typedef struct
{
    uint32_t wtbl[N] __attribute__((aligned(64)));
    uint32_t iwtbl[N] __attribute__((aligned(64)));
    uint32_t brt_table[N] __attribute__((aligned(64)));
    uint32_t n_inv;
    uint32_t mont_one;
    uint32_t brt_table_len;
    uint8_t padding[64 - (3 * sizeof(uint32_t)) % 64]; // Align to cache line
} ntt_ctx_t __attribute__((aligned(64)));

static ntt_ctx_t ntt_ctx;

static void ntt_init(void)
{
    uint32_t w = mont_pow_mod(to_mont(G), (Q - 1U) / N);
    uint32_t iw = mont_pow_mod(w, Q - 2U); // w^{‑1} * R mod Q

    ntt_ctx.wtbl[0] = ntt_ctx.iwtbl[0] = ntt_ctx.mont_one = to_mont(1U);
    for (size_t i = 1; i < N; ++i)
    {
        ntt_ctx.wtbl[i] = mont_mul(ntt_ctx.wtbl[i - 1], w);
        ntt_ctx.iwtbl[i] = mont_mul(ntt_ctx.iwtbl[i - 1], iw);
    }
    ntt_ctx.n_inv = mont_pow_mod(to_mont(N), Q - 2U); // N^{‑1} * R mod Q

    // cache bit reverse table
    memset(ntt_ctx.brt_table, 0, sizeof(ntt_ctx.brt_table));
    for (unsigned i = 1, j = 0; i < N; ++i)
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
            ntt_ctx.brt_table[ntt_ctx.brt_table_len++] = i;
            ntt_ctx.brt_table[ntt_ctx.brt_table_len++] = j;
        }
    }
}

// bit‑reversal permutation (in‑place)
static inline void bit_reverse(uint32_t *x)
{
    for (unsigned i = 0; i < ntt_ctx.brt_table_len; i += 2)
    {
        unsigned j = ntt_ctx.brt_table[i];
        unsigned k = ntt_ctx.brt_table[i + 1];
        if (j < k)
        {
            uint32_t t = x[j];
            x[j] = x[k];
            x[k] = t;
        }
    }
}

static void ntt_vec(uint32_t *x, int invert)
{
    //  m from N/2 to 1
    for (unsigned m = N >> 1, step = 1; m; m >>= 1, step <<= 1)
    {
        const unsigned seg = m << 1;
        uint32_t *wt = invert ? ntt_ctx.wtbl : ntt_ctx.iwtbl;
#pragma omp parallel for schedule(static)
        for (unsigned s = 0; s < N; s += seg) // 每段各做一次蝴蝶輪
        {
            if (s + seg < N)
            {
                __builtin_prefetch(&x[s + seg], 1, 3);
                __builtin_prefetch(&x[s + seg + m], 1, 3);
            }
            unsigned i = 0;
            for (; i + 7 < m; i += 8)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                // Load twiddle factors for 8 butterflies
                uint32x4_t w1 = {wt[step * i], wt[step * (i + 1)],
                                 wt[step * (i + 2)], wt[step * (i + 3)]};
                uint32x4_t w2 = {wt[step * (i + 4)], wt[step * (i + 5)],
                                 wt[step * (i + 6)], wt[step * (i + 7)]};

                // Load data (8 elements each)
                uint32x4_t u1 = vld1q_u32(&x[j]);
                uint32x4_t u2 = vld1q_u32(&x[j + 4]);
                uint32x4_t v1 = vld1q_u32(&x[j2]);
                uint32x4_t v2 = vld1q_u32(&x[j2 + 4]);

                // Butterfly operations
                uint32x4_t add1 = add_mod_q(u1, v1);
                uint32x4_t add2 = add_mod_q(u2, v2);
                uint32x4_t diff1 = sub_mod_q(u1, v1);
                uint32x4_t diff2 = sub_mod_q(u2, v2);

                uint32x4_t prod1 = mont_mul_vec(diff1, w1);
                uint32x4_t prod2 = mont_mul_vec(diff2, w2);

                // Store results
                vst1q_u32(&x[j], add1);
                vst1q_u32(&x[j + 4], add2);
                vst1q_u32(&x[j2], prod1);
                vst1q_u32(&x[j2 + 4], prod2);
            }
            // ---- tail ( ≤3 butterflies ) – scalar -----------------------
            for (; i < m; ++i)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                uint32_t u = x[j];
                uint32_t v = x[j2];

                x[j] = add_mod(u, v);
                x[j2] = mont_mul(sub_mod(u, v), wt[step * i]);
            }
        }
    }
    bit_reverse(x);

    if (!invert)
        return;

    // Inverse NTT: multiply n^{-1}, and  N is a multiple of 4
    uint32x4_t ninv = vdupq_n_u32(ntt_ctx.n_inv);
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < N; i += 8)
    {
        // Process 8 elements at once
        uint32x4_t t1 = vld1q_u32(&x[i]);
        uint32x4_t t2 = vld1q_u32(&x[i + 4]);

        t1 = mont_mul_vec(t1, ninv);
        t2 = mont_mul_vec(t2, ninv);

        vst1q_u32(&x[i], t1);
        vst1q_u32(&x[i + 4], t2);
    }
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt_vec(fa, 0);
    ntt_vec(fb, 0);

    // Vectorised point-wise multiply, N is a multiple of 4
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < N; i += 8)
    {
        // Prefetch next cache line
        __builtin_prefetch(&fa[i + 8], 0, 3);
        __builtin_prefetch(&fb[i + 8], 0, 3);
        __builtin_prefetch(&dst[i + 8], 1, 3);

        uint32x4_t a1 = vld1q_u32(&fa[i]);
        uint32x4_t a2 = vld1q_u32(&fa[i + 4]);
        uint32x4_t b1 = vld1q_u32(&fb[i]);
        uint32x4_t b2 = vld1q_u32(&fb[i + 4]);

        uint32x4_t c1 = mont_mul_vec(a1, b1);
        uint32x4_t c2 = mont_mul_vec(a2, b2);

        vst1q_u32(&dst[i], c1);
        vst1q_u32(&dst[i + 4], c2);
    }
    ntt_vec(dst, 1);

    for (unsigned k = 0; k < N; ++k)
        dst[k] = from_mont(dst[k]);

    // carry propagation in radix‑2^12
    for (unsigned k = 0; k < N; ++k)
    {
        dst[k + 1] += dst[k] >> BITS_PER_LIMB; // BITS_per_limb = 12
        dst[k] &= RADIX_MASK;
    }
}

void bench_ntt_vec(const uint32_t *A, const uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t A_copy[N] = {0};
    uint32_t B_copy[N] = {0};
    uint32_t fa[N] = {0};
    uint32_t fb[N] = {0};
    uint32_t dst[N + 1] = {0};

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
    print_benchmark_results("Montgomery NTT VEC", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result_ntt("Montgomery NTT multiplication", A, B, dst, LIMBS_NUM);
    printf("------------------------------------------\n");
#endif

    // free(w_cache_table);
    // free(iw_cache_table);
}