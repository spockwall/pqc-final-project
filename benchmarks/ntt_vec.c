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
#include <omp.h>
#include "ntt_helpers.h"
#include "ntt_vec_helpers.h"
#include "ntt_vec.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------
static uint32_t wtbl[N];  // forward roots  w^k * R
static uint32_t iwtbl[N]; // inverse roots  w^{‑k} * R
static uint32_t n_inv;    // n^{‑1} * R mod Q  (for final scaling)
static uint32_t mont_one; // 1 in Montgomery form

static uint32_t brt_table[N];
static uint32_t brt_table_len = 0;

static void ntt_init(void)
{
    uint32_t w = mont_pow_mod(to_mont(G), (Q - 1U) / N);
    uint32_t iw = mont_pow_mod(w, Q - 2U); // w^{‑1} * R mod Q

    wtbl[0] = iwtbl[0] = mont_one = to_mont(1U);
    for (size_t i = 1; i < N; ++i)
    {
        wtbl[i] = mont_mul(wtbl[i - 1], w);
        iwtbl[i] = mont_mul(iwtbl[i - 1], iw);
    }
    n_inv = mont_pow_mod(to_mont(N), Q - 2U); // N^{‑1} * R mod Q

    // cache bit reverse table
    memset(brt_table, 0, sizeof(brt_table));
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
            brt_table[brt_table_len++] = i;
            brt_table[brt_table_len++] = j;
        }
    }
}

// bit‑reversal permutation (in‑place)
static inline void bit_reverse(uint32_t *x)
{
    for (unsigned i = 0; i < brt_table_len; i += 2)
    {
        unsigned j = brt_table[i];
        unsigned k = brt_table[i + 1];
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
        uint32_t *wt = invert ? wtbl : iwtbl;
#pragma omp parallel for schedule(static)
        for (unsigned s = 0; s < N; s += seg) // 每段各做一次蝴蝶輪
        {
            unsigned i = 0;
            for (; i + 3 < m; i += 4)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                // twiddle factors  w^{step*(i+k)}
                uint32x4_t w = {wt[step * i], wt[step * (i + 1)],
                                wt[step * (i + 2)], wt[step * (i + 3)]};

                // load data
                uint32x4_t u = vld1q_u32(&x[j]);
                uint32x4_t v = vld1q_u32(&x[j2]);

                // butterfly
                uint32x4_t add = add_mod_q(u, v);
                uint32x4_t diff = sub_mod_q(u, v);
                uint32x4_t prod = mont_mul_vec(diff, w);

                vst1q_u32(&x[j], add);
                vst1q_u32(&x[j2], prod);
                // cnt += 4; // 4 roots per iteration
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
    uint32x4_t ninv = vdupq_n_u32(n_inv);
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < N; i += 4)
    {
        uint32x4_t t = vld1q_u32(&x[i]);
        t = mont_mul_vec(t, ninv);
        vst1q_u32(&x[i], t);
    }
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt_vec(fa, 0);
    ntt_vec(fb, 0);

    // Vectorised point-wise multiply, N is a multiple of 4
#pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < N; i += 4)
    {
        uint32x4_t a = vld1q_u32(&fa[i]);
        uint32x4_t b = vld1q_u32(&fb[i]);
        uint32x4_t c = mont_mul_vec(a, b);
        vst1q_u32(&dst[i], c);
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

    // w_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint32_t));
    // iw_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint32_t));

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