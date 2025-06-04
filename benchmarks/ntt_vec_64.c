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
// #include "ntt_helpers.h"
// #include "ntt_vec_helpers.h"
#include "ntt_vec_64.h"
#include "ntt_vec_64_helpers.h"

static uint64_t wtbl[N];  // forward roots  w^k * R
static uint64_t iwtbl[N]; // inverse roots  w^{‑k} * R
static uint64_t n_inv;    // n^{‑1} * R mod Q  (for final scaling)
static uint64_t mont_one; // 1 in Montgomery form

static uint64_t brt_table[N];
static uint64_t *w_cache_table;  // cache for Montgomery roots
static uint64_t *iw_cache_table; // cache for Montgomery inverse roots
static uint64_t brt_table_len = 0;

static void ntt_init(void)
{
    uint64_t w = mont_pow_mod(to_mont(G), (Q_64 - 1U) / N);
    uint64_t iw = mont_pow_mod(w, Q_64 - 2U); // w^{‑1} * R mod Q

    wtbl[0] = iwtbl[0] = mont_one = to_mont(1U);
    for (size_t i = 1; i < N; ++i)
    {
        wtbl[i] = mont_mul(wtbl[i - 1], w);
        iwtbl[i] = mont_mul(iwtbl[i - 1], iw);
    }
    n_inv = mont_pow_mod(to_mont(N), Q_64 - 2U); // N^{‑1} * R mod Q

    // cache Montgomery roots, make it serial to avoid cache misses
    int cnt = 0;
    for (unsigned m = N >> 1, step = 1; m; m >>= 1, step <<= 1)
    {
        const unsigned seg = m << 1;
        for (unsigned s = 0; s < N; s += seg)
        {
            for (unsigned i = 0; i < m; ++i)
            {
                w_cache_table[cnt] = wtbl[step * i];
                iw_cache_table[cnt++] = iwtbl[step * i];
            }
        }
    }

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
static inline void bit_reverse(uint64_t *x)
{
    for (unsigned i = 0; i < brt_table_len; i += 2)
    {
        unsigned j = brt_table[i];
        unsigned k = brt_table[i + 1];
        if (j < k)
        {
            uint64_t t = x[j];
            x[j] = x[k];
            x[k] = t;
        }
    }
}

static void ntt_vec(uint64_t *x, int invert)
{
    int cnt = 0;
    // m from N/2 to 1
    for (unsigned m = N >> 1, step = 1; m; m >>= 1, step <<= 1)
    {
        const unsigned seg = m << 1;
        uint64_t *wt = invert ? w_cache_table : iw_cache_table;
        for (unsigned s = 0; s < N; s += seg) // 每段各做一次蝴蝶輪
        {
            unsigned i = 0;
            for (; i + 2 < m; i += 2)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                // twiddle factors  w^{step*(i+k)}
                uint64x2_t w = vld1q_u64(&wt[cnt]);

                // load data
                uint64x2_t u = vld1q_u64(&x[j]);
                uint64x2_t v = vld1q_u64(&x[j2]);

                // butterfly
                uint64x2_t add = add_mod_vec64(u, v);
                uint64x2_t diff = sub_mod_vec64(u, v);
                uint64x2_t prod = mont_mul_vec64(diff, w);

                vst1q_u64(&x[j], add);
                vst1q_u64(&x[j2], prod);
                cnt += 2;
            }

            // ---- tail ( ≤3 butterflies ) – scalar -----------------------
            for (; i < m; ++i)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                uint64_t u = x[j];
                uint64_t v = x[j2];

                x[j] = add_mod(u, v);
                x[j2] = mont_mul(sub_mod(u, v), wt[cnt++]);
            }
        }
    }
    bit_reverse(x);

    if (!invert)
        return;

    // Inverse NTT: multiply n^{-1}, and  N is a multiple of 4
    uint64x2_t ninv = vdupq_n_u64(n_inv);
    for (unsigned i = 0; i + 2 < N; i += 2)
    {
        uint64x2_t t = vld1q_u64(&x[i]);
        t = mont_mul_vec64(t, ninv);
        vst1q_u64(&x[i], t);
    }
}

static void multiply(uint64_t *dst, uint64_t *fa, uint64_t *fb)
{
    ntt_vec(fa, 0);
    ntt_vec(fb, 0);

    // Vectorised point-wise multiply, N is a multiple of 4
    for (unsigned i = 0; i + 2 < N; i += 2)
    {
        uint64x2_t a = vld1q_u64(&fa[i]);
        uint64x2_t b = vld1q_u64(&fb[i]);
        uint64x2_t c = mont_mul_vec64(a, b);
        vst1q_u64(&dst[i], c);
    }
    ntt_vec(dst, 1);

    for (unsigned k = 0; k < N; ++k)
        dst[k] = from_mont(dst[k]);

    // carry propagation in radix‑2^12
    for (unsigned k = 0; k < N; ++k)
    {
        dst[k + 1] += dst[k] >> 24;          // BITS_per_limb = 24
        dst[k] &= ((uint64_t)(1 << 24) - 1); // 0x00ffffff
    }
}

void bench_ntt_vec_64(const uint64_t *A, const uint64_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint64_t A_copy[N] = {0};
    uint64_t B_copy[N] = {0};
    uint64_t fa[N] = {0};
    uint64_t fb[N] = {0};
    uint64_t dst[N + 1] = {0};

    w_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint64_t));
    iw_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint64_t));

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
            memcpy(fa, A_copy, N * sizeof(uint64_t));
            memcpy(fb, B_copy, N * sizeof(uint64_t));
            memset(dst, 0, (N + 1) * sizeof(uint64_t));
            multiply(dst, fa, fb);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            memcpy(fa, A_copy, N * sizeof(uint64_t));
            memcpy(fb, B_copy, N * sizeof(uint64_t));
            memset(dst, 0, (N + 1) * sizeof(uint64_t));
            multiply(dst, fa, fb);
        }
        t1 = get_cyclecounter();
        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("Montgomery ntt vec u64", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result_ntt_u64("Montgomery NTT multiplication", A, B, dst, LIMBS_NUM);
    printf("------------------------------------------\n");
#endif

    free(w_cache_table);
    free(iw_cache_table);
}