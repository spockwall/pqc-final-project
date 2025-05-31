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
#include "ntt_vec_helpers.h"
#include "ntt_vec.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------
static uint32_t wtbl[N];  // forward roots  w^k * R
static uint32_t iwtbl[N]; // inverse roots  w^{‑k} * R
static uint32_t n_inv;    // n^{‑1} * R mod Q  (for final scaling)
static uint32_t mont_one; // 1 in Montgomery form

static uint32_t *w_cache_table;  // cache for Montgomery roots
static uint32_t *iw_cache_table; // cache for Montgomery inverse roots

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

static void ntt_vec(uint32_t *x, int invert)
{
    uint32_t cnt = 0;
    // m from N/2 to 1
    for (unsigned m = N >> 1, step = 1; m; m >>= 1, step <<= 1)
    {
        const unsigned seg = m << 1;
        uint32_t *wt = invert ? w_cache_table : iw_cache_table;
        for (unsigned s = 0; s < N; s += seg) // 每段各做一次蝴蝶輪
        {
            unsigned i = 0;
            for (; i + 3 < m; i += 4)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                // twiddle factors  w^{step*(i+k)}
                uint32x4_t w = vld1q_u32(&wt[cnt]);

                // load data
                uint32x4_t u = vld1q_u32(&x[j]);
                uint32x4_t v = vld1q_u32(&x[j2]);

                // butterfly
                uint32x4_t add = add_mod_q(u, v);
                uint32x4_t diff = sub_mod_q(u, v);
                uint32x4_t prod = mont_mul_vec(diff, w);

                vst1q_u32(&x[j], add);
                vst1q_u32(&x[j2], prod);
                cnt += 4; // 4 roots per iteration
            }

            // ---- tail ( ≤3 butterflies ) – scalar -----------------------
            for (; i < m; ++i)
            {
                unsigned j = s + i;
                unsigned j2 = j + m;

                uint32_t u = x[j];
                uint32_t v = x[j2];

                x[j] = add_mod(u, v);
                x[j2] = mont_mul(sub_mod(u, v), wt[cnt++]);
            }
        }
    }

    bit_reverse(x);

    if (!invert)
        return;

    // ---------------- inverse NTT: multiply n^{-1} ----------------
    uint32x4_t ninv = vdupq_n_u32(n_inv);
    unsigned i = 0;
    for (; i + 3 < N; i += 4)
    {
        uint32x4_t t = vld1q_u32(&x[i]);
        t = mont_mul_vec(t, ninv);
        vst1q_u32(&x[i], t);
    }
    for (; i < N; ++i)
        x[i] = mont_mul(x[i], n_inv);
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt_vec(fa, 0);
    ntt_vec(fb, 0);

    // Vectorised point-wise multiply
    unsigned i = 0;
    for (; i + 3 < N; i += 4)
    {
        uint32x4_t a = vld1q_u32(&fa[i]);
        uint32x4_t b = vld1q_u32(&fb[i]);
        uint32x4_t c = mont_mul_vec(a, b);
        vst1q_u32(&dst[i], c);
    }
    for (; i < N; ++i)
    {
        dst[i] = mont_mul(fa[i], fb[i]);
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

void bench_ntt_vec(uint32_t *A, uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[N + 1] = {0};
    uint32_t fa[N] = {0};
    uint32_t fb[N] = {0};
    w_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint32_t));
    iw_cache_table = malloc((BITS_PER_LIMB * LIMBS_NUM << 1) * sizeof(uint32_t));

    for (unsigned k = 0; k < N / 2; ++k)
    {
        A[k] = to_mont(A[k]);
        B[k] = to_mont(B[k]);
    }

    ntt_init();

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            memcpy(fa, A, N * sizeof(uint32_t));
            memcpy(fb, B, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            multiply(dst, fa, fb);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            memcpy(fa, A, N * sizeof(uint32_t));
            memcpy(fb, B, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            multiply(dst, fa, fb);
        }
        t1 = get_cyclecounter();
        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("Montgomery NTT VEC", cycles);
    print_big_hex(A, LIMBS_NUM);
    print_big_hex(B, LIMBS_NUM);
    print_big_hex(dst, N);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result("Montgomery NTT multiplication", A, B, dst, LIMBS_NUM, 0);
    printf("------------------------------------------\n");
#endif

    free(w_cache_table);
    free(iw_cache_table);
}