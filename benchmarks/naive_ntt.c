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
#include "naive_ntt.h"
#include "ntt_helpers.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------
static uint32_t wtbl[N];  // forward roots  w^k
static uint32_t iwtbl[N]; // inverse roots  w^{‑k}
static uint32_t inv_len;  // n^{‑1} mod Q  (for final scaling)

static void ntt_init(void)
{
    uint32_t w = pow_mod(G, (Q - 1U) / N);
    uint32_t iw = pow_mod(w, Q - 2U); // w^{‑1}
    wtbl[0] = iwtbl[0] = 1U;
    for (size_t i = 1; i < N; ++i)
    {
        wtbl[i] = mul_mod(wtbl[i - 1], w);
        iwtbl[i] = mul_mod(iwtbl[i - 1], iw);
    }
    inv_len = pow_mod(N, Q - 2U);
}

// bit‑reversal permutation (in‑place)
static void bit_reverse(uint32_t *x)
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

// ---------------------------------------------------------
//  Iterative radix‑2 NTT (Cooley–Tukey, same as Python code)
// ---------------------------------------------------------
static void ntt(uint32_t *x, int invert)
{
    uint32_t w = pow_mod(G, (Q - 1) / N);
    if (!invert)
        w = modinv(w);

    for (unsigned k = N; k >= 2; k >>= 1)
    {
        uint32_t wn = 1;
        for (unsigned i = 0; i < k / 2; ++i)
        {
            for (unsigned j = i; j < N; j += k)
            {
                uint32_t u = x[j];
                uint32_t v = x[j + k / 2];
                x[j] = add_mod(u, v);
                x[j + k / 2] = mul_mod(sub_mod(u, v), wn);
            }
            wn = mul_mod(wn, w);
        }
        w = mul_mod(w, w);
    }

    bit_reverse(x);

    if (!invert)
        return;

    uint32_t n_inv = modinv(N);
    for (unsigned i = 0; i < N; ++i)
        x[i] = mul_mod(x[i], n_inv);
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt(fa, 0);
    ntt(fb, 0);
    for (unsigned i = 0; i < N; ++i)
        dst[i] = mul_mod(fa[i], fb[i]);
    ntt(dst, 1);

    // carry propagation in radix‑2^12
    for (unsigned i = 0; i < N - 1; ++i)
    {
        dst[i + 1] += dst[i] >> BITS_PER_LIMB; // BITS_per_limb = 12
        dst[i] &= RADIX_MASK;
    }
}

void bench_naive_ntt(const uint32_t *A, const uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[N + 1] = {0};
    uint32_t fa[N] = {0};
    uint32_t fb[N] = {0};
    // ntt_init();
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
    print_benchmark_results("Naive NTT_vec", cycles);
    print_big_hex(A, LIMBS_NUM);
    print_big_hex(B, LIMBS_NUM);
    print_big_hex(dst, N);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result("Naive NTT multiplication", A, B, dst, LIMBS_NUM, 0);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    for (i = 0; i < LIMBS_NUM << 1; i++)
        dst[i] = 0; // zero the result
}