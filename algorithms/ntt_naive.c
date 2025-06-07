#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hal.h"
#include "lib.h"
#include "ntt_naive.h"
#include "ntt_helpers.h"

//--------------------------------------------------------------------------
// Root tables – we need n = 64 for ≤32 limbs (2× size after convolution)
//--------------------------------------------------------------------------
static uint32_t w_tbl[N];  // forward roots  w^k
static uint32_t i_wtbl[N]; // inverse roots  w^{‑k}
static uint32_t n_inv;     // n^{‑1} mod Q  (for final scaling)

void ntt_naive_init(void)
{
    uint32_t w = pow_mod(G, (Q - 1U) / N);
    uint32_t iw = pow_mod(w, Q - 2U); // w^{‑1}
    w_tbl[0] = i_wtbl[0] = 1U;
    for (size_t i = 1; i < N; ++i)
    {
        w_tbl[i] = mul_mod(w_tbl[i - 1], w);
        i_wtbl[i] = mul_mod(i_wtbl[i - 1], iw);
    }
    n_inv = pow_mod(N, Q - 2U); // N^{‑1} mod Q
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

// ---------------------------------------------------------
//  Iterative radix‑2 NTT (Cooley–Tukey, same as Python code)
// ---------------------------------------------------------
static void ntt(uint32_t *x, int invert)
{
    for (unsigned step = 1, k = N; k >= 2; k >>= 1, step <<= 1)
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
            wn = invert ? w_tbl[step * (i + 1) % N] : i_wtbl[step * (i + 1) % N];
        }
    }
    bit_reverse(x);

    if (!invert)
        return;

    for (unsigned i = 0; i < N; ++i)
        x[i] = mul_mod(x[i], n_inv);
}

void ntt_naive_multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    ntt(fa, 0);
    ntt(fb, 0);
    for (unsigned i = 0; i < N; ++i)
        dst[i] = mul_mod(fa[i], fb[i]);
    ntt(dst, 1);
    // carry propagation in radix‑2^12
    for (unsigned i = 0; i < N; ++i)
    {
        dst[i + 1] += dst[i] >> BITS_PER_LIMB; // BITS_per_limb = 12
        dst[i] &= RADIX_MASK;
    }
}
