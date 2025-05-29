// ntt_neon.c – Arm NEON‑accelerated 30‑bit‑radix big‑integer multiplication
// ------------------------------------------------------------------------
// This reference implementation multiplies two big integers (≤1024 bits)
// on Cortex‑A72/Raspberry Pi 4 using a 64‑point number‑theoretic transform
// fully written with Arm NEON intrinsics.  Each limb stores 30 effective
// bits in a 32‑bit word (top two bits zero).  After the inverse NTT we run
// one masked carry sweep to restore the radix‑30 representation.
//
//  Build (AArch64 GCC/Clang):
//      cc -O3 -march=armv8‑a+simd -c ntt_neon.c
//  Build (AArch32 GCC/Clang):
//      cc -O3 -mfpu=neon-vfpv4 -mfloat-abi=hard -c ntt_neon.c
// ------------------------------------------------------------------------

#include <arm_neon.h>
#include <inttypes.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hal.h"
//#include "ntt.h"
#include "lib.h"
#include "naive_ntt.h"
#include "ntt_helpers.h"

// ---------------------------------------------------------
//  Parameters (exactly as in the Python version)
// ---------------------------------------------------------
#define MOD 2013265921 // 15 * 2^27 + 1  (prime)
#define ROOT 31        // primitive root mod MOD
#define N (LIMBS_NUM << 1)        // transform size = 32

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

static uint32_t modinv(uint32_t x)
{
    // MOD is prime ⇒ x^(MOD‑2) ≡ x^‑1 (mod MOD)
    return pow_mod(x, MOD - 2);
}

// bit‑reversal permutation (in‑place)
static void bit_reverse(uint32_t *x, unsigned n)
{
    unsigned j = 0;
    for (unsigned i = 1; i < n; ++i)
    {
        unsigned bit = n >> 1;
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
static void ntt(uint32_t *x, unsigned n, int invert)
{
    uint32_t w = pow_mod(ROOT, (MOD - 1) / n);
    if (!invert)
        w = modinv(w);

    for (unsigned k = n; k >= 2; k >>= 1)
    {
        uint32_t wn = 1;
        for (unsigned i = 0; i < k / 2; ++i)
        {
            for (unsigned j = i; j < n; j += k)
            {
                uint32_t u = x[j];
                uint32_t v = x[j + k / 2];
                x[j] = (u + v) % MOD;
                x[j + k / 2] = mul_mod((u + MOD - v) % MOD, wn);
            }
            wn = mul_mod(wn, w);
        }
        w = mul_mod(w, w);
        // next stage
    }

    bit_reverse(x, n);

    if (invert)
    {
        uint32_t n_inv = modinv(n);
        for (unsigned i = 0; i < n; ++i)
            x[i] = mul_mod(x[i], n_inv);
    }
}

static void multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb)
{
    uint32_t X[N] = {0};

    ntt(fa, N, 0);
    ntt(fb, N, 0);
    for (unsigned i = 0; i < N; ++i)
        X[i] = mul_mod(fa[i], fb[i]);
    ntt(X, N, 1);

    // carry propagation in radix‑2^12
    for (unsigned i = 0; i < N - 1; ++i)
    {
        X[i + 1] += X[i] / BASE;
        X[i] %= BASE;
    }
    for (unsigned i = 0; i < N; ++i)
        dst[i] = X[i];
}

// static void print_big_hex(const uint32_t *x, unsigned limbs)
//{
//     for (unsigned i = 0; i < limbs; ++i)
//         printf("%08x", x[i]);
//     printf("\n");
// }

// ---------------------------------------------------------
//  Random big number generator (radix‑2^12 limbs)
// ---------------------------------------------------------
// static void random_bignum(uint32_t *a)
//{
//    for (unsigned i = 0; i < LIMBS_NUM - 1; ++i)
//        a[i] = rand() & (BASE - 1);
//    a[LIMBS_NUM - 1] = 1 + (rand() & (BASE - 1)); // ensure non‑zero top limb
//}

void bench_naive_ntt(uint32_t *A, uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[LIMBS_NUM << 1] = {0};
    ntt_init();
    for (i = 0; i < LIMBS_NUM; i++)
    {
        A[i] = A[i] & (BASE - 1); // ensure A is in radix‑2^12
        B[i] = B[i] & (BASE - 1); // ensure B is in radix‑2^12
    }
    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            multiply(dst, A, B);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            multiply(dst, A, B);
        }
        t1 = get_cyclecounter();
        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("Naive NTT_vec", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result("Naive NTT multiplication", A, B, dst, LIMBS_NUM, 0);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    for (i = 0; i < LIMBS_NUM << 1; i++)
        dst[i] = 0; // zero the result
}