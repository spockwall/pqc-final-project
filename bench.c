#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "hal.h"
#include "lib.h"
#include "benchmarks/ntt_helpers.h"
#include "benchmarks/gmp_mul.h"
#include "benchmarks/karatsuba.h"
#include "benchmarks/ntt.h"
#include "benchmarks/naive_ntt.h"
#include "benchmarks/ntt_vec.h"

int main()
{

    if (LIMBS_NUM % 8 != 0 || LIMBS_NUM < 1 || LIMBS_NUM > MAX_LIMBS_NUM)
    {
        fprintf(stderr, "LIMBS_NUM must be a multiple of 8 and between 1 and %d.\n", MAX_LIMBS_NUM);
        return 1;
    }
    // operand size in bits
    uint32_t A[LIMBS_NUM << 1] = {0};
    uint32_t B[LIMBS_NUM << 1] = {0};
    uint64_t A_u64[LIMBS_NUM << 1] = {0};
    uint64_t B_u64[LIMBS_NUM << 1] = {0};

    // generate random big integers A and B
    srand(42);

    enable_cyclecounter();

    // ----big integers with 32 bits data per limbs-----------
    generate_random_bigint(A, LIMBS_NUM * 32, 0);
    generate_random_bigint(B, LIMBS_NUM * 32, 0);

    // GMP multiplication benchmark
    bench_gmp(A, B);

    //// Karatsuba multiplication benchmark
    bench_karatsuba(A, B);

    // -----------big integers with masked limbs-------------
    generate_random_bigint(A, LIMBS_NUM * BITS_PER_LIMB, 1);
    generate_random_bigint(B, LIMBS_NUM * BITS_PER_LIMB, 1);

    // Naive NTT multiplication benchmark
    bench_naive_ntt(A, B);

    // NTT multiplication benchmark
    bench_ntt(A, B);

    // NTT multiplication using arm neon benchmark
    bench_ntt_vec(A, B);

    disable_cyclecounter();
    return 0;
}