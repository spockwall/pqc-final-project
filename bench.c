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

    // generate random big integers A and B
    srand(42);
    generate_random_bigint(A, LIMBS_NUM * BITS_PER_LIMB);
    generate_random_bigint(B, LIMBS_NUM * BITS_PER_LIMB);

    enable_cyclecounter();

    // Karatsuba multiplication benchmark
    bench_karatsuba(A, B);

    // Naive NTT multiplication benchmark
    bench_naive_ntt(A, B);

    // NTT multiplication benchmark
    bench_ntt(A, B);

    // GMP multiplication benchmark
    bench_gmp(A, B);

    disable_cyclecounter();
    return 0;
}