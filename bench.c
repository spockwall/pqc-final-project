#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "hal.h"
#include "lib.h"
#include "benchmarks/gmp_mul.h"
#include "benchmarks/karatsuba.h"

int main()
{
    // operand size in bits
    uint32_t A[LIMBS_NUM] = {0};
    uint32_t B[LIMBS_NUM] = {0};

    // generate random big integers A and B
    srand((unsigned int)time(NULL));
    generate_random_bigint(A, LIMBS_NUM * BITS_PER_LIMB);
    generate_random_bigint(B, LIMBS_NUM * BITS_PER_LIMB);

    // GMP multiplication benchmark
    enable_cyclecounter();
    bench_gmp(A, B);
    disable_cyclecounter();

    // Karatsuba multiplication benchmark
    enable_cyclecounter();
    bench_karatsuba(A, B);
    disable_cyclecounter();
    return 0;
}