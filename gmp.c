#include <gmp.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hal.h"
#include "lib.h"

static int bench_gmp()
{
    int i, j;
    int n = LIMBS_NUM * BITS_PER_LIMB;
    uint64_t t0, t1;
    mpz_t a, b, result;
    uint64_t cycles[NTESTS];

    // Initialize variables
    gmp_rand_operand_gen(a, n);
    gmp_rand_operand_gen(b, n);
    mpz_init(result); // Initialize result

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            // Multiply a and b, store in result
            mpz_mul(result, a, b);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            mpz_mul(result, a, b);
        }
        t1 = get_cyclecounter();

        cycles[i] = t1 - t0;
    }

    // Stdout result
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("mpz_mul", cycles);

    // Clear memory
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);

    return 0;
}

int main(void)
{
    enable_cyclecounter();
    bench_gmp();
    disable_cyclecounter();

    return 0;
}