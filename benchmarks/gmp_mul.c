#include <gmp.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "hal.h"
#include "lib.h"
#include "gmp_mul.h"

void bench_gmp(uint32_t *A, uint32_t *B)
{
    int i, j;
    // int n = LIMBS_NUM * BITS_PER_LIMB;
    uint64_t cycles[NTESTS];
    uint64_t t0, t1;

    // gmp initialization
    mpz_t a, b, result;
    mpz_init(a);
    mpz_init(b);
    mpz_init(result);
    mpz_import(a, LIMBS_NUM, -1, sizeof(uint32_t), 0, 0, A);
    mpz_import(b, LIMBS_NUM, -1, sizeof(uint32_t), 0, 0, B);

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

#ifdef VERBOSE
    printf("------------------------------------------\n");
    gmp_printf("A = %Zd\n", a);
    gmp_printf("B = %Zd\n", b);
    gmp_printf("Result = %Zd\n", result);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(result);
}