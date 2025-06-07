#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "hal.h"
#include "lib.h"
#include "karatsuba.h"
#include "../algorithms/karatsuba.h"

void bench_karatsuba(uint32_t *A, uint32_t *B)
{
    // gnereate random A and B
    if (LIMBS_NUM % 8 != 0)
    {
        fprintf(stderr, "Error: n must be a multiple of 8 limbs\n");
        exit(EXIT_FAILURE);
    }

    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[LIMBS_NUM << 1] = {0};

    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            karatsuba32_vec(dst, A, B, LIMBS_NUM);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            karatsuba32_vec(dst, A, B, LIMBS_NUM);
        }
        t1 = get_cyclecounter();

        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("karatsuba32_vec", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result("Karatsuba multiplication", A, B, dst, LIMBS_NUM, 0);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    for (i = 0; i < LIMBS_NUM << 1; i++)
        dst[i] = 0; // zero the result
}