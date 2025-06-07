#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "hal.h"
#include "lib.h"
#include "ntt_naive.h"
#include "../algorithms/ntt_naive.h"

void bench_ntt_naive(const uint32_t *A, const uint32_t *B)
{
    // gnereate random A and B
    int i, j;
    uint64_t t0, t1;
    uint64_t cycles[NTESTS];
    uint32_t dst[N + 1] = {0};
    uint32_t fa[N] = {0};
    uint32_t fb[N] = {0};
    ntt_naive_init();
    for (i = 0; i < NTESTS; i++)
    {
        for (j = 0; j < NWARMUP; j++)
        {
            memcpy(fa, A, N * sizeof(uint32_t));
            memcpy(fb, B, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            ntt_naive_multiply(dst, fa, fb);
        }

        t0 = get_cyclecounter();

        for (j = 0; j < NITERATIONS; j++)
        {
            memcpy(fa, A, N * sizeof(uint32_t));
            memcpy(fb, B, N * sizeof(uint32_t));
            memset(dst, 0, (N + 1) * sizeof(uint32_t));
            ntt_naive_multiply(dst, fa, fb);
        }
        t1 = get_cyclecounter();
        cycles[i] = t1 - t0;
    }
    qsort(cycles, NTESTS, sizeof(uint64_t), cmp_uint64_t);
    print_benchmark_results("Naive NTT_vec", cycles);

#ifdef VERBOSE
    printf("------------------------------------------\n");
    print_computation_result_ntt("Naive NTT multiplication", A, B, dst, LIMBS_NUM);
    printf("------------------------------------------\n");
#endif

    // Clear memory
    for (i = 0; i < LIMBS_NUM << 1; i++)
        dst[i] = 0; // zero the result
}