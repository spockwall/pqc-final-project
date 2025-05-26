#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "lib.h"

// compare two uint64_t values for qsort
int cmp_uint64_t(const void *a, const void *b)
{
    return (int)((*((const uint64_t *)a)) - (*((const uint64_t *)b)));
}

// generate a random operand with n_bits bits using GMP
void gmp_rand_operand_gen(mpz_t output, int n_bits)
{
    gmp_randstate_t state;
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, (unsigned long)time(NULL));

    mpz_init(output); // 初始化每個 mpz_t
    mpz_urandomb(output, state, n_bits);

    gmp_randclear(state);
}

// gmp_printf a big integer in either hexadecimal or decimal format
// 0: decimal format
// 1: hexadecimal format
void print_bigint(uint32_t *a, size_t n, int fmt)
{
    mpz_t z;
    mpz_init(z);

    /* mpz_import: limbs are least-significant first, 32-bit, native endian */
    //| `mpz_import` arg |                                                      |
    //| ---------------- | ---------------------------------------------------- |
    //| `order =  1`     | *most-significant* word first (big-endian array)     |
    //| `order = -1`     | *least-significant* word first (little-endian array) |
    mpz_import(z, n, -1,            /* count, order=LSW first    */
               sizeof(uint32_t), 0, /* size, endian = native     */
               0,                   /* nails                     */
               a);                  /* source pointer            */

    if (fmt == 0)
        gmp_printf("%Zd\n", z); // print in decimal format
    else
        gmp_printf("%ZX\n", z); // print in hexadecimal format
    mpz_clear(z);
}

// generate a random big integer with n_bits bits
void generate_random_bigint(uint32_t *output, int n_bits)
{
    int n_limbs = (n_bits + 31) / 32;
    for (int i = 0; i < n_limbs; i++)
    {
        output[i] = rand() % BASE;
        output[i] &= RADIX30; // ensure each limb is in the range [0, 2^30)
    }
}

// print the median of the benchmarking results
static void print_median(const char *txt, uint64_t cyc[NTESTS])
{
    printf("%10s cycles = %" PRIu64 "\n", txt, cyc[NTESTS >> 1] / NITERATIONS);
}

int percentiles[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99};

// print the legend for percentiles of the benchmarking results
static void print_percentile_legend(void)
{
    unsigned i;
    printf("%21s", "percentile");
    for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
    {
        printf("%7d", percentiles[i]);
    }
    printf("\n");
}

// print the percentiles of the benchmarking results
static void print_percentiles(const char *txt, uint64_t cyc[NTESTS])
{
    unsigned i;
    printf("%10s percentiles:", txt);
    for (i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); i++)
    {
        printf("%7" PRIu64, (cyc)[NTESTS * percentiles[i] / 100] / NITERATIONS);
    }
    printf("\n");
}

void print_benchmark_results(const char *txt, uint64_t cycles[NTESTS])
{
    print_median(txt, cycles);
    printf("\n");
    print_percentile_legend();
    print_percentiles(txt, cycles);
}

// gmp_printf computation result in either hexadecimal or decimal format
// 0: decimal format
// 1: hexadecimal format
void print_computation_result(const char *txt, uint32_t *A, uint32_t *B, uint32_t *dst, size_t n_limbs, int fmt)
{
    printf("%s:\n", txt);
    printf("A = ");
    print_bigint(A, n_limbs, fmt);
    printf("B = ");
    print_bigint(B, n_limbs, fmt);
    printf("Result =");
    print_bigint(dst, n_limbs << 1, fmt);
}