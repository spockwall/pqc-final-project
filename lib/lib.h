#include <gmp.h>
#include <stdint.h>

// benchmarks
#define NWARMUP 50
#define NITERATIONS 300
#define NTESTS 500

// big integer parameters
#define BITS_PER_LIMB 12 // bits in a single limb
#define BASE ((uint64_t)1 << BITS_PER_LIMB)
#define MAX_LIMBS_NUM 2048 // support up to 65536-bits (2048 * 32-bit)
#define N (LIMBS_NUM << 1) // NTT length, degree of polynomial

/*   Every limb holds only the *low* 30 bits, i.e.   0 ≤ a[i] < 2³⁰.
 *   The mask we need after additions/subtractions is therefore           */
#define RADIX30 ((uint32_t)0x3fffffff)
#define RADIX_MASK ((uint32_t)(BASE - 1)) // 0xfff if BITS_PER_LIMB = 12

#ifndef LIMBS_NUM
#define LIMBS_NUM 16
#endif

void gmp_rand_operand_gen(mpz_t output, int n_bits);

void print_bigint(uint32_t *a, size_t n, int fmt);

void generate_random_bigint(uint32_t *output, int n_bits);

// Benchmarking functions
int cmp_uint64_t(const void *a, const void *b);
void print_benchmark_results(const char *txt, uint64_t cycles[NTESTS]);
void print_computation_result(const char *txt, uint32_t *A, uint32_t *B, uint32_t *dst, size_t n_limbs, int fmt);

void print_big_hex(const uint32_t *x, unsigned limbs);