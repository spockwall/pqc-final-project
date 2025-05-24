#include <gmp.h>
#include <stdint.h>

// benchmarks
#define NWARMUP 50
#define NITERATIONS 300
#define NTESTS 500

// Karatsuba parameters
#define BITS_PER_LIMB 32
#define BASE ((uint64_t)1 << BITS_PER_LIMB)
#define MAX_LIMBS_NUM 32 // 支援最多 1024-bit (32 * 32-bit)
#define LIMBS_NUM 16

void gmp_rand_operand_gen(mpz_t output, int n_bits);

void print_bigint_in_hex(uint32_t *a, size_t n);

void print_bigint_in_dec(const uint32_t *a, size_t n);

void generate_random_bigint(uint32_t *output, int n_bits);

int cmp_uint64_t(const void *a, const void *b);

// Benchmarking functions
void print_median(const char *txt, uint64_t cyc[NTESTS]);
void print_percentile_legend(void);
void print_percentiles(const char *txt, uint64_t cyc[NTESTS]);
