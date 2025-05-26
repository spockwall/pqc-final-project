#include <stddef.h>
#include <stdint.h>

// Convenient NTT prime: p = 15·2²⁷ + 1 = 2 013 265 921
#define Q ((uint32_t)2013265921u)

// * −p⁻¹ mod 2³², for Montgomery
#define QINV ((uint32_t)1837919009u)

// primitive root modulo Q
#define G ((uint32_t)31u)

void bench_ntt(uint32_t *A, uint32_t *B);
