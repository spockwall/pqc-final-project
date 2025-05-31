#pragma once
#include <stddef.h>
#include <stdint.h>

// Convenient NTT prime: p = 15·2²⁷ + 1 = 2 013 265 921
#define Q ((uint32_t)2013265921u)

// Q * QINV = = 4,294,967,295 = -1 mod 2³², for Montgomery
#define QINV ((uint32_t)2013265919u)

// primitive root modulo Q
#define G ((uint32_t)31u)

// R = 2^32 for Montgomery reduction
#define R ((uint64_t)1ull << 32)

#define RINV (uint32_t)(R % Q)

#define R2INV ((uint32_t)((uint64_t)RINV * RINV % Q))

void bench_ntt(const uint32_t *A, const uint32_t *B);
