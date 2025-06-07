#include <stddef.h>
#include <stdint.h>

void karatsuba32_vec(uint32_t *restrict dst, const uint32_t *restrict A, const uint32_t *restrict B, size_t n);