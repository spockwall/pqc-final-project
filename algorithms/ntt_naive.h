#include <stddef.h>
#include <stdint.h>

void ntt_naive_init(void);

void ntt_naive_multiply(uint32_t *dst, uint32_t *fa, uint32_t *fb);