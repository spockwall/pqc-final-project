#include <stdint.h>

// ---------- mod-arithmetic --------------------------
static inline uint32_t add_mod(uint32_t a, uint32_t b);
static inline uint32_t sub_mod(uint32_t a, uint32_t b);
static inline uint32_t montmul(uint32_t a, uint32_t b);

// 32×32-bit Montgomery multiply, branch-free
static uint32_t pow_mod(uint32_t g_, uint32_t e);