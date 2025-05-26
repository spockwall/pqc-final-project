#include <stdint.h>

// ---------- mod-arithmetic --------------------------
static inline uint32_t add_mod(uint32_t a, uint32_t b);
static inline uint32_t sub_mod(uint32_t a, uint32_t b);
static inline uint32_t montmul(uint32_t a, uint32_t b);

// 32×32-bit Montgomery multiply, branch-free
static inline uint32_t pow_mod(uint32_t g_, uint32_t e);

// check an uint32_t value is a power of 2
static inline int is_power_of_2(uint32_t x);

// compute log n
static inline unsigned log2n(uint32_t n);