#define LIMB_BITS 32
#define BASE ((uint64_t)1 << LIMB_BITS)
#define MAX_LIMBS 32 // 支援最多 1024-bit (32 * 32-bit)
#define M_PI 3.14159265358979323846

void fft(complex double *a, size_t n, int invert);

void fft_multiply(uint32_t *a, size_t na, uint32_t *b, size_t nb, uint32_t *result);

void fft_rand_operand_gen(uint32_t *output, int n_bits);

void print_fft(uint32_t *a, size_t n);

int test_fft(int n_bits);