
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "fft.h"
#include "hal.h"

void fft(complex double *a, size_t n, int invert)
{
    if (n == 1)
        return;

    complex double *a0 = malloc(n / 2 * sizeof(complex double));
    complex double *a1 = malloc(n / 2 * sizeof(complex double));
    for (size_t i = 0; 2 * i < n; i++)
    {
        a0[i] = a[i * 2];
        a1[i] = a[i * 2 + 1];
    }

    fft(a0, n >> 1, invert);
    fft(a1, n >> 1, invert);

    for (size_t k = 0; 2 * k < n; k++)
    {
        double angle = 2 * M_PI * k / n * (invert ? -1 : 1);
        complex double w = cexp(I * angle);
        a[k] = a0[k] + w * a1[k];
        a[k + (n >> 1)] = a0[k] - w * a1[k];
    }

    free(a0);
    free(a1);
}

void fft_multiply(uint32_t *a, size_t na, uint32_t *b, size_t nb, uint32_t *result)
{
    size_t n = 1;
    while (n < na + nb)
        n <<= 1;

    complex double *fa = calloc(n, sizeof(complex double));
    complex double *fb = calloc(n, sizeof(complex double));

    for (size_t i = 0; i < na; i++)
        fa[i] = a[i];
    for (size_t i = 0; i < nb; i++)
        fb[i] = b[i];

    fft(fa, n, 0);
    fft(fb, n, 0);
    for (size_t i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, n, 1);

    // uint64_t *temp = calloc(n, sizeof(uint64_t));
    uint64_t *temp = calloc(n + 2, sizeof(uint64_t)); // 預留額外空間
    for (size_t i = 0; i < n; i++)
    {
        temp[i] = (uint64_t)round(creal(fa[i]) / n);
    }

    // carry propagation
    for (size_t i = 0; i < n; i++)
    {
        if (temp[i] >= BASE)
        {
            temp[i + 1] += temp[i] / BASE;
            temp[i] %= BASE;
        }
        result[i] = (uint32_t)temp[i];
    }

    free(fa);
    free(fb);
    free(temp);
}

void print_fft(uint32_t *a, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        printf("%08x ", a[i]);
    }
    printf("\n");
}

void fft_rand_operand_gen(uint32_t *output, int n_bits)
{
    // seed for rand

    int n_limbs = (n_bits + 31) / 32;
    for (int i = 0; i < n_limbs; i++)
    {
        output[i] = rand() % BASE;
    }
}

int test_fft(int n_bits)
{
    srand((unsigned int)time(NULL));
    uint32_t A[MAX_LIMBS] = {0}, B[MAX_LIMBS] = {0}, C[MAX_LIMBS * 2] = {0};

    fft_rand_operand_gen(A, n_bits);
    fft_rand_operand_gen(B, n_bits);
    fft_multiply(A, MAX_LIMBS, B, MAX_LIMBS, C);

    return 0;
}